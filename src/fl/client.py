import flwr as fl
import torch
from collections import OrderedDict
from losses import HuberSMAPELoss
from src.models.lstm import LightweightLSTM
from src.fl.privacy import get_noisy_feature_importance

SHARED_LAYER_PREFIXES = ("lstm.",)
HEAD_LAYER_PREFIXES = ("fc.",)


class FedStockClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid,
        train_loader,
        val_loader,
        X_train,
        y_train,
        input_size,
        hidden_size=32,
        epsilon=10.0,
        learning_rate=0.001,
        y_scaler=None,
    ):
        self.cid = cid
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.X_train = X_train
        self.y_train = y_train
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.y_scaler = y_scaler
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LightweightLSTM(input_size=input_size, hidden_size=hidden_size).to(self.device)
        self.criterion = HuberSMAPELoss(target_scaler=y_scaler).to(self.device)

    def get_parameters(self, config):
        # Extract weights from PyTorch model to numpy arrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def _parameter_keys(self):
        return list(self.model.state_dict().keys())

    def select_parameters(self, parameters, prefixes):
        return [
            value
            for key, value in zip(self._parameter_keys(), parameters)
            if key.startswith(prefixes)
        ]

    def get_shared_parameters(self):
        return self.select_parameters(self.get_parameters({}), SHARED_LAYER_PREFIXES)

    def set_parameters(self, parameters):
        # Load weights from numpy arrays to PyTorch model
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def set_shared_parameters(self, parameters):
        state_dict = self.model.state_dict()
        shared_keys = [key for key in state_dict if key.startswith(SHARED_LAYER_PREFIXES)]
        for key, value in zip(shared_keys, parameters):
            state_dict[key] = torch.tensor(value)
        self.model.load_state_dict(state_dict, strict=True)

    def _set_trainable_layers(self, prefixes):
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = name.startswith(prefixes)

    def _reset_trainable_layers(self):
        for parameter in self.model.parameters():
            parameter.requires_grad = True

    def _make_optimizer(self):
        return torch.optim.Adam(
            (param for param in self.model.parameters() if param.requires_grad),
            lr=self.learning_rate,
        )

    def _train_epochs(self, epochs):
        self.model.train()
        optimizer = self._make_optimizer()
        for _ in range(epochs):
            for batch_X, batch_y in self.train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

    def fit(self, parameters, config):
        # Apply weights from server
        self.set_parameters(parameters)
        epochs = config.get("epochs", 5)
        self._reset_trainable_layers()
        self._train_epochs(epochs)

        # Return updated weights and number of samples
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def fit_shared_lstm(self, parameters, config):
        # Apply shared LSTM weights while preserving this client's local output head.
        self.set_shared_parameters(parameters)
        epochs = config.get("epochs", 5)
        self._reset_trainable_layers()
        self._train_epochs(epochs)
        return self.get_shared_parameters(), len(self.train_loader.dataset), {}

    def fit_head(self, parameters, config):
        # Fine-tune only the personalized output head on top of shared LSTM weights.
        self.set_shared_parameters(parameters)
        epochs = config.get("epochs", 1)
        self._set_trainable_layers(HEAD_LAYER_PREFIXES)
        self._train_epochs(epochs)
        self._reset_trainable_layers()
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        # Apply weights from server
        self.set_parameters(parameters)
        return self._evaluate_current()

    def evaluate_shared_lstm(self, parameters, config):
        # Evaluate with shared LSTM weights and this client's local output head.
        self.set_shared_parameters(parameters)
        return self._evaluate_current()

    def _evaluate_current(self):
        # Evaluate locally
        self.model.eval()
        total_loss = 0.0
        y_true_list = []
        y_pred_list = []
        
        with torch.no_grad():
            for batch_X, batch_y in self.val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item() * batch_X.size(0)
                
                y_true_list.append(batch_y.cpu().numpy())
                y_pred_list.append(outputs.cpu().numpy())
                
        avg_loss = total_loss / len(self.val_loader.dataset)
        
        # Calculate SMAPE and RMSE (simplified)
        import numpy as np
        y_true = np.concatenate(y_true_list)
        y_pred = np.concatenate(y_pred_list)
        
        if self.y_scaler is not None:
            y_true = self.y_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
            y_pred = self.y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
        
        metrics = {"rmse": float(rmse), "smape": float(smape)}
        return float(avg_loss), len(self.val_loader.dataset), metrics

    def extract_noisy_importance(self):
        """
        Custom method for Step 1 of PA-CFL: Return noisy feature importance.
        """
        # Flatten time series features for XGBoost
        # Reshape X_train: (samples, seq_len, features) -> (samples, seq_len * features)
        n_samples = self.X_train.shape[0]
        X_flat = self.X_train.reshape(n_samples, -1)
        noisy_importance, _ = get_noisy_feature_importance(X_flat, self.y_train, epsilon=self.epsilon)
        return noisy_importance
