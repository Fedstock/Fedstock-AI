import flwr as fl
import torch
from collections import OrderedDict
from src.models.lstm import LightweightLSTM
from src.fl.privacy import get_noisy_feature_importance

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
        epsilon=1.0,
        learning_rate=0.001,
    ):
        self.cid = cid
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.X_train = X_train
        self.y_train = y_train
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LightweightLSTM(input_size=input_size, hidden_size=hidden_size).to(self.device)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def get_parameters(self, config):
        # Extract weights from PyTorch model to numpy arrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        # Load weights from numpy arrays to PyTorch model
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # Apply weights from server
        self.set_parameters(parameters)
        
        # Train locally
        self.model.train()
        epochs = config.get("epochs", 5)
        for epoch in range(epochs):
            for batch_X, batch_y in self.train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
        # Return updated weights and number of samples
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        # Apply weights from server
        self.set_parameters(parameters)
        
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
