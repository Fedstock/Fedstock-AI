import json
import os
import platform
import subprocess
import sys
import tempfile
import flwr as fl
import torch
from collections import OrderedDict
from src.losses import HuberSMAPELoss
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
        target_transform="identity",
    ):
        self.cid = cid
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.X_train = X_train
        self.y_train = y_train
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.y_scaler = y_scaler
        self.target_transform = target_transform

        self.device = self._select_device()
        self.model = LightweightLSTM(input_size=input_size, hidden_size=hidden_size).to(self.device)
        self.criterion = HuberSMAPELoss(
            target_scaler=y_scaler,
            target_transform=target_transform,
        ).to(self.device)

    @staticmethod
    def _select_device():
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

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

    def _make_optimizer(self, lr):
        return torch.optim.Adam(
            (param for param in self.model.parameters() if param.requires_grad),
            lr=lr,
            weight_decay=1e-4,
        )

    def _train_epochs(self, epochs, current_round=None, total_rounds=None):
        self.model.train()
        
        # Calculate decayed learning rate based on current round
        lr = self.learning_rate
        if current_round is not None and total_rounds is not None and total_rounds > 0:
            import math
            # Cosine decay of the base learning rate across rounds (down to 10% of base LR)
            eta_min = self.learning_rate * 0.1
            lr = eta_min + 0.5 * (self.learning_rate - eta_min) * (1 + math.cos(math.pi * (current_round - 1) / total_rounds))
            
        optimizer = self._make_optimizer(lr)
        
        # Local scheduler within epochs (e.g. CosineAnnealingLR)
        scheduler = None
        if epochs > 1:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.5)
            
        for _ in range(epochs):
            for batch_X, batch_y in self.train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

    def fit(self, parameters, config):
        # Apply weights from server
        self.set_parameters(parameters)
        epochs = config.get("epochs", 5)
        current_round = config.get("current_round")
        total_rounds = config.get("total_rounds")
        self._reset_trainable_layers()
        self._train_epochs(epochs, current_round, total_rounds)

        # Return updated weights and number of samples
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def fit_shared_lstm(self, parameters, config):
        # Apply shared LSTM weights while preserving this client's local output head.
        self.set_shared_parameters(parameters)
        epochs = config.get("epochs", 5)
        current_round = config.get("current_round")
        total_rounds = config.get("total_rounds")
        self._reset_trainable_layers()
        self._train_epochs(epochs, current_round, total_rounds)
        return self.get_shared_parameters(), len(self.train_loader.dataset), {}

    def fit_head(self, parameters, config):
        # Fine-tune the personalized output head and micro-tune shared LSTM weights.
        self.set_shared_parameters(parameters)
        epochs = config.get("epochs", 1)
        current_round = config.get("current_round")
        total_rounds = config.get("total_rounds")
        
        self._reset_trainable_layers()
        
        # Calculate base decayed learning rate
        lr = self.learning_rate
        if current_round is not None and total_rounds is not None and total_rounds > 0:
            import math
            eta_min = self.learning_rate * 0.1
            lr = eta_min + 0.5 * (self.learning_rate - eta_min) * (1 + math.cos(math.pi * (current_round - 1) / total_rounds))
            
        # Create parameter groups: 10% learning rate for LSTM backbone, 100% for FC head
        lstm_params = [p for n, p in self.model.named_parameters() if n.startswith(SHARED_LAYER_PREFIXES)]
        head_params = [p for n, p in self.model.named_parameters() if n.startswith(HEAD_LAYER_PREFIXES)]
        
        optimizer = torch.optim.Adam([
            {'params': lstm_params, 'lr': lr * 0.1},
            {'params': head_params, 'lr': lr}
        ], weight_decay=1e-4)
        
        scheduler = None
        if epochs > 1:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.5)
            
        self.model.train()
        for _ in range(epochs):
            for batch_X, batch_y in self.train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
                
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
        
        # Calculate Metrics
        import numpy as np
        y_true = np.concatenate(y_true_list)
        y_pred = np.concatenate(y_pred_list)

        y_true = self.inverse_target_array(y_true)
        y_pred = self.inverse_target_array(y_pred)
        
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
        wmape = np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-8)
        
        y_train_unscaled = self.inverse_target_array(self.y_train.copy())
            
        if len(y_train_unscaled) > 1:
            naive_mae = np.mean(np.abs(np.diff(y_train_unscaled)))
            naive_mae = max(naive_mae, 1e-8)
        else:
            naive_mae = 1e-8
            
        mase = mae / naive_mae
        
        metrics = {
            "rmse": float(rmse), 
            "smape": float(smape), 
            "mae": float(mae), 
            "wmape": float(wmape), 
            "mase": float(mase)
        }
        return float(avg_loss), len(self.val_loader.dataset), metrics

    def inverse_target_array(self, values):
        import numpy as np

        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if self.y_scaler is not None:
            values = self.y_scaler.inverse_transform(values.reshape(-1, 1)).flatten()
        if self.target_transform == "log1p":
            values = np.expm1(values)
            values = np.clip(values, a_min=0.0, a_max=None)
        return values

    def extract_noisy_importance(self):
        """
        Custom method for Step 1 of PA-CFL: Return noisy feature importance.
        """
        if platform.system() == "Darwin":
            return self._extract_noisy_importance_via_subprocess()

        # Flatten time series features for XGBoost
        # Reshape X_train: (samples, seq_len, features) -> (samples, seq_len * features)
        n_samples = self.X_train.shape[0]
        seq_len = self.X_train.shape[1]
        num_features = self.X_train.shape[2]
        X_flat = self.X_train.reshape(n_samples, -1)
        noisy_importance, _ = get_noisy_feature_importance(
            X_flat, self.y_train, epsilon=self.epsilon, seq_len=seq_len, num_features=num_features
        )
        return noisy_importance

    def _extract_noisy_importance_via_subprocess(self):
        worker_path = os.path.join(os.path.dirname(__file__), "extract_importance_worker.py")
        env = os.environ.copy()
        # Keep the XGBoost worker strictly single-threaded on macOS to reduce
        # native crashes from oversubscription in Apple Silicon environments.
        env.update(
            {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )

        with tempfile.TemporaryDirectory(prefix=f"fedstock_{self.cid}_") as tmpdir:
            payload_path = os.path.join(tmpdir, "payload.npz")
            output_path = os.path.join(tmpdir, "importance.json")

            np_payload = {
                "X_train": self.X_train.astype("float32"),
                "y_train": self.y_train.astype("float32"),
                "epsilon": [float(self.epsilon)],
            }
            import numpy as np
            np.savez_compressed(payload_path, **np_payload)

            completed = subprocess.run(
                [
                    sys.executable,
                    worker_path,
                    "--input-npz",
                    payload_path,
                    "--output-json",
                    output_path,
                ],
                capture_output=True,
                text=True,
                env=env,
            )

            if completed.returncode != 0:
                stderr = completed.stderr.strip()
                stdout = completed.stdout.strip()
                details = stderr or stdout or f"exit code {completed.returncode}"
                raise RuntimeError(
                    f"Feature importance extraction failed for client {self.cid}: {details}"
                )

            with open(output_path, "r") as f:
                payload = json.load(f)
            import numpy as np
            return np.asarray(payload["importance"], dtype=np.float32)
