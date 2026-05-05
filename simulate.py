import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from src.fl.client import FedStockClient
from src.fl.server import BubbleServer

def generate_synthetic_data(num_samples=500, seq_len=14, num_features=8, pattern_type=0):
    """
    Generate synthetic sequence data for demand forecasting.
    Different pattern_types simulate data heterogeneity.
    """
    X = []
    y = []
    
    # Base pattern
    time = np.arange(num_samples + seq_len)
    if pattern_type == 0:
        # Pattern 0: Sine wave + noise
        base_series = np.sin(time * 0.1) * 10 + 20 + np.random.normal(0, 2, len(time))
    elif pattern_type == 1:
        # Pattern 1: Cosine wave + linear trend
        base_series = np.cos(time * 0.2) * 5 + time * 0.05 + 10 + np.random.normal(0, 1, len(time))
    else:
        # Pattern 2: High variance random walk
        base_series = np.cumsum(np.random.normal(0, 1, len(time))) + 30
        
    for i in range(num_samples):
        # The features are variations of the base series
        seq = np.zeros((seq_len, num_features))
        for f in range(num_features):
            seq[:, f] = base_series[i:i+seq_len] + np.random.normal(0, 0.5, seq_len)
            
        target = base_series[i+seq_len]
        X.append(seq)
        y.append([target])
        
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    # Split train/val
    split_idx = int(num_samples * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=32, shuffle=False)
    
    return train_loader, val_loader, X_train, y_train

def main():
    print("Setting up PA-CFL Simulation...")
    num_clients = 6
    seq_len = 14
    num_features = 8
    
    clients_dict = {}
    
    # Create clients with heterogeneous data
    # 2 clients of type 0, 3 clients of type 1, 1 client of type 2 (isolated)
    pattern_assignments = [0, 0, 1, 1, 1, 2]
    
    for i in range(num_clients):
        cid = f"Client_{i}"
        train_loader, val_loader, X_train, y_train = generate_synthetic_data(
            num_samples=300, 
            seq_len=seq_len, 
            num_features=num_features, 
            pattern_type=pattern_assignments[i]
        )
        
        client = FedStockClient(
            cid=cid,
            train_loader=train_loader,
            val_loader=val_loader,
            X_train=X_train,
            y_train=y_train,
            input_size=num_features,
            hidden_size=32,
            epsilon=10.0 # moderate privacy
        )
        clients_dict[cid] = client
        print(f"Created {cid} with data pattern {pattern_assignments[i]}")
        
    server = BubbleServer(clients_dict)
    
    # Run pipeline
    server.step_1_collect_and_cluster()
    server.step_3_federated_learning(num_rounds=3, epochs_per_round=5)

if __name__ == "__main__":
    main()
