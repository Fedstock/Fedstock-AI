import os
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

from src.dataset import load_client_data
from src.fl.client import FedStockClient
from src.fl.server import BubbleServer

def create_sequences(X, y, seq_len):
    xs, ys = [], []
    if len(X) <= seq_len:
        return np.array(xs), np.array(ys)
    for i in range(len(X) - seq_len):
        xs.append(X[i:(i + seq_len)])
        ys.append(y[i + seq_len])
    return np.array(xs), np.array(ys)

def setup_client(client_id, data_dir, seq_len=14):
    """
    Load real data for a client, split into train/val, and initialize FedStockClient.
    """
    print(f"Loading data for {client_id}...")
    X_scaled, y_raw, scaler = load_client_data(client_id, data_dir=data_dir)
    
    # Scale Target (y_raw)
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y_raw.reshape(-1, 1)).flatten()
    
    # Fast test: Use subset if dataset is too large
    max_samples = 10000 
    if len(X_scaled) > max_samples:
        X_scaled = X_scaled[-max_samples:]
        y_scaled = y_scaled[-max_samples:]
        
    split_idx = int(len(X_scaled) * 0.8)
    
    X_train, y_train = X_scaled[:split_idx], y_scaled[:split_idx]
    X_val, y_val = X_scaled[split_idx:], y_scaled[split_idx:]
    
    # Apply Sliding Window
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_len)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_len)
    
    # Create DataLoaders
    if len(X_train_seq) == 0:
        # Fallback if extremely small dataset
        X_train_seq, y_train_seq = np.zeros((1, seq_len, X_train.shape[1])), np.zeros((1,))
        X_val_seq, y_val_seq = np.zeros((1, seq_len, X_val.shape[1])), np.zeros((1,))
        
    X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32).unsqueeze(-1)
    X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_seq, dtype=torch.float32).unsqueeze(-1)
    
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=256, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=256, shuffle=False)
    
    # Input size is number of features
    input_size = X_train.shape[1]
    
    client = FedStockClient(
        cid=client_id,
        train_loader=train_loader,
        val_loader=val_loader,
        X_train=X_train_seq,
        y_train=y_train_seq,
        input_size=input_size,
        hidden_size=32,
        epsilon=1.0,
        y_scaler=y_scaler
    )
    return client

def main():
    print("=== Starting PA-CFL Real Data Execution ===")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "src", "fedstock_data", "data", "clients")
    clustering_json = os.path.join(current_dir, "outputs", "clustering_results.json")
    
    # 1. Initialize Clients
    # Dynamically find clients from the dataset directory (.db files)
    client_ids = []
    if os.path.exists(data_dir):
        for f in os.listdir(data_dir):
            if f.startswith("client_") and f.endswith(".db"):
                cid = f.replace("client_", "").replace(".db", "")
                client_ids.append(cid)
    client_ids.sort()
    
    # For quick testing, we might limit the number of clients
    client_ids = client_ids[:20]  # Just use 20 clients for faster evaluation
    
    print(f"Found {len(client_ids)} clients in dataset.")
    clients_dict = {}
    
    for cid in client_ids:
        clients_dict[cid] = setup_client(cid, data_dir)
        
    # 2. Initialize Server
    server = BubbleServer(clients_dict)
    
    # 3. Load pre-computed clustering
    if os.path.exists(clustering_json):
        server.load_clustering_results(clustering_json)
    else:
        print("Clustering JSON not found. Falling back to calculate from scratch.")
        server.step_1_collect_and_cluster()
        
    # 4. Federated Learning for multi-client bubbles
    server.step_3_federated_learning(num_rounds=3, epochs_per_round=1)
    
    # 5. Personalized Learning for isolated clients (e.g., CA_2)
    server.step_4_personalized_learning(epochs=3)

if __name__ == "__main__":
    main()
