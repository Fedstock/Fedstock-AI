import os
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.dataset import load_client_data
from src.fl.client import FedStockClient
from src.fl.server import BubbleServer

def setup_client(client_id, data_dir, seq_len=1):
    """
    Load real data for a client, split into train/val, and initialize FedStockClient.
    """
    print(f"Loading data for {client_id}...")
    X_scaled, y_raw, scaler = load_client_data(client_id, data_dir=data_dir)
    
    # Split into Train (80%) and Val (20%)
    split_idx = int(len(X_scaled) * 0.8)
    
    # Fast test: Use subset if dataset is too large (e.g. 5M rows -> 100k rows)
    max_samples = 100000 
    if len(X_scaled) > max_samples:
        X_scaled = X_scaled[-max_samples:]
        y_raw = y_raw[-max_samples:]
        split_idx = int(len(X_scaled) * 0.8)
        
    X_train, y_train = X_scaled[:split_idx], y_raw[:split_idx]
    X_val, y_val = X_scaled[split_idx:], y_raw[split_idx:]
    
    # Create DataLoaders
    # Note: X needs to be (batch, seq_len, features) for LSTM. 
    # Current X_scaled is (samples, features). We add seq_len=1 dimension.
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=256, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=256, shuffle=False)
    
    # Input size is number of features
    input_size = X_train.shape[1]
    
    client = FedStockClient(
        cid=client_id,
        train_loader=train_loader,
        val_loader=val_loader,
        X_train=X_train_tensor.numpy(),
        y_train=y_train_tensor.numpy(),
        input_size=input_size,
        hidden_size=32,
        epsilon=1.0
    )
    return client

def main():
    print("=== Starting PA-CFL Real Data Execution ===")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "src", "fedstock_data", "outputs", "clients")
    clustering_json = os.path.join(current_dir, "outputs", "clustering_results.json")
    
    # 1. Initialize Clients
    # Using the 10 clients from the dataset
    client_ids = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]
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
