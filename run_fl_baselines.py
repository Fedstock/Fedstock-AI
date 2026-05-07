import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
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
    
    # Use subset for faster evaluation but large enough
    max_samples = 5000 
    if len(X_scaled) > max_samples:
        X_scaled = X_scaled[-max_samples:]
        y_raw = y_raw[-max_samples:]
        
    split_idx = int(len(X_scaled) * 0.8)
    
    X_train, y_train_raw = X_scaled[:split_idx], y_raw[:split_idx]
    X_val, y_val_raw = X_scaled[split_idx:], y_raw[split_idx:]

    # Fit the target scaler on train only to reduce outlier impact without leaking validation targets.
    y_scaler = RobustScaler()
    y_train = y_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
    y_val = y_scaler.transform(y_val_raw.reshape(-1, 1)).flatten()
    
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
        epsilon=10.0,
        y_scaler=y_scaler
    )
    return client

def aggregate_metrics(history_list):
    if not history_list:
        return 0.0, 0.0
    total_samples = sum(h["num_samples"] for h in history_list)
    avg_rmse = sum(h["rmse"] * h["num_samples"] for h in history_list) / total_samples
    avg_smape = sum(h["smape"] * h["num_samples"] for h in history_list) / total_samples
    return avg_rmse, avg_smape

def main():
    print("=== Starting PA-CFL Evaluation Pipeline ===")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "src", "fedstock_data", "data", "clients")
    
    # 1. Initialize Clients
    client_ids = []
    if os.path.exists(data_dir):
        for f in os.listdir(data_dir):
            if f.startswith("client_") and f.endswith(".db"):
                cid = f.replace("client_", "").replace(".db", "")
                client_ids.append(cid)
    client_ids.sort()
    
    # Use all clients!
    print(f"Found {len(client_ids)} clients in dataset.")
    if len(client_ids) == 0:
        print("No clients found. Exiting.")
        return

    clients_dict = {}
    for cid in client_ids:
        clients_dict[cid] = setup_client(cid, data_dir)
        
    # Hyperparameters
    num_rounds = 10
    epochs_per_round = 2
    
    # We will evaluate 3 strategies
    results = {}
    
    # --- Strategy 1: Local Training ---
    print("\n\n=== Strategy 1: Local Training ===")
    local_clients = {cid: copy.deepcopy(c) for cid, c in clients_dict.items()}
    local_server = BubbleServer(local_clients)
    local_server.isolated = list(local_clients.keys())
    local_server.bubbles = []
    # step_4 acts as local training for all clients
    local_history = local_server.step_4_personalized_learning(epochs=num_rounds * epochs_per_round)
    local_rmse, local_smape = aggregate_metrics(local_history)
    results["Local"] = {"rmse": local_rmse, "smape": local_smape}
    print(f"[Local] RMSE: {local_rmse:.4f}, SMAPE: {local_smape:.4f}")
    
    # --- Strategy 2: Global FedAvg ---
    print("\n\n=== Strategy 2: Global FedAvg ===")
    fedavg_clients = {cid: copy.deepcopy(c) for cid, c in clients_dict.items()}
    fedavg_server = BubbleServer(fedavg_clients)
    fedavg_server.bubbles = [list(fedavg_clients.keys())]
    fedavg_server.isolated = []
    fedavg_history = fedavg_server.step_3_federated_learning(num_rounds=num_rounds, epochs_per_round=epochs_per_round)
    # The last round's global history contains the evaluation across all clients in the bubble
    last_round_fedavg = [h for h in fedavg_history if h["round"] == num_rounds]
    fedavg_rmse, fedavg_smape = aggregate_metrics(last_round_fedavg)
    results["Global FedAvg"] = {"rmse": fedavg_rmse, "smape": fedavg_smape}
    print(f"[Global FedAvg] RMSE: {fedavg_rmse:.4f}, SMAPE: {fedavg_smape:.4f}")

    # --- Strategy 3: PA-CFL ---
    print("\n\n=== Strategy 3: PA-CFL ===")
    pacfl_clients = {cid: copy.deepcopy(c) for cid, c in clients_dict.items()}
    pacfl_server = BubbleServer(pacfl_clients)
    pacfl_server.step_1_collect_and_cluster()
    pacfl_fed_history = pacfl_server.step_3_federated_learning(
        num_rounds=num_rounds,
        epochs_per_round=epochs_per_round,
        global_warmup_rounds=1,
        head_finetune_epochs=1,
        personalize_head=True,
    )
    pacfl_pers_history = pacfl_server.step_4_personalized_learning(epochs=num_rounds * epochs_per_round)
    
    # Prefer personalized head metrics when available; otherwise use last shared-LSTM round.
    pacfl_head_metrics = [h for h in pacfl_fed_history if h["stage"] == "head_finetune"]
    pacfl_bubble_metrics = pacfl_head_metrics or [h for h in pacfl_fed_history if h["round"] == num_rounds]
    pacfl_final_metrics = pacfl_bubble_metrics + pacfl_pers_history
    pacfl_rmse, pacfl_smape = aggregate_metrics(pacfl_final_metrics)
    results["PA-CFL"] = {"rmse": pacfl_rmse, "smape": pacfl_smape}
    print(f"[PA-CFL] RMSE: {pacfl_rmse:.4f}, SMAPE: {pacfl_smape:.4f}")
    
    # Visualization
    print("\n=== Generating Visualizations ===")
    strategies = list(results.keys())
    rmses = [results[s]["rmse"] for s in strategies]
    smapes = [results[s]["smape"] for s in strategies]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(strategies))
    width = 0.35
    
    rects1 = ax1.bar(x - width/2, rmses, width, label='RMSE', color='skyblue')
    ax1.set_ylabel('RMSE', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')
    
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, smapes, width, label='SMAPE', color='salmon')
    ax2.set_ylabel('SMAPE', color='salmon')
    ax2.tick_params(axis='y', labelcolor='salmon')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies)
    ax1.set_title('Comparison of FL Strategies (70 Clients)')
    
    # Add values on top of bars
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1, ax1)
    autolabel(rects2, ax2)
    
    fig.tight_layout()
    output_png = os.path.join(current_dir, "outputs", "baseline_comparison.png")
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png)
    print(f"Visualization saved to {output_png}")
    
    # Save a report
    report_md = os.path.join(current_dir, "outputs", "evaluation_report.md")
    with open(report_md, "w") as f:
        f.write("# Federated Learning Strategies Evaluation Report\n")
        f.write("## Overview\n")
        f.write(f"- **Total Clients:** {len(client_ids)}\n")
        f.write(f"- **Rounds:** {num_rounds}\n")
        f.write(f"- **Epochs per round:** {epochs_per_round}\n")
        f.write("\n## Results\n")
        for s in strategies:
            f.write(f"- **{s}**: RMSE = {results[s]['rmse']:.4f}, SMAPE = {results[s]['smape']:.4f}\n")

if __name__ == "__main__":
    main()
