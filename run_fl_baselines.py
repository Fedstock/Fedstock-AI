import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import copy
import csv
import json
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.data import TensorDataset, DataLoader

from src.dataset import CANDIDATE_FEATURE_COLS, load_client_data, make_group_time_split_indices
from src.fl.client import FedStockClient
from src.fl.extract_features import compute_anova_feature_selection, save_feature_selection
from src.fl.server import BubbleServer

def seed_everything(seed=42):
    """
    재현성을 보장하기 위해 모든 난수 시드를 고정하고 결정론적(deterministic) 연산을 설정합니다.
    random, numpy, torch, CUDA 및 Laplace RNG(np.random.laplace 등)의 
    동작을 명시적으로 제어하여 동일한 환경에서 동일한 결과를 얻도록 합니다.
    """
    # 1. Python 기본 난수 제어
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2. Numpy 난수 제어 (Laplace RNG 등 np.random 기반 함수들에 적용)
    np.random.seed(seed)

    # 3. PyTorch 난수 제어
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 4. CUDA 결정론적 연산 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 5. PyTorch 결정론적 알고리즘 강제 (CUDA >= 10.2 환경 변수 포함)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)


def create_sequences(X, y, seq_len):
    xs, ys = [], []
    if len(X) <= seq_len:
        return np.array(xs), np.array(ys)
    for i in range(len(X) - seq_len):
        xs.append(X[i:(i + seq_len)])
        ys.append(y[i + seq_len])
    return np.array(xs), np.array(ys)

def create_grouped_sequences(X, y, item_ids, indices, seq_len):
    xs, ys = [], []
    indices = np.asarray(indices, dtype=int)
    if len(indices) == 0:
        return np.array(xs), np.array(ys)

    split_item_ids = item_ids[indices]
    for item_id in np.unique(split_item_ids):
        group_indices = indices[split_item_ids == item_id]
        if len(group_indices) <= seq_len:
            continue
        for i in range(len(group_indices) - seq_len):
            xs.append(X[group_indices[i:(i + seq_len)]])
            ys.append(y[group_indices[i + seq_len]])

    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def _make_loader(X_seq, y_seq, batch_size=1024, shuffle=False):
    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
    y_tensor = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(-1)
    return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle)


def _write_json(data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


def _write_csv(rows, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not rows:
        with open(output_path, "w", newline="") as f:
            f.write("")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def setup_client(
    client_id,
    data_dir,
    seq_len=14,
    selected_features=None,
    train_ratio=0.7,
    val_ratio=0.15,
):
    """
    Load real data for a client and initialize FedStockClient.

    Preprocessing is leakage-safe:
    - rows are split chronologically inside each item_id;
    - X scaler is fit on train rows only;
    - y scaler is fit on train rows only;
    - LSTM windows never cross item_id or split boundaries;
    - final strategy metrics use the held-out test split.
    """
    print(f"Loading data for {client_id}...")
    X_raw, y_raw, _, metadata = load_client_data(
        client_id,
        data_dir=data_dir,
        feature_cols=selected_features,
        scale=False,
        return_metadata=True,
    )

    item_ids = np.asarray(metadata["item_id"])
    split_indices = make_group_time_split_indices(
        item_ids,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    train_idx = split_indices["train"]
    val_idx = split_indices["val"]
    test_idx = split_indices["test"]

    if len(train_idx) == 0:
        raise ValueError(f"No train rows available for client {client_id}")

    x_scaler = StandardScaler()
    X_scaled = np.empty_like(X_raw, dtype=np.float32)
    X_scaled[train_idx] = x_scaler.fit_transform(X_raw[train_idx]).astype(np.float32)
    if len(val_idx) > 0:
        X_scaled[val_idx] = x_scaler.transform(X_raw[val_idx]).astype(np.float32)
    if len(test_idx) > 0:
        X_scaled[test_idx] = x_scaler.transform(X_raw[test_idx]).astype(np.float32)

    # Fit the target scaler on train only to reduce outlier impact without leaking validation targets.
    y_scaler = RobustScaler()
    y_scaled = np.empty_like(y_raw, dtype=np.float32)
    y_scaled[train_idx] = y_scaler.fit_transform(y_raw[train_idx].reshape(-1, 1)).flatten()
    if len(val_idx) > 0:
        y_scaled[val_idx] = y_scaler.transform(y_raw[val_idx].reshape(-1, 1)).flatten()
    if len(test_idx) > 0:
        y_scaled[test_idx] = y_scaler.transform(y_raw[test_idx].reshape(-1, 1)).flatten()

    X_train_seq, y_train_seq = create_grouped_sequences(X_scaled, y_scaled, item_ids, train_idx, seq_len)
    X_val_seq, y_val_seq = create_grouped_sequences(X_scaled, y_scaled, item_ids, val_idx, seq_len)
    X_test_seq, y_test_seq = create_grouped_sequences(X_scaled, y_scaled, item_ids, test_idx, seq_len)

    if len(X_train_seq) == 0:
        X_train_seq, y_train_seq = np.zeros((1, seq_len, X_raw.shape[1]), dtype=np.float32), np.zeros((1,), dtype=np.float32)
    if len(X_test_seq) == 0:
        # Prefer a real validation sequence over a synthetic test fallback.
        if len(X_val_seq) > 0:
            X_test_seq, y_test_seq = X_val_seq, y_val_seq
        else:
            X_test_seq, y_test_seq = np.zeros((1, seq_len, X_raw.shape[1]), dtype=np.float32), np.zeros((1,), dtype=np.float32)

    train_loader = _make_loader(X_train_seq, y_train_seq, shuffle=True)
    # FedStockClient calls this val_loader, but the pipeline now uses held-out
    # test sequences for reported strategy metrics.
    eval_loader = _make_loader(X_test_seq, y_test_seq, shuffle=False)

    input_size = X_raw.shape[1]
    
    client = FedStockClient(
        cid=client_id,
        train_loader=train_loader,
        val_loader=eval_loader,
        X_train=X_train_seq,
        y_train=y_train_seq,
        input_size=input_size,
        hidden_size=32,
        epsilon=10.0,
        y_scaler=y_scaler
    )
    client.split_stats = {
        "raw_rows": int(len(X_raw)),
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "test_rows": int(len(test_idx)),
        "train_sequences": int(len(X_train_seq)),
        "val_sequences": int(len(X_val_seq)),
        "test_sequences": int(len(X_test_seq)),
        "num_items": int(len(np.unique(item_ids))),
    }
    return client

def aggregate_metrics(history_list):
    if not history_list:
        return 0.0, 0.0
    total_samples = sum(h["num_samples"] for h in history_list)
    avg_rmse = sum(h["rmse"] * h["num_samples"] for h in history_list) / total_samples
    avg_smape = sum(h["smape"] * h["num_samples"] for h in history_list) / total_samples
    return avg_rmse, avg_smape


def create_run_dir(base_output_dir):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = os.path.join(base_output_dir, "runs", run_id)
    os.makedirs(run_dir, exist_ok=False)
    latest_path = os.path.join(base_output_dir, "latest_run.txt")
    os.makedirs(base_output_dir, exist_ok=True)
    with open(latest_path, "w") as f:
        f.write(run_dir + "\n")
    return run_id, run_dir


def flatten_per_client_metrics(histories):
    rows = []
    for strategy, history in histories.items():
        for record in history:
            base = {
                "strategy": strategy,
                "stage": record.get("stage"),
                "round": record.get("round"),
                "bubble": record.get("bubble"),
                "epochs": record.get("epochs"),
            }

            if "per_client_metrics" in record:
                for metric in record["per_client_metrics"]:
                    row = dict(base)
                    row.update(metric)
                    rows.append(row)
            elif "client" in record:
                row = dict(base)
                row.update(
                    {
                        "client": record["client"],
                        "num_samples": record.get("num_samples"),
                        "train_samples": record.get("train_samples"),
                        "rmse": record.get("rmse"),
                        "smape": record.get("smape"),
                    }
                )
                rows.append(row)

    return rows


def build_run_manifest(run_id, run_dir):
    files = {
        "config": "config.json",
        "feature_selection": "feature_selection.json",
        "split_summary": "split_summary.json",
        "feature_importances": "feature_importances.json",
        "clustering_results": "clustering_results.json",
        "metrics_history": "metrics_history.json",
        "per_client_metrics_json": "per_client_metrics.json",
        "per_client_metrics_csv": "per_client_metrics.csv",
        "final_results": "final_results.json",
        "evaluation_report": "evaluation_report.md",
        "baseline_comparison": "baseline_comparison.png",
        "models": "models/",
        "run_manifest": "run_manifest.json",
    }
    return {
        "schema_version": 1,
        "run_id": run_id,
        "run_dir": run_dir,
        "files": {
            name: os.path.join(run_dir, relative_path)
            for name, relative_path in files.items()
        },
        "notes": [
            "Per-client metrics are saved as both nested histories and a flat table.",
            "Clustering records include per-bubble client membership and client-to-cluster assignments.",
            "Reported final metrics use held-out test sequences.",
        ],
    }

def main():
    # 재현성을 위한 전체 시드 고정 및 결정론적 설정 적용
    seed_everything(seed=42)

    print("=== Starting PA-CFL Evaluation Pipeline ===")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "src", "fedstock_data", "data", "clients")
    output_base_dir = os.path.join(current_dir, "outputs")
    run_id, run_dir = create_run_dir(output_base_dir)
    print(f"Run outputs will be saved to {run_dir}")

    # Hyperparameters and evaluation protocol
    seq_len = 14
    train_ratio = 0.70
    val_ratio = 0.15
    num_rounds = 100
    epochs_per_round = 5
    global_warmup_rounds = 1
    head_finetune_epochs = 1
    recluster_interval = 10
    feature_top_k = 12
    feature_alpha = 0.10
    
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

    config = {
        "run_id": run_id,
        "data_dir": data_dir,
        "output_dir": run_dir,
        "num_clients": len(client_ids),
        "seq_len": seq_len,
        "split_policy": "chronological split inside each item_id",
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": 1.0 - train_ratio - val_ratio,
        "reported_eval_split": "test",
        "x_scaler_fit_scope": "train rows only",
        "y_scaler_fit_scope": "train rows only",
        "feature_selection_scope": "train rows only",
        "num_rounds": num_rounds,
        "epochs_per_round": epochs_per_round,
        "global_warmup_rounds": global_warmup_rounds,
        "head_finetune_epochs": head_finetune_epochs,
        "recluster_interval": recluster_interval,
        "feature_top_k": feature_top_k,
        "feature_alpha": feature_alpha,
    }
    _write_json(config, os.path.join(run_dir, "config.json"))

    feature_selection_path = os.path.join(run_dir, "feature_selection.json")
    feature_selection = compute_anova_feature_selection(
        clients=client_ids,
        data_dir=data_dir,
        candidate_features=CANDIDATE_FEATURE_COLS,
        top_k=feature_top_k,
        alpha=feature_alpha,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    save_feature_selection(feature_selection, feature_selection_path)
    selected_features = feature_selection["selected_features"]
    config["selected_features"] = selected_features
    _write_json(config, os.path.join(run_dir, "config.json"))
    print(f"Selected {len(selected_features)} ANOVA features: {selected_features}")

    clients_dict = {}
    for cid in client_ids:
        clients_dict[cid] = setup_client(
            cid,
            data_dir,
            seq_len=seq_len,
            selected_features=selected_features,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
    split_summary = {cid: client.split_stats for cid, client in clients_dict.items()}
    _write_json(split_summary, os.path.join(run_dir, "split_summary.json"))
    
    # We will evaluate 3 strategies
    results = {}
    
    # --- Strategy 1: Local Training ---
    print("\n\n=== Strategy 1: Local Training ===")
    local_clients = {cid: copy.deepcopy(c) for cid, c in clients_dict.items()}
    local_server = BubbleServer(local_clients, output_dir=run_dir)
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
    fedavg_server = BubbleServer(fedavg_clients, output_dir=run_dir)
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
    pacfl_server = BubbleServer(pacfl_clients, output_dir=run_dir)
    pacfl_server.step_1_collect_and_cluster()
    pacfl_fed_history = pacfl_server.step_3_federated_learning(
        num_rounds=num_rounds,
        epochs_per_round=epochs_per_round,
        global_warmup_rounds=global_warmup_rounds,
        head_finetune_epochs=head_finetune_epochs,
        personalize_head=True,
        recluster_interval=recluster_interval,
    )
    pacfl_pers_history = pacfl_server.step_4_personalized_learning(epochs=num_rounds * epochs_per_round)
    
    # Prefer personalized head metrics when available; otherwise use last shared-LSTM round.
    pacfl_head_metrics = [h for h in pacfl_fed_history if h["stage"] == "head_finetune"]
    pacfl_bubble_metrics = pacfl_head_metrics or [h for h in pacfl_fed_history if h["round"] == num_rounds]
    pacfl_final_metrics = pacfl_bubble_metrics + pacfl_pers_history
    pacfl_rmse, pacfl_smape = aggregate_metrics(pacfl_final_metrics)
    results["PA-CFL"] = {"rmse": pacfl_rmse, "smape": pacfl_smape}
    print(f"[PA-CFL] RMSE: {pacfl_rmse:.4f}, SMAPE: {pacfl_smape:.4f}")
    
    # Save final models for PA-CFL
    pacfl_server.save_models(output_dir=os.path.join(run_dir, "models"))
    histories = {
        "Local": local_history,
        "Global FedAvg": fedavg_history,
        "PA-CFL": pacfl_fed_history + pacfl_pers_history,
    }
    per_client_metrics = flatten_per_client_metrics(histories)
    _write_json(
        {
            "results": results,
            "histories": histories,
        },
        os.path.join(run_dir, "metrics_history.json"),
    )
    _write_json(per_client_metrics, os.path.join(run_dir, "per_client_metrics.json"))
    _write_csv(per_client_metrics, os.path.join(run_dir, "per_client_metrics.csv"))
    _write_json(
        {
            "run_id": run_id,
            "results": results,
            "evaluation_split": "test",
            "metric_weight": "evaluation sequence count",
        },
        os.path.join(run_dir, "final_results.json"),
    )

    
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
    ax1.set_title(f'Comparison of FL Strategies ({len(client_ids)} Clients, Test Split)')
    
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
    output_png = os.path.join(run_dir, "baseline_comparison.png")
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png)
    print(f"Visualization saved to {output_png}")
    
    # Save a report
    report_md = os.path.join(run_dir, "evaluation_report.md")
    with open(report_md, "w") as f:
        f.write("# Federated Learning Strategies Evaluation Report\n")
        f.write("## Overview\n")
        f.write(f"- **Run ID:** {run_id}\n")
        f.write(f"- **Total Clients:** {len(client_ids)}\n")
        f.write(f"- **Rounds:** {num_rounds}\n")
        f.write(f"- **Epochs per round:** {epochs_per_round}\n")
        f.write("- **Evaluation split:** held-out test split\n")
        f.write("- **Sequence policy:** item_id-grouped windows; no item or split boundary crossing\n")
        f.write("- **Scaler policy:** X and y scalers fit on train rows only\n")
        f.write("- **Feature selection:** ANOVA fit on train rows only\n")
        f.write("\n## Results\n")
        for s in strategies:
            f.write(f"- **{s}**: RMSE = {results[s]['rmse']:.4f}, SMAPE = {results[s]['smape']:.4f}\n")
    print(f"Evaluation report saved to {report_md}")

    _write_json(build_run_manifest(run_id, run_dir), os.path.join(run_dir, "run_manifest.json"))

if __name__ == "__main__":
    main()
