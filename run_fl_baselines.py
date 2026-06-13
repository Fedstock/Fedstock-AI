import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse
import copy
import csv
import json
import random
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.data import TensorDataset, DataLoader

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

from src.dataset import CANDIDATE_FEATURE_COLS, load_client_data, make_group_time_split_indices
from src.fl.client import FedStockClient
from src.fl.extract_features import compute_anova_feature_selection, save_feature_selection
from src.fl.server import BubbleServer


def get_runtime_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 4. CUDA 결정론적 연산 설정
    if torch.cuda.is_available():
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


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_precomputed_feature_importances(input_path, expected_client_ids):
    payload = _load_json(input_path)
    if not isinstance(payload, dict):
        raise ValueError("Precomputed feature importances file must contain a JSON object.")

    client_vectors = {
        str(client_id): value
        for client_id, value in payload.items()
        if isinstance(value, list)
    }

    missing = [cid for cid in expected_client_ids if cid not in client_vectors]
    if missing:
        raise ValueError(
            f"Precomputed feature importances missing client IDs: {missing}"
        )

    return {cid: np.asarray(client_vectors[cid], dtype=np.float32) for cid in expected_client_ids}


def export_feature_importances_only(clients_dict, output_path):
    print("\n=== Exporting Precomputed PA-CFL Feature Importances Only ===")
    feature_importances = {}
    for cid, client in clients_dict.items():
        print(f"Extracting feature importances for {cid}...")
        feature_importances[cid] = client.extract_noisy_importance().tolist()
    _write_json(feature_importances, output_path)
    print(f"Saved precomputed feature importances to {output_path}")


def _load_prediction_frames(client_id, data_dir):
    client_dir = os.path.join(data_dir, client_id)
    train_path = os.path.join(client_dir, "train.csv")
    valid_path = os.path.join(client_dir, "valid.csv")
    if not os.path.exists(train_path) or not os.path.exists(valid_path):
        raise FileNotFoundError(f"Missing train/valid CSV for client {client_id}")

    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)

    for frame in (train_df, valid_df):
        if "sales" in frame.columns and "quantity" not in frame.columns:
            frame["quantity"] = frame["sales"]
        if "event_flag" in frame.columns and "is_holiday" not in frame.columns:
            frame["is_holiday"] = frame["event_flag"]
        frame["date"] = pd.to_datetime(frame["date"])
        frame["dayofweek"] = frame["date"].dt.dayofweek
        frame["month"] = frame["date"].dt.month
        if "rolling_std_28" not in frame.columns:
            frame["rolling_std_28"] = 0.0

    train_df["_origin"] = "train"
    valid_df["_origin"] = "valid"
    return train_df, valid_df


def build_validation_prediction_rows(client, client_id, data_dir):
    train_df, valid_df = _load_prediction_frames(client_id, data_dir)
    full_df = (
        pd.concat([train_df, valid_df], ignore_index=True)
        .sort_values(["item_id", "date"])
        .reset_index(drop=True)
    )

    feature_cols = list(client.selected_features or CANDIDATE_FEATURE_COLS)
    target_col = "target_7d" if "target_7d" in full_df.columns else "quantity"
    X_all = full_df[feature_cols].values.astype(np.float32)
    X_scaled = client.x_scaler.transform(X_all).astype(np.float32)

    rows = []
    sequences = []
    seq_len = int(client.seq_len)

    for item_id, group in full_df.groupby("item_id", sort=False):
        group_indices = group.index.to_numpy()
        for pos in range(seq_len, len(group_indices)):
            row_idx = group_indices[pos]
            if full_df.at[row_idx, "_origin"] != "valid":
                continue
            seq_indices = group_indices[pos - seq_len:pos]
            sequences.append(X_scaled[seq_indices])
            rows.append(
                {
                    "client_id": client_id,
                    "item_id": str(item_id),
                    "date": str(full_df.at[row_idx, "date"].date()),
                    "sales_label": float(full_df.at[row_idx, target_col]),
                }
            )

    if not sequences:
        return rows

    X_tensor = torch.tensor(np.asarray(sequences, dtype=np.float32), dtype=torch.float32).to(client.device)
    client.model.eval()
    with torch.no_grad():
        predictions = client.model(X_tensor).cpu().numpy().reshape(-1)

    predictions = client.inverse_target_array(predictions)
    for row, prediction in zip(rows, predictions):
        row["prediction"] = float(prediction)
    return rows


def save_client_prediction_csv(output_path, client_id, strategy_to_rows, prediction_date=None):
    if not strategy_to_rows:
        return

    strategy_names = list(strategy_to_rows.keys())
    first_rows = strategy_to_rows[strategy_names[0]]
    merged_rows = []

    for idx, base in enumerate(first_rows):
        row = {
            "client_id": client_id,
            "item_id": base["item_id"],
            "date": base["date"],
            "sales_label": base["sales_label"],
        }
        for strategy in strategy_names:
            strategy_rows = strategy_to_rows[strategy]
            if idx < len(strategy_rows):
                row[f"prediction_{strategy.lower().replace(' ', '_')}"] = strategy_rows[idx].get("prediction")
        merged_rows.append(row)

    if prediction_date:
        merged_rows = [
            row for row in merged_rows
            if row.get("date") == prediction_date
        ]

    _write_csv(merged_rows, output_path)
    print(f"Saved validation prediction CSV to {output_path}")


def setup_client(
    client_id,
    data_dir,
    seq_len=14,
    selected_features=None,
    train_ratio=0.7,
    val_ratio=0.15,
    target_transform="identity",
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
    if target_transform == "log1p":
        y_source = np.log1p(np.clip(y_raw, a_min=0.0, a_max=None)).astype(np.float32)
    else:
        y_source = y_raw.astype(np.float32)

    y_scaler = RobustScaler()
    y_scaled = np.empty_like(y_source, dtype=np.float32)
    y_scaled[train_idx] = y_scaler.fit_transform(y_source[train_idx].reshape(-1, 1)).flatten()
    if len(val_idx) > 0:
        y_scaled[val_idx] = y_scaler.transform(y_source[val_idx].reshape(-1, 1)).flatten()
    if len(test_idx) > 0:
        y_scaled[test_idx] = y_scaler.transform(y_source[test_idx].reshape(-1, 1)).flatten()

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
        epsilon=30.0,
        y_scaler=y_scaler,
        target_transform=target_transform,
    )
    client.x_scaler = x_scaler
    client.selected_features = list(selected_features) if selected_features is not None else None
    client.seq_len = seq_len
    client.target_transform = target_transform
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
        return {k: 0.0 for k in ["rmse", "smape", "mae", "wmape", "mase"]}
    total_samples = sum(h["num_samples"] for h in history_list)
    result = {}
    for metric in ["rmse", "smape", "mae", "wmape", "mase"]:
        result[metric] = sum(h.get(metric, 0.0) * h["num_samples"] for h in history_list) / total_samples
    return result


def create_run_dir(base_output_dir):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = os.path.join(base_output_dir, "runs", run_id)
    os.makedirs(run_dir, exist_ok=False)
    
    latest_path = os.path.join(base_output_dir, "latest_run.txt")
    os.makedirs(base_output_dir, exist_ok=True)
    with open(latest_path, "w") as f:
        f.write(run_dir + "\n")

    # Update .gitignore dynamically to point to the latest run directory
    try:
        model_dir = os.path.dirname(base_output_dir)
        gitignore_path = os.path.join(model_dir, ".gitignore")
        if os.path.exists(gitignore_path):
            with open(gitignore_path, "r") as f:
                lines = f.readlines()
            
            new_lines = []
            skip = False
            for line in lines:
                if "# Ignore all runs except the latest" in line:
                    new_lines.append(line)
                    new_lines.append("outputs/runs/*\n")
                    new_lines.append(f"!outputs/runs/{run_id}/\n")
                    skip = True
                    continue
                if skip:
                    if line.strip().startswith("outputs/runs/*") or line.strip().startswith("!outputs/runs/"):
                        continue
                    else:
                        skip = False
                new_lines.append(line)
                
            with open(gitignore_path, "w") as f:
                f.writelines(new_lines)
            print(f"Updated .gitignore with latest run: {run_id}")
    except Exception as e:
        print(f"Failed to update .gitignore: {e}")

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
                        "mae": record.get("mae"),
                        "wmape": record.get("wmape"),
                        "mase": record.get("mase"),
                    }
                )
                rows.append(row)

    return rows


def save_client_feature_importance_artifacts(noisy_importances, feature_names, output_dir):
    rows = []
    rankings = {}

    for client_id in sorted(noisy_importances.keys()):
        vector = np.asarray(noisy_importances[client_id], dtype=np.float32).reshape(-1)
        ranking = []
        for rank, feature_idx in enumerate(np.argsort(vector)[::-1], start=1):
            feature_name = (
                feature_names[int(feature_idx)]
                if int(feature_idx) < len(feature_names)
                else f"feature_{int(feature_idx)}"
            )
            importance = float(vector[int(feature_idx)])
            rows.append(
                {
                    "client_id": client_id,
                    "rank": rank,
                    "feature": feature_name,
                    "importance": importance,
                }
            )
            ranking.append(
                {
                    "rank": rank,
                    "feature": feature_name,
                    "importance": importance,
                }
            )
        rankings[client_id] = ranking

    _write_csv(rows, os.path.join(output_dir, "feature_importances_by_client.csv"))
    _write_json(rankings, os.path.join(output_dir, "feature_importance_rankings.json"))


def build_run_manifest(run_id, run_dir):
    files = {
        "config": "config.json",
        "feature_selection": "feature_selection.json",
        "split_summary": "split_summary.json",
        "feature_importances": "feature_importances.json",
        "feature_importances_by_client_csv": "feature_importances_by_client.csv",
        "feature_importance_rankings_json": "feature_importance_rankings.json",
        "clustering_results": "clustering_results.json",
        "metrics_history": "metrics_history.json",
        "per_client_metrics_json": "per_client_metrics.json",
        "per_client_metrics_csv": "per_client_metrics.csv",
        "final_results": "final_results.json",
        "evaluation_report": "evaluation_report.md",
        "baseline_comparison": "baseline_comparison.png",
        "models_local": "models_local/",
        "models_fedavg": "models_fedavg/",
        "models_pacfl": "models_pacfl/",
        "logs": "logs/",
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
            "models_local stores client models trained only with local data before any FL aggregation.",
            "models_fedavg stores the fully shared global baseline models.",
            "models_pacfl stores the final personalized PA-CFL models after clustering, bubble FL, and personalization.",
            "Feature importance artifacts include DP-noisy XGBoost vectors and per-client ranked feature tables.",
        ],
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run FL baselines or split PA-CFL feature extraction across machines."
    )
    parser.add_argument(
        "--extract-feature-importances-only",
        action="store_true",
        help="Only compute and save PA-CFL noisy feature importances, then exit.",
    )
    parser.add_argument(
        "--feature-importances-output",
        default=None,
        help="Where to save extracted feature importances JSON. Defaults to <run_dir>/feature_importances.json.",
    )
    parser.add_argument(
        "--precomputed-feature-importances",
        default=None,
        help="Path to a precomputed feature_importances.json file to reuse instead of recomputing in PA-CFL.",
    )
    parser.add_argument(
        "--exclude-client-id",
        default=None,
        help="Optional client ID to exclude from Local/FedAvg/PA-CFL training.",
    )
    parser.add_argument(
        "--prediction-client-id",
        default=None,
        help="Optional client ID whose valid.csv labels/predictions should be exported at the end.",
    )
    parser.add_argument(
        "--prediction-date",
        default=None,
        help="Optional YYYY-MM-DD date filter for validation prediction export.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    # 재현성을 위한 전체 시드 고정 및 결정론적 설정 적용
    seed_everything(seed=42)

    print("=== Starting PA-CFL Evaluation Pipeline ===")
    print(f"=== Runtime Device: {get_runtime_device()} ===")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data", "clients")
    output_base_dir = os.path.join(current_dir, "outputs")
    run_id, run_dir = create_run_dir(output_base_dir)
    
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    sys.stdout = Logger(os.path.join(log_dir, "training_full.log"))
    
    print(f"Run outputs will be saved to {run_dir}")

    # Hyperparameters and evaluation protocol
    seq_len = 14
    train_ratio = 0.70
    val_ratio = 0.15
    num_rounds = 60
    epochs_per_round = 3
    global_warmup_rounds = 10
    head_finetune_epochs = 10
    recluster_interval = 10
    local_epochs = 40
    feature_top_k = 12
    feature_alpha = 0.10
    
    # 1. Initialize Clients
    client_ids = []
    if os.path.exists(data_dir):
        for f in os.listdir(data_dir):
            if os.path.isdir(os.path.join(data_dir, f)) and os.path.exists(os.path.join(data_dir, f, "train.csv")):
                client_ids.append(f)
    client_ids.sort()
    all_client_ids = list(client_ids)
    if args.exclude_client_id:
        if args.exclude_client_id not in all_client_ids:
            raise ValueError(
                f"exclude-client-id '{args.exclude_client_id}' not found in data/clients."
            )
        client_ids = [cid for cid in client_ids if cid != args.exclude_client_id]
        print(f"Excluding client from training: {args.exclude_client_id}")
    
    # Use all clients!
    print(f"Found {len(client_ids)} clients in dataset.")
    if len(client_ids) == 0:
        print("No clients found. Exiting.")
        return

    config = {
        "run_id": run_id,
        "data_dir": data_dir,
        "output_dir": run_dir,
        "runtime_device": get_runtime_device(),
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
        "local_epochs": local_epochs,
        "feature_top_k": feature_top_k,
        "feature_alpha": feature_alpha,
        "pacfl_flow": [
            "client-side XGBoost feature importance extraction",
            "DP noise injection",
            "server-side clustering",
            "bubble FL",
            "personalization",
        ],
        "extract_feature_importances_only": bool(args.extract_feature_importances_only),
        "precomputed_feature_importances": args.precomputed_feature_importances,
        "exclude_client_id": args.exclude_client_id,
        "prediction_client_id": args.prediction_client_id,
        "prediction_date": args.prediction_date,
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

    if args.extract_feature_importances_only:
        export_path = args.feature_importances_output or os.path.join(run_dir, "feature_importances.json")
        export_feature_importances_only(clients_dict, export_path)
        save_client_feature_importance_artifacts(
            _load_json(export_path),
            selected_features,
            run_dir,
        )
        print("Feature importances exported. Exiting before Local/FedAvg/PA-CFL training.")
        _write_json(build_run_manifest(run_id, run_dir), os.path.join(run_dir, "run_manifest.json"))
        return
    
    # We will evaluate 3 strategies
    results = {}
    
    # --- Strategy 1: Local Training ---
    print("\n\n=== Strategy 1: Local Training ===")
    local_clients = {cid: copy.deepcopy(c) for cid, c in clients_dict.items()}
    local_server = BubbleServer(local_clients, output_dir=run_dir)
    local_server.isolated = list(local_clients.keys())
    local_server.bubbles = []
    # step_4 acts as local training for all clients
    local_history = local_server.step_4_personalized_learning(epochs=local_epochs)
    local_server.save_models(output_dir=os.path.join(run_dir, "models_local"))
    local_metrics = aggregate_metrics(local_history)
    results["Local"] = local_metrics
    print(f"[Local] RMSE: {local_metrics['rmse']:.4f}, SMAPE: {local_metrics['smape']:.4f}, MAE: {local_metrics['mae']:.4f}, WMAPE: {local_metrics['wmape']:.4f}, MASE: {local_metrics['mase']:.4f}")
    
    # --- Strategy 2: Global FedAvg ---
    print("\n\n=== Strategy 2: Global FedAvg ===")
    fedavg_clients = {cid: copy.deepcopy(c) for cid, c in clients_dict.items()}
    fedavg_server = BubbleServer(fedavg_clients, output_dir=run_dir)
    fedavg_server.bubbles = [list(fedavg_clients.keys())]
    fedavg_server.isolated = []
    fedavg_history = fedavg_server.step_3_federated_learning(
        num_rounds=num_rounds,
        epochs_per_round=epochs_per_round,
    )
    fedavg_server.save_models(output_dir=os.path.join(run_dir, "models_fedavg"))
    last_round_fedavg = [h for h in fedavg_history if h["round"] == num_rounds]
    fedavg_metrics = aggregate_metrics(last_round_fedavg)
    results["Global FedAvg"] = fedavg_metrics
    print(
        f"[Global FedAvg] RMSE: {fedavg_metrics['rmse']:.4f}, "
        f"SMAPE: {fedavg_metrics['smape']:.4f}, MAE: {fedavg_metrics['mae']:.4f}, "
        f"WMAPE: {fedavg_metrics['wmape']:.4f}, MASE: {fedavg_metrics['mase']:.4f}"
    )

    # --- Strategy 3: PA-CFL ---
    print("\n\n=== Strategy 3: PA-CFL ===")
    pacfl_clients = {cid: copy.deepcopy(c) for cid, c in clients_dict.items()}
    pacfl_server = BubbleServer(pacfl_clients, output_dir=run_dir)
    if args.precomputed_feature_importances:
        print(f"Loading precomputed feature importances from {args.precomputed_feature_importances}")
        precomputed_importances = load_precomputed_feature_importances(
            args.precomputed_feature_importances,
            client_ids,
        )
        pacfl_server.cluster_from_noisy_importances(
            noisy_importances=precomputed_importances,
            client_ids=client_ids,
            stage="initial_clustering_precomputed",
            round_num=0,
        )
    else:
        # PA-CFL default flow:
        # client-side XGBoost importance -> DP noise -> server-side clustering.
        pacfl_server.step_1_collect_and_cluster()
    save_client_feature_importance_artifacts(
        pacfl_server.noisy_importances,
        selected_features,
        run_dir,
    )
    # After clustering, each bubble runs FL with a shared LSTM backbone and
    # client-specific heads, followed by a final personalization stage.
    pacfl_fed_history = pacfl_server.step_3_federated_learning(
        num_rounds=num_rounds,
        epochs_per_round=epochs_per_round,
        global_warmup_rounds=global_warmup_rounds,
        head_finetune_epochs=head_finetune_epochs,
        personalize_head=True,
        recluster_interval=recluster_interval,
    )
    pacfl_pers_history = pacfl_server.step_4_personalized_learning(epochs=local_epochs)
    
    # Prefer personalized head metrics when available; otherwise use last shared-LSTM round.
    pacfl_head_metrics = [h for h in pacfl_fed_history if h["stage"] == "head_finetune"]
    pacfl_bubble_metrics = pacfl_head_metrics or [h for h in pacfl_fed_history if h["round"] == num_rounds]
    pacfl_final_metrics = pacfl_bubble_metrics + pacfl_pers_history
    pacfl_metrics = aggregate_metrics(pacfl_final_metrics)
    results["PA-CFL"] = pacfl_metrics
    print(f"[PA-CFL] RMSE: {pacfl_metrics['rmse']:.4f}, SMAPE: {pacfl_metrics['smape']:.4f}, MAE: {pacfl_metrics['mae']:.4f}, WMAPE: {pacfl_metrics['wmape']:.4f}, MASE: {pacfl_metrics['mase']:.4f}")
    
    # Save final models for PA-CFL
    pacfl_server.save_models(output_dir=os.path.join(run_dir, "models_pacfl"))
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
    maes = [results[s]["mae"] for s in strategies]
    smapes = [results[s]["smape"] for s in strategies]
    wmapes = [results[s]["wmape"] for s in strategies]
    mases = [results[s]["mase"] for s in strategies]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(strategies))
    width = 0.35
    
    # Subplot 1: RMSE and MAE (Absolute Errors)
    rects1 = ax1.bar(x - width/2, rmses, width, label='RMSE', color='skyblue')
    rects2 = ax1.bar(x + width/2, maes, width, label='MAE', color='lightgreen')
    ax1.set_ylabel('Absolute Error')
    ax1.set_title('RMSE and MAE Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies)
    ax1.legend()
    
    # Subplot 2: SMAPE, WMAPE
    # SMAPE is usually 0-100, WMAPE is usually 0-1 (or 0-100). 
    # To plot them together effectively, we'll use a twin axis.
    rects3 = ax2.bar(x - width/2, smapes, width, label='SMAPE (%)', color='salmon')
    ax2.set_ylabel('SMAPE (%)', color='salmon')
    ax2.tick_params(axis='y', labelcolor='salmon')
    
    ax3 = ax2.twinx()
    rects4 = ax3.bar(x + width/2, wmapes, width, label='WMAPE', color='gold')
    ax3.set_ylabel('WMAPE', color='gold')
    ax3.tick_params(axis='y', labelcolor='gold')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies)
    ax2.set_title('SMAPE and WMAPE Comparison')
    
    # Add values on top of bars
    def autolabel(rects, ax, fmt='{:.2f}'):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(fmt.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1, ax1)
    autolabel(rects2, ax1)
    autolabel(rects3, ax2)
    autolabel(rects4, ax3)
    
    fig.suptitle(f'Comparison of FL Strategies ({len(client_ids)} Clients, Test Split)', fontsize=16)
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
        f.write("- **PA-CFL pipeline:** client-side XGBoost importance, DP noise, server-side clustering, bubble FL, personalization\n")
        f.write("\n## Results\n")
        for s in strategies:
            f.write(f"- **{s}**: RMSE = {results[s]['rmse']:.4f}, SMAPE = {results[s]['smape']:.4f}, MAE = {results[s]['mae']:.4f}, WMAPE = {results[s]['wmape']:.4f}, MASE = {results[s]['mase']:.4f}\n")
    print(f"Evaluation report saved to {report_md}")

    if args.prediction_client_id:
        if args.prediction_client_id not in all_client_ids:
            raise ValueError(
                f"prediction-client-id '{args.prediction_client_id}' not found in data/clients."
            )

        prediction_rows = {}

        if args.prediction_client_id in local_server.clients:
            local_prediction_client = local_server.clients[args.prediction_client_id]
        else:
            local_prediction_client = setup_client(
                args.prediction_client_id,
                data_dir,
                seq_len=seq_len,
                selected_features=selected_features,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
            )
            local_prediction_client.fit(
                parameters=local_prediction_client.get_parameters({}),
                config={"epochs": local_epochs, "current_round": 1, "total_rounds": 1},
            )
        prediction_rows["Local"] = build_validation_prediction_rows(
            local_prediction_client,
            args.prediction_client_id,
            data_dir,
        )

        fedavg_prediction_client = setup_client(
            args.prediction_client_id,
            data_dir,
            seq_len=seq_len,
            selected_features=selected_features,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        representative_global_client = next(iter(fedavg_server.clients.values()))
        fedavg_prediction_client.set_parameters(
            representative_global_client.get_parameters({})
        )
        prediction_rows["Global FedAvg"] = build_validation_prediction_rows(
            fedavg_prediction_client,
            args.prediction_client_id,
            data_dir,
        )

        if args.prediction_client_id in pacfl_server.clients:
            pacfl_prediction_client = pacfl_server.clients[args.prediction_client_id]
        else:
            pacfl_prediction_client = setup_client(
                args.prediction_client_id,
                data_dir,
                seq_len=seq_len,
                selected_features=selected_features,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
            )
            pacfl_server.add_client(
                args.prediction_client_id,
                pacfl_prediction_client,
                warm_start=True,
                train_epochs=local_epochs,
                save_results=False,
            )
        prediction_rows["PA-CFL"] = build_validation_prediction_rows(
            pacfl_prediction_client,
            args.prediction_client_id,
            data_dir,
        )

        prediction_csv_path = os.path.join(
            run_dir,
            (
                f"validation_predictions_{args.prediction_client_id}_{args.prediction_date}.csv"
                if args.prediction_date
                else f"validation_predictions_{args.prediction_client_id}.csv"
            ),
        )
        save_client_prediction_csv(
            prediction_csv_path,
            args.prediction_client_id,
            prediction_rows,
            prediction_date=args.prediction_date,
        )

    _write_json(build_run_manifest(run_id, run_dir), os.path.join(run_dir, "run_manifest.json"))

if __name__ == "__main__":
    main()
