import argparse
import copy
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor


CLIENT_IDS = ("CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3")
FEATURE_COLS = [
    "sales",
    "sell_price",
    "wday",
    "month",
    "year",
    "is_weekend",
    "event_flag",
    "snap",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_7",
    "zero_ratio",
    "total_sales",
]
TARGET_COL = "target_1d"
INDEX_COLS = ["item_id", "date"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"csv_pipeline_{timestamp}.log"
    logger = logging.getLogger("csv_pa_cfl")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger, log_path


class LightweightLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


def load_split_csv(client_dir, split):
    path = client_dir / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {split}.csv: {path}")
    columns = INDEX_COLS + FEATURE_COLS + [TARGET_COL]
    df = pd.read_csv(path, usecols=columns)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df["date"] = pd.to_datetime(df["date"])
    return df


def sample_frame(df, max_samples, seed):
    if max_samples and len(df) > max_samples:
        return df.sample(n=max_samples, random_state=seed)
    return df


def extract_feature_importances(data_dir, output_path, max_samples, xgb_estimators, seed, logger):
    logger.info("Extracting feature importances from train.csv only")
    feature_importances = {}
    for client_id in CLIENT_IDS:
        train_df = load_split_csv(data_dir / client_id, "train")
        train_df = sample_frame(train_df, max_samples, seed)
        X = train_df[FEATURE_COLS].to_numpy(dtype=np.float32)
        y = train_df[TARGET_COL].to_numpy(dtype=np.float32)

        model = XGBRegressor(
            n_estimators=xgb_estimators,
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            objective="reg:squarederror",
        )
        model.fit(X, y)
        feature_importances[client_id] = model.feature_importances_.astype(float).tolist()
        logger.info("%s feature importance rows=%d features=%d", client_id, len(train_df), len(FEATURE_COLS))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(feature_importances, f, indent=4)
    return feature_importances


def normalize_importances(importances):
    normalized = []
    for imp in importances:
        non_negative = np.maximum(np.asarray(imp, dtype=float), 0.0)
        total = non_negative.sum()
        if total <= 0:
            normalized.append(np.ones_like(non_negative) / len(non_negative))
        else:
            normalized.append(non_negative / total)
    return np.asarray(normalized)


def run_server_clustering(feature_importances, output_path, config, logger):
    pyc_path = Path(__file__).resolve().parent / "src" / "fl" / "__pycache__" / "server_clustering.cpython-314.pyc"
    import importlib.util

    spec = importlib.util.spec_from_file_location("server_clustering_pyc", pyc_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    client_ids = list(feature_importances.keys())
    noisy_importances = np.array([feature_importances[cid] for cid in client_ids], dtype=float)
    labels, k_star, multi_bubbles, isolated = module.perform_clustering(
        noisy_importances,
        max_clusters=config["max_clusters"],
        complexity_penalty=config["complexity_penalty"],
        singleton_penalty=config["singleton_penalty"],
        isolation_std_multiplier=config["isolation_std_multiplier"],
    )
    assignments = {cid: int(labels[i]) for i, cid in enumerate(client_ids)}
    isolated_clients = [client_ids[i] for i in isolated]

    output = {
        "k_star": int(k_star),
        "assignments": assignments,
        "isolated_clients": isolated_clients,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    clusters = {}
    for cid, label in assignments.items():
        clusters.setdefault(label, []).append(cid)
    logger.info("Server clustering k_star=%d", k_star)
    for label in sorted(clusters):
        logger.info("Cluster %s: %s", label, clusters[label])
    logger.info("Isolated clients: %s", isolated_clients)
    logger.info("Multi-client bubble labels: %s", multi_bubbles)
    return output, clusters


def fit_global_scaler(data_dir, logger):
    scaler = StandardScaler()
    total_rows = 0
    for client_id in CLIENT_IDS:
        train_path = data_dir / client_id / "train.csv"
        if not train_path.exists():
            raise FileNotFoundError(f"Missing train.csv: {train_path}")
        df = pd.read_csv(train_path, usecols=FEATURE_COLS)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        scaler.partial_fit(df.to_numpy(dtype=np.float32))
        total_rows += len(df)
    logger.info("Global scaler fitted on train.csv only: rows=%d clients=%d", total_rows, len(CLIENT_IDS))
    return scaler


def make_sequences(context_df, eval_df, scaler, config):
    seq_len = config["sequence_length"]
    context = context_df.copy()
    context["_is_eval"] = False
    eval_part = eval_df.copy()
    eval_part["_is_eval"] = True
    combined = pd.concat([context, eval_part], ignore_index=True)
    combined = combined.sort_values(["item_id", "date"]).reset_index(drop=True)

    X_parts = []
    y_parts = []
    for _, group in combined.groupby("item_id", sort=False):
        features = scaler.transform(group[FEATURE_COLS].to_numpy(dtype=np.float32)).astype(np.float32)
        targets = group[TARGET_COL].to_numpy(dtype=np.float32)
        eval_mask = group["_is_eval"].to_numpy(dtype=bool)
        eval_positions = np.flatnonzero(eval_mask)
        valid_positions = eval_positions[eval_positions >= seq_len - 1]
        if len(valid_positions) == 0:
            continue

        item_X = np.stack([features[pos - seq_len + 1 : pos + 1] for pos in valid_positions])
        item_y = targets[valid_positions]
        X_parts.append(item_X)
        y_parts.append(item_y)

    if not X_parts:
        raise ValueError(f"No sequences produced. sequence_length={seq_len} may be too large.")
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


def prepare_client_data(data_dir, client_id, scaler, config):
    client_dir = data_dir / client_id
    train_df = load_split_csv(client_dir, "train")
    valid_df = load_split_csv(client_dir, "valid")
    test_df = load_split_csv(client_dir, "test")

    X_train, y_train = make_sequences(pd.DataFrame(columns=train_df.columns), train_df, scaler, config)
    X_valid, y_valid = make_sequences(train_df, valid_df, scaler, config)
    X_test, y_test = make_sequences(pd.concat([train_df, valid_df], ignore_index=True), test_df, scaler, config)

    if config["train_max_samples"] and len(y_train) > config["train_max_samples"]:
        rng = np.random.default_rng(config["seed"])
        indices = rng.choice(len(y_train), size=config["train_max_samples"], replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]

    def make_loader(X, y, shuffle):
        xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        return DataLoader(TensorDataset(xt, yt), batch_size=config["batch_size"], shuffle=shuffle)

    return {
        "train_loader": make_loader(X_train, y_train, True),
        "valid_loader": make_loader(X_valid, y_valid, False),
        "test_loader": make_loader(X_test, y_test, False),
        "train_samples": len(y_train),
        "valid_samples": len(y_valid),
        "test_samples": len(y_test),
    }


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total = 0
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.item()) * len(y)
        total += len(y)
    return total_loss / max(total, 1)


def evaluate(model, loader, device):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for X, y in loader:
            out = model(X.to(device)).cpu().numpy().reshape(-1)
            preds.append(np.maximum(out, 0.0))
            targets.append(y.numpy().reshape(-1))
    pred = np.concatenate(preds)
    target = np.concatenate(targets)
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    mae = float(np.mean(np.abs(pred - target)))
    smape = float(np.mean(2.0 * np.abs(pred - target) / (np.abs(target) + np.abs(pred) + 1e-8)) * 100.0)
    return {"rmse": rmse, "mae": mae, "smape": smape, "num_samples": int(len(target))}


def average_state_dicts(states, weights):
    total = float(sum(weights))
    averaged = {}
    for key in states[0]:
        averaged[key] = sum(state[key] * (weight / total) for state, weight in zip(states, weights))
    return averaged


def weighted_metrics(rows):
    total = sum(row["num_samples"] for row in rows)
    return {
        "rmse": sum(row["rmse"] * row["num_samples"] for row in rows) / total,
        "mae": sum(row["mae"] * row["num_samples"] for row in rows) / total,
        "smape": sum(row["smape"] * row["num_samples"] for row in rows) / total,
        "num_samples": int(total),
    }


def train_pipeline(data_dir, clusters, isolated_clients, config, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training device: %s", device)
    scaler = fit_global_scaler(data_dir, logger)
    clients = {}
    for client_id in CLIENT_IDS:
        clients[client_id] = prepare_client_data(data_dir, client_id, scaler, config)
        logger.info(
            "%s prepared sequences train=%d valid=%d test=%d seq_len=%d",
            client_id,
            clients[client_id]["train_samples"],
            clients[client_id]["valid_samples"],
            clients[client_id]["test_samples"],
            config["sequence_length"],
        )

    input_size = len(FEATURE_COLS)
    criterion = nn.MSELoss()
    final_models = {}
    history = []
    isolated_set = set(isolated_clients)

    for label, members in sorted(clusters.items()):
        if len(members) == 1 and members[0] in isolated_set:
            continue
        logger.info("[Bubble %s] FedAvg members=%s", label, members)
        global_model = LightweightLSTM(input_size=input_size, hidden_size=config["hidden_size"]).to(device)
        for round_idx in range(1, config["num_rounds"] + 1):
            local_states = []
            weights = []
            for client_id in members:
                local_model = copy.deepcopy(global_model).to(device)
                optimizer = torch.optim.Adam(local_model.parameters(), lr=config["learning_rate"])
                for _ in range(config["epochs_per_round"]):
                    train_one_epoch(local_model, clients[client_id]["train_loader"], optimizer, criterion, device)
                local_states.append({k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()})
                weights.append(clients[client_id]["train_samples"])
            global_model.load_state_dict(average_state_dicts(local_states, weights))

            val_rows = []
            for client_id in members:
                row = evaluate(global_model, clients[client_id]["valid_loader"], device)
                row.update({"client_id": client_id, "cluster": int(label), "stage": "federated_valid", "round": round_idx})
                val_rows.append(row)
            summary = weighted_metrics(val_rows)
            logger.info(
                "[Bubble %s] round %d/%d valid RMSE=%.4f MAE=%.4f SMAPE=%.4f",
                label,
                round_idx,
                config["num_rounds"],
                summary["rmse"],
                summary["mae"],
                summary["smape"],
            )
            history.extend(val_rows)

        for client_id in members:
            final_models[client_id] = global_model

    for client_id in isolated_clients:
        logger.info("[Personalized] local training client=%s", client_id)
        model = LightweightLSTM(input_size=input_size, hidden_size=config["hidden_size"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        for epoch in range(1, config["personalized_epochs"] + 1):
            train_loss = train_one_epoch(model, clients[client_id]["train_loader"], optimizer, criterion, device)
            val = evaluate(model, clients[client_id]["valid_loader"], device)
            logger.info(
                "[Personalized %s] epoch %d/%d train_loss=%.4f valid RMSE=%.4f MAE=%.4f SMAPE=%.4f",
                client_id,
                epoch,
                config["personalized_epochs"],
                train_loss,
                val["rmse"],
                val["mae"],
                val["smape"],
            )
        final_models[client_id] = model

    test_rows = []
    for client_id in CLIENT_IDS:
        row = evaluate(final_models[client_id], clients[client_id]["test_loader"], device)
        row.update({"client_id": client_id, "cluster": int(next(k for k, v in clusters.items() if client_id in v))})
        test_rows.append(row)
        logger.info(
            "[Test %s] cluster=%s RMSE=%.4f MAE=%.4f SMAPE=%.4f",
            client_id,
            row["cluster"],
            row["rmse"],
            row["mae"],
            row["smape"],
        )
    return history, test_rows, weighted_metrics(test_rows)


def main():
    parser = argparse.ArgumentParser(description="Run CSV-based PA-CFL pipeline.")
    parser.add_argument("--max-samples", type=int, default=200000)
    parser.add_argument("--xgb-estimators", type=int, default=200)
    parser.add_argument("--train-max-samples", type=int, default=0)
    parser.add_argument("--num-rounds", type=int, default=4)
    parser.add_argument("--epochs-per-round", type=int, default=1)
    parser.add_argument("--personalized-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--sequence-length", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "src" / "fedstock_data" / "outputs" / "clients"
    outputs_dir = project_root / "outputs"
    logger, log_path = setup_logger(outputs_dir / "logs")
    set_seed(args.seed)

    config = {
        "seed": args.seed,
        "max_samples": args.max_samples,
        "xgb_estimators": args.xgb_estimators,
        "max_clusters": 3,
        "complexity_penalty": 0.10,
        "singleton_penalty": 0.35,
        "isolation_std_multiplier": 1.0,
        "train_max_samples": args.train_max_samples or None,
        "num_rounds": args.num_rounds,
        "epochs_per_round": args.epochs_per_round,
        "personalized_epochs": args.personalized_epochs,
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "learning_rate": args.learning_rate,
        "sequence_length": args.sequence_length,
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "server_clustering_source": "train.csv only",
        "scaler": "global StandardScaler fitted on all clients' train.csv only",
    }
    logger.info("CSV PA-CFL pipeline started")
    logger.info("Config: %s", json.dumps(config, ensure_ascii=False, sort_keys=True))

    started = time.time()
    feature_path = outputs_dir / "feature_importances.json"
    clustering_path = outputs_dir / "clustering_results.json"
    feature_importances = extract_feature_importances(
        data_dir,
        feature_path,
        args.max_samples,
        args.xgb_estimators,
        args.seed,
        logger,
    )
    clustering_result, clusters = run_server_clustering(
        feature_importances,
        clustering_path,
        config,
        logger,
    )
    history, test_rows, test_summary = train_pipeline(
        data_dir,
        clusters,
        clustering_result["isolated_clients"],
        config,
        logger,
    )

    elapsed = time.time() - started
    result = {
        "elapsed_seconds": elapsed,
        "config": config,
        "clustering": clustering_result,
        "validation_history": history,
        "test_by_client": test_rows,
        "test_summary": test_summary,
        "log_path": str(log_path),
    }
    results_dir = outputs_dir / "experiments"
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"csv_pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    logger.info("CSV PA-CFL pipeline finished in %.2fs", elapsed)
    logger.info("Final test summary: %s", json.dumps(test_summary, sort_keys=True))
    logger.info("Saved results: %s", result_path)
    logger.info("Log file: %s", log_path)


if __name__ == "__main__":
    main()
