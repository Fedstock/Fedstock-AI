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
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

from losses import HuberSMAPELoss


CLIENT_IDS = ("CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3")
CANDIDATE_FEATURE_COLS = [
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
COMMON_LAYER_PREFIXES = ("lstm.",)
HEAD_LAYER_PREFIXES = ("fc.",)


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


def clean_frame(df):
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def sample_frame(df, max_samples, seed):
    if max_samples and len(df) > max_samples:
        return df.sample(n=max_samples, random_state=seed)
    return df


def load_split_csv(client_dir, split, feature_cols):
    path = client_dir / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {split}.csv: {path}")
    columns = INDEX_COLS + feature_cols + [TARGET_COL]
    df = pd.read_csv(path, usecols=columns)
    df = clean_frame(df)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_numeric_split(client_dir, split, feature_cols):
    path = client_dir / f"{split}.csv"
    columns = feature_cols + [TARGET_COL]
    return clean_frame(pd.read_csv(path, usecols=columns))


def select_features(data_dir, config, output_path, logger):
    frames = []
    selection_seed = config["seeds"][0]
    for client_id in CLIENT_IDS:
        df = load_numeric_split(data_dir / client_id, "train", CANDIDATE_FEATURE_COLS)
        df = sample_frame(df, config["feature_selection_max_samples_per_client"], selection_seed)
        frames.append(df)
    train_df = pd.concat(frames, ignore_index=True)

    X = train_df[CANDIDATE_FEATURE_COLS].to_numpy(dtype=np.float32)
    y = train_df[TARGET_COL].to_numpy(dtype=np.float32)
    f_values, p_values = f_regression(X, y)
    f_values = np.nan_to_num(f_values, nan=0.0, posinf=0.0, neginf=0.0)
    p_values = np.nan_to_num(p_values, nan=1.0, posinf=1.0, neginf=1.0)
    scores = {col: float(score) for col, score in zip(CANDIDATE_FEATURE_COLS, f_values)}
    pvals = {col: float(pval) for col, pval in zip(CANDIDATE_FEATURE_COLS, p_values)}

    corr = train_df[CANDIDATE_FEATURE_COLS].corr().abs()
    dropped = set()
    for i, left in enumerate(CANDIDATE_FEATURE_COLS):
        for right in CANDIDATE_FEATURE_COLS[i + 1 :]:
            if left in dropped or right in dropped:
                continue
            value = corr.loc[left, right]
            if pd.notna(value) and value >= config["corr_threshold"]:
                drop = left if scores[left] < scores[right] else right
                dropped.add(drop)

    pruned = [col for col in CANDIDATE_FEATURE_COLS if col not in dropped]
    top_k = min(config["anova_top_k"], len(pruned))
    selected = sorted(pruned, key=lambda col: scores[col], reverse=True)[:top_k]
    selected = [col for col in CANDIDATE_FEATURE_COLS if col in selected]

    result = {
        "candidate_features": CANDIDATE_FEATURE_COLS,
        "correlation_threshold": config["corr_threshold"],
        "correlation_pruned_features": sorted(dropped),
        "anova_top_k": config["anova_top_k"],
        "selected_features": selected,
        "f_scores": scores,
        "p_values": pvals,
        "rows_used": int(len(train_df)),
        "source": "all clients' train.csv only",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
    logger.info("Feature selection rows=%d selected=%s pruned=%s", len(train_df), selected, sorted(dropped))
    return selected, result


def extract_feature_importances(data_dir, output_path, feature_cols, config, logger):
    logger.info("Extracting feature importances from train.csv only")
    feature_importances = {}
    for client_id in CLIENT_IDS:
        train_df = load_numeric_split(data_dir / client_id, "train", feature_cols)
        train_df = sample_frame(train_df, config["max_samples"], config["seed"])
        X = train_df[feature_cols].to_numpy(dtype=np.float32)
        y = train_df[TARGET_COL].to_numpy(dtype=np.float32)

        model = XGBRegressor(
            n_estimators=config["xgb_estimators"],
            random_state=config["seed"],
            n_jobs=-1,
            tree_method="hist",
            objective="reg:squarederror",
        )
        model.fit(X, y)
        feature_importances[client_id] = model.feature_importances_.astype(float).tolist()
        logger.info("%s feature importance rows=%d features=%d", client_id, len(train_df), len(feature_cols))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(feature_importances, f, indent=4)
    return feature_importances


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
        "max_clusters": int(config["max_clusters"]),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    clusters = {}
    for cid, label in assignments.items():
        clusters.setdefault(label, []).append(cid)
    logger.info("Server clustering k_star=%d max_clusters=%d", k_star, config["max_clusters"])
    for label in sorted(clusters):
        logger.info("Cluster %s: %s", label, clusters[label])
    logger.info("Isolated clients: %s", isolated_clients)
    logger.info("Multi-client bubble labels: %s", multi_bubbles)
    return output, clusters


def cluster_vectors(client_vectors, config):
    pyc_path = Path(__file__).resolve().parent / "src" / "fl" / "__pycache__" / "server_clustering.cpython-314.pyc"
    import importlib.util

    spec = importlib.util.spec_from_file_location("server_clustering_pyc", pyc_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    client_ids = list(client_vectors.keys())
    vectors = np.array([client_vectors[cid] for cid in client_ids], dtype=float)
    labels, k_star, multi_bubbles, isolated = module.perform_clustering(
        vectors,
        max_clusters=config["max_clusters"],
        complexity_penalty=config["complexity_penalty"],
        singleton_penalty=config["singleton_penalty"],
        isolation_std_multiplier=config["isolation_std_multiplier"],
    )
    assignments = {cid: int(labels[i]) for i, cid in enumerate(client_ids)}
    isolated_clients = [client_ids[i] for i in isolated]
    clusters = {}
    for cid, label in assignments.items():
        clusters.setdefault(int(label), []).append(cid)
    return {
        "k_star": int(k_star),
        "assignments": assignments,
        "isolated_clients": isolated_clients,
        "multi_bubbles": [int(label) for label in multi_bubbles],
    }, clusters


def fit_client_scalers(data_dir, client_id, feature_cols, logger):
    feature_scaler = StandardScaler()
    target_scaler = RobustScaler()
    train_path = data_dir / client_id / "train.csv"
    df = clean_frame(pd.read_csv(train_path, usecols=feature_cols + [TARGET_COL]))
    feature_scaler.fit(df[feature_cols].to_numpy(dtype=np.float32))
    target_scaler.fit(df[[TARGET_COL]].to_numpy(dtype=np.float32))
    logger.info(
        "Client-specific scalers fitted on train.csv only: client=%s rows=%d feature=StandardScaler target=RobustScaler",
        client_id,
        len(df),
    )
    return feature_scaler, target_scaler


def make_sequences(context_df, eval_df, feature_scaler, target_scaler, feature_cols, config):
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
        features = feature_scaler.transform(group[feature_cols].to_numpy(dtype=np.float32)).astype(np.float32)
        targets = target_scaler.transform(group[[TARGET_COL]].to_numpy(dtype=np.float32)).reshape(-1).astype(np.float32)
        eval_mask = group["_is_eval"].to_numpy(dtype=bool)
        eval_positions = np.flatnonzero(eval_mask)
        valid_positions = eval_positions[eval_positions >= seq_len - 1]
        if len(valid_positions) == 0:
            continue

        X_parts.append(np.stack([features[pos - seq_len + 1 : pos + 1] for pos in valid_positions]))
        y_parts.append(targets[valid_positions])

    if not X_parts:
        raise ValueError(f"No sequences produced. sequence_length={seq_len} may be too large.")
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


def prepare_client_data(data_dir, client_id, feature_scaler, target_scaler, feature_cols, config):
    client_dir = data_dir / client_id
    train_df = load_split_csv(client_dir, "train", feature_cols)
    valid_df = load_split_csv(client_dir, "valid", feature_cols)
    test_df = load_split_csv(client_dir, "test", feature_cols)

    X_train, y_train = make_sequences(pd.DataFrame(columns=train_df.columns), train_df, feature_scaler, target_scaler, feature_cols, config)
    X_valid, y_valid = make_sequences(train_df, valid_df, feature_scaler, target_scaler, feature_cols, config)
    X_test, y_test = make_sequences(pd.concat([train_df, valid_df], ignore_index=True), test_df, feature_scaler, target_scaler, feature_cols, config)

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
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
    }


def prepare_all_clients(data_dir, feature_cols, config, logger):
    clients = {}
    for client_id in CLIENT_IDS:
        feature_scaler, target_scaler = fit_client_scalers(data_dir, client_id, feature_cols, logger)
        clients[client_id] = prepare_client_data(data_dir, client_id, feature_scaler, target_scaler, feature_cols, config)
        logger.info(
            "%s prepared sequences train=%d valid=%d test=%d seq_len=%d",
            client_id,
            clients[client_id]["train_samples"],
            clients[client_id]["valid_samples"],
            clients[client_id]["test_samples"],
            config["sequence_length"],
        )
    return clients


def build_loss(target_scaler, config, device):
    return HuberSMAPELoss(
        target_scaler=target_scaler,
        huber_delta=config["huber_delta"],
        smape_weight=config["smape_loss_weight"],
    ).to(device)


def inverse_target(values, target_scaler):
    if target_scaler is None:
        return values
    return target_scaler.inverse_transform(values.reshape(-1, 1)).reshape(-1)


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


def evaluate(model, loader, target_scaler, device):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for X, y in loader:
            out = model(X.to(device)).cpu().numpy().reshape(-1)
            preds.append(out)
            targets.append(y.numpy().reshape(-1))
    pred = np.maximum(inverse_target(np.concatenate(preds), target_scaler), 0.0)
    target = inverse_target(np.concatenate(targets), target_scaler)
    sse = float(np.sum((pred - target) ** 2))
    sum_y = float(np.sum(target))
    sum_y2 = float(np.sum(target**2))
    n = int(len(target))
    rmse = float(np.sqrt(sse / n))
    mae = float(np.mean(np.abs(pred - target)))
    smape = float(np.mean(2.0 * np.abs(pred - target) / (np.abs(target) + np.abs(pred) + 1e-8)) * 100.0)
    sst = sum_y2 - (sum_y * sum_y / n)
    r2 = float(1.0 - sse / sst) if sst > 0 else 0.0
    return {
        "rmse": rmse,
        "mae": mae,
        "smape": smape,
        "r2": r2,
        "num_samples": n,
        "sse": sse,
        "sum_y": sum_y,
        "sum_y2": sum_y2,
    }


def average_state_dicts(states, weights):
    total = float(sum(weights))
    averaged = {}
    for key in states[0]:
        averaged[key] = sum(state[key] * (weight / total) for state, weight in zip(states, weights))
    return averaged


def aggregate_metrics(rows):
    total = sum(row["num_samples"] for row in rows)
    sse = sum(row["sse"] for row in rows)
    sum_y = sum(row["sum_y"] for row in rows)
    sum_y2 = sum(row["sum_y2"] for row in rows)
    sst = sum_y2 - (sum_y * sum_y / total)
    return {
        "rmse": float(np.sqrt(sse / total)),
        "mae": float(sum(row["mae"] * row["num_samples"] for row in rows) / total),
        "smape": float(sum(row["smape"] * row["num_samples"] for row in rows) / total),
        "r2": float(1.0 - sse / sst) if sst > 0 else 0.0,
        "num_samples": int(total),
        "sse": float(sse),
        "sum_y": float(sum_y),
        "sum_y2": float(sum_y2),
    }


def compact_metrics(row):
    return {key: row[key] for key in ("rmse", "mae", "smape", "r2", "num_samples")}


def clone_model_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def clone_state_subset(model, prefixes):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if key.startswith(prefixes)
    }


def load_state_subset(model, partial_state):
    state = model.state_dict()
    for key, value in partial_state.items():
        state[key] = value.to(state[key].device)
    model.load_state_dict(state)


def flatten_state_delta(local_state, reference_state):
    delta = np.concatenate(
        [
            (local_state[key] - reference_state[key]).detach().cpu().numpy().reshape(-1)
            for key in sorted(reference_state)
        ]
    )
    return np.concatenate([np.maximum(delta, 0.0), np.maximum(-delta, 0.0)])


def clone_partial_state(state):
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def set_trainable_layers(model, trainable_prefixes):
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(trainable_prefixes)


def reset_trainable_layers(model):
    for parameter in model.parameters():
        parameter.requires_grad = True


def make_optimizer(model, learning_rate):
    return torch.optim.Adam((param for param in model.parameters() if param.requires_grad), lr=learning_rate)


def train_head_only(model, loader, criterion, config, device):
    set_trainable_layers(model, HEAD_LAYER_PREFIXES)
    optimizer = make_optimizer(model, config["learning_rate"])
    for _ in range(config["head_finetune_epochs"]):
        train_one_epoch(model, loader, optimizer, criterion, device)
    reset_trainable_layers(model)


def train_pa_cfl_shared_global_model(clients, feature_cols, config, device, logger):
    global_model = LightweightLSTM(input_size=len(feature_cols), hidden_size=config["hidden_size"]).to(device)
    warmup_rounds = config["pa_cfl_global_warmup_rounds"]
    logger.info(
        "[PA-CFL Shared Global] warmup_rounds=%d members=%s",
        warmup_rounds,
        list(CLIENT_IDS),
    )

    for round_idx in range(1, warmup_rounds + 1):
        local_states = []
        weights = []
        for client_id in CLIENT_IDS:
            local_model = copy.deepcopy(global_model).to(device)
            optimizer = torch.optim.Adam(local_model.parameters(), lr=config["learning_rate"])
            criterion = build_loss(clients[client_id]["target_scaler"], config, device)
            for _ in range(config["epochs_per_round"]):
                train_one_epoch(local_model, clients[client_id]["train_loader"], optimizer, criterion, device)
            local_states.append(clone_model_state(local_model))
            weights.append(clients[client_id]["train_samples"])
        global_model.load_state_dict(average_state_dicts(local_states, weights))
        logger.info("[PA-CFL Shared Global] warmup round %d/%d complete", round_idx, warmup_rounds)

    return clone_model_state(global_model)


def train_fedavg_group(label, members, clients, feature_cols, config, device, logger, stage, initial_state=None):
    global_model = LightweightLSTM(input_size=len(feature_cols), hidden_size=config["hidden_size"]).to(device)
    if initial_state is not None:
        global_model.load_state_dict(initial_state)
    history = []
    for round_idx in range(1, config["num_rounds"] + 1):
        local_states = []
        weights = []
        for client_id in members:
            local_model = copy.deepcopy(global_model).to(device)
            optimizer = torch.optim.Adam(local_model.parameters(), lr=config["learning_rate"])
            criterion = build_loss(clients[client_id]["target_scaler"], config, device)
            for _ in range(config["epochs_per_round"]):
                train_one_epoch(local_model, clients[client_id]["train_loader"], optimizer, criterion, device)
            local_states.append(clone_model_state(local_model))
            weights.append(clients[client_id]["train_samples"])
        global_model.load_state_dict(average_state_dicts(local_states, weights))

        val_rows = []
        for client_id in members:
            row = evaluate(global_model, clients[client_id]["valid_loader"], clients[client_id]["target_scaler"], device)
            row.update({"client_id": client_id, "cluster": int(label), "round": round_idx, "stage": stage})
            val_rows.append(row)
        summary = aggregate_metrics(val_rows)
        logger.info(
            "[%s %s] round %d/%d valid RMSE=%.4f MAE=%.4f R2=%.4f SMAPE=%.4f",
            stage,
            label,
            round_idx,
            config["num_rounds"],
            summary["rmse"],
            summary["mae"],
            summary["r2"],
            summary["smape"],
        )
        history.extend(val_rows)
    return {client_id: global_model for client_id in members}, history


def train_personalized_head_group(label, members, clients, feature_cols, config, device, logger, stage, initial_state):
    global_model = LightweightLSTM(input_size=len(feature_cols), hidden_size=config["hidden_size"]).to(device)
    global_model.load_state_dict(initial_state)
    client_models = {}
    for client_id in members:
        model = LightweightLSTM(input_size=len(feature_cols), hidden_size=config["hidden_size"]).to(device)
        model.load_state_dict(initial_state)
        client_models[client_id] = model

    history = []
    for round_idx in range(1, config["num_rounds"] + 1):
        local_common_states = []
        weights = []
        shared_common_state = clone_state_subset(global_model, COMMON_LAYER_PREFIXES)
        for client_id in members:
            local_model = client_models[client_id]
            load_state_subset(local_model, shared_common_state)
            reset_trainable_layers(local_model)
            optimizer = make_optimizer(local_model, config["learning_rate"])
            criterion = build_loss(clients[client_id]["target_scaler"], config, device)
            for _ in range(config["epochs_per_round"]):
                train_one_epoch(local_model, clients[client_id]["train_loader"], optimizer, criterion, device)
            local_common_states.append(clone_state_subset(local_model, COMMON_LAYER_PREFIXES))
            weights.append(clients[client_id]["train_samples"])

        averaged_common_state = average_state_dicts(local_common_states, weights)
        load_state_subset(global_model, averaged_common_state)

        val_rows = []
        for client_id in members:
            load_state_subset(client_models[client_id], averaged_common_state)
            row = evaluate(client_models[client_id], clients[client_id]["valid_loader"], clients[client_id]["target_scaler"], device)
            row.update({"client_id": client_id, "cluster": int(label), "round": round_idx, "stage": stage})
            val_rows.append(row)
        summary = aggregate_metrics(val_rows)
        logger.info(
            "[%s %s] round %d/%d valid RMSE=%.4f MAE=%.4f R2=%.4f SMAPE=%.4f",
            stage,
            label,
            round_idx,
            config["num_rounds"],
            summary["rmse"],
            summary["mae"],
            summary["r2"],
            summary["smape"],
        )
        history.extend(val_rows)

    final_common_state = clone_state_subset(global_model, COMMON_LAYER_PREFIXES)
    if config["head_finetune_epochs"] > 0:
        for client_id in members:
            model = client_models[client_id]
            load_state_subset(model, final_common_state)
            criterion = build_loss(clients[client_id]["target_scaler"], config, device)
            train_head_only(model, clients[client_id]["train_loader"], criterion, config, device)
            row = evaluate(model, clients[client_id]["valid_loader"], clients[client_id]["target_scaler"], device)
            row.update({"client_id": client_id, "cluster": int(label), "round": config["head_finetune_epochs"], "stage": f"{stage}_head_finetune"})
            history.append(row)
            logger.info(
                "[%s Head %s] client=%s epochs=%d valid RMSE=%.4f MAE=%.4f R2=%.4f SMAPE=%.4f",
                stage,
                label,
                client_id,
                config["head_finetune_epochs"],
                row["rmse"],
                row["mae"],
                row["r2"],
                row["smape"],
            )

    return client_models, history


def train_dynamic_pa_cfl(clients, initial_clusters, feature_cols, config, device, logger):
    shared_initial_state = train_pa_cfl_shared_global_model(clients, feature_cols, config, device, logger)
    initial_common_state = {
        key: value
        for key, value in shared_initial_state.items()
        if key.startswith(COMMON_LAYER_PREFIXES)
    }
    client_models = {}
    for client_id in CLIENT_IDS:
        model = LightweightLSTM(input_size=len(feature_cols), hidden_size=config["hidden_size"]).to(device)
        model.load_state_dict(shared_initial_state)
        client_models[client_id] = model

    current_clusters = {int(label): list(members) for label, members in initial_clusters.items()}
    common_states = {label: clone_partial_state(initial_common_state) for label in current_clusters}
    history = []
    recluster_events = []

    for round_idx in range(1, config["num_rounds"] + 1):
        next_common_states = {}
        latest_client_common_states = {}
        latest_update_vectors = {}

        for label, members in sorted(current_clusters.items()):
            if not members:
                continue
            shared_common_state = common_states.get(label, initial_common_state)
            local_common_states = []
            weights = []

            for client_id in members:
                local_model = client_models[client_id]
                load_state_subset(local_model, shared_common_state)
                reset_trainable_layers(local_model)
                optimizer = make_optimizer(local_model, config["learning_rate"])
                criterion = build_loss(clients[client_id]["target_scaler"], config, device)
                for _ in range(config["epochs_per_round"]):
                    train_one_epoch(local_model, clients[client_id]["train_loader"], optimizer, criterion, device)

                local_common_state = clone_state_subset(local_model, COMMON_LAYER_PREFIXES)
                local_common_states.append(local_common_state)
                latest_client_common_states[client_id] = local_common_state
                latest_update_vectors[client_id] = flatten_state_delta(local_common_state, shared_common_state)
                weights.append(clients[client_id]["train_samples"])

            averaged_common_state = average_state_dicts(local_common_states, weights)
            next_common_states[label] = averaged_common_state

            val_rows = []
            for client_id in members:
                load_state_subset(client_models[client_id], averaged_common_state)
                row = evaluate(client_models[client_id], clients[client_id]["valid_loader"], clients[client_id]["target_scaler"], device)
                row.update({"client_id": client_id, "cluster": int(label), "round": round_idx, "stage": "pa_cfl_valid"})
                val_rows.append(row)
            summary = aggregate_metrics(val_rows)
            logger.info(
                "[PA-CFL Bubble %s] round %d/%d valid RMSE=%.4f MAE=%.4f R2=%.4f SMAPE=%.4f members=%s",
                label,
                round_idx,
                config["num_rounds"],
                summary["rmse"],
                summary["mae"],
                summary["r2"],
                summary["smape"],
                members,
            )
            history.extend(val_rows)

        common_states = next_common_states

        should_recluster = (
            config["recluster_interval"] > 0
            and round_idx % config["recluster_interval"] == 0
            and round_idx < config["num_rounds"]
        )
        if should_recluster:
            clustering_meta, reclustered = cluster_vectors(latest_update_vectors, config)
            global_common_state = average_state_dicts(
                [latest_client_common_states[client_id] for client_id in CLIENT_IDS],
                [clients[client_id]["train_samples"] for client_id in CLIENT_IDS],
            )
            current_clusters = {int(label): list(members) for label, members in reclustered.items()}
            common_states = {label: clone_partial_state(global_common_state) for label in current_clusters}
            event = {
                "stage": "pa_cfl_recluster",
                "round": round_idx,
                "assignments": clustering_meta["assignments"],
                "isolated_clients": clustering_meta["isolated_clients"],
                "k_star": clustering_meta["k_star"],
            }
            recluster_events.append(event)
            logger.info(
                "[PA-CFL Recluster] round=%d k_star=%d assignments=%s isolated=%s",
                round_idx,
                clustering_meta["k_star"],
                clustering_meta["assignments"],
                clustering_meta["isolated_clients"],
            )

    for label, members in sorted(current_clusters.items()):
        final_common_state = common_states[label]
        if config["head_finetune_epochs"] <= 0:
            continue
        for client_id in members:
            model = client_models[client_id]
            load_state_subset(model, final_common_state)
            criterion = build_loss(clients[client_id]["target_scaler"], config, device)
            train_head_only(model, clients[client_id]["train_loader"], criterion, config, device)
            row = evaluate(model, clients[client_id]["valid_loader"], clients[client_id]["target_scaler"], device)
            row.update({"client_id": client_id, "cluster": int(label), "round": config["head_finetune_epochs"], "stage": "pa_cfl_head_finetune"})
            history.append(row)
            logger.info(
                "[PA-CFL Head %s] client=%s epochs=%d valid RMSE=%.4f MAE=%.4f R2=%.4f SMAPE=%.4f",
                label,
                client_id,
                config["head_finetune_epochs"],
                row["rmse"],
                row["mae"],
                row["r2"],
                row["smape"],
            )

    return client_models, history, current_clusters, recluster_events


def train_local_models(clients, feature_cols, config, device, logger):
    final_models = {}
    history = []
    total_epochs = config["num_rounds"] * config["epochs_per_round"]
    for client_id in CLIENT_IDS:
        model = LightweightLSTM(input_size=len(feature_cols), hidden_size=config["hidden_size"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        criterion = build_loss(clients[client_id]["target_scaler"], config, device)
        for epoch in range(1, total_epochs + 1):
            train_one_epoch(model, clients[client_id]["train_loader"], optimizer, criterion, device)
            row = evaluate(model, clients[client_id]["valid_loader"], clients[client_id]["target_scaler"], device)
            row.update({"client_id": client_id, "cluster": -1, "round": epoch, "stage": "local_valid"})
            history.append(row)
        final_models[client_id] = model
        logger.info("[Local %s] epochs=%d valid RMSE=%.4f MAE=%.4f R2=%.4f", client_id, total_epochs, row["rmse"], row["mae"], row["r2"])
    return final_models, history


def train_full_fedavg(clients, feature_cols, config, device, logger):
    logger.info("[Full FedAvg] members=%s", list(CLIENT_IDS))
    return train_fedavg_group(0, list(CLIENT_IDS), clients, feature_cols, config, device, logger, "full_fedavg_valid")


def train_pa_cfl(clients, clusters, isolated_clients, feature_cols, config, device, logger):
    return train_dynamic_pa_cfl(clients, clusters, feature_cols, config, device, logger)


def evaluate_test_models(models, clients, clusters, device, logger, method):
    rows = []
    for client_id in CLIENT_IDS:
        row = evaluate(models[client_id], clients[client_id]["test_loader"], clients[client_id]["target_scaler"], device)
        cluster = int(next((k for k, v in clusters.items() if client_id in v), -1))
        row.update({"client_id": client_id, "cluster": cluster, "method": method})
        rows.append(row)
        logger.info(
            "[Test %s %s] cluster=%s RMSE=%.4f MAE=%.4f R2=%.4f SMAPE=%.4f",
            method,
            client_id,
            cluster,
            row["rmse"],
            row["mae"],
            row["r2"],
            row["smape"],
        )
    return rows, aggregate_metrics(rows)


def run_single_seed(data_dir, outputs_dir, feature_cols, feature_selection, seed, base_config, logger):
    config = dict(base_config)
    config["seed"] = seed
    set_seed(seed)
    logger.info("=== Repeat seed=%d started ===", seed)

    feature_path = outputs_dir / f"feature_importances_seed_{seed}.json"
    clustering_path = outputs_dir / f"clustering_results_seed_{seed}.json"
    feature_importances = extract_feature_importances(data_dir, feature_path, feature_cols, config, logger)
    clustering_result, pa_clusters = run_server_clustering(feature_importances, clustering_path, config, logger)
    if seed == base_config["seeds"][0]:
        with (outputs_dir / "feature_importances.json").open("w", encoding="utf-8") as f:
            json.dump(feature_importances, f, indent=4)
        with (outputs_dir / "clustering_results.json").open("w", encoding="utf-8") as f:
            json.dump(clustering_result, f, indent=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training device: %s", device)
    clients = prepare_all_clients(data_dir, feature_cols, config, logger)

    local_models, local_history = train_local_models(clients, feature_cols, config, device, logger)
    local_rows, local_summary = evaluate_test_models(local_models, clients, {-1: list(CLIENT_IDS)}, device, logger, "local")

    full_models, full_history = train_full_fedavg(clients, feature_cols, config, device, logger)
    full_rows, full_summary = evaluate_test_models(full_models, clients, {0: list(CLIENT_IDS)}, device, logger, "full_fedavg")

    pa_models, pa_history, final_pa_clusters, recluster_events = train_pa_cfl(
        clients,
        pa_clusters,
        clustering_result["isolated_clients"],
        feature_cols,
        config,
        device,
        logger,
    )
    pa_rows, pa_summary = evaluate_test_models(pa_models, clients, final_pa_clusters, device, logger, "pa_cfl")

    summaries = {
        "local": compact_metrics(local_summary),
        "full_fedavg": compact_metrics(full_summary),
        "pa_cfl": compact_metrics(pa_summary),
    }
    logger.info("=== Repeat seed=%d summary: %s ===", seed, json.dumps(summaries, sort_keys=True))
    return {
        "seed": seed,
        "config": config,
        "feature_selection": feature_selection,
        "clustering": clustering_result,
        "dynamic_reclustering": recluster_events,
        "final_pa_clusters": {str(label): members for label, members in final_pa_clusters.items()},
        "validation_history": local_history + full_history + pa_history,
        "test_by_client": {
            "local": [compact_metrics(row) | {"client_id": row["client_id"], "cluster": row["cluster"]} for row in local_rows],
            "full_fedavg": [compact_metrics(row) | {"client_id": row["client_id"], "cluster": row["cluster"]} for row in full_rows],
            "pa_cfl": [compact_metrics(row) | {"client_id": row["client_id"], "cluster": row["cluster"]} for row in pa_rows],
        },
        "test_summary": summaries,
    }


def summarize_repeats(repeats):
    methods = ("local", "full_fedavg", "pa_cfl")
    metrics = ("rmse", "mae", "smape", "r2")
    summary = {}
    for method in methods:
        summary[method] = {}
        for metric in metrics:
            values = np.array([repeat["test_summary"][method][metric] for repeat in repeats], dtype=float)
            summary[method][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
        summary[method]["num_samples"] = repeats[0]["test_summary"][method]["num_samples"]
    return summary


def parse_seeds(value):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run CSV-based PA-CFL with baselines and repeated evaluation.")
    parser.add_argument("--max-samples", type=int, default=200000)
    parser.add_argument("--xgb-estimators", type=int, default=200)
    parser.add_argument("--train-max-samples", type=int, default=0)
    parser.add_argument("--num-rounds", type=int, default=4)
    parser.add_argument("--epochs-per-round", type=int, default=1)
    parser.add_argument("--personalized-epochs", type=int, default=4)
    parser.add_argument("--pa-cfl-global-warmup-rounds", type=int, default=1)
    parser.add_argument("--head-finetune-epochs", type=int, default=1)
    parser.add_argument("--recluster-interval", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--smape-loss-weight", type=float, default=0.1)
    parser.add_argument("--sequence-length", type=int, default=28)
    parser.add_argument("--max-clusters", type=int, default=5)
    parser.add_argument("--corr-threshold", type=float, default=0.95)
    parser.add_argument("--anova-top-k", type=int, default=12)
    parser.add_argument("--feature-selection-max-samples-per-client", type=int, default=0)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "src" / "fedstock_data" / "outputs" / "clients"
    outputs_dir = project_root / "outputs"
    logger, log_path = setup_logger(outputs_dir / "logs")

    seeds = parse_seeds(args.seeds)
    base_config = {
        "seeds": seeds,
        "max_samples": args.max_samples,
        "xgb_estimators": args.xgb_estimators,
        "max_clusters": args.max_clusters,
        "complexity_penalty": 0.001,
        "singleton_penalty": 0.05,
        "isolation_std_multiplier": 1.0,
        "train_max_samples": args.train_max_samples or None,
        "num_rounds": args.num_rounds,
        "epochs_per_round": args.epochs_per_round,
        "personalized_epochs": args.personalized_epochs,
        "pa_cfl_global_warmup_rounds": args.pa_cfl_global_warmup_rounds,
        "head_finetune_epochs": args.head_finetune_epochs,
        "recluster_interval": args.recluster_interval,
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "learning_rate": args.learning_rate,
        "huber_delta": args.huber_delta,
        "smape_loss_weight": args.smape_loss_weight,
        "sequence_length": args.sequence_length,
        "candidate_feature_cols": CANDIDATE_FEATURE_COLS,
        "target_col": TARGET_COL,
        "server_clustering_source": "train.csv only",
        "feature_scaler": "client-specific StandardScaler fitted on each client's train.csv only",
        "target_scaler": "client-specific RobustScaler fitted on each client's train.csv only",
        "loss": "HuberSMAPELoss: Huber on RobustScaler-transformed targets plus SMAPE term on original target scale",
        "pa_cfl_personalization": "bubble-level FedAvg shares only LSTM parameters; each client keeps and fine-tunes an independent fc head",
        "pa_cfl_dynamic_reclustering": "every recluster_interval rounds, compare latest client LSTM update vectors, rebuild bubbles, update the global LSTM, and deploy it before continuing training",
        "corr_threshold": args.corr_threshold,
        "anova_top_k": args.anova_top_k,
        "feature_selection_max_samples_per_client": args.feature_selection_max_samples_per_client or None,
    }
    logger.info("CSV PA-CFL repeated experiment started")
    logger.info("Base config: %s", json.dumps(base_config, ensure_ascii=False, sort_keys=True))

    started = time.time()
    feature_selection_path = outputs_dir / "feature_selection.json"
    feature_cols, feature_selection = select_features(data_dir, base_config, feature_selection_path, logger)
    base_config["feature_cols"] = feature_cols

    repeats = []
    for seed in seeds:
        repeats.append(run_single_seed(data_dir, outputs_dir, feature_cols, feature_selection, seed, base_config, logger))

    aggregate = summarize_repeats(repeats)
    elapsed = time.time() - started
    result = {
        "elapsed_seconds": elapsed,
        "config": base_config,
        "feature_selection": feature_selection,
        "repeats": repeats,
        "aggregate_summary": aggregate,
        "log_path": str(log_path),
    }
    results_dir = outputs_dir / "experiments"
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"csv_baseline_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    logger.info("CSV PA-CFL repeated experiment finished in %.2fs", elapsed)
    logger.info("Aggregate summary: %s", json.dumps(aggregate, sort_keys=True))
    logger.info("Saved results: %s", result_path)
    logger.info("Log file: %s", log_path)


if __name__ == "__main__":
    main()
