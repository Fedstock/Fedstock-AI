import argparse
import json
import logging
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.dataset import load_client_data
from src.fl.client import FedStockClient
from src.fl.server import BubbleServer


CLIENT_IDS = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"experiment_{timestamp}.log")

    logger = logging.getLogger("pa_cfl_experiment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger, log_path


def build_client(client_id, data_dir, config, logger):
    logger.info("Loading data for %s", client_id)
    X_scaled, y_raw, _ = load_client_data(client_id, data_dir=data_dir)

    max_samples = config["max_samples"]
    if len(X_scaled) > max_samples:
        X_scaled = X_scaled[-max_samples:]
        y_raw = y_raw[-max_samples:]

    split_idx = int(len(X_scaled) * config["train_ratio"])
    X_train, y_train = X_scaled[:split_idx], y_raw[:split_idx]
    X_val, y_val = X_scaled[split_idx:], y_raw[split_idx:]

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=config["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val_tensor, y_val_tensor),
        batch_size=config["batch_size"],
        shuffle=False,
    )

    logger.info(
        "%s dataset prepared: train=%d, val=%d, batch_size=%d",
        client_id,
        len(train_loader.dataset),
        len(val_loader.dataset),
        config["batch_size"],
    )

    return FedStockClient(
        cid=client_id,
        train_loader=train_loader,
        val_loader=val_loader,
        X_train=X_train_tensor.numpy(),
        y_train=y_train_tensor.numpy(),
        input_size=X_train.shape[1],
        hidden_size=config["hidden_size"],
        epsilon=config["epsilon"],
        learning_rate=config["learning_rate"],
    )


def weighted_final_metrics(history):
    federated = [row for row in history if row["stage"] == "federated"]
    personalized = [row for row in history if row["stage"] == "personalized"]

    final_by_bubble = {}
    for row in federated:
        final_by_bubble[row["bubble"]] = row

    final_rows = list(final_by_bubble.values()) + personalized
    if not final_rows:
        return {"rmse": None, "smape": None}

    total_samples = sum(row["num_samples"] for row in final_rows)
    return {
        "rmse": float(sum(row["rmse"] * row["num_samples"] for row in final_rows) / total_samples),
        "smape": float(sum(row["smape"] * row["num_samples"] for row in final_rows) / total_samples),
        "num_samples": int(total_samples),
    }


def run_one_experiment(name, config, data_dir, clustering_json, logger):
    logger.info("=== Experiment %s started ===", name)
    logger.info("Hyperparameters: %s", json.dumps(config, sort_keys=True))
    set_seed(config["seed"])

    clients = {
        client_id: build_client(client_id, data_dir, config, logger)
        for client_id in CLIENT_IDS
    }

    server = BubbleServer(clients)
    server.load_clustering_results(clustering_json, logger=logger)

    started_at = time.time()
    fl_history = server.step_3_federated_learning(
        num_rounds=config["num_rounds"],
        epochs_per_round=config["epochs_per_round"],
        logger=logger,
    )
    personalized_history = server.step_4_personalized_learning(
        epochs=config["personalized_epochs"],
        logger=logger,
    )
    elapsed = time.time() - started_at

    history = fl_history + personalized_history
    summary = weighted_final_metrics(history)
    summary.update({"experiment": name, "elapsed_seconds": elapsed})

    logger.info("=== Experiment %s finished in %.2fs ===", name, elapsed)
    logger.info("Final summary: %s", json.dumps(summary, sort_keys=True))
    return {"config": config, "history": history, "summary": summary}


def default_experiments(max_samples):
    return {
        "tuned": {
            "seed": 42,
            "train_ratio": 0.8,
            "max_samples": max_samples,
            "batch_size": 512,
            "hidden_size": 48,
            "learning_rate": 0.003,
            "epsilon": 1.0,
            "num_rounds": 4,
            "epochs_per_round": 2,
            "personalized_epochs": 4,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Run PA-CFL hyperparameter experiments.")
    parser.add_argument("--max-samples", type=int, default=20000)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, "src", "fedstock_data", "outputs", "clients")
    clustering_json = os.path.join(project_root, "outputs", "clustering_results.json")
    log_dir = os.path.join(project_root, "outputs", "logs")
    results_dir = os.path.join(project_root, "outputs", "experiments")
    os.makedirs(results_dir, exist_ok=True)

    logger, log_path = setup_logger(log_dir)
    logger.info("Experiment log file: %s", log_path)

    experiments = default_experiments(args.max_samples)

    results = {}
    for name, config in experiments.items():
        results[name] = run_one_experiment(name, config, data_dir, clustering_json, logger)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(results_dir, f"experiment_results_{timestamp}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    logger.info("Saved experiment results: %s", results_path)
    logger.info("Log file retained at: %s", log_path)


if __name__ == "__main__":
    main()
