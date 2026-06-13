import os
import sqlite3

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


CANDIDATE_FEATURE_COLS = [
    "dayofweek",
    "month",
    "is_weekend",
    "is_holiday",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_28",
    "rolling_std_28",
    "price_change_rate",
    "sell_price",
    "week_of_year",
    "is_month_start",
    "is_month_end",
]


class FedStockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_group_time_split_indices(item_ids, train_ratio=0.7, val_ratio=0.15):
    """
    Return chronological train/validation/test row indices inside each item_id.

    The input rows are sorted by item_id and date in load_client_data. Splitting
    inside each item prevents LSTM windows and preprocessing fits from mixing
    unrelated products or future evaluation rows into training state.
    """
    item_ids = np.asarray(item_ids)
    train_indices = []
    val_indices = []
    test_indices = []

    for item_id in pd.unique(item_ids):
        group_indices = np.where(item_ids == item_id)[0]
        n_rows = len(group_indices)
        train_end = int(n_rows * train_ratio)
        val_end = int(n_rows * (train_ratio + val_ratio))

        train_indices.extend(group_indices[:train_end])
        val_indices.extend(group_indices[train_end:val_end])
        test_indices.extend(group_indices[val_end:])

    return {
        "train": np.asarray(train_indices, dtype=int),
        "val": np.asarray(val_indices, dtype=int),
        "test": np.asarray(test_indices, dtype=int),
    }


def _load_client_frame(client_id, data_dir):
    db_path = os.path.join(data_dir, f"client_{client_id}.db")
    client_path = os.path.join(data_dir, client_id)
    features_path = os.path.join(client_path, "features.parquet")
    csv_train_path = os.path.join(client_path, "train.csv")
    csv_valid_path = os.path.join(client_path, "valid.csv")

    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        query = (
            "SELECT f.*, s.sales as quantity "
            "FROM FEATURES f "
            "JOIN SALES_RECORDS s "
            "ON f.item_id = s.item_id AND f.sale_date = s.sale_date"
        )
        df = pd.read_sql_query(query, conn)
        conn.close()
        df["date"] = pd.to_datetime(df["sale_date"])
    elif os.path.exists(csv_train_path):
        df_train = pd.read_csv(csv_train_path)
        df_valid = pd.read_csv(csv_valid_path)
        df = pd.concat([df_train, df_valid], ignore_index=True)
        if "sales" in df.columns:
            df = df.rename(columns={"sales": "quantity"})
        if "event_flag" in df.columns:
            df = df.rename(columns={"event_flag": "is_holiday"})
        df["date"] = pd.to_datetime(df["date"])
        if "rolling_std_28" not in df.columns:
            df["rolling_std_28"] = 0.0
    else:
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Cannot find data for client: {client_id}")
        df = pd.read_parquet(features_path)
        date_col = "date" if "date" in df.columns else "sale_date"
        df["date"] = pd.to_datetime(df[date_col])

    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    return df.sort_values(by=["item_id", "date"]).reset_index(drop=True)


def load_client_data(
    client_id,
    data_dir="src/4/data/clients",
    sequence_length=1,
    feature_cols=None,
    scale=False,
    return_metadata=False,
):
    """
    Load one client's tabular rows.

    The default returns raw features. Callers should split rows first, then fit
    scalers on train rows only. Pass scale=True only for legacy one-shot loaders
    that do not report validation/test metrics.
    """
    df = _load_client_frame(client_id, data_dir)
    feature_cols = list(feature_cols) if feature_cols is not None else list(CANDIDATE_FEATURE_COLS)

    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing feature columns for {client_id}: {missing_cols}")

    target_col = "target_7d" if "target_7d" in df.columns else "quantity"
    df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

    X_raw = df[feature_cols].values.astype(np.float32)
    y_raw = df[target_col].values.astype(np.float32)

    if scale:
        scaler = StandardScaler()
        X_values = scaler.fit_transform(X_raw).astype(np.float32)
    else:
        scaler = None
        X_values = X_raw

    if sequence_length > 1:
        # Sequence creation is intentionally handled by the caller so it can
        # respect train/test boundaries and item_id groups.
        pass

    if return_metadata:
        metadata = {
            "item_id": df["item_id"].astype(str).values,
            "date": pd.to_datetime(df["date"]).values,
            "target_col": target_col,
            "feature_cols": feature_cols,
        }
        return X_values, y_raw, scaler, metadata

    return X_values, y_raw, scaler


def get_dataloader(
    client_id,
    batch_size=64,
    shuffle=True,
    data_dir="src/fedstock_data/data/clients",
    feature_cols=None,
):
    X, y, scaler = load_client_data(client_id, data_dir, feature_cols=feature_cols, scale=True)
    dataset = FedStockDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, scaler


if __name__ == "__main__":
    print("Testing data loader for CA_1_HOBBIES_1...")
    dl, scaler = get_dataloader("CA_1_HOBBIES_1", data_dir="src/fedstock_data/data/clients")
    print(f"Total batches: {len(dl)}")
    for X_batch, y_batch in dl:
        print(f"X_batch shape: {X_batch.shape}")
        print(f"y_batch shape: {y_batch.shape}")
        break
