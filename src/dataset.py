import os
import sqlite3
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

CANDIDATE_FEATURE_COLS = [
    'dayofweek', 'month', 'is_weekend', 'is_holiday',
    'lag_7', 'lag_14', 'lag_28',
    'rolling_mean_7', 'rolling_std_7', 'rolling_mean_28', 'rolling_std_28',
    'price_change_rate', 'sell_price', 'week_of_year', 'is_month_start', 'is_month_end'
]

class FedStockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_client_data(client_id, data_dir="src/4/data/clients", sequence_length=1, feature_cols=None):
    """
    특정 클라이언트의 데이터를 로드하고 Data Leakage 방지를 위해 전처리를 수행합니다.
    """
    db_path = os.path.join(data_dir, f"client_{client_id}.db")
    client_path = os.path.join(data_dir, client_id)
    features_path = os.path.join(client_path, "features.parquet")
    csv_train_path = os.path.join(client_path, "train.csv")
    csv_valid_path = os.path.join(client_path, "valid.csv")
    
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        query = "SELECT f.*, s.sales as quantity FROM FEATURES f JOIN SALES_RECORDS s ON f.item_id = s.item_id AND f.sale_date = s.sale_date"
        df = pd.read_sql_query(query, conn)
        conn.close()
        df['date'] = pd.to_datetime(df['sale_date'])
        df['dayofweek'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        # 1. 정렬: 시계열 처리를 위해 item_id와 date 기준으로 정렬
        df = df.sort_values(by=['item_id', 'date']).reset_index(drop=True)
    elif os.path.exists(csv_train_path):
        df_train = pd.read_csv(csv_train_path)
        df_valid = pd.read_csv(csv_valid_path)
        df = pd.concat([df_train, df_valid])
        if 'sales' in df.columns:
            df = df.rename(columns={'sales': 'quantity'})
        if 'event_flag' in df.columns:
            df = df.rename(columns={'event_flag': 'is_holiday'})
        df['date'] = pd.to_datetime(df['date'])
        df['dayofweek'] = df['date'].dt.dayofweek
        if 'rolling_std_28' not in df.columns:
            df['rolling_std_28'] = 0.0 # Fill missing column for compatibility
        df = df.sort_values(by=['item_id', 'date']).reset_index(drop=True)
    else:
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"데이터를 찾을 수 없습니다: {client_id}")
            
        df = pd.read_parquet(features_path)
        # 1. 정렬: 시계열 처리를 위해 item_id와 date 기준으로 정렬
        df = df.sort_values(by=['item_id', 'date']).reset_index(drop=True)
    
    # Data Leakage 방지를 위한 .shift(1)은 데이터셋 생성 단계(src/4)에서 이미 처리됨
    # 결측치(각 아이템의 첫 번째 행 등) 제거
    df = df.dropna().reset_index(drop=True)
    
    # 3. 특성과 타겟 분리
    # 사용할 feature 목록 (store_id, item_id, date는 학습 특성에서 제외)
    feature_cols = list(feature_cols) if feature_cols is not None else list(CANDIDATE_FEATURE_COLS)
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing feature columns for {client_id}: {missing_cols}")
    
    # 데이터셋에 포함된 target_7d가 있으면 사용, 없으면 quantity(당일 수요) 사용
    target_col = 'target_7d' if 'target_7d' in df.columns else 'quantity'
    
    X_raw = df[feature_cols].values
    y_raw = df[target_col].values
    
    # 4. 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # LSTM 등 시계열 모델을 위한 시퀀스 데이터 생성 로직 (선택적)
    if sequence_length > 1:
        # 이 부분은 단순화를 위해 생략되었으나, 향후 필요시 window 슬라이딩을 구현할 수 있습니다.
        # XGBoost의 경우 sequence_length=1로 사용하면 됩니다.
        pass
        
    return X_scaled, y_raw, scaler

def get_dataloader(client_id, batch_size=64, shuffle=True, data_dir="src/fedstock_data/data/clients", feature_cols=None):
    """
    PyTorch DataLoader를 반환합니다.
    """
    X, y, scaler = load_client_data(client_id, data_dir, feature_cols=feature_cols)
    
    dataset = FedStockDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader, scaler

if __name__ == "__main__":
    # 간단한 테스트 코드
    print("Testing data loader for CA_1_HOBBIES_1...")
    # 경로 조정: 현재 실행 위치가 workspace root라고 가정
    dl, scaler = get_dataloader("CA_1_HOBBIES_1", data_dir="src/fedstock_data/data/clients")
    print(f"Total batches: {len(dl)}")
    for X_batch, y_batch in dl:
        print(f"X_batch shape: {X_batch.shape}")
        print(f"y_batch shape: {y_batch.shape}")
        break
