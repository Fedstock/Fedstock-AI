import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class FedStockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_client_data(client_id, data_dir="src/fedstock_data/outputs/clients", sequence_length=1):
    """
    특정 클라이언트의 데이터를 로드하고 Data Leakage 방지를 위해 전처리를 수행합니다.
    """
    client_path = os.path.join(data_dir, client_id)
    features_path = os.path.join(client_path, "features.parquet")
    
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"데이터를 찾을 수 없습니다: {features_path}")
        
    df = pd.read_parquet(features_path)
    
    # 1. 정렬: 시계열 처리를 위해 item_id와 date 기준으로 정렬
    df = df.sort_values(by=['item_id', 'date']).reset_index(drop=True)
    
    # 2. Data Leakage 방지: rolling feature들을 .shift(1) 처리 (item_id 그룹별로 수행하여 아이템 간 누수 방지)
    rolling_cols = ['rolling_mean_7', 'rolling_std_7', 'rolling_mean_28', 'rolling_std_28']
    
    # 아이템별로 shift 적용
    df[rolling_cols] = df.groupby('item_id')[rolling_cols].shift(1)
    
    # shift로 인해 발생한 결측치(각 아이템의 첫 번째 행) 제거
    df = df.dropna().reset_index(drop=True)
    
    # 3. 특성과 타겟 분리
    # 사용할 feature 목록 (store_id, item_id, date는 학습 특성에서 제외)
    feature_cols = [
        'dayofweek', 'month', 'is_weekend', 'is_holiday',
        'lag_7', 'lag_14', 'lag_28',
        'rolling_mean_7', 'rolling_std_7', 'rolling_mean_28', 'rolling_std_28'
    ]
    target_col = 'quantity'
    
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

def get_dataloader(client_id, batch_size=64, shuffle=True, data_dir="src/fedstock_data/outputs/clients"):
    """
    PyTorch DataLoader를 반환합니다.
    """
    X, y, scaler = load_client_data(client_id, data_dir)
    
    dataset = FedStockDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader, scaler

if __name__ == "__main__":
    # 간단한 테스트 코드
    print("Testing data loader for CA_1...")
    # 경로 조정: 현재 실행 위치가 workspace root라고 가정
    dl, scaler = get_dataloader("CA_1", data_dir="src/fedstock_data/outputs/clients")
    print(f"Total batches: {len(dl)}")
    for X_batch, y_batch in dl:
        print(f"X_batch shape: {X_batch.shape}")
        print(f"y_batch shape: {y_batch.shape}")
        break
