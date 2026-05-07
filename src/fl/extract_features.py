import os
import json
import time
import numpy as np
import xgboost as xgb
import sys

# 프로젝트 루트 경로 추가 (sys.path)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.dataset import load_client_data
from src.fl.privacy import get_noisy_feature_importance

def extract_features_for_all_clients(clients, data_dir, output_file, max_samples=100000, epsilon=10.0):
    """
    각 클라이언트별로 데이터를 로드하여 XGBoost를 통해 피처 중요도를 추출하고,
    차등 정보 보호(Laplace Noise)가 적용된 중요도 벡터를 JSON 형태로 저장합니다.
    """
    results = {}
    
    print(f"Starting Local Feature Extraction for {len(clients)} clients...")
    print(f"Data directory: {data_dir}")
    print(f"Epsilon (Privacy budget): {epsilon}")
    
    for client_id in clients:
        start_time = time.time()
        print(f"\n[{client_id}] Processing...")
        
        try:
            # 1. DataLoader의 전처리 함수를 사용해 데이터 로드 (Data Leakage 방지 적용됨)
            X, y, scaler = load_client_data(client_id, data_dir=data_dir)
            
            # 2. 성능 향상을 위한 데이터 샘플링 (데이터가 너무 큰 경우)
            if len(X) > max_samples:
                print(f"  - Subsampling {len(X)} rows to {max_samples} for faster extraction...")
                indices = np.random.choice(len(X), max_samples, replace=False)
                X_sample = X[indices]
                y_sample = y[indices]
            else:
                X_sample = X
                y_sample = y
            
            # 3. XGBoost 피처 중요도 및 노이즈 계산 (privacy.py 활용)
            # XGBRegressor의 tree_method='hist' 등을 privacy.py 내에서 설정하면 더 빠르지만
            # 여기서는 privacy 모듈을 그대로 활용합니다.
            noisy_importance, sensitivity = get_noisy_feature_importance(X_sample, y_sample, epsilon=epsilon)
            
            # 4. 결과 저장
            results[client_id] = noisy_importance.tolist()
            
            elapsed = time.time() - start_time
            print(f"  - Completed in {elapsed:.2f}s")
            print(f"  - Top 3 Important Features (noisy): {np.argsort(noisy_importance)[-3:][::-1]}")
            
        except Exception as e:
            print(f"  - Error processing {client_id}: {e}")
            
    # 5. 추출된 피처 중요도를 파일로 저장 (Server Clustering에서 사용됨)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nFeature extraction complete! Saved to {output_file}")

if __name__ == "__main__":
    # 설정값: 전체 10개 매장(클라이언트)
    CLIENTS = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]
    DATA_DIR = os.path.join(project_root, "src/fedstock_data/outputs/clients")
    OUTPUT_FILE = os.path.join(project_root, "outputs/feature_importances.json")
    
    # 모듈 실행 (속도와 성능의 균형을 위해 max_samples=50000 권장)
    extract_features_for_all_clients(CLIENTS, DATA_DIR, OUTPUT_FILE, max_samples=50000, epsilon=10.0)
