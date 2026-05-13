# Fedstock-AI

PA-CFL(Privacy-Adaptive Clustered Federated Learning)를 기반으로 매장별 판매량을 예측하는 실험 파이프라인입니다. SQLite 클라이언트 데이터 소스를 그대로 사용하며, ANOVA 기반 feature selection, 초기 feature clustering, bubble 단위 federated learning, 동적 재클러스터링, isolated client 개인화 학습을 한 번에 실행합니다.

## 실행

```bash
python run_fl_baselines.py
```

실행 시 세 가지 전략을 비교합니다.

- `Local`: 클라이언트별 독립 학습
- `Global FedAvg`: 전체 클라이언트를 하나의 글로벌 FedAvg 그룹으로 학습
- `PA-CFL`: ANOVA feature selection, bubble clustering, shared LSTM 학습, 동적 재클러스터링, head fine-tuning 적용

주요 하이퍼파라미터는 [run_fl_baselines.py](run_fl_baselines.py) 안에서 조정합니다.

| 파라미터 | 기본값 | 설명 |
| :--- | :---: | :--- |
| `num_rounds` | 100 | federated learning 라운드 수 |
| `epochs_per_round` | 3 | 라운드당 local epoch 수 |
| `seq_len` | 14 | LSTM 입력 시퀀스 길이 |
| `global_warmup_rounds` | 20 | 공통 글로벌 warm-up 라운드 수 |
| `recluster_interval` | 20 | 동적 재클러스터링 주기 |
| `head_finetune_epochs` | 20 | 개인화 head fine-tuning epoch 수 |

## 파이프라인

1. 데이터 로딩
   - `src/fedstock_data/data/clients/client_*.db`의 SQLite 클라이언트 DB를 사용합니다.
   - `src.dataset.load_client_data()`가 기존 데이터 소스 선택 방식을 유지합니다.

2. ANOVA feature selection
   - 16개 후보 feature에 대해 ANOVA F-score와 p-value를 계산합니다.
   - `alpha=0.10`, `top_k=12` 기준으로 유의미한 12개 feature를 선택합니다.
   - 결과는 `outputs/feature_selection.json`에 저장됩니다.

3. 초기 clustering
   - 각 클라이언트의 noisy feature importance를 수집합니다.
   - EMD 거리와 Agglomerative Clustering으로 bubble을 구성합니다.

4. PA-CFL 학습
   - 초기 warm-up에서는 공통 글로벌 모델을 학습합니다.
   - bubble 내부에서는 shared LSTM body를 FedAvg로 학습하고, output head는 클라이언트별로 유지합니다.
   - 동적 재클러스터링 이후 다중 클러스터에는 클러스터별 글로벌 모델을 배포합니다.
   - 공통 글로벌 모델은 계속 업데이트하되, 초기 warm-up과 isolated client에만 배포합니다.

5. 결과 저장
   - 평가 리포트, 시각화, feature selection, clustering history, 모델 가중치, 학습 로그가 `outputs/` 아래에 저장됩니다.

## Feature Selection 출력

`outputs/feature_selection.json`은 다음 정보를 포함합니다.

- `candidate_features`: 16개 후보 feature
- `selected_features`: ANOVA로 선택된 12개 feature
- `ranked_features`: feature별 `rank`, `f_score`, `p_value`, `significant`, `selected`
- `selected_significant_count`: 선택된 feature 중 유의성 기준을 통과한 개수
- `total_samples`, `num_clients`, `client_sample_count_stats`: 계산에 사용된 데이터 요약

현재 선택 feature는 다음과 같습니다.

```text
rolling_mean_28, rolling_mean_7, rolling_std_28, lag_7,
rolling_std_7, lag_14, lag_28, sell_price,
is_month_end, is_month_start, is_holiday, month
```

## Clustering 출력

`outputs/clustering_results.json`은 최종 결과만 덮어쓰지 않고, 초기 clustering과 모든 동적 재클러스터링 결과를 `records` 배열에 순서대로 저장합니다.

파일 크기를 줄이기 위해 클라이언트별 assignment와 다중 bubble의 클라이언트 목록은 생략합니다. 대신 다음 요약을 제공합니다.

- `sequence`, `stage`, `round`
- `k_star`, `num_clusters`
- `cluster_sizes`, `cluster_size_stats`
- `num_multi_client_bubbles`, `num_isolated_clients`
- `isolated_clients`

## 디렉토리 구조

최상위 폴더에는 실행 진입점과 문서만 두고, 코드와 산출물은 하위 폴더로 분리합니다.

```text
model/
├─ .gitignore
├─ README.md
├─ run_fl_baselines.py
├─ src/
│  ├─ dataset.py
│  ├─ losses.py
│  ├─ fl/
│  │  ├─ client.py
│  │  ├─ extract_features.py
│  │  ├─ privacy.py
│  │  ├─ server.py
│  │  └─ server_clustering.py
│  ├─ models/
│  │  └─ lstm.py
│  └─ fedstock_data/
├─ outputs/
│  ├─ runs/
│  │  ├─ (Latest experimental results folder)/
│  │  │  ├─ models/
│  │  │  │  ├─ bubbles/
│  │  │  │  └─ clients/
│  └─ └─ └─ (Output files)
└─ temp/
   └─ reference and experiment scratch files
```

## 주요 모듈

- `src/dataset.py`: SQLite/CSV/parquet 데이터 로딩, 후보 feature 정의, scaling
- `src/losses.py`: `HuberSMAPELoss`
- `src/models/lstm.py`: Lightweight LSTM 모델
- `src/fl/client.py`: `FedStockClient`, shared LSTM/body-head 분리 학습
- `src/fl/server.py`: PA-CFL orchestration, warm-up, bubble FedAvg, 동적 재클러스터링, 결과 저장
- `src/fl/server_clustering.py`: EMD 거리와 Agglomerative Clustering
- `src/fl/extract_features.py`: ANOVA feature selection과 noisy feature importance 추출
- `src/fl/privacy.py`: XGBoost importance와 Laplace noise 기반 DP 처리

## 산출물

| 경로 | 설명 |
| :--- | :--- |
| `outputs/evaluation_report.md` | 전략별 최종 RMSE/SMAPE 요약 |
| `outputs/baseline_comparison.png` | 전략별 성능 비교 차트 |
| `outputs/feature_selection.json` | ANOVA feature selection 결과 |
| `outputs/feature_importances.json` | 클라이언트별 noisy feature importance |
| `outputs/clustering_results.json` | 초기 및 동적 clustering history |
| `outputs/models/clients/` | 클라이언트별 최종 모델 |
| `outputs/models/bubbles/` | bubble별 shared LSTM body |
| `outputs/logs/training_full.log` | 전체 학습 로그 |

## 의존성

주요 패키지는 다음과 같습니다.

- Python 3.12+
- PyTorch
- NumPy, Pandas, Scikit-learn
- Matplotlib
- Flower(`flwr`)
- XGBoost

ANOVA feature selection은 Scikit-learn만 필요합니다. XGBoost는 noisy feature importance 추출 경로에서 사용됩니다.
