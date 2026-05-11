# Fedstock-AI: PA-CFL 기반 다중 매장 판매량 예측 시스템

**Privacy-Adaptive Clustered Federated Learning** 프레임워크를 활용하여, 이질적인 유통 데이터 환경에서 프라이버시를 보호하면서 최적의 판매량 예측 모델을 구축하는 시스템입니다.

---

## 📋 전체 파이프라인 개요

```
[Step 0] 데이터 로드 & 전처리
    ↓   SQLite DB에서 매장별 데이터 로드 → RobustScaler 적용 → Sliding Window(14일) 시퀀스 생성
[Step 1] 피처 중요도 추출 (각 클라이언트)
    ↓   XGBoost로 피처 중요도 산출 → Gradient Clipping + Laplace 노이즈로 DP 적용 → 서버에 전송
[Step 2] EMD 기반 클러스터링 (서버)
    ↓   노이즈 적용된 피처 중요도 분포 간 EMD(Earth Mover's Distance) 거리 행렬 계산
    ↓   Agglomerative Clustering + DBI로 최적 클러스터 수(k*) 자동 결정
    ↓   유사 매장끼리 '버블(Bubble)'로 그룹화
[Step 3] 연합학습 (버블 내 FedAvg)
    ↓   Global Warmup → 버블별 LSTM Body 공유 학습 → 동적 재클러스터링 → Head Fine-tuning
[Step 4] 개인화 학습 (고립 클라이언트)
        고립(Isolated) 매장은 공유 LSTM Body 위에 독립적 Head를 학습
```

---

## ⚙️ 실행 방법

### 전체 파이프라인 (베이스라인 비교 평가 포함)

```bash
python run_fl_baselines.py
```

- **비교 대상**: Local Training / Global FedAvg / PA-CFL (제안 방식)
- **결과 출력**: `outputs/evaluation_report.md`, `outputs/baseline_comparison.png`

### 주요 하이퍼파라미터 (`run_fl_baselines.py` 내부)

| 파라미터 | 기본값 | 설명 |
| :--- | :---: | :--- |
| `num_rounds` | 100 | 연합학습 라운드 수 |
| `epochs_per_round` | 5 | 라운드당 로컬 학습 에포크 수 |
| `seq_len` | 14 | LSTM 시퀀스 길이 (과거 참조 일수) |
| `hidden_size` | 32 | LSTM 은닉 유닛 수 |
| `global_warmup_rounds` | 1 | 전역 워밍업 라운드 수 |
| `recluster_interval` | 10 | 동적 재클러스터링 주기 (라운드) |
| `head_finetune_epochs` | 1 | Head 미세 조정 에포크 수 |

---

## 🏗️ 주요 구현 내용

### 1. PA-CFL (Privacy-Adaptive Clustered Federated Learning)
- **프라이버시 보호**: 각 매장은 XGBoost 피처 중요도에 Laplace 노이즈를 추가하여 서버에 전송 (ε-차등 정보 보호)
- **적응형 클러스터링**: EMD 거리 + Agglomerative Clustering으로 유사 매장을 '버블'로 자동 분류
- **버블 내 연합학습**: FedAvg로 버블 내 모델 파라미터를 가중 평균하여 공유 모델 생성

### 2. Lightweight LSTM 아키텍처
- 매장 POS 기기의 저사양 환경을 고려한 **경량 LSTM** 채택 (기존 논문의 Transformer 대체)
- 구조: `LSTM(input, hidden=32, layers=1)` → `Linear(hidden, 1)`
- Body(LSTM)/Head(FC) 분리 설계로 개인화 연합학습에 최적화

### 3. HuberSMAPE 손실 함수
- Huber Loss와 SMAPE의 가중 결합으로, 이상치에 강건하면서도 비율 기반 정확도를 동시에 최적화
- 타겟 스케일러의 center/scale 정보를 활용하여 역스케일 기반 SMAPE를 손실에 직접 반영

---

## 🔬 추가 적용 세부 기법

### Body/Head 분리 학습 (Personalized FL)
- **LSTM Body (공유)**: 버블 내 모든 매장이 공통으로 학습하는 시계열 패턴 인코더
- **FC Head (개인화)**: 각 매장이 독립적으로 유지하는 출력 레이어
- `fit_shared_lstm()`: Body만 서버와 동기화, Head는 각 매장이 보존
- `fit_head()`: Body를 고정하고 Head만 미세 조정

### Global Warmup
- 클러스터링 이전에 전체 클라이언트로 1~2라운드 글로벌 FedAvg를 수행
- 모든 버블이 **동일한 전역 기초 지식**에서 출발하여, 소규모 버블의 학습 격리 문제를 해소

### 동적 재클러스터링 (Dynamic Re-clustering)
- 학습 중 `recluster_interval` 주기마다 각 클라이언트의 **가중치 업데이트 벡터(Delta)**를 수집
- Delta 벡터 간 유사도를 기반으로 버블을 재구성하여, 학습 과정에서 변화하는 클라이언트 특성을 반영

### 타겟 데이터 스케일링
- `RobustScaler`를 사용하여 이상치에 강건한 타겟(y) 정규화
- 학습 데이터로만 fit하여 검증 데이터 누수(Data Leakage)를 방지

### Gradient Clipping 기반 DP
- 피처 중요도 벡터의 L2 Norm을 `clip_norm`으로 제한하여 민감도(Sensitivity) 상한을 확정
- 확정된 민감도에 비례하는 Laplace 노이즈를 추가하여 ε-차등 정보 보호를 구현

---

## 📂 프로젝트 구조

```text
📦 model
 ┣ 📜 run_fl_baselines.py       # 전체 파이프라인 실행 (Local / FedAvg / PA-CFL 비교)
 ┣ 📜 losses.py                 # HuberSMAPE 손실 함수
 ┣ 📂 src
 │ ┣ 📜 dataset.py              # SQLite DB 로드 및 피처 전처리
 │ ┣ 📂 fl                      # 연합학습 핵심 모듈
 │ │ ┣ 📜 client.py             # FedStockClient (Body/Head 분리, 로컬 학습/평가)
 │ │ ┣ 📜 server.py             # BubbleServer (Global Warmup, FedAvg, 동적 재클러스터링)
 │ │ ┣ 📜 server_clustering.py  # EMD 거리 행렬 + Agglomerative Clustering
 │ │ ┣ 📜 privacy.py            # Gradient Clipping DP 기반 노이즈 생성
 │ │ ┗ 📜 extract_features.py   # 독립 실행용 피처 추출 스크립트
 │ ┣ 📂 models
 │ │ ┗ 📜 lstm.py               # Lightweight LSTM 모델
 │ ┗ 📂 fedstock_data/data/clients  # SQLite (.db) 매장별 데이터셋
 ┣ 📂 outputs                   # 학습 결과 및 시각화
 │ ┣ 📜 evaluation_report.md    # 최종 평가 리포트
 │ ┣ 📜 baseline_comparison.png # 전략 비교 시각화
 │ ┣ 📜 clustering_results.json # 클러스터링 결과
 │ ┣ 📜 feature_importances.json # 피처 중요도 결과
 │ ┗ 📜 feature_selection.json  # 피처 선택 결과
 ┣ 📂 temp                      # 비필수 파일 (문서, 이전 스크립트 등)
 ┗ 📜 README.md
```

---

## 🛠 기술 스택

| 분류 | 기술 |
| :--- | :--- |
| **Language** | Python 3.12+ |
| **Deep Learning** | PyTorch (CUDA 12.x+) |
| **Feature Extraction** | XGBoost |
| **Data Processing** | Pandas, NumPy, Scikit-learn, SQLite3 |
| **FL Framework** | Custom PA-CFL BubbleServer + Flower (flwr) |

---

## 📝 참고 문헌
- Privacy-Adaptive Clustered Federated Learning for Transformer-Based Sales Forecasting on Heterogeneous Retail Data (기반 논문)
- M5 Forecasting - Accuracy (Kaggle Dataset)