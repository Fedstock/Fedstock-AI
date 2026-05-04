# Fedstock-AI: PA-CFL 기반 다중 매장 판매량 예측 시스템

본 프로젝트는 이질적인 유통 데이터 환경에서 프라이버시를 보호하면서 최적의 예측 모델을 구축하기 위한 **PA-CFL (Privacy-Adaptive Clustered Federated Learning)** 프레임워크의 파이토치(PyTorch) 구현체입니다. 기존 논문에서 제안된 트랜스포머(Transformer) 모델 대신 **경량화된 LSTM(Lightweight LSTM) 아키텍처**를 채택하여 실제 매장 POS 기기에서도 학습이 가능하도록 학습 효율을 높였습니다.

## 🚀 주요 특징 (Key Features)

1. **로컬 피처 중요도 추출 (Local Feature Extraction)**
   * 각 클라이언트(매장)는 로컬 데이터를 외부로 노출하지 않고, XGBoost를 사용하여 판매량 예측에 영향을 미치는 주요 특성(Feature Importance)을 추출합니다.
2. **차등 정보 보호 (Differential Privacy)**
   * 추출된 피처 중요도에 라플라스 노이즈(Laplace Noise)를 추가하여 역공학을 통한 로컬 데이터 유출을 원천 차단합니다.
3. **EMD 기반 어글로머러티브 클러스터링 (Agglomerative Clustering via EMD)**
   * 중앙 서버는 노이즈가 추가된 피처 중요도를 수집하고, **Earth Mover's Distance (EMD)** 를 활용해 데이터 분포가 유사한 매장들을 동일한 클러스터(Bubble)로 묶습니다.
4. **클러스터 기반 연합학습 및 개인화 (Clustered & Personalized FL)**
   * **다중 클라이언트 클러스터:** 클러스터 내에서 FedAvg를 수행하여 협업 모델을 학습합니다.
   * **단일(고립) 클라이언트:** 다른 매장과 특성이 크게 다른 매장은 억지로 통합하지 않고, 개인화 연합학습(Personalized FL)을 통해 독자적인 로컬 학습을 수행합니다.
---

## 📂 프로젝트 구조 (Project Structure)

```text
📦 model
 ┣ 📂 src
 │ ┣ 📂 fedstock_data    # 전처리 완료된 M5 기반 매장별(Client) 분할 데이터
 │ ┣ 📂 fl               # 연합학습 및 PA-CFL 핵심 모듈
 │ │ ┣ 📜 client.py             # 클라이언트 모델 및 로컬 학습 모듈
 │ │ ┣ 📜 server.py             # 연합학습 서버(BubbleServer) 모듈
 │ │ ┣ 📜 privacy.py            # XGBoost 피처 추출 및 라플라스 노이즈 모듈
 │ │ ┣ 📜 extract_features.py   # 전 매장 피처 중요도 추출 실행 스크립트
 │ │ ┗ 📜 server_clustering.py  # EMD 기반 클러스터링 모듈
 │ ┣ 📂 models           # 신경망 아키텍처
 │ │ ┗ 📜 lstm.py               # Lightweight LSTM 모델 정의
 │ ┗ 📜 dataset.py       # PyTorch DataLoader 및 전처리 로직 (Leakage Prevention)
 ┣ 📂 outputs            # 추출된 특징 및 클러스터링 결과물 (git 제외)
 ┣ 📜 run_fl.py          # 전체 FL 파이프라인 통합 구동 스크립트
 ┣ 📜 simulate.py        # 합성 데이터 기반 시뮬레이션용 레거시 스크립트
 ┗ 📜 README.md          # 프로젝트 설명서
```

---

## ⚙️ 실행 방법 (How to Run)

전체 파이프라인은 크게 3단계로 진행됩니다.

### Step 1: 로컬 피처 중요도 추출 및 노이즈 추가 (Client-side)
각 매장별로 데이터를 로드하여 XGBoost를 학습시키고, 노이즈가 추가된 피처 중요도(`feature_importances.json`)를 생성합니다.
```bash
python src/fl/extract_features.py
```
*(참고: 속도를 위해 내부적으로 `max_samples=50000` 등 샘플링 기법이 적용되어 있습니다.)*

### Step 2: EMD 기반 서버 클러스터링 (Server-side)
생성된 피처 중요도를 서버가 읽어 들여 EMD 거리를 계산하고, Davies-Bouldin Index 기반으로 최적의 클러스터를 구성(`clustering_results.json`)합니다.
```bash
python src/fl/server_clustering.py
```

### Step 3: 클러스터별 LSTM 연합학습 및 개인화 학습
실제 시계열 데이터를 로드하여, 구성된 클러스터 기반으로 다중 매장 연합학습(FedAvg)과 단일 매장의 개인화 학습(Personalized FL)을 수행합니다.
```bash
python run_fl.py
```

---

## 🛠 기술 스택 (Tech Stack)
* **Language:** Python 3.x
* **Machine Learning:** PyTorch, XGBoost, Scikit-learn
* **Data Processing:** Pandas, Numpy, PyArrow (Parquet)
* **Federated Learning:** Flower (flwr) - *현재 임시로 커스텀 BubbleServer 로직으로 구현됨*

## 📝 참고 문헌 (References)
* Privacy-Adaptive Clustered Federated Learning for Transformer-Based Sales Forecasting on Heterogeneous Retail Data (본 프로젝트의 기반 논문)
* M5 Forecasting - Accuracy (Kaggle Dataset)