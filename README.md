# Fedstock-AI: PA-CFL 기반 다중 매장 판매량 예측 시스템

본 프로젝트는 이질적인 유통 데이터 환경에서 프라이버시를 보호하면서 최적의 예측 모델을 구축하기 위한 **PA-CFL (Privacy-Adaptive Clustered Federated Learning)** 프레임워크의 파이토치(PyTorch) 구현체입니다. 기존 논문에서 제안된 트랜스포머(Transformer) 모델 대신 **경량화된 LSTM(Lightweight LSTM) 아키텍처**를 채택하여 실제 매장 POS 기기에서도 학습이 가능하도록 구성했습니다.

## 🚀 최신 업데이트 사항 (Latest Pipeline Features)

최근 업데이트를 통해 성능과 효율성을 극대화하기 위한 여러 최적화 기법이 도입되었습니다.

1. **Sliding Window 기반 시계열 학습 (Sequence Length = 14)**
   * 단일 스텝 입력 구조에서 벗어나, 과거 14일치 데이터를 하나의 시퀀스로 묶어 LSTM에 공급함으로써 시간적 맥락(Temporal Dependency)을 정확히 학습합니다.
2. **타겟 데이터 스케일링 (Target StandardScaler)**
   * 분산이 크고 Zero-inflated 특성이 있는 판매량 타겟 데이터(`y`)를 정규화하여 학습 안정성을 높이고, 오차율(SMAPE)을 기존 190%대에서 70%대 이하로 획기적으로 낮췄습니다.
3. **Gradient Clipping 기반 DP 로직 최적화**
   * 기존 반복적(Leave-one-out) Empirical Sensitivity 연산의 극심한 병목을 제거하고, 피처 중요도 벡터의 L2 Norm을 제한(Gradient Clipping)하는 방식으로 프라이버시 로직을 개선하여 연산 시간을 단축했습니다.
4. **정교한 하위 클러스터링 최적화**
   * 서버의 클러스터링 알고리즘 페널티를 대폭 완화(`complexity_penalty=0.001`, `max_clusters=15`)하여, 단순히 1~2개의 거대한 그룹이 아닌 도메인과 특성이 진짜 유사한 매장끼리 세밀하게 묶이도록(10개 내외의 다중 버블) 개선했습니다.
5. **GPU 가속 최신화 호환**
   * CUDA 12.x(RTX 5070 Ti 등 최신 아키텍처 지원) 호환 PyTorch 업데이트 반영이 완료되었습니다.

---

## 📂 프로젝트 구조 (Project Structure)

```text
📦 model
 ┣ 📂 src
 │ ┣ 📂 3/data/clients         # SQLite (.db) 기반의 전처리 완료된 매장별 데이터셋
 │ ┣ 📂 fl                     # 연합학습 및 PA-CFL 핵심 모듈
 │ │ ┣ 📜 client.py             # 클라이언트 모델 (Target 역스케일링 로직 포함)
 │ │ ┣ 📜 server.py             # 연합학습 서버(BubbleServer)
 │ │ ┣ 📜 privacy.py            # Gradient Clipping 기반 프라이버시 보호 노이즈 생성
 │ │ ┗ 📜 server_clustering.py  # EMD 거리 및 Agglomerative 클러스터링 모듈
 │ ┣ 📂 models                 # 신경망 아키텍처
 │ │ ┗ 📜 lstm.py               # Lightweight LSTM 모델
 │ ┗ 📜 dataset.py             # PyTorch DataLoader 및 SQLite 전처리 로직
 ┣ 📂 outputs                  # 실험 결과 및 로그, 클러스터링 JSON 결과물
 ┣ 📜 run_fl.py                # 전체 FL 파이프라인 (클러스터링 + 슬라이딩 윈도우 + FL) 통합 실행 스크립트
 ┗ 📜 README.md                # 프로젝트 설명서
```

---

## ⚙️ 실행 방법 (How to Run)

최신 파이프라인은 `run_fl.py` 스크립트 하나로 전체 과정을 통합 관리합니다. SQLite DB 파일을 자동으로 스캔하여 클라이언트 구성을 동적으로 세팅합니다.

### 전체 파이프라인(클러스터링 및 연합학습) 실행
터미널에서 아래의 명령어를 입력하여 파이프라인을 구동합니다.
```bash
python run_fl.py
```
*(실행 시 자동으로 데이터 로드 -> 피처 중요도 추출(Gradient Clipping) -> 서버 클러스터링(EMD 기반, Bubble 생성) -> 연합 학습(FedAvg) 과정이 연달아 진행됩니다.)*

### 📊 주요 파라미터 튜닝 포인트 (`run_fl.py` 내부)
- `seq_len`: LSTM의 과거 데이터 참조 일수 (기본값: 14)
- `max_clusters`, `complexity_penalty`: `src/fl/server.py` 내부 파라미터 조정을 통해 생성될 버블(클러스터)의 개수 조절

---

## 🛠 기술 스택 (Tech Stack)
* **Language:** Python 3.12+
* **Machine Learning:** PyTorch (CUDA 12.x+), XGBoost, Scikit-learn
* **Data Processing:** Pandas, Numpy, SQLite3
* **Federated Learning Framework:** Custom PA-CFL BubbleServer 구현

## 📝 참고 문헌 (References)
* Privacy-Adaptive Clustered Federated Learning for Transformer-Based Sales Forecasting on Heterogeneous Retail Data (기반 논문)
* M5 Forecasting - Accuracy (Kaggle Dataset)