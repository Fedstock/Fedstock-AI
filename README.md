# Fedstock Server (Central FL Orchestrator)

이 저장소는 Fedstock 서비스의 중앙 서버 측 연합학습(FL) 시스템 모듈을 포함하고 있습니다. 연합학습 전체 라운드를 오케스트레이션하고 모델 가중치를 집계하는 FL Server 및 EMD 기반 매장 버블 군집화 코어를 제공합니다.

> [!NOTE]
> 중앙 백엔드 API 서비스(Spring Boot)는 별도의 저장소인 [Fedstock-Backend](https://github.com/Fedstock/Fedstock-Backend)에서 독립적으로 관리 및 개발되고 있습니다.

## 📂 디렉토리 구조 (Directory Structure)

```text
├── outputs/         # 글로벌 및 버블 모델 가중치, 클러스터링 이력 리소스
└── src/             # 서버 측 핵심 파이썬 소스코드
    ├── fl/
    │   ├── server.py             # BubbleServer 연합학습 오케스트레이터 및 FedAvg 집계
    │   └── server_clustering.py  # EMD 거리 계산 및 Davies-Bouldin Index 기반 최적 클러스터링 (k*)
    ├── models/
    │   └── lstm.py               # 글로벌/버블 가중치 관리용 Lightweight LSTM 모델 정의 (공유)
    └── losses.py                 # HuberSMAPELoss 정의
```

## 🚀 구동 방법 (Quick Start)

### 1. Spring Boot 백엔드 API 서버 실행
중앙 백엔드 API 서버의 소스코드 및 구동 방법은 별도의 저장소인 [Fedstock-Backend](https://github.com/Fedstock/Fedstock-Backend)를 참고해 주시기 바랍니다.

### 2. FL Server & Clustering (Python)
Flower 프레임워크와 PyTorch를 결합하여 버블 단위의 FedAvg 연합학습을 학습 라운드별로 조정하고 동적으로 매장을 버블로 재군집화(Dynamic Reclustering)합니다.

* `src/fl/server_clustering.py`를 실행하여 초기 EMD 거리 계산 및 클러스터 할당을 수행할 수 있습니다.
* `src/fl/server.py` 내의 `BubbleServer` 클래스가 전체 communication round와 parameter aggregation을 오케스트레이션합니다.

---

## 💾 중앙 데이터베이스 구조 (PostgreSQL)
중앙 데이터베이스 스키마 및 테이블 생성 등은 백엔드 시스템에 종속되어 있습니다. 세부적인 데이터베이스 구성 및 스키마에 대해서는 [Fedstock-Backend](https://github.com/Fedstock/Fedstock-Backend) 저장소를 참고하여 주시기 바랍니다.
