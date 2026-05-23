# Fedstock Server (Central FL Orchestrator & Backend API)

이 디렉토리는 Fedstock 서비스의 중앙 서버 측 시스템 모듈을 포함하고 있습니다. 연합학습 전체 라운드를 중계하고 모델 가중치를 집계하는 FL Server, EMD 기반 매장 버블 군집화 코어와 함께, 중앙 백엔드 서비스(Spring Boot API)를 제공합니다.

## 📂 디렉토리 구조 (Directory Structure)

```text
server/
├── backend/         # Spring Boot 기반 중앙 백엔드 API 서버 (Modular Monolith)
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
중앙 서버는 매장 등록, JWT 인증, 연합학습 라운드 정보 제어, 가중치 업로드 및 대시보드 통계 API를 관리합니다.

#### local 실행 (JDK 21 및 PostgreSQL 필요)
```bash
# backend 디렉토리로 이동
cd server/backend

# 스프링 부트 애플리케이션 실행
./gradlew bootRun
```

#### Docker Compose로 실행 (DB 및 App 동시 실행)
```bash
# backend 디렉토리로 이동 후 실행 스크립트 구동
cd server/backend
./run.sh up
```
- Swagger API Docs: `http://localhost:8080/swagger-ui.html`
- OpenAPI JSON: `http://localhost:8080/v3/api-docs`
- Health Check: `http://localhost:8080/health` 또는 `/actuator/health`

### 2. FL Server & Clustering (Python)
Flower 프레임워크와 PyTorch를 결합하여 버블 단위의 FedAvg 연합학습을 학습 라운드별로 조정하고 동적으로 매장을 버블로 재군집화(Dynamic Reclustering)합니다.

* `src/fl/server_clustering.py`를 실행하여 초기 EMD 거리 계산 및 클러스터 할당을 수행할 수 있습니다.
* `src/fl/server.py` 내의 `BubbleServer` 클래스가 전체 communication round와 parameter aggregation을 오케스트레이션합니다.

---

## 💾 중앙 데이터베이스 구조 (PostgreSQL)
`backend` 프로젝트가 PostgreSQL 컨테이너와 연결되어 동작하며, 주요 테이블은 다음과 같습니다.
- `STORES`: 가입된 매장 정보 및 메타데이터
- `CLUSTERS`: 생성된 버블(Cluster) 메타데이터
- `STORE_CLUSTER_MAP`: 매장별 소속 버블 매핑 관계
- `FEATURE_IMPORTANCE`: 수집된 Laplace 노이즈 feature importance 벡터
- `GLOBAL_MODELS`: 각 버블 및 전체 글로벌 모델 정보
- `FL_ROUNDS`: 라운드별 진행 상태 및 참여 스토어 현황
