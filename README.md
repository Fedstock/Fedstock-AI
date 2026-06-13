# Fedstock Server (Central FL Orchestrator)

## 운영 모델 Artifact 원칙

운영용 모델 파일(`.pt`, `.pth`, `.pkl`, `.pickle`, `.joblib`)은 Git 저장소에 두지 않습니다. Backend가 모델 artifact를 S3에 업로드하고, AI API에는 S3 URI와 라운드/클러스터/버전 metadata만 전달합니다. AI 서버는 필요한 artifact를 `MODEL_LOCAL_DIR`(기본 `/tmp/models`)에 임시 다운로드해서 처리한 뒤, 생성된 global model, cluster model, evaluation report를 다시 S3에 업로드합니다.

필수 런타임 환경변수:

```env
AWS_REGION=ap-northeast-2
ARTIFACT_BUCKET=mlops-artifacts-995370109104-ap-northeast-2
MODEL_LOCAL_DIR=/tmp/models
```

ECS 운영 환경에서는 IAM task role로 S3 권한을 부여하며, AWS access key/secret key는 코드나 `.env`에 넣지 않습니다.

### Backend -> AI S3 URI 기반 FL aggregation 예시

```http
POST /clients/fl-model/aggregate
Content-Type: application/json
```

```json
{
  "scope": "all_clients",
  "roundId": "fl-sync-20260613-001",
  "clusterId": "0",
  "modelVersion": "v20260613-001",
  "expectedClientCount": 3,
  "outputPrefixUri": "s3://mlops-artifacts-995370109104-ap-northeast-2/models/clusters/fl-sync-20260613-001",
  "models": [
    {
      "clientId": "CA_1_FOODS_3",
      "sampleCount": 13004,
      "modelArtifactUri": "s3://mlops-artifacts-995370109104-ap-northeast-2/updates/fl-sync-20260613-001/clients/CA_1_FOODS_3.pt"
    },
    {
      "clientId": "CA_2_FOODS_7",
      "sampleCount": 9821,
      "modelArtifactUri": "s3://mlops-artifacts-995370109104-ap-northeast-2/updates/fl-sync-20260613-001/clients/CA_2_FOODS_7.pt"
    }
  ]
}
```

응답은 로컬 파일 경로가 아니라 S3 URI metadata를 반환합니다.

```json
{
  "ok": true,
  "roundId": "fl-sync-20260613-001",
  "clusterId": "0",
  "modelVersion": "v20260613-001",
  "receivedClientCount": 2,
  "aggregatedModelUri": "s3://mlops-artifacts-995370109104-ap-northeast-2/models/clusters/fl-sync-20260613-001/cluster-0.pt",
  "aggregation": "sample_count_weighted_fedavg"
}
```

### S3 Key Convention

```text
s3://{ARTIFACT_BUCKET}/models/global/{modelVersion}/global.pt
s3://{ARTIFACT_BUCKET}/models/clusters/{roundId}/cluster-{clusterId}.pt
s3://{ARTIFACT_BUCKET}/updates/{roundId}/clients/{clientId}.pt
s3://{ARTIFACT_BUCKET}/reports/{roundId}/evaluation.json
```

### Batch Upload 호환 API

`POST /clients/fl-model/batch`는 Backend DB bytea 기반 전달을 위한 multipart 호환 API입니다. 수신 파일은 Git repo가 아니라 `MODEL_LOCAL_DIR/{round_id}/clients` 아래에 임시 저장되며, 응답에는 로컬 path를 포함하지 않습니다.

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
