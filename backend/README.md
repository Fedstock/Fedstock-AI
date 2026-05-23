# Fedstock Backend

Fedstock 서비스의 백엔드 API 서버입니다.
현재 단계는 실제 도메인 API와 ERD가 확정되기 전, 팀원이 같은 환경에서 개발을 시작할 수 있도록 기본 서버, DB 연결, 문서화, 샘플 CRUD를 준비하는 단계입니다.

## Tech Stack

| Category | Stack |
| --- | --- |
| Language | Java 21 |
| Framework | Spring Boot 3.5.14 |
| Build | Gradle Wrapper |
| API | Spring Web |
| Validation | Spring Validation |
| Database | PostgreSQL |
| ORM | Spring Data JPA, Hibernate |
| API Docs | springdoc-openapi, Swagger UI |
| Monitoring | Spring Boot Actuator |
| Container | Docker, Docker Compose |
| Test | JUnit 5, Spring Boot Test |

## Architecture

이 프로젝트는 **Modular Monolith + Clean Architecture Lite** 구조를 사용합니다.

```text
client
  -> api
  -> application
  -> domain
  <- infrastructure
```

기능별 패키지를 하나의 작은 모듈처럼 관리하고, 각 기능 내부에서 요청 처리, 유스케이스, 도메인 규칙, 기술 구현을 분리합니다.
자세한 폴더링과 코드 작성 규칙은 [docs/CONVENTION.md](docs/CONVENTION.md)를 확인합니다.

## Project Structure

```text
src/main/java/com/fedstock/backend
├── main
│   ├── api
│   └── error
├── demo
│   ├── api
│   │   └── dto
│   ├── application
│   ├── domain
│   └── infrastructure
└── example_reservation
    ├── api
    ├── application
    ├── domain
    └── infrastructure
```

## Environment

로컬 실행 기본값은 `.env.example`과 같습니다.

```text
DB_HOST=localhost
DB_PORT=5432
DB_NAME=fedstock
DB_USERNAME=fedstock
DB_PASSWORD=fedstock
```

Spring profile:

| Profile | Purpose |
| --- | --- |
| `local` | 로컬 개발 |
| `docker` | Docker Compose 실행 |

## Run with Docker

```bash
./run.sh up
```

백그라운드 실행:

```bash
./run.sh up-bg
```

종료:

```bash
./run.sh down
```

PostgreSQL 데이터까지 삭제:

```bash
./run.sh clean
```

## Run Locally

JDK 21과 PostgreSQL이 필요합니다.

```bash
./gradlew bootRun
```

Docker Compose로 DB만 먼저 실행하고 애플리케이션은 로컬에서 실행할 수도 있습니다.

```bash
./run.sh db
./run.sh app
```

## API Docs

서버 실행 후 Swagger UI에서 API를 확인합니다.

```text
http://localhost:8080/swagger-ui.html
```

OpenAPI JSON:

```text
http://localhost:8080/v3/api-docs
```

## Health Check

```bash
curl http://localhost:8080/health
```

Actuator health:

```bash
curl http://localhost:8080/actuator/health
```

## Demo API

실제 ERD와 도메인 API가 확정되기 전까지 서버 동작, DB 연결, Swagger 문서화를 확인하기 위한 샘플 CRUD입니다.

```bash
curl -X POST http://localhost:8080/api/demos \
  -H "Content-Type: application/json" \
  -d '{
    "title": "first demo",
    "content": "demo content"
  }'
```

```bash
curl http://localhost:8080/api/demos
curl http://localhost:8080/api/demos/1
```

```bash
curl -X PUT http://localhost:8080/api/demos/1 \
  -H "Content-Type: application/json" \
  -d '{
    "title": "updated demo",
    "content": "updated content"
  }'
```

```bash
curl -X DELETE http://localhost:8080/api/demos/1
```

## Commit Convention

커밋 메시지는 영어 한 줄로 간결하게 작성합니다.
scope 괄호는 사용하지 않습니다.

```text
feat: add demo crud api
fix: handle demo not found
docs: update backend readme
```

## Before Real Domain Work

실제 API 개발 전 필요한 결정 사항:

- ERD 확정
- 도메인별 API 명세 확정
- 인증/인가 방식 확정
- 운영 DB 계정 및 배포 환경 변수 분리
- 마이그레이션 도구 적용 여부 결정
