# Fedstock Backend 폴더링 및 코드 컨벤션

이 문서는 팀원이 같은 방식으로 기능을 추가하기 위한 작업 지침입니다.  
정답을 강제하기보다, 초급~중급 개발자가 헷갈리지 않게 일관된 기준을 제공하는 것이 목적입니다.

## 기본 구조

```text
src/main/java/com/fedstock/backend
├── FedstockBackendApplication.java
├── main
│   ├── api
│   └── error
└── example_reservation
    ├── api
    │   └── dto
    ├── application
    ├── domain
    └── infrastructure
```

## 패키지 역할

### `main`

프로젝트 전체에 공통으로 필요한 코드를 둡니다.

- 애플리케이션 상태 확인 API
- 공통 예외 응답
- 전역 설정 또는 공통 유틸

주의할 점:

- 특정 기능의 비즈니스 로직은 `main`에 두지 않습니다.
- 여러 기능에서 정말 함께 쓰는 코드인지 먼저 확인합니다.

### `example_reservation`

새 기능을 만들 때 참고하는 예시 기능입니다.  
실제 기능을 추가할 때는 같은 구조로 새 패키지를 만듭니다.

예시:

```text
stock_order
├── api
├── application
├── domain
└── infrastructure
```

현재 `demo` 패키지는 PostgreSQL, JPA, Swagger 연결을 확인하기 위한 샘플 CRUD입니다.
실제 도메인 API와 ERD가 확정되면 `demo`는 참고용으로만 사용하거나 제거합니다.

## 계층별 작성 기준

### `api`

외부 요청과 응답을 담당합니다.

- Controller
- Request DTO
- Response DTO
- URL, HTTP Method, Validation

작성 원칙:

- 비즈니스 판단 로직을 넣지 않습니다.
- Request DTO는 `@Valid`로 입력값을 검증합니다.
- API 경로는 kebab-case와 복수형을 사용합니다.

예시:

```text
POST /api/example-reservations
GET  /api/example-reservations/{reservationId}
```

### `application`

하나의 사용자 행동 또는 유스케이스를 처리합니다.

- Service
- Command
- Query
- Transaction 범위

작성 원칙:

- Controller에서 받은 요청을 도메인 흐름으로 연결합니다.
- 여러 도메인 객체나 저장소 호출을 조합합니다.
- 외부 API, DB 구현 세부사항을 직접 알지 않도록 합니다.

### `domain`

핵심 비즈니스 규칙을 둡니다.

- 도메인 모델
- Enum
- Repository 인터페이스
- 도메인 정책 메서드

작성 원칙:

- 가능한 한 Spring 어노테이션에 의존하지 않습니다.
- 상태 변경 규칙은 도메인 객체 안에서 표현합니다.
- DB 저장 방식 때문에 도메인 규칙이 흔들리지 않게 합니다.

### `infrastructure`

기술 구현체를 둡니다.

- Repository 구현체
- 외부 API Client
- 파일, 메시지 큐, 캐시 연동
- JPA Entity 또는 DB 매핑 코드

작성 원칙:

- 기술 선택이 바뀌어도 `domain`과 `application`의 변경을 최소화합니다.
- 현재 예시는 DB 설계 전 단계라 인메모리 저장소를 사용합니다.

## 이름 규칙

- 패키지: 기능 루트는 `example_reservation`처럼 lower_snake_case를 사용합니다.
- 하위 패키지: `api`, `application`, `domain`, `infrastructure`로 통일합니다.
- 클래스: PascalCase를 사용합니다.
- 메서드와 변수: camelCase를 사용합니다.
- 상수: UPPER_SNAKE_CASE를 사용합니다.
- Request DTO: `CreateSomethingRequest`
- Response DTO: `SomethingResponse`
- Command: `CreateSomethingCommand`
- Service: `SomethingService`
- Repository 인터페이스: `SomethingRepository`

## 기능 추가 순서

1. 기능 루트 패키지를 만듭니다.
2. `domain`에 핵심 모델과 Repository 인터페이스를 작성합니다.
3. `application`에 유스케이스 Service와 Command를 작성합니다.
4. `infrastructure`에 저장소 또는 외부 연동 구현체를 작성합니다.
5. `api`에 Controller와 Request/Response DTO를 작성합니다.
6. 필요한 검증, 예외, 테스트를 추가합니다.

## 작업 전 체크리스트

- Controller에 비즈니스 로직이 들어가지 않았나요?
- 입력값 검증이 Request DTO에 있나요?
- Service가 하나의 명확한 유스케이스를 표현하나요?
- 도메인 규칙이 단순 데이터 저장 코드에 묻혀 있지 않나요?
- 새 기능 구조가 `example_reservation`과 같은 흐름을 따르나요?
- 에러 응답이 팀에서 이해할 수 있는 형태인가요?

## Docker 사용 기준

Docker는 팀원이 같은 환경에서 서버를 실행하기 위한 기본 실행 수단입니다.

- `Dockerfile`은 애플리케이션 이미지를 만드는 기준 파일입니다.
- `docker-compose.yml`은 로컬에서 서버를 쉽게 실행하기 위한 파일입니다.
- `.dockerignore`는 이미지 빌드에 불필요한 파일을 제외합니다.
- Docker 실행 시 `SPRING_PROFILES_ACTIVE=docker`를 사용합니다.
- 비밀번호, 토큰, 운영 DB 주소 같은 민감 정보는 이미지나 Git에 넣지 않습니다.

기본 실행:

```bash
docker compose up --build
```

Docker 설정을 바꿀 때는 README의 실행 방법도 함께 갱신합니다.

## 커밋 메시지 규칙

커밋은 영어 한 줄로 간결하게 작성합니다.  
scope 괄호는 사용하지 않습니다.

좋은 예:

```text
feat: add reservation create api
fix: handle reservation not found
docs: update foldering guide
```

피할 예:

```text
feat(api): add reservation api
feat: reservation api and refactor and docs
```
