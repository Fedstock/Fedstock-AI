from __future__ import annotations

import json
import math
import os
import re
import shutil
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from fastapi import Body, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ValidationError
from sklearn.preprocessing import RobustScaler, StandardScaler

from app.config import get_settings
from app.storage import StorageError, build_s3_uri, download_s3_file, parse_s3_uri, upload_s3_file
from src.fl.server_clustering import assign_new_client, perform_clustering
from src.models.lstm import LightweightLSTM


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT_DIR / "outputs"
DEFAULT_RUN_ID = "20260522_025148_061195"

SEQ_LEN = 14
FORECAST_HORIZON_DAYS = 1
FORECAST_TARGET_COLUMN = "target_1d"
FORECAST_UNIT = "next_day_sales"
HISTORY_WINDOW_PER_ITEM = 35
DEFAULT_LEAD_TIME_DAYS = 4


def _resolve_run_dir() -> Path:
    latest_path = OUTPUTS_DIR / "latest_run.txt"
    if latest_path.exists():
        raw_latest = latest_path.read_text().strip()
        if raw_latest:
            local_candidate = OUTPUTS_DIR / "runs" / Path(raw_latest).name
            if local_candidate.exists():
                return local_candidate

    default_candidate = OUTPUTS_DIR / "runs" / DEFAULT_RUN_ID
    if default_candidate.exists():
        return default_candidate

    run_dirs = sorted((OUTPUTS_DIR / "runs").glob("*"))
    if run_dirs:
        return run_dirs[-1]

    return default_candidate


RUN_DIR = _resolve_run_dir()
CLIENT_MODEL_DIR = RUN_DIR / "models" / "clients"
CONFIG_PATH = RUN_DIR / "config.json"
FEATURE_IMPORTANCES_PATH = RUN_DIR / "feature_importances.json"
CLUSTERING_RESULTS_PATH = RUN_DIR / "clustering_results.json"
ITEM_MASTER_PATH = ROOT_DIR / "data" / "item_master.csv"
MODEL_LOCAL_DIR = Path(get_settings().model_local_dir)


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default


def _load_selected_features() -> list[str]:
    config = _load_json(CONFIG_PATH, {})
    features = config.get("selected_features")
    if isinstance(features, list) and features:
        return [str(feature) for feature in features]

    return [
        "rolling_mean_28",
        "rolling_mean_7",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_std_28",
        "rolling_std_7",
        "sell_price",
        "is_month_end",
        "is_month_start",
        "month",
        "week_of_year",
    ]


SELECTED_FEATURES = _load_selected_features()


def _load_item_master() -> dict[str, dict[str, Any]]:
    if not ITEM_MASTER_PATH.exists():
        return {}

    frame = pd.read_csv(ITEM_MASTER_PATH).fillna("")
    if "item_id" not in frame.columns:
        return {}

    master: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        item_id = str(row["item_id"]).strip()
        if not item_id:
            continue
        weather_tags = [
            tag.strip()
            for tag in str(row.get("weather_tags", "")).split("|")
            if tag.strip()
        ]
        master[item_id] = {
            "itemName": str(row.get("item_name_ko", "")).strip(),
            "category": str(row.get("display_category", "")).strip(),
            "weatherTags": weather_tags,
        }
    return master


ITEM_MASTER = _load_item_master()


class ClusterAssignmentRequest(BaseModel):
    scope: str
    roundId: str
    clientId: str
    sampleCount: float
    featureNames: list[str]
    featureImportance: list[float]
    expectedClientCount: int | None = None


class ClusterAssignmentResponse(BaseModel):
    status: str
    clientId: str
    scope: str
    assignedTo: str | None
    clusterId: int | None
    clusterMembers: list[str]
    queueStatus: str
    distance: float | None
    threshold: float | None
    message: str


CLUSTER_ASSIGNMENT_QUEUES: dict[str, dict[str, Any]] = {}
CLUSTER_ASSIGNMENT_COMPLETED: dict[str, dict[str, dict[str, Any]]] = {}
CLIENT_CLUSTER_ASSIGNMENTS: dict[str, dict[str, Any]] = {}
FL_MODEL_SYNC_QUEUES: dict[str, dict[int, dict[str, Any]]] = {}


app = FastAPI(title="Fedstock AI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _format_number(value: float | int) -> str:
    if isinstance(value, float) and not value.is_integer():
        return f"{value:,.1f}"
    return f"{int(round(value)):,}"


def _format_currency(value: float | int) -> str:
    return f"US${int(round(value)):,}"


def _validation_item(
    column: str,
    label: str,
    ok: bool,
    required: bool = True,
    warning_message: str | None = None,
) -> dict[str, Any]:
    status = "passed" if ok else "failed"
    message = "확인됐습니다." if ok else "예상 판매량 계산에 필요한 항목입니다."
    if not ok and not required:
        status = "warning"
        message = warning_message or "없어도 계산은 진행됩니다."

    return {
        "column": column,
        "label": label,
        "required": required,
        "status": status,
        "message": message,
    }


def _safe_path_component(value: str, field_name: str) -> str:
    cleaned = str(value).strip()
    if not cleaned or not re.fullmatch(r"[A-Za-z0-9_.-]+", cleaned):
        raise HTTPException(status_code=400, detail=f"{field_name} may contain only letters, numbers, '_', '-', and '.'.")
    return cleaned


def _ensure_model_local_dir() -> Path:
    MODEL_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    return MODEL_LOCAL_DIR


def _load_tensor_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to read PyTorch model artifact: {exc}") from exc
    if not isinstance(state, dict) or not state:
        raise HTTPException(status_code=422, detail="Model artifact must be a non-empty state_dict.")
    tensor_state: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            raise HTTPException(status_code=422, detail=f"State dict value must be a tensor: {key}")
def _verify_bearer_token(authorization: str | None = Header(default=None)) -> None:
    expected_token = get_settings().api_bearer_token
    if not expected_token:
        raise HTTPException(status_code=500, detail="API_BEARER_TOKEN이 설정되어 있지 않습니다.")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Bearer Token 인증이 필요합니다.")
    supplied_token = authorization.removeprefix("Bearer ").strip()
    if supplied_token != expected_token:
        raise HTTPException(status_code=401, detail="Bearer Token이 올바르지 않습니다.")


def _parse_cluster_assignment_request(raw: dict[str, Any]) -> ClusterAssignmentRequest:
    required = ["scope", "roundId", "clientId", "sampleCount", "featureNames", "featureImportance"]
    missing = [field for field in required if field not in raw or raw[field] is None]
    if missing:
        raise HTTPException(status_code=400, detail=f"필수 필드가 누락되었습니다: {', '.join(missing)}")
    try:
        payload = ClusterAssignmentRequest(**raw)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=f"요청 형식이 올바르지 않습니다: {exc}") from exc

    if payload.scope not in {"single_client", "all_clients"}:
        raise HTTPException(status_code=400, detail="지원하지 않는 scope입니다.")
    if not payload.roundId.strip():
        raise HTTPException(status_code=400, detail="roundId는 비어 있을 수 없습니다.")
    if not payload.clientId.strip():
        raise HTTPException(status_code=400, detail="clientId는 비어 있을 수 없습니다.")
    if payload.sampleCount <= 0:
        raise HTTPException(status_code=400, detail="sampleCount는 0보다 커야 합니다.")
    if payload.scope == "all_clients" and (payload.expectedClientCount is None or payload.expectedClientCount <= 0):
        raise HTTPException(status_code=400, detail="all_clients scope에는 expectedClientCount가 필요합니다.")
    return payload


def _importance_vector_from_payload(payload: ClusterAssignmentRequest) -> np.ndarray:
    if len(payload.featureNames) != len(payload.featureImportance):
        raise HTTPException(status_code=400, detail="featureNames와 featureImportance 길이가 일치하지 않습니다.")

    feature_names = [str(name) for name in payload.featureNames]
    if len(set(feature_names)) != len(feature_names):
        raise HTTPException(status_code=400, detail="featureNames에 중복 항목이 있습니다.")

    selected = set(SELECTED_FEATURES)
    supplied = set(feature_names)
    if supplied != selected:
        missing = sorted(selected - supplied)
        extra = sorted(supplied - selected)
        detail = "서버 selectedFeatures와 featureNames가 일치하지 않습니다."
        if missing:
            detail += f" 누락: {', '.join(missing)}."
        if extra:
            detail += f" 추가: {', '.join(extra)}."
        raise HTTPException(status_code=400, detail=detail)

    by_feature = {name: float(value) for name, value in zip(feature_names, payload.featureImportance)}
    vector = np.asarray([by_feature[name] for name in SELECTED_FEATURES], dtype=np.float32)
    if not np.isfinite(vector).all():
        raise HTTPException(status_code=400, detail="featureImportance에는 유한한 숫자만 사용할 수 있습니다.")
    return vector


def _safe_path_component(value: str, field_name: str) -> str:
    cleaned = str(value).strip()
    if not cleaned or not re.fullmatch(r"[A-Za-z0-9_.-]+", cleaned):
        raise HTTPException(status_code=400, detail=f"{field_name}에는 영문, 숫자, _, -, . 만 사용할 수 있습니다.")
    return cleaned


def _validate_pt_upload(model_file: UploadFile) -> None:
    filename = model_file.filename or ""
    if not filename.lower().endswith(".pt"):
        raise HTTPException(status_code=400, detail="model_file은 .pt 파일이어야 합니다.")


def _fl_sync_root(round_id: str) -> Path:
    safe_round = _safe_path_component(round_id, "round_id")
    return OUTPUTS_DIR / "fl_sync" / safe_round


def _save_uploaded_model(model_file: UploadFile, client_id: str, round_id: str) -> Path:
    _validate_pt_upload(model_file)
    safe_client = _safe_path_component(client_id, "client_id")
    upload_dir = _fl_sync_root(round_id) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = upload_dir / f"client_{safe_client}.pt"
    with upload_path.open("wb") as handle:
        shutil.copyfileobj(model_file.file, handle)
    if upload_path.stat().st_size <= 0:
        raise HTTPException(status_code=400, detail="model_file이 비어 있습니다.")
    return upload_path


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f".pt 모델 파일을 읽을 수 없습니다: {exc}") from exc
    if not isinstance(state, dict) or not state:
        raise HTTPException(status_code=400, detail=".pt 모델 파일은 비어 있지 않은 state_dict 형식이어야 합니다.")
    tensor_state = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            raise HTTPException(status_code=400, detail=f"state_dict 값은 Tensor여야 합니다: {key}")
        tensor_state[str(key)] = value.detach().cpu().float()
    return tensor_state


def _load_tensor_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to read PyTorch model artifact: {exc}") from exc
    if not isinstance(state, dict) or not state:
        raise HTTPException(status_code=422, detail="Model artifact must be a non-empty state_dict.")
    tensor_state: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            raise HTTPException(status_code=422, detail=f"State dict value must be a tensor: {key}")
        tensor_state[str(key)] = value.detach().cpu().float()
    return tensor_state


def _weighted_average_state_dicts(
    weighted_states: list[tuple[dict[str, torch.Tensor], float]]
) -> dict[str, torch.Tensor]:
    if not weighted_states:
        raise HTTPException(status_code=422, detail="No model artifacts were provided for aggregation.")
    reference_keys = set(weighted_states[0][0])
    for state, _ in weighted_states:
        if set(state) != reference_keys:
            raise HTTPException(status_code=422, detail="Model state_dict keys do not match.")
    total_weight = sum(max(0.0, float(weight)) for _, weight in weighted_states)
    if total_weight <= 0:
        raise HTTPException(status_code=422, detail="Total sample count must be greater than zero.")

    averaged: dict[str, torch.Tensor] = {}
    for key in sorted(reference_keys):
        reference_shape = weighted_states[0][0][key].shape
        weighted_sum = None
        for state, weight in weighted_states:
            tensor = state[key]
            if tensor.shape != reference_shape:
                raise HTTPException(status_code=422, detail=f"Tensor shape mismatch for state_dict key: {key}")
def _weighted_average_state_dicts(items: list[tuple[dict[str, torch.Tensor], float]]) -> dict[str, torch.Tensor]:
    if not items:
        raise HTTPException(status_code=500, detail="평균할 모델이 없습니다.")
    reference_keys = set(items[0][0])
    for state, _ in items:
        if set(state) != reference_keys:
            raise HTTPException(status_code=500, detail="업로드된 모델 state_dict key가 서로 다릅니다.")
    total_weight = sum(max(0.0, float(weight)) for _, weight in items)
    if total_weight <= 0:
        raise HTTPException(status_code=400, detail="sample_weight 합계는 0보다 커야 합니다.")

    averaged: dict[str, torch.Tensor] = {}
    for key in sorted(reference_keys):
        weighted_sum = None
        reference_shape = items[0][0][key].shape
        for state, weight in items:
            tensor = state[key]
            if tensor.shape != reference_shape:
                raise HTTPException(status_code=500, detail=f"업로드된 모델 Tensor shape가 서로 다릅니다: {key}")
            contribution = tensor * (float(weight) / total_weight)
            weighted_sum = contribution if weighted_sum is None else weighted_sum + contribution
        averaged[key] = weighted_sum
    return averaged


def _weighted_average_state_dicts(items: list[tuple[dict[str, torch.Tensor], float]]) -> dict[str, torch.Tensor]:
    if not items:
        raise HTTPException(status_code=422, detail="No model artifacts were provided for aggregation.")
    reference_keys = set(items[0][0])
    for state, _ in items:
        if set(state) != reference_keys:
            raise HTTPException(status_code=422, detail="Model state_dict keys do not match.")
    total_weight = sum(max(0.0, float(weight)) for _, weight in items)
    if total_weight <= 0:
        raise HTTPException(status_code=422, detail="Total sample count must be greater than zero.")

    averaged: dict[str, torch.Tensor] = {}
    for key in sorted(reference_keys):
        weighted_sum = None
        reference_shape = items[0][0][key].shape
        for state, weight in items:
            tensor = state[key]
            if tensor.shape != reference_shape:
                raise HTTPException(status_code=422, detail=f"Tensor shape mismatch for state_dict key: {key}")
            contribution = tensor * (float(weight) / total_weight)
            weighted_sum = contribution if weighted_sum is None else weighted_sum + contribution
        averaged[key] = weighted_sum
    return averaged


def _json_field(payload: dict[str, Any], *names: str, required: bool = True, default: Any = None) -> Any:
    for name in names:
        if name in payload and payload[name] is not None:
            return payload[name]
    if required:
        raise HTTPException(status_code=400, detail=f"Missing required field: {names[0]}")
    return default


def _join_s3_prefix(prefix_uri: str, filename: str) -> str:
    bucket, key = parse_s3_uri(prefix_uri.rstrip("/") + "/placeholder")
    prefix_key = key.rsplit("/", 1)[0].strip("/")
    return build_s3_uri(bucket, f"{prefix_key}/{filename}" if prefix_key else filename)


def _storage_http_error(exc: StorageError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=exc.message)


def _csv_status(
    file_name: str,
    rows: int,
    product_count: int,
    date_range: str | None,
    validation: list[dict[str, Any]],
    issues: list[dict[str, str]],
    state: str = "loaded",
) -> dict[str, Any]:
    return {
        "state": state,
        "fileName": file_name,
        "rowCount": rows,
        "productCount": product_count,
        "dateRange": date_range,
        "uploadedAt": datetime.now().strftime("%Y. %m. %d. %H:%M:%S"),
        "validation": validation,
        "issues": issues,
    }


def _normalize_column_name(value: str) -> str:
    return (
        str(value)
        .lstrip("\ufeff")
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace(".", "")
    )


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {_normalize_column_name(column): column for column in df.columns}
    for candidate in candidates:
        key = _normalize_column_name(candidate)
        if key in normalized:
            return normalized[key]
    return None


def _read_csv(file: UploadFile, content: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(BytesIO(content))
    except UnicodeDecodeError:
        return pd.read_csv(BytesIO(content), encoding="cp949")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"CSV를 읽을 수 없습니다: {exc}") from exc


def _next_day_sales(values: pd.Series) -> pd.Series:
    return values.shift(-1)


def _derive_dept_from_item_id(item_id: pd.Series) -> pd.Series:
    parts = item_id.astype(str).str.extract(r"^([^_]+_[^_]+)")
    return parts[0].fillna("UNKNOWN")


def _prepare_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, str]], bool]:
    item_col = _find_column(df, ["item_id", "itemId", "id", "상품 ID", "상품ID", "상품 번호", "상품번호"])
    date_col = _find_column(df, ["sale_date", "date", "판매일", "판매 날짜", "판매날짜", "날짜", "일자"])
    sales_col = _find_column(df, ["sales", "quantity", "판매량", "판매 수량", "판매수량", "수량"])
    stock_col = _find_column(df, ["current_stock", "stock", "현재 재고", "현재재고", "재고", "재고 수량", "재고수량"])
    price_col = _find_column(df, ["sell_price", "price", "판매가", "가격", "상품 가격", "상품가격"])
    client_col = _find_column(df, ["client_id", "clientId", "클라이언트 ID", "클라이언트ID", "매장 클라이언트", "매장클라이언트"])
    store_col = _find_column(df, ["store_id", "storeId", "매장 ID", "매장ID", "매장"])
    dept_col = _find_column(df, ["dept_id", "deptId", "department", "상품군", "부서"])
    target_col = _find_column(df, ["target_1d", "target1d", "1일 판매량", "1일판매량", "다음날 판매량", "익일 판매량"])
    legacy_target_col = _find_column(df, ["target_7d", "target7d", "7일 판매량", "7일판매량", "주간 판매량", "주간판매량"])

    validation = [
        _validation_item("item_id", "상품 정보", item_col is not None),
        _validation_item("sale_date", "판매일", date_col is not None),
        _validation_item("sales", "판매량", sales_col is not None),
        _validation_item("sell_price", "판매가", price_col is not None),
        _validation_item(
            "client_id",
            "매장 구분 정보",
            client_col is not None or store_col is not None,
            required=False,
            warning_message="매장 정보가 없으면 업로드 파일 기준으로 묶어서 계산합니다.",
        ),
    ]
    missing = [item["label"] for item in validation if item["status"] == "failed"]
    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"예상 판매량 계산에 필요한 항목이 없습니다: {', '.join(missing)}",
                "validation": validation,
            },
        )

    prepared = df.copy()
    prepared["item_id"] = prepared[item_col].astype(str)

    dept = prepared[dept_col].astype(str) if dept_col else _derive_dept_from_item_id(prepared["item_id"])
    if client_col:
        prepared["client_id"] = prepared[client_col].astype(str)
    elif store_col:
        prepared["client_id"] = prepared[store_col].astype(str) + "_" + dept
    else:
        prepared["client_id"] = "uploaded_client"

    item_name_col = _find_column(df, ["item_name", "name", "상품명", "상품 이름", "상품이름"])
    prepared["item_name"] = prepared[item_name_col].astype(str) if item_name_col else prepared["item_id"]

    category_col = _find_column(df, ["category", "cat_id", "dept_id", "카테고리", "상품 분류", "상품분류", "분류"])
    if category_col:
        prepared["category"] = prepared[category_col].astype(str)
    else:
        prepared["category"] = dept

    prepared["sale_date"] = pd.to_datetime(prepared[date_col], errors="coerce")
    prepared["sales"] = pd.to_numeric(prepared[sales_col], errors="coerce")
    prepared["current_stock"] = pd.to_numeric(prepared[stock_col], errors="coerce") if stock_col else np.nan
    prepared["sell_price"] = pd.to_numeric(prepared[price_col], errors="coerce")
    prepared["lead_time_days"] = DEFAULT_LEAD_TIME_DAYS
    prepared["ordered_qty"] = 0
    prepared["is_holiday"] = 0

    prepared = prepared.dropna(subset=["item_id", "client_id", "sale_date", "sales", "sell_price"])
    if prepared.empty:
        raise HTTPException(status_code=400, detail="예상 판매량 계산에 사용할 수 있는 기록이 없습니다.")

    prepared = prepared.sort_values(["client_id", "item_id", "sale_date"]).reset_index(drop=True)
    prepared["dayofweek"] = prepared["sale_date"].dt.dayofweek
    prepared["month"] = prepared["sale_date"].dt.month
    prepared["week_of_year"] = prepared["sale_date"].dt.isocalendar().week.astype(int)
    prepared["is_weekend"] = prepared["dayofweek"].isin([5, 6]).astype(int)
    prepared["is_month_start"] = prepared["sale_date"].dt.is_month_start.astype(int)
    prepared["is_month_end"] = prepared["sale_date"].dt.is_month_end.astype(int)

    group_cols = ["client_id", "item_id"]
    grouped_sales = prepared.groupby(group_cols, sort=False)["sales"]
    computed_target = grouped_sales.transform(_next_day_sales)
    if target_col:
        prepared[FORECAST_TARGET_COLUMN] = pd.to_numeric(prepared[target_col], errors="coerce").combine_first(computed_target)
    else:
        prepared[FORECAST_TARGET_COLUMN] = computed_target
    if legacy_target_col and legacy_target_col != FORECAST_TARGET_COLUMN:
        prepared = prepared.drop(columns=[legacy_target_col], errors="ignore")

    if "lag_7" not in prepared.columns:
        prepared["lag_7"] = grouped_sales.shift(7)
    else:
        prepared["lag_7"] = pd.to_numeric(prepared["lag_7"], errors="coerce")
    if "lag_14" not in prepared.columns:
        prepared["lag_14"] = grouped_sales.shift(14)
    else:
        prepared["lag_14"] = pd.to_numeric(prepared["lag_14"], errors="coerce")
    if "lag_28" not in prepared.columns:
        prepared["lag_28"] = grouped_sales.shift(28)
    else:
        prepared["lag_28"] = pd.to_numeric(prepared["lag_28"], errors="coerce")
    if "rolling_mean_7" not in prepared.columns:
        prepared["rolling_mean_7"] = grouped_sales.transform(lambda values: values.shift(1).rolling(7, min_periods=1).mean())
    else:
        prepared["rolling_mean_7"] = pd.to_numeric(prepared["rolling_mean_7"], errors="coerce")
    if "rolling_mean_28" not in prepared.columns:
        prepared["rolling_mean_28"] = grouped_sales.transform(lambda values: values.shift(1).rolling(28, min_periods=1).mean())
    else:
        prepared["rolling_mean_28"] = pd.to_numeric(prepared["rolling_mean_28"], errors="coerce")
    if "rolling_std_7" not in prepared.columns:
        prepared["rolling_std_7"] = grouped_sales.transform(lambda values: values.shift(1).rolling(7, min_periods=2).std()).fillna(0)
    else:
        prepared["rolling_std_7"] = pd.to_numeric(prepared["rolling_std_7"], errors="coerce").fillna(0)
    if "rolling_std_28" not in prepared.columns:
        prepared["rolling_std_28"] = grouped_sales.transform(lambda values: values.shift(1).rolling(28, min_periods=2).std()).fillna(0)
    else:
        prepared["rolling_std_28"] = pd.to_numeric(prepared["rolling_std_28"], errors="coerce").fillna(0)
    if "price_change_rate" not in prepared.columns:
        prepared["price_change_rate"] = (
            prepared.groupby(group_cols, sort=False)["sell_price"]
            .pct_change()
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
        )
    else:
        prepared["price_change_rate"] = pd.to_numeric(prepared["price_change_rate"], errors="coerce").fillna(0)

    usable = prepared.dropna(subset=SELECTED_FEATURES + ["sales"]).reset_index(drop=True)
    issues: list[dict[str, str]] = []
    if not target_col:
        issues.append({
            "severity": "warning",
            "message": "판매 이력을 바탕으로 다음 1일 판매 기준을 계산했습니다.",
        })

    dropped = len(prepared) - len(usable)
    if dropped > 0:
        issues.append({
            "severity": "warning",
            "message": f"최근 판매 흐름을 계산하기 어려운 초기 {dropped:,}개 기록은 제외했습니다.",
        })

    if usable.empty or usable.groupby(group_cols).size().max() <= SEQ_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"상품별로 최소 약 {SEQ_LEN + 29}일 이상의 판매 이력이 필요합니다.",
        )

    return usable, validation, issues, stock_col is not None


def _load_feature_importances() -> dict[str, np.ndarray]:
    raw = _load_json(FEATURE_IMPORTANCES_PATH, {})
    return {str(cid): np.asarray(values, dtype=np.float32) for cid, values in raw.items()}


def _load_cluster_state() -> tuple[list[list[str]], list[str], dict[str, int]]:
    data = _load_json(CLUSTERING_RESULTS_PATH, {})
    records = data.get("records", []) if isinstance(data, dict) else []
    latest = records[-1] if records else data

    bubbles = []
    for bubble in latest.get("multi_client_bubbles", []):
        clients = bubble.get("clients", []) if isinstance(bubble, dict) else []
        if clients:
            bubbles.append([str(cid) for cid in clients])

    isolated = [str(cid) for cid in latest.get("isolated_clients", [])]
    assignments = {str(cid): int(cluster) for cid, cluster in latest.get("assignments", {}).items()}
    if not assignments:
        for idx, bubble in enumerate(bubbles):
            for cid in bubble:
                assignments[cid] = idx
        for cid in isolated:
            assignments[cid] = len(bubbles)

    return bubbles, isolated, assignments


def _safe_cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 1.0
    return float(np.clip(1.0 - float(np.dot(a, b) / (norm_a * norm_b)), 0.0, 2.0))


def _client_importance_from_frame(client_df: pd.DataFrame) -> np.ndarray:
    frame = client_df.dropna(subset=SELECTED_FEATURES + [FORECAST_TARGET_COLUMN])
    if frame.empty:
        values = client_df[SELECTED_FEATURES].to_numpy(dtype=np.float32)
        variances = np.nan_to_num(np.nanvar(values, axis=0), nan=0.0)
        norm = np.linalg.norm(variances)
        return (variances / norm).astype(np.float32) if norm > 0 else np.ones(len(SELECTED_FEATURES), dtype=np.float32)

    target = frame[FORECAST_TARGET_COLUMN].to_numpy(dtype=np.float32)
    vector: list[float] = []
    for feature in SELECTED_FEATURES:
        feature_values = frame[feature].to_numpy(dtype=np.float32)
        if np.nanstd(feature_values) < 1e-12 or np.nanstd(target) < 1e-12:
            vector.append(0.0)
            continue
        corr = np.corrcoef(feature_values, target)[0, 1]
        vector.append(float(np.nan_to_num(corr, nan=0.0)))

    importance = np.asarray(vector, dtype=np.float32)
    if np.linalg.norm(importance) < 1e-12:
        importance = np.ones(len(SELECTED_FEATURES), dtype=np.float32)
    return importance


def _nearest_clients(
    importance: np.ndarray,
    candidates: list[str],
    existing_importances: dict[str, np.ndarray],
    limit: int = 5,
) -> list[dict[str, Any]]:
    ranked = []
    for cid in candidates:
        if cid not in existing_importances:
            continue
        distance = _safe_cosine_distance(importance, existing_importances[cid])
        ranked.append({"clientId": cid, "distance": round(distance, 4)})
    return sorted(ranked, key=lambda item: item["distance"])[:limit]


FEATURE_SIGNAL_LABELS = {
    "rolling_mean_28": "최근 4주 판매 흐름",
    "rolling_mean_7": "최근 1주 판매 흐름",
    "lag_7": "지난주 같은 시점",
    "lag_14": "2주 전 같은 시점",
    "lag_28": "4주 전 같은 시점",
    "rolling_std_28": "최근 판매 변동성",
    "rolling_std_7": "단기 판매 변동성",
    "sell_price": "판매가 변화",
    "is_month_end": "월말 흐름",
    "is_month_start": "월초 흐름",
    "month": "월별 계절성",
    "week_of_year": "주차별 흐름",
}


def _cluster_feature_signals(
    members: list[str],
    existing_importances: dict[str, np.ndarray],
    limit: int = 3,
) -> list[str]:
    unique_members = sorted({member for member in members if member in existing_importances})
    if len(unique_members) < 3:
        return []

    matrix = np.vstack([np.abs(existing_importances[member]) for member in unique_members])
    mean_importance = np.nan_to_num(matrix.mean(axis=0), nan=0.0)
    ranked_indices = np.argsort(mean_importance)[::-1]
    signals = []
    for index in ranked_indices:
        feature = SELECTED_FEATURES[int(index)]
        label = FEATURE_SIGNAL_LABELS.get(feature, feature)
        if label not in signals:
            signals.append(label)
        if len(signals) >= limit:
            break
    return signals


def _cluster_assignment_for_client(
    client_id: str,
    client_df: pd.DataFrame,
    existing_importances: dict[str, np.ndarray],
    bubbles: list[list[str]],
    isolated: list[str],
    known_assignments: dict[str, int],
) -> dict[str, Any]:
    importance = _client_importance_from_frame(client_df)

    if client_id in existing_importances:
        cluster_id = known_assignments.get(client_id)
        if cluster_id is not None and 0 <= cluster_id < len(bubbles):
            members = bubbles[cluster_id]
        else:
            members = [client_id]
        signal_members = [cid for cid in members if cid in existing_importances]
        similar = _nearest_clients(importance, [cid for cid in members if cid != client_id] or list(existing_importances), existing_importances)
        representative = similar[0]["clientId"] if similar else client_id
        return {
            "clientId": client_id,
            "isKnownClient": True,
            "assignedTo": "known_client",
            "clusterId": cluster_id,
            "bubbleIndex": cluster_id,
            "clusterSize": len(signal_members),
            "privacySafe": len(signal_members) >= 3,
            "featureSignals": _cluster_feature_signals(signal_members, existing_importances),
            "representativeClientId": representative,
            "similarClients": similar,
            "distance": None,
            "threshold": None,
            "importance": importance,
        }

    result = assign_new_client(
        importance,
        existing_importances,
        bubbles,
        isolated,
        metric="cosine",
        new_client_id=client_id,
    )

    bubble_index = result.get("bubble_index")
    if bubble_index is not None and 0 <= int(bubble_index) < len(result["bubbles"]):
        members = [cid for cid in result["bubbles"][int(bubble_index)] if cid != client_id]
    else:
        members = list(existing_importances)

    signal_members = [cid for cid in members if cid in existing_importances]
    similar = _nearest_clients(importance, members, existing_importances)
    representative = similar[0]["clientId"] if similar else next(iter(existing_importances))
    return {
        "clientId": client_id,
        "isKnownClient": False,
        "assignedTo": result.get("assigned_to"),
        "clusterId": bubble_index,
        "bubbleIndex": bubble_index,
        "clusterSize": len(signal_members),
        "privacySafe": len(signal_members) >= 3,
        "featureSignals": _cluster_feature_signals(signal_members, existing_importances),
        "representativeClientId": representative,
        "similarClients": similar,
        "distance": round(float(result["distance"]), 4) if result.get("distance") is not None else None,
        "threshold": round(float(result["threshold"]), 4) if result.get("threshold") is not None else None,
        "importance": importance,
    }


def _cluster_assignment_for_importance(
    client_id: str,
    importance: np.ndarray,
    existing_importances: dict[str, np.ndarray],
    bubbles: list[list[str]],
    isolated: list[str],
    known_assignments: dict[str, int],
) -> dict[str, Any]:
    if not existing_importances or not bubbles:
        raise HTTPException(status_code=404, detail="기준 클러스터 또는 기준 feature importance가 없습니다.")

    for existing_client_id, existing_vector in existing_importances.items():
        if len(existing_vector) != len(importance):
            raise HTTPException(
                status_code=500,
                detail=f"기준 feature importance 길이가 현재 selectedFeatures와 맞지 않습니다: {existing_client_id}",
            )

    if client_id in existing_importances:
        cluster_id = known_assignments.get(client_id)
        if cluster_id is not None and 0 <= cluster_id < len(bubbles):
            members = list(bubbles[cluster_id])
        else:
            members = [client_id]
        return {
            "assignedTo": "known_client",
            "clusterId": cluster_id,
            "clusterMembers": members,
            "distance": None,
            "threshold": None,
        }

    result = assign_new_client(
        importance,
        existing_importances,
        bubbles,
        isolated,
        metric="cosine",
        new_client_id=client_id,
    )
    bubble_index = result.get("bubble_index")
    if bubble_index is not None and 0 <= int(bubble_index) < len(result["bubbles"]):
        members = list(result["bubbles"][int(bubble_index)])
    else:
        members = [client_id]

    return {
        "assignedTo": result.get("assigned_to"),
        "clusterId": int(bubble_index) if bubble_index is not None else None,
        "clusterMembers": members,
        "distance": round(float(result["distance"]), 4) if result.get("distance") is not None else None,
        "threshold": round(float(result["threshold"]), 4) if result.get("threshold") is not None else None,
    }


def _assignment_response(
    *,
    status: str,
    payload: ClusterAssignmentRequest,
    assigned_to: str | None,
    cluster_id: int | None,
    cluster_members: list[str],
    queue_status: str,
    distance: float | None,
    threshold: float | None,
    message: str,
) -> dict[str, Any]:
    return ClusterAssignmentResponse(
        status=status,
        clientId=payload.clientId,
        scope=payload.scope,
        assignedTo=assigned_to,
        clusterId=cluster_id,
        clusterMembers=cluster_members,
        queueStatus=queue_status,
        distance=distance,
        threshold=threshold,
        message=message,
    ).model_dump()


def _handle_single_client_assignment(payload: ClusterAssignmentRequest, importance: np.ndarray) -> dict[str, Any]:
    existing_importances = _load_feature_importances()
    bubbles, isolated, known_assignments = _load_cluster_state()
    assignment = _cluster_assignment_for_importance(
        payload.clientId,
        importance,
        existing_importances,
        bubbles,
        isolated,
        known_assignments,
    )
    CLIENT_CLUSTER_ASSIGNMENTS[payload.clientId] = {
        "assignedTo": assignment["assignedTo"],
        "clusterId": assignment["clusterId"],
        "clusterMembers": assignment["clusterMembers"],
    }
    return _assignment_response(
        status="assigned",
        payload=payload,
        assigned_to=assignment["assignedTo"],
        cluster_id=assignment["clusterId"],
        cluster_members=assignment["clusterMembers"],
        queue_status="completed",
        distance=assignment["distance"],
        threshold=assignment["threshold"],
        message="클러스터 배정이 완료되었습니다.",
    )


def _handle_all_clients_assignment(payload: ClusterAssignmentRequest, importance: np.ndarray) -> dict[str, Any]:
    completed_round = CLUSTER_ASSIGNMENT_COMPLETED.get(payload.roundId)
    if completed_round is not None:
        if payload.clientId in completed_round:
            raise HTTPException(status_code=409, detail="이미 완료된 roundId/clientId 요청입니다.")
        raise HTTPException(status_code=409, detail="이미 완료된 roundId입니다.")

    round_state = CLUSTER_ASSIGNMENT_QUEUES.setdefault(
        payload.roundId,
        {
            "expectedClientCount": payload.expectedClientCount,
            "clients": {},
        },
    )
    if round_state["expectedClientCount"] != payload.expectedClientCount:
        raise HTTPException(status_code=409, detail="동일 roundId의 expectedClientCount가 일치하지 않습니다.")

    clients: dict[str, dict[str, Any]] = round_state["clients"]
    if payload.clientId in clients:
        raise HTTPException(status_code=409, detail="동일 roundId에 같은 clientId가 이미 등록되어 있습니다.")
    clients[payload.clientId] = {
        "payload": payload,
        "importance": importance,
    }

    expected_count = int(payload.expectedClientCount or 0)
    if len(clients) < expected_count:
        return _assignment_response(
            status="queued",
            payload=payload,
            assigned_to=None,
            cluster_id=None,
            cluster_members=[],
            queue_status="waiting",
            distance=None,
            threshold=None,
            message=f"클러스터링 대기 중입니다. ({len(clients)}/{expected_count})",
        )

    client_ids = list(clients)
    vectors = np.array([clients[cid]["importance"] for cid in client_ids], dtype=np.float32)
    labels, _, _, _ = perform_clustering(
        vectors,
        max_clusters=8,
        complexity_penalty=0.15,
        singleton_penalty=0.20,
        metric="cosine",
    )
    grouped: dict[int, list[str]] = {}
    for index, label in enumerate(labels):
        grouped.setdefault(int(label), []).append(client_ids[index])

    completed: dict[str, dict[str, Any]] = {}
    for index, cid in enumerate(client_ids):
        cluster_id = int(labels[index])
        members = grouped[cluster_id]
        completed[cid] = {
            "assignedTo": "bubble" if len(members) > 1 else "isolated",
            "clusterId": cluster_id if len(members) > 1 else None,
            "clusterMembers": members,
            "distance": None,
            "threshold": None,
        }
        CLIENT_CLUSTER_ASSIGNMENTS[cid] = {
            "assignedTo": completed[cid]["assignedTo"],
            "clusterId": completed[cid]["clusterId"],
            "clusterMembers": completed[cid]["clusterMembers"],
        }

    CLUSTER_ASSIGNMENT_COMPLETED[payload.roundId] = completed
    CLUSTER_ASSIGNMENT_QUEUES.pop(payload.roundId, None)
    assignment = completed[payload.clientId]
    return _assignment_response(
        status="assigned",
        payload=payload,
        assigned_to=assignment["assignedTo"],
        cluster_id=assignment["clusterId"],
        cluster_members=assignment["clusterMembers"],
        queue_status="completed",
        distance=assignment["distance"],
        threshold=assignment["threshold"],
        message="전체 클라이언트 초기 클러스터링이 완료되었습니다.",
    )


def _resolve_client_assignment(client_id: str) -> dict[str, Any]:
    if client_id in CLIENT_CLUSTER_ASSIGNMENTS:
        assignment = CLIENT_CLUSTER_ASSIGNMENTS[client_id]
        return {
            "clusterId": assignment.get("clusterId"),
            "clusterMembers": list(assignment.get("clusterMembers", [client_id])),
            "assignedTo": assignment.get("assignedTo"),
        }

    bubbles, isolated, assignments = _load_cluster_state()
    if client_id in assignments:
        cluster_id = assignments[client_id]
        if 0 <= cluster_id < len(bubbles):
            return {
                "clusterId": cluster_id,
                "clusterMembers": list(bubbles[cluster_id]),
                "assignedTo": "bubble" if len(bubbles[cluster_id]) > 1 else "isolated",
            }
        return {
            "clusterId": None,
            "clusterMembers": [client_id],
            "assignedTo": "isolated",
        }

    if client_id in isolated:
        return {
            "clusterId": None,
            "clusterMembers": [client_id],
            "assignedTo": "isolated",
        }
    raise HTTPException(status_code=404, detail="클라이언트의 클러스터 배정 정보를 찾을 수 없습니다.")


def _model_response(
    path: Path,
    *,
    client_id: str,
    cluster_id: int | None,
    model_scope: str,
) -> FileResponse:
    if not path.exists():
        raise HTTPException(status_code=404, detail="다운로드할 FL 모델을 찾지 못했습니다.")
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=path.name,
        headers={
            "X-Fedstock-Client-Id": client_id,
            "X-Fedstock-Cluster-Id": "" if cluster_id is None else str(cluster_id),
            "X-Fedstock-Model-Scope": model_scope,
            "X-Fedstock-Storage-Path": str(path),
        },
    )


def _select_single_client_model(client_id: str, assignment: dict[str, Any], uploaded_path: Path) -> tuple[Path, str]:
    cluster_id = assignment.get("clusterId")
    if cluster_id is not None:
        bubble_path = RUN_DIR / "models" / "bubbles" / f"bubble_{int(cluster_id)}.pt"
        if bubble_path.exists():
            return bubble_path, "cluster"

    client_path = CLIENT_MODEL_DIR / f"client_{client_id}.pt"
    if client_path.exists():
        return client_path, "client"
    return uploaded_path, "fallback"


def _handle_all_clients_fl_model(
    *,
    client_id: str,
    round_id: str,
    sample_weight: float,
    uploaded_path: Path,
    state_dict: dict[str, torch.Tensor],
    assignment: dict[str, Any],
) -> FileResponse:
    cluster_id = assignment.get("clusterId")
    if cluster_id is None:
        raise HTTPException(status_code=404, detail="클러스터 모델 업데이트를 위한 클러스터 배정 정보가 없습니다.")

    cluster_members = set(str(member) for member in assignment.get("clusterMembers", []) if str(member))
    if not cluster_members:
        raise HTTPException(status_code=404, detail="클러스터 구성원 정보를 찾지 못했습니다.")

    round_state = FL_MODEL_SYNC_QUEUES.setdefault(round_id, {})
    cluster_state = round_state.setdefault(
        int(cluster_id),
        {
            "expectedMembers": cluster_members,
            "clients": {},
        },
    )
    if set(cluster_state["expectedMembers"]) != cluster_members:
        raise HTTPException(status_code=409, detail="동일 round_id의 클러스터 구성원 정보가 일치하지 않습니다.")

    clients: dict[str, dict[str, Any]] = cluster_state["clients"]
    if client_id in clients:
        raise HTTPException(status_code=409, detail="동일 round_id에 같은 client_id 모델이 이미 업로드되었습니다.")

    clients[client_id] = {
        "path": uploaded_path,
        "state": state_dict,
        "sampleWeight": sample_weight,
    }

    if set(clients) != cluster_members:
        raise HTTPException(status_code=409, detail="클러스터 모델 업데이트가 아직 완료되지 않았습니다.")

    averaged = _weighted_average_state_dicts(
        [(item["state"], float(item["sampleWeight"])) for item in clients.values()]
    )
    cluster_dir = _fl_sync_root(round_id) / "clusters"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    output_path = cluster_dir / f"bubble_{int(cluster_id)}.pt"
    torch.save(averaged, output_path)
    return _model_response(output_path, client_id=client_id, cluster_id=int(cluster_id), model_scope="cluster")


def _load_model(model_path: Path) -> LightweightLSTM:
    state_dict = torch.load(model_path, map_location="cpu")
    input_size = int(state_dict["lstm.weight_ih_l0"].shape[1])
    hidden_size = int(state_dict["lstm.weight_hh_l0"].shape[1])
    if input_size != len(SELECTED_FEATURES):
        raise HTTPException(
            status_code=500,
            detail="예상 판매량 계산 기준이 현재 파일과 맞지 않습니다.",
        )
    model = LightweightLSTM(input_size=input_size, hidden_size=hidden_size)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


_MODEL_CACHE: dict[str, LightweightLSTM] = {}


def _get_model_for_representative(representative_client_id: str) -> tuple[LightweightLSTM, Path]:
    raise HTTPException(status_code=503, detail="Local repository model artifacts are not used in production.")
    if not candidate.exists():
        candidates = []
        if not candidates:
            raise HTTPException(status_code=500, detail="예상 판매량 계산에 필요한 기준 정보를 찾지 못했습니다.")
        candidate = candidates[0]

    key = str(candidate)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = _load_model(candidate)
    return _MODEL_CACHE[key], candidate


def _model_artifact_uri_for_assignment(assignment: dict[str, Any]) -> str:
    settings = get_settings()
    if not settings.artifact_bucket:
        raise HTTPException(status_code=503, detail="ARTIFACT_BUCKET is required for operational model artifact loading.")

    run_id = RUN_DIR.name
    cluster_id = assignment.get("clusterId")
    if cluster_id is not None:
        return build_s3_uri(settings.artifact_bucket, f"models/clusters/{run_id}/cluster-{cluster_id}.pt")

    representative = str(assignment["representativeClientId"])
    return build_s3_uri(settings.artifact_bucket, f"updates/{run_id}/clients/{representative}.pt")


def _get_model_for_assignment(assignment: dict[str, Any]) -> tuple[LightweightLSTM, str]:
    artifact_uri = _model_artifact_uri_for_assignment(assignment)
    safe_name = _safe_path_component(
        f"cluster-{assignment['clusterId']}" if assignment.get("clusterId") is not None else str(assignment["representativeClientId"]),
        "model artifact",
    )
    local_path = _ensure_model_local_dir() / "analyze-csv" / RUN_DIR.name / f"{safe_name}.pt"
    try:
        download_s3_file(artifact_uri, local_path)
    except StorageError as exc:
        raise _storage_http_error(exc) from exc

    key = artifact_uri
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = _load_model(local_path)
    return _MODEL_CACHE[key], artifact_uri


def _predict_sales(
    df: pd.DataFrame,
    assignments: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    historical_rows: list[dict[str, Any]] = []
    latest_rows: list[dict[str, Any]] = []
    used_model_artifacts: set[str] = set()

    with torch.no_grad():
        for client_id, client_df in df.groupby("client_id", sort=False):
            client_df = client_df.copy()
            assignment = assignments[str(client_id)]
            model, model_artifact_uri = _get_model_for_assignment(assignment)
            used_model_artifacts.add(model_artifact_uri)

            target_values = client_df[FORECAST_TARGET_COLUMN].dropna().to_numpy(dtype=np.float32)
            if len(target_values) == 0 or len(client_df) <= SEQ_LEN:
                continue

            x_scaler = StandardScaler()
            y_scaler = RobustScaler()
            scaled_features = x_scaler.fit_transform(client_df[SELECTED_FEATURES].to_numpy(dtype=np.float32)).astype(np.float32)
            y_scaler.fit(target_values.reshape(-1, 1))
            client_df["_feature_row"] = list(scaled_features)

            for item_id, group in client_df.groupby("item_id", sort=False):
                group = group.sort_values("sale_date").reset_index(drop=True)
                if len(group) <= SEQ_LEN:
                    continue

                group_features = np.stack(group["_feature_row"].to_numpy())
                start_idx = max(SEQ_LEN, len(group) - HISTORY_WINDOW_PER_ITEM)
                for target_idx in range(start_idx, len(group)):
                    actual_target = group.loc[target_idx, FORECAST_TARGET_COLUMN]
                    if pd.isna(actual_target):
                        continue
                    sequence = group_features[target_idx - SEQ_LEN:target_idx]
                    tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
                    scaled_prediction = model(tensor).cpu().numpy()
                    prediction = float(y_scaler.inverse_transform(scaled_prediction.reshape(-1, 1))[0, 0])
                    historical_rows.append({
                        "client_id": str(client_id),
                        "item_id": item_id,
                        "sale_date": group.loc[target_idx, "sale_date"],
                        "actual": max(0.0, float(actual_target)),
                        "predicted": max(0.0, prediction),
                    })

                latest_sequence = group_features[-SEQ_LEN:]
                tensor = torch.tensor(latest_sequence, dtype=torch.float32).unsqueeze(0)
                scaled_prediction = model(tensor).cpu().numpy()
                prediction = float(y_scaler.inverse_transform(scaled_prediction.reshape(-1, 1))[0, 0])
                latest = group.iloc[-1].to_dict()
                latest["forecast_qty"] = max(0.0, prediction)
                latest["model_artifact_uri"] = model_artifact_uri
                latest["assigned_cluster"] = assignment["clusterId"]
                latest["representative_client_id"] = assignment["representativeClientId"]
                latest_rows.append(latest)

    if not latest_rows:
        raise HTTPException(status_code=400, detail="예상 판매량을 계산할 수 있는 상품 이력이 없습니다.")

    return pd.DataFrame(historical_rows), pd.DataFrame(latest_rows), sorted(used_model_artifacts)


def _trend_from_predictions(source: pd.DataFrame, historical_predictions: pd.DataFrame) -> list[dict[str, Any]]:
    actual_source = source.dropna(subset=[FORECAST_TARGET_COLUMN])
    actual_by_date = actual_source.groupby(actual_source["sale_date"].dt.date).agg(sales=(FORECAST_TARGET_COLUMN, "sum")).reset_index()
    pred_by_date = historical_predictions.groupby(historical_predictions["sale_date"].dt.date).agg(forecast=("predicted", "sum")).reset_index()
    merged = actual_by_date.merge(pred_by_date, on="sale_date", how="inner").tail(30)
    return [
        {
            "date": pd.to_datetime(row["sale_date"]).strftime("%m/%d"),
            "sales": int(round(row["sales"])),
            "forecast": int(round(row["forecast"])),
            "revenue": 0,
        }
        for _, row in merged.iterrows()
    ]


def _daily_forecast_points(
    source: pd.DataFrame,
    client_id: str,
    item_id: str,
    forecast_qty: float,
    forecast_start: pd.Timestamp,
) -> list[dict[str, Any]]:
    forecast_dates = [forecast_start + timedelta(days=offset) for offset in range(FORECAST_HORIZON_DAYS)]
    item_history = source[
        (source["client_id"].astype(str) == client_id)
        & (source["item_id"].astype(str) == item_id)
    ].sort_values("sale_date").tail(28)

    weights = np.ones(FORECAST_HORIZON_DAYS, dtype=np.float32)
    if not item_history.empty and float(item_history["sales"].sum()) > 0:
        history = item_history.copy()
        history["dayofweek"] = history["sale_date"].dt.dayofweek
        weekday_mean = history.groupby("dayofweek")["sales"].mean()
        weights = np.asarray(
            [max(0.0, float(weekday_mean.get(date.dayofweek, 0.0))) for date in forecast_dates],
            dtype=np.float32,
        )
        if float(weights.sum()) <= 0:
            weights = np.ones(FORECAST_HORIZON_DAYS, dtype=np.float32)

    weights = weights / float(weights.sum())
    daily_values = weights * float(forecast_qty)
    rounded_values = np.round(daily_values, 1)
    rounded_values[-1] = round(max(0.0, float(forecast_qty) - float(rounded_values[:-1].sum())), 1)

    return [
        {
            "date": date.strftime("%m/%d"),
            "isoDate": date.strftime("%Y-%m-%d"),
            "sales": round(float(rounded_values[idx]), 1),
        }
        for idx, date in enumerate(forecast_dates)
    ]


def _resolve_item_display(raw_item_id: str, uploaded_item_name: Any, uploaded_category: Any) -> dict[str, Any]:
    master = ITEM_MASTER.get(raw_item_id, {})
    uploaded_name = str(uploaded_item_name).strip() if uploaded_item_name is not None else ""
    item_name = uploaded_name if uploaded_name and uploaded_name != raw_item_id else master.get("itemName") or raw_item_id
    category = master.get("category") or str(uploaded_category)

    return {
        "itemName": item_name,
        "category": category,
        "weatherTags": master.get("weatherTags", []),
    }


def _build_dashboard(
    file_name: str,
    source: pd.DataFrame,
    latest_predictions: pd.DataFrame,
    historical_predictions: pd.DataFrame,
    validation: list[dict[str, Any]],
    issues: list[dict[str, str]],
    used_model_artifacts: list[str],
    stock_available: bool,
    cluster_assignments: list[dict[str, Any]],
) -> dict[str, Any]:
    forecast_items = []
    top_products = []
    forecast_daily_series = []
    date_min = source["sale_date"].min()
    date_max = source["sale_date"].max()
    date_range = f"{date_min.strftime('%Y. %m. %d.')} - {date_max.strftime('%Y. %m. %d.')}"
    forecast_start = date_max + timedelta(days=1)
    forecast_end = date_max + timedelta(days=FORECAST_HORIZON_DAYS)
    forecast_window = {
        "anchorDate": date_max.strftime("%Y-%m-%d"),
        "startDate": forecast_start.strftime("%Y-%m-%d"),
        "endDate": forecast_end.strftime("%Y-%m-%d"),
        "horizonDays": FORECAST_HORIZON_DAYS,
        "label": f"{forecast_start.strftime('%Y. %m. %d.')} - {forecast_end.strftime('%Y. %m. %d.')}",
    }

    for _, row in latest_predictions.iterrows():
        forecast_qty = max(0.0, float(row["forecast_qty"]))
        forecast_daily_qty = forecast_qty / FORECAST_HORIZON_DAYS
        rolling_mean_7 = float(row["rolling_mean_7"]) if not pd.isna(row["rolling_mean_7"]) else forecast_daily_qty
        rolling_mean_28 = float(row["rolling_mean_28"]) if not pd.isna(row["rolling_mean_28"]) else rolling_mean_7
        sell_price = max(0.0, float(row["sell_price"]))

        trend_gap = ((rolling_mean_7 - rolling_mean_28) / rolling_mean_28 * 100) if rolling_mean_28 > 0 else 0
        trend = "up" if trend_gap > 8 else "down" if trend_gap < -8 else "stable"
        client_id = str(row.get("client_id", "default"))
        raw_item_id = str(row["item_id"])
        display = _resolve_item_display(raw_item_id, row.get("item_name"), row.get("category"))
        item = {
            "itemId": f"{client_id}:{raw_item_id}" if client_id != "default" else raw_item_id,
            "itemName": display["itemName"],
            "category": f"{client_id} · {display['category']}" if client_id != "default" else display["category"],
            "rawItemId": raw_item_id,
            "weatherTags": display["weatherTags"],
        }
        forecast_items.append({
            **item,
            "forecastQty": round(forecast_qty, 1),
            "forecastDailyQty": round(forecast_daily_qty, 1),
            "forecastHorizonDays": FORECAST_HORIZON_DAYS,
            "rollingMean7": round(rolling_mean_7, 1),
            "rollingMean28": round(rolling_mean_28, 1),
            "wowChangePct": round(trend_gap, 1),
            "trend": trend,
            "confidence": 84,
        })
        top_products.append({
            **item,
            "sales": round(forecast_qty),
            "revenue": round(forecast_qty * sell_price),
        })
        forecast_daily_series.append({
            **item,
            "forecastQty": round(forecast_qty, 1),
            "forecastHorizonDays": FORECAST_HORIZON_DAYS,
            "points": _daily_forecast_points(source, client_id, raw_item_id, forecast_qty, forecast_start),
        })

    forecast_total = sum(item["forecastQty"] for item in forecast_items)
    revenue_estimate = sum(item["revenue"] for item in top_products)
    overview_metrics = [
        {"label": f"{FORECAST_HORIZON_DAYS}일 예상 판매량", "value": f"{_format_number(forecast_total)}개", "helper": f"{forecast_window['label']} 예상", "trend": "up", "iconKey": "TrendingUp", "tone": "primary"},
        {"label": "계산한 상품 수", "value": f"{len(forecast_items)}개", "helper": "계산 완료", "iconKey": "Package", "tone": "info"},
        {"label": "예측 기간", "value": f"{FORECAST_HORIZON_DAYS}일", "helper": f"기준일 {date_max.strftime('%Y. %m. %d.')}", "iconKey": "TrendingUp", "tone": "primary"},
        {"label": f"{FORECAST_HORIZON_DAYS}일 예상 매출", "value": _format_currency(revenue_estimate), "helper": "예상 판매량 × 판매가", "iconKey": "DollarSign", "tone": "success"},
    ]

    data = {
        "source": "ai",
        "overviewMetrics": overview_metrics,
        "forecastWindow": forecast_window,
        "salesTrend": [],
        "topProducts": sorted(top_products, key=lambda item: item["sales"], reverse=True)[:12],
        "forecastItems": sorted(forecast_items, key=lambda item: item["forecastQty"], reverse=True),
        "forecastDailySeries": sorted(forecast_daily_series, key=lambda item: item["forecastQty"], reverse=True),
        "forecastSeries": [],
        "inventoryMetrics": [],
        "inventoryItems": [],
        "orderMetrics": [],
        "orderRecommendations": [],
        "clusterAssignments": cluster_assignments,
    }

    status = _csv_status(
        file_name=file_name,
        rows=len(source),
        product_count=source["item_id"].nunique(),
        date_range=date_range,
        validation=validation,
        issues=issues,
    )
    return {
        "status": status,
        "data": data,
        "model": {
            "artifactStorage": "s3" if get_settings().artifact_bucket else "not_configured",
            "modelCount": len(used_model_artifacts),
            "artifactUris": used_model_artifacts[:10],
            "selectedFeatures": SELECTED_FEATURES,
            "sequenceLength": SEQ_LEN,
            "forecastTarget": FORECAST_TARGET_COLUMN,
            "forecastHorizonDays": FORECAST_HORIZON_DAYS,
            "forecastUnit": FORECAST_UNIT,
            "stockAvailable": stock_available,
            "runId": RUN_DIR.name,
        },
    }


@app.post("/clients/fl-model/aggregate")
def aggregate_fl_models(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    scope = str(_json_field(payload, "scope")).strip()
    if scope != "all_clients":
        raise HTTPException(status_code=400, detail="scope must be all_clients.")

    round_id = str(_json_field(payload, "roundId", "round_id")).strip()
    safe_round_id = _safe_path_component(round_id, "roundId")
    output_prefix_uri = str(_json_field(payload, "outputPrefixUri", "output_prefix_uri")).strip()
    models = _json_field(payload, "models")
    if not isinstance(models, list) or not models:
        raise HTTPException(status_code=400, detail="models must be a non-empty array.")

    expected_count = _json_field(
        payload,
        "expectedClientCount",
        "expected_client_count",
        required=False,
        default=len(models),
    )
    if int(expected_count) != len(models):
        raise HTTPException(status_code=409, detail="expectedClientCount does not match models length.")

    cluster_id = _json_field(payload, "clusterId", "cluster_id", required=False)
    model_version = _json_field(payload, "modelVersion", "model_version", required=False)
    cluster_id_text = None if cluster_id is None else str(cluster_id).strip()
    if cluster_id_text:
        _safe_path_component(cluster_id_text, "clusterId")

    local_root = _ensure_model_local_dir() / safe_round_id
    download_dir = local_root / "downloads"
    weighted_states: list[tuple[dict[str, torch.Tensor], float]] = []

    try:
        parse_s3_uri(output_prefix_uri.rstrip("/") + "/placeholder")
        for item in models:
            if not isinstance(item, dict):
                raise HTTPException(status_code=400, detail="models entries must be objects.")
            client_id = str(_json_field(item, "clientId", "client_id")).strip()
            safe_client_id = _safe_path_component(client_id, "clientId")
            sample_count = float(_json_field(item, "sampleCount", "sample_count", "sampleWeight", "sample_weight"))
            artifact_uri = str(_json_field(item, "modelArtifactUri", "model_artifact_uri")).strip()
            local_path = download_dir / f"client_{safe_client_id}.pt"
            download_s3_file(artifact_uri, local_path)
            weighted_states.append((_load_tensor_state_dict(local_path), sample_count))
    except StorageError as exc:
        raise _storage_http_error(exc) from exc

    averaged = _weighted_average_state_dicts(weighted_states)
    output_filename = f"cluster-{cluster_id_text}.pt" if cluster_id_text else "global.pt"
    output_path = local_root / "aggregated" / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(averaged, output_path)

    try:
        aggregated_uri = upload_s3_file(output_path, _join_s3_prefix(output_prefix_uri, output_filename))
    except StorageError as exc:
        raise _storage_http_error(exc) from exc

    response: dict[str, Any] = {
        "ok": True,
        "roundId": round_id,
        "receivedClientCount": len(models),
        "aggregatedModelUri": aggregated_uri,
        "aggregation": "sample_count_weighted_fedavg",
    }
    if cluster_id_text is not None:
        response["clusterId"] = cluster_id_text
    if model_version is not None:
        response["modelVersion"] = str(model_version)
    return response


def _parse_batch_metadata(metadata: str | bytes | None) -> dict[str, Any]:
    if metadata is None:
        raise HTTPException(status_code=400, detail="metadata is required.")
    raw = metadata.decode("utf-8") if isinstance(metadata, bytes) else str(metadata)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"metadata must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="metadata must be a JSON object.")
    return parsed


@app.post("/clients/fl-model/batch")
async def receive_fl_model_batch(
    metadata: str | None = Form(default=None),
    model_files: list[UploadFile] | None = File(default=None),
) -> dict[str, Any]:
    parsed = _parse_batch_metadata(metadata)
    scope = str(_json_field(parsed, "scope")).strip()
    if scope != "all_clients":
        raise HTTPException(status_code=400, detail="scope must be all_clients.")
    round_id = str(_json_field(parsed, "round_id", "roundId")).strip()
    safe_round_id = _safe_path_component(round_id, "round_id")
    expected_client_count = int(_json_field(parsed, "expected_client_count", "expectedClientCount"))
    model_items = _json_field(parsed, "models")
    if not isinstance(model_items, list) or not model_items:
        raise HTTPException(status_code=400, detail="models must be a non-empty array.")
    if not model_files:
        raise HTTPException(status_code=400, detail="model_files is required.")
    if expected_client_count != len(model_files):
        raise HTTPException(status_code=409, detail="expected_client_count does not match uploaded model_files count.")
    if len(model_items) != len(model_files):
        raise HTTPException(status_code=400, detail="models metadata count does not match uploaded model_files count.")

    expected_by_filename: dict[str, dict[str, Any]] = {}
    for item in model_items:
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail="models entries must be objects.")
        client_id = str(_json_field(item, "client_id", "clientId")).strip()
        filename = str(_json_field(item, "filename")).strip()
        _safe_path_component(client_id, "client_id")
        safe_filename = _safe_path_component(filename, "filename")
        if not safe_filename.lower().endswith(".pt"):
            raise HTTPException(status_code=400, detail="model files must use .pt extension.")
        sample_weight = float(_json_field(item, "sample_weight", "sampleWeight", required=False, default=0))
        if sample_weight < 0:
            raise HTTPException(status_code=400, detail="sample_weight must be greater than or equal to zero.")
        expected_by_filename[safe_filename] = item

    upload_dir = _ensure_model_local_dir() / safe_round_id / "clients"
    upload_dir.mkdir(parents=True, exist_ok=True)
    received = 0
    for model_file in model_files:
        safe_filename = _safe_path_component(model_file.filename or "", "filename")
        if safe_filename not in expected_by_filename:
            raise HTTPException(status_code=400, detail=f"Unexpected uploaded model file: {safe_filename}")
        if not safe_filename.lower().endswith(".pt"):
            raise HTTPException(status_code=400, detail="model_files must use .pt extension.")
        target_path = upload_dir / safe_filename
        content = await model_file.read()
        if not content:
            raise HTTPException(status_code=400, detail=f"Uploaded model file is empty: {safe_filename}")
        target_path.write_bytes(content)
        _load_tensor_state_dict(target_path)
        received += 1

    return {
        "ok": True,
        "round_id": round_id,
        "received_client_count": received,
    }


@app.get("/ai/health")
def health_check() -> dict[str, Any]:
    settings = get_settings()
    model_dir = Path(settings.model_local_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    return {
        "status": "ok",
        "service": "fedstock-ai",
        "env": settings.env,
        "aws_region": settings.aws_region,
        "artifact_bucket_configured": bool(settings.artifact_bucket),
        "model_local_dir_configured": True,
        "analyzeCsvAvailable": True,
        "modelLocalDirReady": model_dir.exists(),
        "featureImportancesExists": FEATURE_IMPORTANCES_PATH.exists(),
        "clusteringResultsExists": CLUSTERING_RESULTS_PATH.exists(),
        "selectedFeatures": SELECTED_FEATURES,
        "forecastTarget": FORECAST_TARGET_COLUMN,
        "forecastHorizonDays": FORECAST_HORIZON_DAYS,
    }


@app.get("/health")
def legacy_health_check() -> dict[str, Any]:
    return health_check()


def _cluster_assignment_handler(raw_payload: dict[str, Any]) -> dict[str, Any]:
    payload = _parse_cluster_assignment_request(raw_payload)
    importance = _importance_vector_from_payload(payload)
    if payload.scope == "single_client":
        return _handle_single_client_assignment(payload, importance)
    return _handle_all_clients_assignment(payload, importance)


@app.post("/clients/cluster-assignment")
@app.post("/ai/clients/cluster-assignment")
def cluster_assignment(
    raw_payload: dict[str, Any] = Body(...),
    _: None = Depends(_verify_bearer_token),
) -> dict[str, Any]:
    return _cluster_assignment_handler(raw_payload)


@app.post("/clients/{client_id}/fl-model")
@app.post("/ai/clients/{client_id}/fl-model")
def fl_model_sync(
    client_id: str,
    model_file: UploadFile = File(...),
    form_client_id: str = Form(..., alias="client_id"),
    scope: str = Form(...),
    round_id: str = Form(...),
    sample_weight: float | None = Form(default=None),
    _: None = Depends(_verify_bearer_token),
) -> FileResponse:
    if client_id != form_client_id:
        raise HTTPException(status_code=400, detail="Path의 client_id와 Body의 client_id가 일치하지 않습니다.")
    if scope not in {"single_client", "all_clients"}:
        raise HTTPException(status_code=400, detail="지원하지 않는 scope입니다.")
    if not round_id.strip():
        raise HTTPException(status_code=400, detail="round_id는 비어 있을 수 없습니다.")

    effective_weight = 1.0 if sample_weight is None else float(sample_weight)
    if effective_weight <= 0:
        raise HTTPException(status_code=400, detail="sample_weight는 0보다 커야 합니다.")

    _safe_path_component(client_id, "client_id")
    _safe_path_component(round_id, "round_id")
    uploaded_path = _save_uploaded_model(model_file, client_id, round_id)
    state_dict = _load_state_dict(uploaded_path)
    assignment = _resolve_client_assignment(client_id)

    if scope == "single_client":
        selected_path, model_scope = _select_single_client_model(client_id, assignment, uploaded_path)
        return _model_response(
            selected_path,
            client_id=client_id,
            cluster_id=assignment.get("clusterId"),
            model_scope=model_scope,
        )

    return _handle_all_clients_fl_model(
        client_id=client_id,
        round_id=round_id,
        sample_weight=effective_weight,
        uploaded_path=uploaded_path,
        state_dict=state_dict,
        assignment=assignment,
    )


@app.post("/analyze-csv")
async def analyze_csv(file: UploadFile = File(...)) -> dict[str, Any]:
    content = await file.read()
    raw_df = _read_csv(file, content)
    prepared, validation, issues, stock_available = _prepare_frame(raw_df)

    existing_importances = _load_feature_importances()
    bubbles, isolated, known_assignments = _load_cluster_state()
    if not existing_importances or not bubbles:
        raise HTTPException(
            status_code=500,
            detail="예상 판매량 계산에 필요한 기준 정보를 확인하지 못했습니다.",
        )

    assignments_by_client: dict[str, dict[str, Any]] = {}
    serializable_assignments: list[dict[str, Any]] = []
    for client_id, client_df in prepared.groupby("client_id", sort=False):
        assignment = _cluster_assignment_for_client(
            str(client_id),
            client_df,
            existing_importances,
            bubbles,
            isolated,
            known_assignments,
        )
        assignments_by_client[str(client_id)] = assignment
        serializable_assignments.append({
            "isKnownClient": assignment["isKnownClient"],
            "assignedTo": assignment["assignedTo"],
            "clusterId": assignment["clusterId"],
            "clusterSize": assignment["clusterSize"],
            "privacySafe": assignment["privacySafe"],
            "featureSignals": assignment["featureSignals"],
            "distance": assignment["distance"],
            "threshold": assignment["threshold"],
        })

    known_count = sum(1 for item in serializable_assignments if item["isKnownClient"])
    new_count = len(serializable_assignments) - known_count
    if known_count:
        issues.append({
            "severity": "warning",
            "message": f"{known_count}개 매장 구분은 기존 판매 패턴과 일치해 기존 기준으로 계산했습니다.",
        })
    if new_count:
        issues.append({
            "severity": "warning",
            "message": f"{new_count}개 신규 매장 구분을 비슷한 판매 패턴의 매장 유형에 연결했습니다.",
        })

    historical_predictions, latest_predictions, used_model_artifacts = _predict_sales(prepared, assignments_by_client)
    return _build_dashboard(
        file_name=file.filename or "uploaded.csv",
        source=prepared,
        latest_predictions=latest_predictions,
        historical_predictions=historical_predictions,
        validation=validation,
        issues=issues,
        used_model_artifacts=used_model_artifacts,
        stock_available=stock_available,
        cluster_assignments=serializable_assignments,
    )
