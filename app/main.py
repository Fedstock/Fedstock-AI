from __future__ import annotations

import math
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.models.lstm import LightweightLSTM


ROOT_DIR = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT_DIR / "outputs" / "runs" / "20260522_025148_061195"
CLIENT_MODEL_DIR = RUN_DIR / "models" / "clients"
CONFIG_PATH = RUN_DIR / "config.json"

SEQ_LEN = 14
DEFAULT_LEAD_TIME_DAYS = 4
HISTORY_WINDOW_PER_ITEM = 35


def _load_selected_features() -> list[str]:
    if CONFIG_PATH.exists():
        config = pd.read_json(CONFIG_PATH, typ="series").to_dict()
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


app = FastAPI(title="Fedstock AI API", version="0.1.0")
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


def _validation_item(
    column: str,
    label: str,
    ok: bool,
    required: bool = True,
    warning_message: str | None = None,
) -> dict[str, Any]:
    status = "passed" if ok else "failed"
    message = "AI 분석 입력 확인" if ok else "AI 분석에 필요한 항목 누락"
    if not ok and not required:
        status = "warning"
        message = warning_message or "없으면 백엔드에서 추정합니다."

    return {
        "column": column,
        "label": label,
        "required": required,
        "status": status,
        "message": message,
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
    except Exception as exc:  # pragma: no cover - message is surfaced to UI
        raise HTTPException(status_code=400, detail=f"CSV를 읽을 수 없습니다: {exc}") from exc


def _prepare_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, str]], bool]:
    item_col = _find_column(df, ["item_id", "itemId", "id", "상품 ID", "상품ID", "상품 번호", "상품번호"])
    date_col = _find_column(df, ["sale_date", "date", "판매일", "판매 날짜", "판매날짜", "날짜", "일자"])
    sales_col = _find_column(df, ["sales", "quantity", "판매량", "판매 수량", "판매수량", "수량"])
    stock_col = _find_column(df, ["current_stock", "stock", "현재 재고", "현재재고", "재고", "재고 수량", "재고수량"])
    price_col = _find_column(df, ["sell_price", "price", "판매가", "가격", "상품 가격", "상품가격"])
    client_col = _find_column(df, ["client_id", "clientId", "클라이언트 ID", "클라이언트ID", "매장 클라이언트", "매장클라이언트"])

    stock_missing = stock_col is None
    validation = [
        _validation_item("item_id", "상품 ID", item_col is not None),
        _validation_item("sale_date", "판매일", date_col is not None),
        _validation_item("sales", "판매량", sales_col is not None),
        _validation_item(
            "current_stock",
            "현재 재고",
            stock_col is not None,
            required=False,
            warning_message="재고가 없으면 판매량 예측만 표시하고 재고/발주는 계산하지 않습니다.",
        ),
        _validation_item("sell_price", "판매가", price_col is not None),
    ]
    missing = [item["label"] for item in validation if item["status"] == "failed"]
    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"AI 모델 입력에 필요한 컬럼이 없습니다: {', '.join(missing)}",
                "validation": validation,
            },
        )

    prepared = df.copy()
    prepared["item_id"] = prepared[item_col].astype(str)
    prepared["client_id"] = prepared[client_col].astype(str) if client_col else "default"
    prepared["item_name"] = prepared[_find_column(df, ["item_name", "name", "상품명", "상품 이름", "상품이름"])].astype(str) if _find_column(df, ["item_name", "name", "상품명", "상품 이름", "상품이름"]) else prepared["item_id"]
    category_col = _find_column(df, ["category", "cat_id", "dept_id", "카테고리", "상품 분류", "상품분류", "분류"])
    prepared["category"] = prepared[category_col].astype(str) if category_col else prepared["item_id"].str.split("_").str[0].fillna("UNKNOWN")
    prepared["sale_date"] = pd.to_datetime(prepared[date_col], errors="coerce")
    prepared["sales"] = pd.to_numeric(prepared[sales_col], errors="coerce")
    prepared["current_stock"] = pd.to_numeric(prepared[stock_col], errors="coerce") if stock_col else np.nan
    prepared["sell_price"] = pd.to_numeric(prepared[price_col], errors="coerce")
    prepared["lead_time_days"] = pd.to_numeric(prepared[_find_column(df, ["lead_time_days", "입고 기간", "입고기간", "리드타임", "lead time"])], errors="coerce") if _find_column(df, ["lead_time_days", "입고 기간", "입고기간", "리드타임", "lead time"]) else DEFAULT_LEAD_TIME_DAYS
    prepared["ordered_qty"] = pd.to_numeric(prepared[_find_column(df, ["ordered_qty", "on_order_qty", "입고 예정 수량", "입고예정수량", "발주 중 수량", "발주중수량"])], errors="coerce") if _find_column(df, ["ordered_qty", "on_order_qty", "입고 예정 수량", "입고예정수량", "발주 중 수량", "발주중수량"]) else 0
    prepared["is_holiday"] = pd.to_numeric(prepared[_find_column(df, ["is_holiday", "휴일 여부", "휴일여부", "공휴일"])], errors="coerce") if _find_column(df, ["is_holiday", "휴일 여부", "휴일여부", "공휴일"]) else 0

    prepared = prepared.dropna(subset=["item_id", "sale_date", "sales", "sell_price"])
    if prepared.empty:
        raise HTTPException(status_code=400, detail="AI 분석에 사용할 수 있는 행이 없습니다.")

    prepared = prepared.sort_values(["item_id", "sale_date"]).reset_index(drop=True)
    prepared["dayofweek"] = prepared["sale_date"].dt.dayofweek
    prepared["month"] = prepared["sale_date"].dt.month
    prepared["week_of_year"] = prepared["sale_date"].dt.isocalendar().week.astype(int)
    prepared["is_weekend"] = prepared["dayofweek"].isin([5, 6]).astype(int)
    prepared["is_month_start"] = prepared["sale_date"].dt.is_month_start.astype(int)
    prepared["is_month_end"] = prepared["sale_date"].dt.is_month_end.astype(int)

    grouped_sales = prepared.groupby("item_id")["sales"]
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
            prepared.groupby("item_id")["sell_price"]
            .pct_change()
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
        )
    else:
        prepared["price_change_rate"] = pd.to_numeric(prepared["price_change_rate"], errors="coerce").fillna(0)

    usable = prepared.dropna(subset=SELECTED_FEATURES + ["sales"]).reset_index(drop=True)
    issues: list[dict[str, str]] = []
    if stock_missing:
        issues.append({
            "severity": "warning",
            "message": "업로드 파일에 현재 재고가 없어 AI 판매량 예측만 표시합니다. 재고 점검과 발주 추천은 current_stock 컬럼이 있을 때 계산됩니다.",
        })
    dropped = len(prepared) - len(usable)
    if dropped > 0:
        issues.append({
            "severity": "warning",
            "message": f"lag/rolling feature 생성을 위해 초기 {dropped:,}개 행은 AI 추론에서 제외했습니다.",
        })
    if usable.groupby("item_id").size().max() <= SEQ_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"AI 추론에는 lag/rolling feature 생성을 포함해 상품별 약 {SEQ_LEN + 29}일 이상의 판매 이력이 필요합니다.",
        )

    issues.append({
        "severity": "warning",
        "message": "업로드된 CSV를 로컬 PA-CFL LSTM 모델로 추론했습니다. 프론트 계산값은 사용하지 않았습니다.",
    })
    return usable, validation, issues, not stock_missing


def _choose_model_path(df: pd.DataFrame) -> Path:
    env_model = os.getenv("FEDSTOCK_MODEL_PATH")
    if env_model:
        path = Path(env_model)
        if path.exists():
            return path

    client_col = _find_column(df, ["client_id", "clientId", "클라이언트 ID", "클라이언트ID", "매장 클라이언트", "매장클라이언트"])
    if client_col:
        client_id = str(df[client_col].mode().iloc[0])
        candidate = CLIENT_MODEL_DIR / f"client_{client_id}.pt"
        if candidate.exists():
            return candidate

    env_client = os.getenv("FEDSTOCK_CLIENT_ID")
    if env_client:
        candidate = CLIENT_MODEL_DIR / f"client_{env_client}.pt"
        if candidate.exists():
            return candidate

    candidates = sorted(CLIENT_MODEL_DIR.glob("client_*.pt"))
    if not candidates:
        raise HTTPException(status_code=500, detail="저장된 AI 모델(.pt)을 찾지 못했습니다.")
    return candidates[0]


def _load_model(model_path: Path) -> LightweightLSTM:
    state_dict = torch.load(model_path, map_location="cpu")
    weight = state_dict["lstm.weight_ih_l0"]
    hidden_size = int(state_dict["lstm.weight_hh_l0"].shape[1])
    input_size = int(weight.shape[1])
    if input_size != len(SELECTED_FEATURES):
        raise HTTPException(
            status_code=500,
            detail=f"모델 입력 크기({input_size})와 선택 feature 수({len(SELECTED_FEATURES)})가 다릅니다.",
        )

    model = LightweightLSTM(input_size=input_size, hidden_size=hidden_size)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


_MODEL_CACHE: dict[str, LightweightLSTM] = {}


def _get_model_for_client(client_id: str, fallback_model_path: Path) -> tuple[LightweightLSTM, Path]:
    candidate = CLIENT_MODEL_DIR / f"client_{client_id}.pt"
    model_path = candidate if candidate.exists() else fallback_model_path
    cache_key = str(model_path)
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = _load_model(model_path)
    return _MODEL_CACHE[cache_key], model_path


def _predict_sales(df: pd.DataFrame, fallback_model_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    historical_rows: list[dict[str, Any]] = []
    latest_rows: list[dict[str, Any]] = []
    used_model_paths: set[Path] = set()

    with torch.no_grad():
        for client_id, client_df in df.groupby("client_id", sort=False):
            client_df = client_df.copy()
            model, model_path = _get_model_for_client(str(client_id), fallback_model_path)
            used_model_paths.add(model_path)

            features = client_df[SELECTED_FEATURES].to_numpy(dtype=np.float32)
            sales = client_df["sales"].to_numpy(dtype=np.float32)
            if len(features) <= SEQ_LEN:
                continue

            x_scaler = StandardScaler()
            y_scaler = RobustScaler()
            scaled_features = x_scaler.fit_transform(features).astype(np.float32)
            y_scaler.fit(sales.reshape(-1, 1))
            client_df["_feature_row"] = list(scaled_features)

            for item_id, group in client_df.groupby("item_id", sort=False):
                group = group.sort_values("sale_date").reset_index(drop=True)
                if len(group) <= SEQ_LEN:
                    continue

                group_features = np.stack(group["_feature_row"].to_numpy())
                start_idx = max(SEQ_LEN, len(group) - HISTORY_WINDOW_PER_ITEM)
                for target_idx in range(start_idx, len(group)):
                    sequence = group_features[target_idx - SEQ_LEN:target_idx]
                    tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
                    scaled_prediction = model(tensor).cpu().numpy()
                    prediction = float(y_scaler.inverse_transform(scaled_prediction.reshape(-1, 1))[0, 0])
                    historical_rows.append(
                        {
                            "client_id": str(client_id),
                            "item_id": item_id,
                            "sale_date": group.loc[target_idx, "sale_date"],
                            "actual": float(group.loc[target_idx, "sales"]),
                            "predicted": max(0.0, prediction),
                        }
                    )

                latest_sequence = group_features[-SEQ_LEN:]
                tensor = torch.tensor(latest_sequence, dtype=torch.float32).unsqueeze(0)
                scaled_prediction = model(tensor).cpu().numpy()
                prediction = float(y_scaler.inverse_transform(scaled_prediction.reshape(-1, 1))[0, 0])
                latest = group.iloc[-1].to_dict()
                latest["forecast_qty"] = max(0.0, prediction)
                latest["model_path"] = str(model_path)
                latest_rows.append(latest)

    if not latest_rows:
        raise HTTPException(status_code=400, detail="AI 모델이 예측할 수 있는 상품 시퀀스가 없습니다.")

    return pd.DataFrame(historical_rows), pd.DataFrame(latest_rows), sorted(used_model_paths)


def _trend_from_predictions(source: pd.DataFrame, historical_predictions: pd.DataFrame) -> list[dict[str, Any]]:
    actual_by_date = source.groupby(source["sale_date"].dt.date).agg(sales=("sales", "sum"), revenue=("sales", "sum")).reset_index()
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


def _build_dashboard(
    file_name: str,
    source: pd.DataFrame,
    latest_predictions: pd.DataFrame,
    historical_predictions: pd.DataFrame,
    validation: list[dict[str, Any]],
    issues: list[dict[str, str]],
    used_model_paths: list[Path],
    stock_available: bool,
) -> dict[str, Any]:
    forecast_items = []
    inventory_items = []
    order_recommendations = []
    top_products = []

    for _, row in latest_predictions.iterrows():
        forecast_qty = max(0.0, float(row["forecast_qty"]))
        rolling_mean_7 = float(row["rolling_mean_7"]) if not pd.isna(row["rolling_mean_7"]) else forecast_qty
        rolling_mean_28 = float(row["rolling_mean_28"]) if not pd.isna(row["rolling_mean_28"]) else rolling_mean_7
        rolling_std_28 = max(0.0, float(row["rolling_std_28"]) if not pd.isna(row["rolling_std_28"]) else 0)
        sell_price = max(0.0, float(row["sell_price"]))

        trend_gap = ((rolling_mean_7 - rolling_mean_28) / rolling_mean_28 * 100) if rolling_mean_28 > 0 else 0
        trend = "up" if trend_gap > 8 else "down" if trend_gap < -8 else "stable"
        client_id = str(row.get("client_id", "default"))
        raw_item_id = str(row["item_id"])
        category = str(row["category"])
        item = {
            "itemId": f"{client_id}:{raw_item_id}" if client_id != "default" else raw_item_id,
            "itemName": str(row["item_name"]),
            "category": f"{client_id} · {category}" if client_id != "default" else category,
        }
        forecast_items.append({
            **item,
            "forecastQty": round(forecast_qty, 1),
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

        if stock_available:
            current_stock = max(0.0, float(row["current_stock"]))
            lead_time_days = max(1.0, float(row["lead_time_days"]) if not pd.isna(row["lead_time_days"]) else DEFAULT_LEAD_TIME_DAYS)
            ordered_qty = max(0.0, float(row["ordered_qty"]) if not pd.isna(row["ordered_qty"]) else 0)
            days_until_stockout = current_stock / forecast_qty if forecast_qty > 0 else None
            if days_until_stockout is None:
                status = "normal"
            elif days_until_stockout <= 3:
                status = "critical"
            elif days_until_stockout <= 7:
                status = "warning"
            elif days_until_stockout > 28:
                status = "overstock"
            else:
                status = "normal"

            safety_stock = math.ceil(max(rolling_std_28 * math.sqrt(lead_time_days) * 1.65, forecast_qty * lead_time_days * 0.2))
            reorder_point = math.ceil(forecast_qty * lead_time_days + safety_stock)
            recommended_order_qty = max(0, math.ceil(reorder_point + safety_stock - current_stock - ordered_qty))
            priority = "high" if (days_until_stockout is not None and days_until_stockout <= lead_time_days) else "medium" if current_stock < reorder_point else "low"

            inventory_items.append({
                **item,
                "currentStock": round(current_stock),
                "expectedDailySales": round(forecast_qty, 1),
                "daysUntilStockout": round(days_until_stockout, 1) if days_until_stockout is not None else None,
                "status": status,
                "trend": trend,
            })
            order_recommendations.append({
                **item,
                "currentStock": round(current_stock),
                "reorderPoint": reorder_point,
                "safetyStock": safety_stock,
                "leadTimeDays": round(lead_time_days, 1),
                "orderedQty": round(ordered_qty),
                "recommendedOrderQty": recommended_order_qty,
                "priority": priority,
                "reason": "AI 예측 판매량을 기준으로 계산된 발주 추천입니다.",
            })

    sales_trend = _trend_from_predictions(source, historical_predictions)
    if not sales_trend:
        raise HTTPException(status_code=400, detail="차트에 표시할 AI 예측 시계열이 부족합니다.")

    order_items = [item for item in order_recommendations if item["recommendedOrderQty"] > 0]
    critical_count = sum(1 for item in inventory_items if item["status"] == "critical")
    normal_count = sum(1 for item in inventory_items if item["status"] == "normal")
    forecast_total = sum(item["forecastQty"] for item in forecast_items)
    order_total = sum(item["recommendedOrderQty"] for item in order_items)
    revenue_estimate = sum(item["revenue"] for item in top_products)
    average_lead_time = sum(item["leadTimeDays"] for item in order_recommendations) / max(1, len(order_recommendations))

    date_min = source["sale_date"].min()
    date_max = source["sale_date"].max()
    date_range = f"{date_min.strftime('%Y. %m. %d.')} - {date_max.strftime('%Y. %m. %d.')}"

    if stock_available:
        overview_metrics = [
            {"label": "AI 예상 판매량", "value": f"{_format_number(forecast_total)}개", "helper": "PA-CFL LSTM 추론", "trend": "up", "iconKey": "TrendingUp", "tone": "primary"},
            {"label": "품절 위험 상품", "value": f"{critical_count}개", "helper": "AI 수요 대비 재고", "iconKey": "AlertCircle", "tone": "danger"},
            {"label": "추천 발주 총량", "value": f"{_format_number(order_total)}개", "helper": "AI 예측 기반", "iconKey": "ShoppingCart", "tone": "warning"},
            {"label": "예상 매출", "value": _format_currency(revenue_estimate), "helper": "예측 판매량 × 판매가", "iconKey": "DollarSign", "tone": "success"},
        ]
        inventory_metrics = [
            {"label": "품절 위험 상품", "value": f"{critical_count}개", "helper": "3일 이내 소진 예상", "iconKey": "AlertCircle", "tone": "danger"},
            {"label": "주의 상품", "value": f"{sum(1 for item in inventory_items if item['status'] == 'warning')}개", "helper": "7일 이내 점검", "iconKey": "Package", "tone": "warning"},
            {"label": "정상 상품", "value": f"{normal_count}개", "helper": "운영 범위 내", "iconKey": "Package", "tone": "success"},
            {"label": "여유 재고", "value": f"{sum(1 for item in inventory_items if item['status'] == 'overstock')}개", "helper": "28일 이상 보유", "iconKey": "Package", "tone": "info"},
        ]
        order_metrics = [
            {"label": "발주 필요 상품", "value": f"{len(order_items)}개", "helper": "AI 예측 기반", "iconKey": "ShoppingCart", "tone": "primary"},
            {"label": "추천 발주 수량", "value": f"{_format_number(order_total)}개", "helper": "오늘 제안 수량", "iconKey": "ShoppingCart", "tone": "warning"},
            {"label": "먼저 발주할 상품", "value": f"{sum(1 for item in order_items if item['priority'] == 'high')}개", "helper": "입고 전 소진 가능", "iconKey": "Truck", "tone": "danger"},
            {"label": "평균 입고 기간", "value": f"{_format_number(average_lead_time)}일", "helper": "등록 상품 기준", "iconKey": "Truck", "tone": "info"},
        ]
    else:
        overview_metrics = [
            {"label": "AI 예상 판매량", "value": f"{_format_number(forecast_total)}개", "helper": "PA-CFL LSTM 추론", "trend": "up", "iconKey": "TrendingUp", "tone": "primary"},
            {"label": "분석 상품 수", "value": f"{len(forecast_items)}개", "helper": "AI 예측 완료", "iconKey": "Package", "tone": "info"},
            {"label": "예측 시계열", "value": f"{len(sales_trend)}일", "helper": "차트 표시 구간", "iconKey": "TrendingUp", "tone": "primary"},
            {"label": "예상 매출", "value": _format_currency(revenue_estimate), "helper": "예측 판매량 × 판매가", "iconKey": "DollarSign", "tone": "success"},
        ]
        inventory_metrics = []
        order_metrics = []

    data = {
        "source": "ai",
        "overviewMetrics": overview_metrics,
        "salesTrend": sales_trend,
        "topProducts": sorted(top_products, key=lambda item: item["sales"], reverse=True)[:12],
        "forecastSeries": [
            {"date": point["date"], "actual": point["sales"], "predicted": point["forecast"]}
            for point in sales_trend
        ],
        "inventoryMetrics": inventory_metrics,
        "inventoryItems": inventory_items,
        "orderMetrics": order_metrics,
        "orderRecommendations": sorted(order_recommendations, key=lambda item: (item["recommendedOrderQty"], item["currentStock"]), reverse=True),
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
            "paths": [str(path.relative_to(ROOT_DIR)) for path in used_model_paths[:10]],
            "modelCount": len(used_model_paths),
            "selectedFeatures": SELECTED_FEATURES,
            "sequenceLength": SEQ_LEN,
            "stockAvailable": stock_available,
        },
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "modelDirExists": CLIENT_MODEL_DIR.exists(),
        "selectedFeatures": SELECTED_FEATURES,
    }


@app.post("/analyze-csv")
async def analyze_csv(file: UploadFile = File(...)) -> dict[str, Any]:
    content = await file.read()
    raw_df = _read_csv(file, content)
    prepared, validation, issues, stock_available = _prepare_frame(raw_df)
    fallback_model_path = _choose_model_path(raw_df)
    historical_predictions, latest_predictions, used_model_paths = _predict_sales(prepared, fallback_model_path)
    return _build_dashboard(
        file_name=file.filename or "uploaded.csv",
        source=prepared,
        latest_predictions=latest_predictions,
        historical_predictions=historical_predictions,
        validation=validation,
        issues=issues,
        used_model_paths=used_model_paths,
        stock_available=stock_available,
    )
