import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.environ["MODEL_LOCAL_DIR"] = os.path.join(PROJECT_ROOT, "outputs", "_test_models")

from app import main


def _sample_frame(include_target_1d=False, include_target_7d=False):
    rows = []
    for day in range(1, 46):
        row = {
            "item_id": "FOODS_3_080",
            "sale_date": f"2014-10-{day:02d}" if day <= 31 else f"2014-11-{day - 31:02d}",
            "client_id": "CA1_Foods_3",
            "sales": float(day),
            "sell_price": 1.5,
        }
        if include_target_1d:
            row["target_1d"] = 1000.0 + day
        if include_target_7d:
            row["target_7d"] = 7000.0 + day
        rows.append(row)
    return pd.DataFrame(rows)


def test_prepare_frame_prefers_uploaded_target_1d_over_target_7d():
    prepared, _, issues, _ = main._prepare_frame(
        _sample_frame(include_target_1d=True, include_target_7d=True)
    )

    ordered = prepared.sort_values(["client_id", "item_id", "sale_date"]).reset_index(drop=True)
    assert (ordered["target_1d"] == ordered["sales"] + 1000.0).all()
    assert "target_7d" not in prepared.columns
    assert not any("다음 1일" in issue["message"] for issue in issues)


def test_prepare_frame_computes_target_1d_from_next_sales_when_missing():
    prepared, _, issues, _ = main._prepare_frame(_sample_frame())

    ordered = prepared.sort_values(["client_id", "item_id", "sale_date"]).reset_index(drop=True)
    assert ordered.loc[0, "target_1d"] == ordered.loc[1, "sales"]
    assert any("다음 1일" in issue["message"] for issue in issues)


def test_dashboard_and_health_report_next_day_forecast_metadata():
    source, validation, issues, stock_available = main._prepare_frame(_sample_frame(include_target_1d=True))
    latest_predictions = source.groupby(["client_id", "item_id"], sort=False).tail(1).copy()
    latest_predictions["forecast_qty"] = 12.5
    latest_predictions["model_path"] = "outputs/model.pt"
    latest_predictions["assigned_cluster"] = 0
    latest_predictions["representative_client_id"] = latest_predictions["client_id"]

    historical_predictions = pd.DataFrame(
        {
            "client_id": latest_predictions["client_id"].iloc[:1],
            "item_id": latest_predictions["item_id"].iloc[:1],
            "sale_date": latest_predictions["sale_date"].iloc[:1],
            "actual": [12.0],
            "predicted": [12.5],
        }
    )

    dashboard = main._build_dashboard(
        file_name="sample.csv",
        source=source,
        latest_predictions=latest_predictions,
        historical_predictions=historical_predictions,
        validation=validation,
        issues=issues,
        used_model_artifacts=[],
        stock_available=stock_available,
        cluster_assignments=[],
    )

    assert dashboard["data"]["forecastWindow"]["horizonDays"] == 1
    assert dashboard["model"]["forecastTarget"] == "target_1d"
    assert dashboard["model"]["forecastUnit"] == "next_day_sales"
    assert dashboard["data"]["forecastItems"][0]["forecastHorizonDays"] == 1
    assert "1일 예상 판매량" in dashboard["data"]["overviewMetrics"][0]["label"]

    health = main.health_check()
    assert health["forecastTarget"] == "target_1d"
    assert health["forecastHorizonDays"] == 1


def main_test():
    test_prepare_frame_prefers_uploaded_target_1d_over_target_7d()
    test_prepare_frame_computes_target_1d_from_next_sales_when_missing()
    test_dashboard_and_health_report_next_day_forecast_metadata()


if __name__ == "__main__":
    main_test()
