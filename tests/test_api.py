from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from vayucast.api import create_app
from vayucast.config import load_config
from vayucast.features import FEATURE_COLUMNS


class _ConstantModel:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, x_df: pd.DataFrame) -> np.ndarray:
        assert list(x_df.columns) == FEATURE_COLUMNS
        return np.full(len(x_df), self.value)


def _sample_window() -> dict:
    dates = pd.date_range("2026-11-01", periods=12, freq="D")
    history = [
        {
            "date": d.strftime("%Y-%m-%d"),
            "pm2_5": 75.0 + i,
            "pm10": 100.0 + i,
            "nitrogen_dioxide": 30.0,
            "ozone": 40.0,
            "carbon_monoxide": 550.0,
            "temperature_2m_mean": 21.0,
            "relative_humidity_2m_mean": 58.0,
            "wind_speed_10m_max": 9.5,
            "precipitation_sum": 0.0,
        }
        for i, d in enumerate(dates)
    ]
    forecast = [
        {
            "date": f"2026-11-{13 + i:02d}",
            "temperature_2m_mean": 19.5 - i,
            "relative_humidity_2m_mean": 62.0,
            "wind_speed_10m_max": 8.0,
            "precipitation_sum": 0.0,
        }
        for i in range(3)
    ]
    return {
        "city": "Delhi",
        "fetched_at": "2026-11-13T08:00:00+05:30",
        "anchor_date": "2026-11-12",
        "history": history,
        "weather_forecast": forecast,
    }


@pytest.fixture()
def api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.delenv("AWS_S3_BUCKET", raising=False)
    cfg = load_config().model_copy(deep=True)
    cfg.paths.model_bundle = tmp_path / "model_bundle.joblib"
    cfg.paths.metadata = tmp_path / "metadata.json"
    cfg.paths.latest_cache = tmp_path / "latest_features.json"
    cfg.paths.drift_json = tmp_path / "drift_summary.json"

    bundle = {
        "version": "v-test-001",
        "models": {1: _ConstantModel(72.0), 2: _ConstantModel(95.0), 3: _ConstantModel(140.0)},
        "feature_columns": FEATURE_COLUMNS,
        "horizons": [1, 2, 3],
    }
    joblib.dump(bundle, cfg.paths.model_bundle)
    cfg.paths.metadata.write_text(
        json.dumps({"model_version": "v-test-001", "metrics": {"h1": {"model_mae": 14.5}}}),
        encoding="utf-8",
    )
    cfg.paths.latest_cache.write_text(json.dumps(_sample_window()), encoding="utf-8")
    cfg.paths.drift_json.write_text(
        json.dumps({"drifted_columns_count": 2, "monitored_columns": 9}),
        encoding="utf-8",
    )

    app = create_app(cfg=cfg)
    with TestClient(app) as client:
        yield client


def test_health_endpoint(api_client: TestClient) -> None:
    resp = api_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_version"] == "v-test-001"
    assert body["fallback_cache_ready"] is True


def test_predict_live_open_meteo(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("vayucast.api.fetch_live_inference_window", lambda _cfg: _sample_window())
    resp = api_client.get("/predict?days=3")
    assert resp.status_code == 200
    body = resp.json()
    assert body["data_source"] == "live_open_meteo"
    assert len(body["forecasts"]) == 3
    assert [f["aqi_category"] for f in body["forecasts"]] == ["Moderate", "Poor", "Very Poor"]


def test_predict_falls_back_to_cache_on_live_api_error(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(_cfg):
        raise RuntimeError("Open-Meteo timeout")

    monkeypatch.setattr("vayucast.api.fetch_live_inference_window", _boom)
    resp = api_client.get("/predict?days=2")
    assert resp.status_code == 200
    body = resp.json()
    assert body["data_source"] == "fallback_cache"
    assert len(body["forecasts"]) == 2


def test_predict_rejects_invalid_horizon(api_client: TestClient) -> None:
    assert api_client.get("/predict?days=5").status_code == 422
    assert api_client.get("/predict?days=0").status_code == 422


def test_model_info_and_dashboard(api_client: TestClient) -> None:
    info = api_client.get("/model-info")
    assert info.status_code == 200
    assert info.json()["metadata"]["model_version"] == "v-test-001"
    assert info.json()["drift"]["drifted_columns_count"] == 2

    page = api_client.get("/")
    assert page.status_code == 200
    assert "VayuCast" in page.text