from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vayucast.config import load_config
from vayucast.features import (
    FEATURE_COLUMNS,
    build_horizon_dataset,
    build_inference_features,
    pm25_to_cpcb_aqi,
)
from vayucast.ingest import clean_and_fill_gaps


def _make_sample_df(days: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=days, freq="D")
    idx = np.arange(days, dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "pm2_5": 60.0 + idx * 1.5,
            "pm10": 90.0 + idx * 2.0,
            "nitrogen_dioxide": 25.0 + (idx % 5),
            "ozone": 40.0 + (idx % 7),
            "carbon_monoxide": 500.0 + idx * 3.0,
            "temperature_2m_mean": 20.0 + (idx % 10) * 0.5,
            "relative_humidity_2m_mean": 55.0 + (idx % 6),
            "wind_speed_10m_max": 10.0 + (idx % 4),
            "precipitation_sum": np.where(idx % 9 == 0, 2.5, 0.0),
        }
    )


@pytest.mark.parametrize(
    ("pm25", "expected_cat"),
    [
        (-5.0, "Good"),
        (15.0, "Good"),
        (30.0, "Good"),
        (45.0, "Satisfactory"),
        (75.0, "Moderate"),
        (105.0, "Poor"),
        (180.0, "Very Poor"),
        (320.0, "Severe"),
        (600.0, "Severe"),
    ],
)
def test_pm25_to_cpcb_aqi_bands(pm25: float, expected_cat: str) -> None:
    aqi, category = pm25_to_cpcb_aqi(pm25)
    assert category == expected_cat
    assert 0 <= aqi <= 500


def test_clean_and_fill_gaps_ffills_short_and_drops_long() -> None:
    df = _make_sample_df(15)
    # 2-day gap (rows 3, 4) -> should be forward-filled
    df.loc[[3, 4], "pm2_5"] = np.nan
    # 3-day gap (rows 9, 10, 11) -> 3rd missing day exceeds limit=2 and must be dropped
    df.loc[[9, 10, 11], "pm2_5"] = np.nan

    cleaned = clean_and_fill_gaps(df, max_ffill_days=2)
    assert cleaned["pm2_5"].isna().sum() == 0
    assert len(cleaned) == 14


def test_build_horizon_dataset_has_no_leakage_and_respects_gaps() -> None:
    cfg = load_config()
    df = _make_sample_df(30)
    # Drop a day in the middle to verify shift alignment rejects broken horizons
    df = df.drop(index=18).reset_index(drop=True)

    x_df, y_true, y_base, target_dates = build_horizon_dataset(df, horizon=1, cfg=cfg)

    assert list(x_df.columns) == FEATURE_COLUMNS
    assert not x_df.isna().any().any()
    assert len(x_df) == len(y_true) == len(y_base) == len(target_dates)
    # Persistence baseline must equal anchor-day PM2.5, while target is next day (+1.5 in fixture)
    assert np.allclose(y_base.to_numpy(), x_df["pm2_5"].to_numpy())
    assert np.allclose((y_true - y_base).to_numpy(), 1.5)


def test_build_inference_features_uses_forecast_and_falls_back_gracefully() -> None:
    cfg = load_config()
    df = _make_sample_df(12)
    records = df.assign(date=df["date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records")

    window = {
        "anchor_date": "2026-01-12",
        "history": records,
        "weather_forecast": [
            {
                "date": "2026-01-13",
                "temperature_2m_mean": 11.5,
                "relative_humidity_2m_mean": 77.0,
                "wind_speed_10m_max": 6.2,
                "precipitation_sum": 1.0,
            }
        ],
    }

    row_h1, target_h1 = build_inference_features(window, horizon=1, cfg=cfg)
    assert target_h1 == "2026-01-13"
    assert list(row_h1.columns) == FEATURE_COLUMNS
    assert row_h1["target_temp"].iloc[0] == pytest.approx(11.5)

    # Horizon 2 is missing from weather_forecast -> falls back cleanly to anchor-day weather
    row_h2, target_h2 = build_inference_features(window, horizon=2, cfg=cfg)
    assert target_h2 == "2026-01-14"
    assert not row_h2.isna().any().any()
    assert row_h2["target_temp"].iloc[0] == pytest.approx(records[-1]["temperature_2m_mean"])