from __future__ import annotations

import math
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from vayucast.config import AppConfig

# Official Indian CPCB NAQI 24-hr PM2.5 breakpoints: (c_low, c_high, aqi_low, aqi_high, category)
CPCB_PM25_BANDS: list[tuple[float, float, int, int, str]] = [
    (0.0, 30.0, 0, 50, "Good"),
    (30.0, 60.0, 51, 100, "Satisfactory"),
    (60.0, 90.0, 101, 200, "Moderate"),
    (90.0, 120.0, 201, 300, "Poor"),
    (120.0, 250.0, 301, 400, "Very Poor"),
    (250.0, 380.0, 401, 500, "Severe"),
]

FEATURE_COLUMNS: list[str] = [
    "pm2_5",
    "pm10",
    "nitrogen_dioxide",
    "ozone",
    "carbon_monoxide",
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max",
    "precipitation_sum",
    "pm2_5_lag_1",
    "pm2_5_lag_2",
    "pm2_5_lag_3",
    "pm2_5_lag_7",
    "pm2_5_roll_mean_3",
    "pm2_5_roll_mean_7",
    "pm2_5_roll_std_7",
    "target_temp",
    "target_humidity",
    "target_wind",
    "target_precip",
    "target_month",
    "target_dow",
    "target_doy_sin",
    "target_doy_cos",
]


def pm25_to_cpcb_aqi(pm25: float) -> tuple[int, str]:
    """Map 24-hr PM2.5 (µg/m³) to Indian CPCB sub-index (0-500) and category."""
    val = max(0.0, float(pm25))
    for c_low, c_high, i_low, i_high, category in CPCB_PM25_BANDS:
        if val <= c_high:
            span = c_high - c_low
            aqi = i_low + ((val - c_low) / span) * (i_high - i_low)
            return int(round(aqi)), category
    return 500, "Severe"


def build_base_features(df: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    """Compute strictly backward-looking lags and rolling stats on daily readings."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.sort_values("date").reset_index(drop=True)

    for lag in cfg.features.lags:
        out[f"pm2_5_lag_{lag}"] = out["pm2_5"].shift(lag)

    for window in cfg.features.rolling_windows:
        out[f"pm2_5_roll_mean_{window}"] = out["pm2_5"].rolling(window, min_periods=window).mean()

    out["pm2_5_roll_std_7"] = out["pm2_5"].rolling(7, min_periods=7).std()
    return out


def _add_calendar_features(dates: pd.Series) -> pd.DataFrame:
    doy = dates.dt.dayofyear
    angle = 2.0 * math.pi * doy / 365.25
    return pd.DataFrame(
        {
            "target_month": dates.dt.month.astype(float),
            "target_dow": dates.dt.dayofweek.astype(float),
            "target_doy_sin": np.sin(angle),
            "target_doy_cos": np.cos(angle),
        },
        index=dates.index,
    )


def build_horizon_dataset(
    df: pd.DataFrame, horizon: int, cfg: AppConfig
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Return (X, y_true, y_persistence_baseline, target_dates) for a forecast horizon."""
    base = build_base_features(df, cfg)

    # Target-day weather is available at inference time via Open-Meteo's 3-day weather forecast
    base["target_date"] = base["date"].shift(-horizon)
    base["target_temp"] = base["temperature_2m_mean"].shift(-horizon)
    base["target_humidity"] = base["relative_humidity_2m_mean"].shift(-horizon)
    base["target_wind"] = base["wind_speed_10m_max"].shift(-horizon)
    base["target_precip"] = base["precipitation_sum"].shift(-horizon)
    base["target_pm2_5"] = base["pm2_5"].shift(-horizon)

    # Ensure no multi-day date gap broke the shift alignment
    expected_date = base["date"] + pd.to_timedelta(horizon, unit="D")
    aligned = base[base["target_date"] == expected_date].copy()

    cal = _add_calendar_features(aligned["target_date"])
    aligned = pd.concat([aligned, cal], axis=1)

    required_cols = [*FEATURE_COLUMNS, "target_pm2_5"]
    aligned = aligned.dropna(subset=required_cols).reset_index(drop=True)

    x_df = aligned[FEATURE_COLUMNS].astype(float)
    y_true = aligned["target_pm2_5"].astype(float)
    # Persistence baseline: tomorrow (or day +h) equals today's PM2.5
    y_baseline = aligned["pm2_5"].astype(float)
    target_dates = aligned["target_date"]
    return x_df, y_true, y_baseline, target_dates


def build_inference_features(
    window: dict[str, Any], horizon: int, cfg: AppConfig
) -> tuple[pd.DataFrame, str]:
    """Build a single-row feature DataFrame for live/fallback inference at horizon h."""
    history_df = pd.DataFrame(window["history"])
    base = build_base_features(history_df, cfg)
    latest = base.iloc[-1].copy()

    anchor_date = pd.Timestamp(latest["date"])
    target_ts = anchor_date + timedelta(days=horizon)
    target_date_str = target_ts.strftime("%Y-%m-%d")

    forecast_rows = {row["date"]: row for row in window.get("weather_forecast", [])}
    weather_row = forecast_rows.get(target_date_str)

    # Graceful fallback to anchor-day weather if a stale cache lacks that forecast date
    latest["target_temp"] = float(
        weather_row["temperature_2m_mean"]
        if weather_row and weather_row.get("temperature_2m_mean") is not None
        else latest["temperature_2m_mean"]
    )
    latest["target_humidity"] = float(
        weather_row["relative_humidity_2m_mean"]
        if weather_row and weather_row.get("relative_humidity_2m_mean") is not None
        else latest["relative_humidity_2m_mean"]
    )
    latest["target_wind"] = float(
        weather_row["wind_speed_10m_max"]
        if weather_row and weather_row.get("wind_speed_10m_max") is not None
        else latest["wind_speed_10m_max"]
    )
    latest["target_precip"] = float(
        weather_row["precipitation_sum"]
        if weather_row and weather_row.get("precipitation_sum") is not None
        else latest["precipitation_sum"]
    )

    doy = target_ts.dayofyear
    angle = 2.0 * math.pi * doy / 365.25
    latest["target_month"] = float(target_ts.month)
    latest["target_dow"] = float(target_ts.dayofweek)
    latest["target_doy_sin"] = float(math.sin(angle))
    latest["target_doy_cos"] = float(math.cos(angle))

    row_df = pd.DataFrame([latest[FEATURE_COLUMNS].to_dict()]).astype(float)
    return row_df, target_date_str