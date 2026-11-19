from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from vayucast.config import AppConfig, load_config
from vayucast.storage import upload_file, upload_json

logger = logging.getLogger(__name__)

AQ_HOURLY_VARS = [
    "pm2_5",
    "pm10",
    "nitrogen_dioxide",
    "ozone",
    "carbon_monoxide",
]
WEATHER_DAILY_VARS = [
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max",
    "precipitation_sum",
]
CORE_COLUMNS = ["pm2_5", *WEATHER_DAILY_VARS]
MIN_VALID_HOURS_PER_DAY = 12


def _today_in_tz(tz_name: str) -> date:
    return datetime.now(ZoneInfo(tz_name)).date()


def _get_json(url: str, params: dict[str, Any], timeout: int, retries: int = 3) -> dict[str, Any]:
    last_err: Exception | None = None
    req_timeout = httpx.Timeout(timeout, connect=10.0)
    for attempt in range(1, retries + 1):
        try:
            with httpx.Client(timeout=req_timeout) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                return resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(attempt * 2.0)
    raise RuntimeError(f"Request failed after {retries} attempts: {url}") from last_err


def _date_chunks(start: date, end: date, chunk_days: int = 365) -> list[tuple[date, date]]:
    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def _aggregate_hourly_aq(payload: dict[str, Any]) -> pd.DataFrame:
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return pd.DataFrame(columns=["date", *AQ_HOURLY_VARS])

    df = pd.DataFrame({"timestamp": pd.to_datetime(times)})
    for col in AQ_HOURLY_VARS:
        df[col] = pd.to_numeric(hourly.get(col), errors="coerce")

    df["date"] = df["timestamp"].dt.normalize()
    grouped = df.groupby("date", as_index=False)
    daily = grouped[AQ_HOURLY_VARS].mean()

    # Drop days with fewer than 12 valid hourly PM2.5 readings to avoid partial-day bias
    valid_counts = grouped["pm2_5"].count()["pm2_5"]
    daily.loc[valid_counts < MIN_VALID_HOURS_PER_DAY, "pm2_5"] = pd.NA
    return daily


def _parse_daily_weather(payload: dict[str, Any]) -> pd.DataFrame:
    daily = payload.get("daily", {})
    times = daily.get("time", [])
    if not times:
        return pd.DataFrame(columns=["date", *WEATHER_DAILY_VARS])

    df = pd.DataFrame({"date": pd.to_datetime(times).normalize()})
    for col in WEATHER_DAILY_VARS:
        df[col] = pd.to_numeric(daily.get(col), errors="coerce")
    return df


def fetch_historical_aq(cfg: AppConfig, start: date, end: date) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for c_start, c_end in _date_chunks(start, end):
        params = {
            "latitude": cfg.city.latitude,
            "longitude": cfg.city.longitude,
            "hourly": ",".join(AQ_HOURLY_VARS),
            "timezone": cfg.city.timezone,
            "start_date": c_start.isoformat(),
            "end_date": c_end.isoformat(),
        }
        payload = _get_json(cfg.api.air_quality_url, params, cfg.api.timeout_seconds)
        frames.append(_aggregate_hourly_aq(payload))

    if not frames:
        return pd.DataFrame(columns=["date", *AQ_HOURLY_VARS])
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date")


def fetch_historical_weather(cfg: AppConfig, start: date, end: date) -> pd.DataFrame:
    today = _today_in_tz(cfg.city.timezone)
    archive_cutoff = min(end, today - timedelta(days=5))
    frames: list[pd.DataFrame] = []

    if start <= archive_cutoff:
        for c_start, c_end in _date_chunks(start, archive_cutoff):
            params = {
                "latitude": cfg.city.latitude,
                "longitude": cfg.city.longitude,
                "daily": ",".join(WEATHER_DAILY_VARS),
                "timezone": cfg.city.timezone,
                "start_date": c_start.isoformat(),
                "end_date": c_end.isoformat(),
            }
            payload = _get_json(cfg.api.weather_archive_url, params, cfg.api.timeout_seconds)
            frames.append(_parse_daily_weather(payload))

    # Stitch recent days from Forecast API to cover Open-Meteo's 2-5 day archive lag
    if end > archive_cutoff:
        recent_params = {
            "latitude": cfg.city.latitude,
            "longitude": cfg.city.longitude,
            "daily": ",".join(WEATHER_DAILY_VARS),
            "timezone": cfg.city.timezone,
            "past_days": 14,
            "forecast_days": 1,
        }
        recent_payload = _get_json(
            cfg.api.weather_forecast_url, recent_params, cfg.api.timeout_seconds
        )
        recent_df = _parse_daily_weather(recent_payload)
        recent_df = recent_df[
            (recent_df["date"] >= pd.Timestamp(start)) & (recent_df["date"] <= pd.Timestamp(end))
        ]
        frames.append(recent_df)

    if not frames:
        return pd.DataFrame(columns=["date", *WEATHER_DAILY_VARS])

    combined = pd.concat(frames, ignore_index=True)
    # Keep first non-null value per date (preferring archive over forecast where both exist)
    return (
        combined.groupby("date", as_index=False)
        .first()
        .sort_values("date")
        .reset_index(drop=True)
    )


def clean_and_fill_gaps(df: pd.DataFrame, max_ffill_days: int = 2) -> pd.DataFrame:
    if df.empty:
        return df

    cleaned = df.copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"]).dt.normalize()
    cleaned = cleaned.drop_duplicates(subset=["date"]).sort_values("date")

    full_index = pd.date_range(cleaned["date"].min(), cleaned["date"].max(), freq="D")
    cleaned = cleaned.set_index("date").reindex(full_index)
    cleaned.index.name = "date"

    # Forward-fill short gaps up to max_ffill_days; drop rows with longer missing stretches
    value_cols = [c for c in cleaned.columns if c in (*AQ_HOURLY_VARS, *WEATHER_DAILY_VARS)]
    cleaned[value_cols] = cleaned[value_cols].ffill(limit=max_ffill_days)
    cleaned = cleaned.dropna(subset=CORE_COLUMNS).reset_index()
    return cleaned


def fetch_live_inference_window(cfg: AppConfig) -> dict[str, Any]:
    """Fetch last 14 days of AQ/weather + next 3 days of weather forecast for /predict."""
    today = _today_in_tz(cfg.city.timezone)
    yesterday = today - timedelta(days=1)
    max_horizon = max(cfg.features.horizons)

    aq_params = {
        "latitude": cfg.city.latitude,
        "longitude": cfg.city.longitude,
        "hourly": ",".join(AQ_HOURLY_VARS),
        "timezone": cfg.city.timezone,
        "past_days": 14,
        "forecast_days": 1,
    }
    weather_params = {
        "latitude": cfg.city.latitude,
        "longitude": cfg.city.longitude,
        "daily": ",".join(WEATHER_DAILY_VARS),
        "timezone": cfg.city.timezone,
        "past_days": 14,
        "forecast_days": max_horizon + 1,
    }

    aq_df = _aggregate_hourly_aq(
        _get_json(cfg.api.air_quality_url, aq_params, cfg.api.timeout_seconds)
    )
    weather_df = _parse_daily_weather(
        _get_json(cfg.api.weather_forecast_url, weather_params, cfg.api.timeout_seconds)
    )

    # Use complete days up through yesterday (or today if yesterday unavailable) as history
    history_df = pd.merge(aq_df, weather_df, on="date", how="inner")
    history_df = history_df[history_df["date"] <= pd.Timestamp(yesterday)]
    history_df = clean_and_fill_gaps(history_df, max_ffill_days=cfg.preprocessing.max_ffill_days)
    if history_df.empty:
        raise RuntimeError("No valid historical rows returned from Open-Meteo.")

    anchor_date = history_df["date"].max().date()
    forecast_df = weather_df[
        (weather_df["date"] > pd.Timestamp(anchor_date))
        & (weather_df["date"] <= pd.Timestamp(anchor_date + timedelta(days=max_horizon)))
    ].sort_values("date")

    return {
        "city": cfg.city.name,
        "fetched_at": datetime.now(ZoneInfo(cfg.city.timezone)).isoformat(),
        "anchor_date": anchor_date.isoformat(),
        "history": _df_to_records(history_df.tail(14)),
        "weather_forecast": _df_to_records(forecast_df),
    }


def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return json.loads(out.to_json(orient="records"))

def _build_cache_window(cfg: AppConfig, history_df: pd.DataFrame) -> dict[str, Any]:
    """Build fallback cache from already-ingested history + 1 weather forecast call."""
    tail_df = history_df.tail(14)
    anchor_date = pd.to_datetime(tail_df["date"]).max().date()
    max_horizon = max(cfg.features.horizons)

    forecast_records: list[dict[str, Any]] = []
    try:
        params = {
            "latitude": cfg.city.latitude,
            "longitude": cfg.city.longitude,
            "daily": ",".join(WEATHER_DAILY_VARS),
            "timezone": cfg.city.timezone,
            "forecast_days": max_horizon + 2,
        }
        forecast_df = _parse_daily_weather(
            _get_json(cfg.api.weather_forecast_url, params, cfg.api.timeout_seconds)
        )
        forecast_df = forecast_df[
            (forecast_df["date"] > pd.Timestamp(anchor_date))
            & (forecast_df["date"] <= pd.Timestamp(anchor_date + timedelta(days=max_horizon)))
        ].sort_values("date")
        forecast_records = _df_to_records(forecast_df)
    except Exception as exc:
        logger.warning("Forecast enrichment failed; saving history-only cache: %s", exc)

    return {
        "city": cfg.city.name,
        "fetched_at": datetime.now(ZoneInfo(cfg.city.timezone)).isoformat(),
        "anchor_date": anchor_date.isoformat(),
        "history": _df_to_records(tail_df),
        "weather_forecast": forecast_records,
    }


def run_ingest(cfg: AppConfig | None = None, full_refresh: bool = False) -> pd.DataFrame:
    cfg = cfg or load_config()
    today = _today_in_tz(cfg.city.timezone)
    end_date = today - timedelta(days=1)
    history_start = date.fromisoformat(cfg.api.history_start_date)

    raw_path = cfg.paths.raw_data
    existing_df = pd.DataFrame()
    start_date = history_start

    if raw_path.exists() and not full_refresh:
        existing_df = pd.read_parquet(raw_path)
        if not existing_df.empty:
            last_date = pd.to_datetime(existing_df["date"]).max().date()
            # Re-fetch the last 7 days so recently finalized weather reanalysis replaces forecasts
            start_date = max(history_start, last_date - timedelta(days=7))

    logger.info("Fetching Open-Meteo data for %s: %s -> %s", cfg.city.name, start_date, end_date)
    aq_df = fetch_historical_aq(cfg, start_date, end_date)
    weather_df = fetch_historical_weather(cfg, start_date, end_date)
    fresh_df = pd.merge(aq_df, weather_df, on="date", how="inner")

    if not existing_df.empty:
        existing_df["date"] = pd.to_datetime(existing_df["date"]).dt.normalize()
        merged = pd.concat([existing_df, fresh_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["date"], keep="last")
    else:
        merged = fresh_df

    final_df = clean_and_fill_gaps(merged, max_ffill_days=cfg.preprocessing.max_ffill_days)

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(raw_path, index=False)
    logger.info(
        "Saved %d daily rows (%s to %s) -> %s",
        len(final_df),
        final_df["date"].min().date(),
        final_df["date"].max().date(),
        raw_path,
    )

    # Refresh fallback cache snapshot for demo-day / API-outage resilience
    # Build fallback cache from already-ingested history (avoids redundant AQ API call)
    live_window = _build_cache_window(cfg, final_df)
    cfg.paths.latest_cache.parent.mkdir(parents=True, exist_ok=True)
    cfg.paths.latest_cache.write_text(json.dumps(live_window, indent=2), encoding="utf-8")
    logger.info("Updated fallback cache -> %s", cfg.paths.latest_cache)

    upload_file(raw_path, "data/delhi_daily.parquet")
    upload_json(live_window, "cache/latest_features.json")
    return final_df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Ingest Delhi AQ & weather data from Open-Meteo.")
    parser.add_argument("--full", action="store_true", help="Ignore existing parquet and pull all.")
    args = parser.parse_args()
    run_ingest(full_refresh=args.full)


if __name__ == "__main__":
    main()