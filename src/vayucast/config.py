from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CityConfig(BaseModel):
    name: str
    latitude: float
    longitude: float
    timezone: str


class ApiConfig(BaseModel):
    air_quality_url: str
    weather_archive_url: str
    weather_forecast_url: str
    history_start_date: str
    timeout_seconds: int = 30


class PreprocessingConfig(BaseModel):
    max_ffill_days: int = 2


class FeaturesConfig(BaseModel):
    lags: list[int]
    rolling_windows: list[int]
    horizons: list[int]


class TrainingConfig(BaseModel):
    test_days: int = 90
    model_params: dict[str, Any] = Field(default_factory=dict)


class GateConfig(BaseModel):
    primary_horizon: int = 1
    max_regression_pct: float = 5.0


class DriftConfig(BaseModel):
    reference_days: int = 180
    current_days: int = 30
    drift_share_threshold: float = 0.5


class PathsConfig(BaseModel):
    raw_data: Path
    latest_cache: Path
    model_bundle: Path
    metadata: Path
    reference_data: Path
    drift_json: Path
    drift_html: Path


class AppConfig(BaseModel):
    city: CityConfig
    api: ApiConfig
    preprocessing: PreprocessingConfig
    features: FeaturesConfig
    training: TrainingConfig
    gate: GateConfig
    drift: DriftConfig
    paths: PathsConfig


class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    vayucast_config_path: Path = Path("configs/config.yaml")
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "vayucast-delhi-pm25"
    aws_s3_bucket: str | None = None
    aws_region: str = "ap-south-1"


@lru_cache(maxsize=1)
def get_env() -> EnvSettings:
    return EnvSettings()


@lru_cache(maxsize=1)
def load_config(config_path: Path | None = None) -> AppConfig:
    path = config_path or get_env().vayucast_config_path
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig.model_validate(raw)