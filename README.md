# VayuCast: Automated MLOps Pipeline for Air Quality Forecasting

VayuCast is an end-to-end MLOps system that forecasts the next **1 to 3 days of air quality (PM2.5 and Indian CPCB NAQI category)** for **Delhi** (`28.6139° N, 77.2090° E`).

Rather than stopping at a static notebook, VayuCast automates daily data ingestion, drift monitoring, MLflow experiment tracking, dual quality-gate model promotion, containerized FastAPI serving with cache fallback, and AWS deployment through GitHub Actions.

## Architecture

* **Data Ingestion (`src/vayucast/ingest.py`)**: Pulls daily Delhi air quality and weather history from Open-Meteo (Aug 2022–present), stitches recent forecast reanalysis to avoid archive lag, forward-fills short gaps (≤ 2 days), drops longer gaps, and refreshes a fallback inference cache (`data/processed/latest_features.json`).
  > *Data source note:* Open-Meteo air quality values are derived from CAMS atmospheric reanalysis/forecast models rather than raw ground-station sensors. This guarantees consistent schemas and zero-key reliability across training and live inference.
* **Feature Engineering (`src/vayucast/features.py`)**: Shared train/serve pipeline using backward-looking PM2.5 lags (`1, 2, 3, 7` days), 3- and 7-day rolling statistics, target-day weather forecasts (temperature, humidity, wind, rain), cyclical seasonality, and official **Indian CPCB NAQI** PM2.5 breakpoints (`Good`, `Satisfactory`, `Moderate`, `Poor`, `Very Poor`, `Severe`).
* **Training & Experiment Tracking (`src/vayucast/train.py`)**: Trains three direct **LightGBM** regressors (`h=1, 2, 3`), logs parameters/metrics/models to **MLflow**, and tracks reproducible stages with **DVC** (`dvc.yaml`).
* **Dual Quality Gate (`src/vayucast/gate.py`)**: Promotes a challenger model only if it (1) beats the *"tomorrow equals today"* persistence baseline on `h=1` MAE and (2) does not regress by more than `5%` relative to the current Production champion.
* **Drift Monitoring (`src/vayucast/drift.py`)**: Runs **Evidently** (`DataDriftPreset`) comparing the latest 30-day window against the 180-day training reference window, generating JSON + HTML reports.
* **Serving (`src/vayucast/api.py`)**: Slim containerized **FastAPI** service with automatic fallback to the cached feature snapshot if Open-Meteo is unreachable, plus `POST /reload` for zero-downtime S3 model refreshes.

## Evaluation Results (Delhi, 90-Day Holdout)

| Horizon | LightGBM MAE (µg/m³) | Persistence Baseline MAE | MAE Reduction | CPCB Category Acc | Baseline Category Acc |
|---|---:|---:|---:|---:|---:|
| **Day +1 (`h1`)** | 14.95 | 16.11 | **-7.2%** | 61.1% | 54.4% |
| **Day +2 (`h2`)** | 17.88 | 23.02 | **-22.3%** | 51.1% | 37.8% |
| **Day +3 (`h3`)** | 20.15 | 28.00 | **-28.0%** | 44.4% | 25.6% |

As expected in atmospheric forecasting, absolute error increases with forecast horizon while the gradient-boosted model consistently outperforms the persistence baseline across all three days.

## API Endpoints

* `GET /` — Live forecast dashboard (with a *Test Cache Fallback* button for demo resilience)
* `GET /docs` — Interactive OpenAPI / Swagger UI
* `GET /health` — Service health, loaded model version, and fallback cache status
* `GET /predict?days=3&use_cache=false` — 1 to 3 day PM2.5 + CPCB AQI forecast
* `GET /model-info` — Active model metrics, gate decision, and latest Evidently drift summary
* `POST /reload` — Pulls latest promoted model & cache from S3 (`X-Reload-Token` header)

## Quickstart (Local)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[train,dev]"

# 1. Ingest data, train with quality gate, and check drift
python -m vayucast.ingest
python -m vayucast.train --fail-on-gate
python -m vayucast.drift

# 2. Run tests & linter
pytest -v
ruff check src/ tests/

# 3. Start API locally (or view MLflow UI: mlflow ui --backend-store-uri sqlite:///mlflow.db)
uvicorn vayucast.api:app --reload