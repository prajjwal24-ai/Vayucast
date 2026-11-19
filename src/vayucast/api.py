from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Annotated, Any

import joblib
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from vayucast.config import AppConfig, load_config
from vayucast.features import build_inference_features, pm25_to_cpcb_aqi
from vayucast.ingest import fetch_live_inference_window
from vayucast.storage import download_file, download_json, is_s3_enabled

logger = logging.getLogger(__name__)
LIVE_CACHE_TTL_SECONDS = 900


class DayForecast(BaseModel):
    horizon_days: int
    target_date: str
    predicted_pm25: float
    cpcb_aqi: int
    aqi_category: str


class PredictResponse(BaseModel):
    city: str
    model_version: str
    data_source: str
    anchor_date: str
    anchor_pm25: float
    forecasts: list[DayForecast]


class HealthResponse(BaseModel):
    status: str
    city: str
    model_loaded: bool
    model_version: str | None
    fallback_cache_ready: bool
    s3_enabled: bool


@dataclass
class ServiceState:
    cfg: AppConfig
    bundle: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    drift_summary: dict[str, Any] | None = None
    fallback_window: dict[str, Any] | None = None
    live_window_cache: dict[str, Any] | None = None
    live_window_ts: float = 0.0

    def load_artifacts(self, sync_s3: bool = True) -> None:
        if sync_s3 and is_s3_enabled():
            download_file("production/model_bundle.joblib", self.cfg.paths.model_bundle)
            download_file("production/metadata.json", self.cfg.paths.metadata)
            download_file("cache/latest_features.json", self.cfg.paths.latest_cache)
            download_file("reports/drift_summary.json", self.cfg.paths.drift_json)

        if self.cfg.paths.model_bundle.exists():
            self.bundle = joblib.load(self.cfg.paths.model_bundle)
            logger.info("Loaded model bundle %s", self.bundle.get("version"))
        else:
            self.bundle = None
            logger.warning("Model bundle missing at %s", self.cfg.paths.model_bundle)

        if self.cfg.paths.metadata.exists():
            self.metadata = json.loads(self.cfg.paths.metadata.read_text(encoding="utf-8"))
        else:
            self.metadata = {}

        if self.cfg.paths.drift_json.exists():
            self.drift_summary = json.loads(self.cfg.paths.drift_json.read_text(encoding="utf-8"))
        else:
            self.drift_summary = None

        if self.cfg.paths.latest_cache.exists():
            self.fallback_window = json.loads(
                self.cfg.paths.latest_cache.read_text(encoding="utf-8")
            )
        else:
            self.fallback_window = None

    def get_inference_window(self, force_cache: bool = False) -> tuple[dict[str, Any], str]:
        if not force_cache:
            now = time.monotonic()
            if (
                self.live_window_cache is not None
                and (now - self.live_window_ts) < LIVE_CACHE_TTL_SECONDS
            ):
                return self.live_window_cache, "live_open_meteo"

            try:
                window = fetch_live_inference_window(self.cfg)
                self.live_window_cache = window
                self.live_window_ts = now
                self.fallback_window = window
                return window, "live_open_meteo"
            except Exception as exc:
                logger.warning("Live Open-Meteo fetch failed, using fallback cache: %s", exc)

        if self.fallback_window is not None:
            return self.fallback_window, "fallback_cache"

        if is_s3_enabled():
            remote_cache = download_json("cache/latest_features.json")
            if remote_cache is not None:
                self.fallback_window = remote_cache
                return remote_cache, "fallback_cache"

        raise HTTPException(
            status_code=503,
            detail="Live Open-Meteo API failed and no fallback feature cache is available.",
        )


def create_app(cfg: AppConfig | None = None) -> FastAPI:
    app_cfg = cfg or load_config()
    state = ServiceState(cfg=app_cfg)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state.load_artifacts(sync_s3=True)
        yield

    app = FastAPI(
        title="VayuCast API",
        description="Automated MLOps 1-3 day Delhi PM2.5 & CPCB AQI forecasting service.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.vayucast = state

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def dashboard() -> str:
        return DASHBOARD_HTML

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        loaded = state.bundle is not None
        return HealthResponse(
            status="ok" if loaded else "degraded",
            city=state.cfg.city.name,
            model_loaded=loaded,
            model_version=state.bundle.get("version") if state.bundle else None,
            fallback_cache_ready=state.fallback_window is not None,
            s3_enabled=is_s3_enabled(),
        )

    @app.get("/predict", response_model=PredictResponse)
    def predict(
        days: Annotated[int, Query(ge=1, le=3, description="Forecast horizon (1 to 3 days)")] = 3,
        use_cache: Annotated[
            bool, Query(description="Force fallback cache for offline/demo resilience testing")
        ] = False,
    ) -> PredictResponse:
        if state.bundle is None:
            raise HTTPException(status_code=503, detail="Model bundle is not loaded.")

        window, data_source = state.get_inference_window(force_cache=use_cache)
        history = window.get("history", [])
        if not history:
            raise HTTPException(status_code=503, detail="Feature window contains no history rows.")

        anchor_pm25 = round(float(history[-1]["pm2_5"]), 2)
        models: dict[int, Any] = state.bundle["models"]
        forecasts: list[DayForecast] = []

        for h in range(1, days + 1):
            if h not in models:
                raise HTTPException(status_code=500, detail=f"No trained model for horizon {h}.")
            row_df, target_date = build_inference_features(window, horizon=h, cfg=state.cfg)
            pred_pm25 = max(0.0, float(models[h].predict(row_df)[0]))
            aqi_val, aqi_cat = pm25_to_cpcb_aqi(pred_pm25)
            forecasts.append(
                DayForecast(
                    horizon_days=h,
                    target_date=target_date,
                    predicted_pm25=round(pred_pm25, 2),
                    cpcb_aqi=aqi_val,
                    aqi_category=aqi_cat,
                )
            )

        return PredictResponse(
            city=state.cfg.city.name,
            model_version=str(state.bundle.get("version", "unknown")),
            data_source=data_source,
            anchor_date=str(window["anchor_date"]),
            anchor_pm25=anchor_pm25,
            forecasts=forecasts,
        )

    @app.get("/model-info")
    def model_info() -> dict[str, Any]:
        return {
            "model_loaded": state.bundle is not None,
            "metadata": state.metadata,
            "drift": state.drift_summary,
        }

    @app.post("/reload")
    def reload_artifacts(
        x_reload_token: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        expected = os.getenv("VAYUCAST_RELOAD_TOKEN")
        if expected and x_reload_token != expected:
            raise HTTPException(status_code=401, detail="Invalid reload token.")
        state.live_window_cache = None
        state.load_artifacts(sync_s3=True)
        return {
            "reloaded": True,
            "model_version": state.bundle.get("version") if state.bundle else None,
        }

    return app


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>VayuCast — Delhi Air Quality Forecast</title>
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; max-width: 860px;
           margin: 2rem auto; padding: 0 1rem; background: #0f172a; color: #e2e8f0; }
    h1 { margin: 0 0 .25rem; font-size: 1.75rem; }
    .sub { color: #94a3b8; margin-bottom: 1.5rem; }
    .bar { display: flex; gap: .75rem; align-items: center; flex-wrap: wrap;
           margin-bottom: 1.25rem; }
    .pill { background: #1e293b; border: 1px solid #334155; padding: .3rem .7rem;
            border-radius: 999px; font-size: .85rem; }
    button { background: #38bdf8; color: #0f172a; border: 0; padding: .4rem .85rem;
             border-radius: .45rem; font-weight: 600; cursor: pointer; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem; margin-bottom: 1.5rem; }
    .card { background: #1e293b; border: 1px solid #334155; border-radius: .75rem;
            padding: 1.1rem; }
    .cat { display: inline-block; padding: .2rem .6rem; border-radius: .35rem;
           font-weight: 700; font-size: .85rem; margin-top: .5rem; }
    .val { font-size: 2rem; font-weight: 700; margin: .35rem 0; }
    table { width: 100%; border-collapse: collapse; background: #1e293b;
            border-radius: .6rem; overflow: hidden; }
    th, td { padding: .6rem .8rem; border-bottom: 1px solid #334155; text-align: left; }
    th { color: #94a3b8; font-weight: 600; }
    a { color: #38bdf8; }
  </style>
</head>
<body>
  <h1>VayuCast — Delhi PM2.5 Forecast</h1>
  <div class="sub">Automated MLOps pipeline (LightGBM + Open-Meteo + CPCB NAQI) |
    <a href="/docs">OpenAPI Docs</a></div>
  <div class="bar">
    <span id="meta" class="pill">Loading...</span>
    <span id="drift" class="pill">Drift: checking...</span>
    <button onclick="loadForecast(false)">Live Refresh</button>
    <button onclick="loadForecast(true)">Test Cache Fallback</button>
  </div>
  <div id="cards" class="grid"></div>
  <h3>Model Performance vs. Persistence Baseline</h3>
  <table>
    <thead><tr><th>Horizon</th><th>Model MAE</th><th>Baseline MAE</th>
      <th>CPCB Category Acc</th></tr></thead>
    <tbody id="metrics"></tbody>
  </table>
  <script>
    const COLORS = {
      "Good": "#22c55e", "Satisfactory": "#84cc16", "Moderate": "#eab308",
      "Poor": "#f97316", "Very Poor": "#ef4444", "Severe": "#991b1b"
    };
    async function loadForecast(useCache) {
      document.getElementById("meta").textContent = "Fetching forecast...";
      try {
        const r = await fetch(`/predict?days=3&use_cache=${useCache}`);
        const d = await r.json();
        if (!r.ok) {
          throw new Error(d.detail || `HTTP ${r.status}`);
        }
        document.getElementById("meta").textContent =
          `Model ${d.model_version} | Source: ${d.data_source} | Anchor ${d.anchor_date}`;
        document.getElementById("cards").innerHTML = d.forecasts.map(f => `
          <div class="card">
            <div style="color:#94a3b8">Day +${f.horizon_days} (${f.target_date})</div>
            <div class="val">${f.predicted_pm25} <span style="font-size:.9rem">µg/m³</span></div>
            <div>CPCB AQI: <strong>${f.cpcb_aqi}</strong></div>
            <span class="cat" style="background:${COLORS[f.aqi_category] || '#64748b'};
              color:#0f172a">${f.aqi_category}</span>
          </div>`).join("");
      } catch (err) {
        document.getElementById("meta").textContent = `Error: ${err.message}`;
      }
    }
    async function loadInfo() {
      const r = await fetch("/model-info");
      const d = await r.json();
      const m = d.metadata?.metrics || {};
      document.getElementById("metrics").innerHTML = Object.entries(m).map(([h, v]) => `
        <tr><td>${h.toUpperCase()}</td><td>${v.model_mae}</td>
        <td>${v.baseline_mae}</td><td>${(v.category_accuracy * 100).toFixed(1)}%</td></tr>
      `).join("");
      const dr = d.drift;
      document.getElementById("drift").textContent = dr
        ? `Drift: ${dr.drifted_columns_count}/${dr.monitored_columns} cols ` +
          `(${dr.dataset_drift_detected ? "ALERT" : "OK"})`
        : "Drift: no report yet";
    }
    loadForecast(false);
    loadInfo();
  </script>
</body>
</html>"""

app = create_app()