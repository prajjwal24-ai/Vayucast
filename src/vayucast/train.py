from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

import joblib
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from vayucast.config import AppConfig, get_env, load_config
from vayucast.features import (
    FEATURE_COLUMNS,
    build_base_features,
    build_horizon_dataset,
    pm25_to_cpcb_aqi,
)
from vayucast.gate import GateDecision, evaluate_quality_gate
from vayucast.storage import download_file, download_json, upload_file, upload_json

logger = logging.getLogger(__name__)


def _category_accuracy(y_true: pd.Series, y_pred: np.ndarray) -> float:
    true_cats = [pm25_to_cpcb_aqi(v)[1] for v in y_true]
    pred_cats = [pm25_to_cpcb_aqi(v)[1] for v in y_pred]
    return float(np.mean([t == p for t, p in zip(true_cats, pred_cats, strict=True)]))


def _load_existing_champion_mae(cfg: AppConfig) -> float | None:
    meta = download_json("production/metadata.json")
    if meta is None and cfg.paths.metadata.exists():
        meta = json.loads(cfg.paths.metadata.read_text(encoding="utf-8"))
    if not meta:
        return None
    primary_key = f"h{cfg.gate.primary_horizon}"
    return meta.get("metrics", {}).get(primary_key, {}).get("model_mae")


def run_training(
    cfg: AppConfig | None = None,
    force_promote: bool = False,
) -> tuple[dict[str, Any], GateDecision]:
    cfg = cfg or load_config()
    env = get_env()

    if not cfg.paths.raw_data.exists():
        downloaded = download_file("data/delhi_daily.parquet", cfg.paths.raw_data)
        if not downloaded:
            raise FileNotFoundError(
                f"Dataset not found at {cfg.paths.raw_data}. Run `python -m vayucast.ingest` first."
            )

    df = pd.read_parquet(cfg.paths.raw_data)
    mlflow.set_tracking_uri(env.mlflow_tracking_uri)
    mlflow.set_experiment(env.mlflow_experiment_name)

    models: dict[int, lgb.LGBMRegressor] = {}
    metrics_by_horizon: dict[str, dict[str, float]] = {}
    version_tag = datetime.now(UTC).strftime("v%Y%m%d-%H%M%S")

    with mlflow.start_run(run_name=f"train-{version_tag}") as run:
        mlflow.log_params(
            {
                "city": cfg.city.name,
                "rows_raw": len(df),
                "test_days": cfg.training.test_days,
                **cfg.training.model_params,
            }
        )

        for h in cfg.features.horizons:
            x_df, y_true, y_base, _ = build_horizon_dataset(df, horizon=h, cfg=cfg)
            if len(x_df) <= cfg.training.test_days + 30:
                raise ValueError(f"Not enough rows ({len(x_df)}) for horizon {h}.")

            split_idx = len(x_df) - cfg.training.test_days
            x_train, x_test = x_df.iloc[:split_idx], x_df.iloc[split_idx:]
            y_train, y_test = y_true.iloc[:split_idx], y_true.iloc[split_idx:]
            base_test = y_base.iloc[split_idx:]

            model = lgb.LGBMRegressor(**cfg.training.model_params)
            model.fit(x_train, y_train)
            preds = np.clip(model.predict(x_test), 0.0, None)

            model_mae = float(mean_absolute_error(y_test, preds))
            model_rmse = float(root_mean_squared_error(y_test, preds))
            base_mae = float(mean_absolute_error(y_test, base_test))
            base_rmse = float(root_mean_squared_error(y_test, base_test))
            cat_acc = _category_accuracy(y_test, preds)
            base_cat_acc = _category_accuracy(y_test, base_test.to_numpy())

            metrics_by_horizon[f"h{h}"] = {
                "model_mae": round(model_mae, 3),
                "baseline_mae": round(base_mae, 3),
                "model_rmse": round(model_rmse, 3),
                "baseline_rmse": round(base_rmse, 3),
                "category_accuracy": round(cat_acc, 4),
                "baseline_category_accuracy": round(base_cat_acc, 4),
            }

            mlflow.log_metrics(
                {
                    f"h{h}_model_mae": model_mae,
                    f"h{h}_baseline_mae": base_mae,
                    f"h{h}_model_rmse": model_rmse,
                    f"h{h}_category_accuracy": cat_acc,
                }
            )
            mlflow.lightgbm.log_model(model, name=f"lgbm_h{h}")

            # Refit on 100% of available data for production forecasting
            prod_model = lgb.LGBMRegressor(**cfg.training.model_params)
            prod_model.fit(x_df, y_true)
            models[h] = prod_model

        primary_key = f"h{cfg.gate.primary_horizon}"
        challenger_mae = metrics_by_horizon[primary_key]["model_mae"]
        baseline_mae = metrics_by_horizon[primary_key]["baseline_mae"]
        champion_mae = _load_existing_champion_mae(cfg)

        decision = evaluate_quality_gate(
            challenger_mae=challenger_mae,
            baseline_mae=baseline_mae,
            champion_mae=champion_mae,
            max_regression_pct=cfg.gate.max_regression_pct,
        )

        mlflow.log_metrics({"gate_passed": int(decision.passed)})
        mlflow.set_tags(
            {
                "version": version_tag,
                "gate_passed": str(decision.passed),
                "gate_reason": decision.reason,
            }
        )

        metadata: dict[str, Any] = {
            "model_version": version_tag,
            "mlflow_run_id": run.info.run_id,
            "trained_at_utc": datetime.now(UTC).isoformat(),
            "city": cfg.city.name,
            "data_rows": len(df),
            "data_start": str(pd.to_datetime(df["date"]).min().date()),
            "data_end": str(pd.to_datetime(df["date"]).max().date()),
            "feature_columns": FEATURE_COLUMNS,
            "horizons": cfg.features.horizons,
            "metrics": metrics_by_horizon,
            "gate": decision.model_dump(),
        }

    if decision.passed or force_promote:
        cfg.paths.model_bundle.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "version": version_tag,
            "models": models,
            "feature_columns": FEATURE_COLUMNS,
            "horizons": cfg.features.horizons,
        }
        joblib.dump(bundle, cfg.paths.model_bundle)
        cfg.paths.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        # Save reference window for Evidently drift monitoring
        ref_df = build_base_features(df, cfg).dropna().tail(cfg.drift.reference_days)
        ref_df.to_parquet(cfg.paths.reference_data, index=False)

        upload_file(cfg.paths.model_bundle, "production/model_bundle.joblib")
        upload_json(metadata, "production/metadata.json")
        upload_file(cfg.paths.reference_data, "production/reference.parquet")
        logger.info("Promoted model %s -> %s", version_tag, cfg.paths.model_bundle)
    else:
        logger.warning("Promotion skipped: %s", decision.reason)

    return metadata, decision


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Train VayuCast LightGBM models & evaluate gate.")
    parser.add_argument("--force-promote", action="store_true", help="Promote even if gate fails.")
    parser.add_argument(
        "--fail-on-gate",
        action="store_true",
        help="Exit with code 1 if quality gate does not pass (for CI).",
    )
    args = parser.parse_args()

    metadata, decision = run_training(force_promote=args.force_promote)

    print("\n=== VayuCast Training Summary ===")
    print(f"Version: {metadata['model_version']} | Rows: {metadata['data_rows']}")
    print(f"{'Horizon':<8} {'Model MAE':>10} {'Baseline MAE':>13} {'Cat Acc':>9} {'Base Cat':>9}")
    for h, m in metadata["metrics"].items():
        print(
            f"{h:<8} {m['model_mae']:>10.2f} {m['baseline_mae']:>13.2f} "
            f"{m['category_accuracy'] * 100:>8.1f}% {m['baseline_category_accuracy'] * 100:>8.1f}%"
        )
    print(f"\nQuality Gate: {'PASSED' if decision.passed else 'FAILED'} — {decision.reason}")

    if args.fail_on_gate and not decision.passed:
        sys.exit(1)


if __name__ == "__main__":
    main()