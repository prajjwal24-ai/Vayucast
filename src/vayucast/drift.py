from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from vayucast.config import AppConfig, load_config
from vayucast.storage import download_file, upload_file, upload_json

logger = logging.getLogger(__name__)

MONITORED_COLUMNS: list[str] = [
    "pm2_5",
    "pm10",
    "nitrogen_dioxide",
    "ozone",
    "carbon_monoxide",
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max",
    "precipitation_sum",
]


def _extract_drift_summary(
    result_dict: dict[str, Any], drift_share_threshold: float
) -> tuple[int, float, dict[str, float]]:
    drifted_count = 0
    drift_share = 0.0
    column_scores: dict[str, float] = {}

    for metric in result_dict.get("metrics", []):
        metric_id = str(metric.get("metric_id") or metric.get("metric_name") or "")
        value = metric.get("value")

        if "DriftedColumnsCount" in metric_id and isinstance(value, dict):
            drifted_count = int(round(float(value.get("count", 0))))
            drift_share = float(value.get("share", 0.0))
        elif "ValueDrift" in metric_id and isinstance(value, (int, float)):
            match = re.search(r"column=([^,)]+)", metric_id)
            if match:
                col_name = match.group(1).strip().strip("'\"")
                column_scores[col_name] = round(float(value), 4)

    if not drifted_count and drift_share > 0 and MONITORED_COLUMNS:
        drifted_count = int(round(drift_share * len(MONITORED_COLUMNS)))

    return drifted_count, round(drift_share, 4), column_scores


def run_drift_check(cfg: AppConfig | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()

    if not cfg.paths.raw_data.exists():
        download_file("data/delhi_daily.parquet", cfg.paths.raw_data)
    if not cfg.paths.reference_data.exists():
        download_file("production/reference.parquet", cfg.paths.reference_data)

    if not cfg.paths.raw_data.exists() or not cfg.paths.reference_data.exists():
        raise FileNotFoundError(
            "Both raw dataset and reference dataset are required. Run ingest and train first."
        )

    raw_df = pd.read_parquet(cfg.paths.raw_data).sort_values("date")
    ref_df = pd.read_parquet(cfg.paths.reference_data)

    cols = [c for c in MONITORED_COLUMNS if c in raw_df.columns and c in ref_df.columns]
    reference_slice = ref_df[cols].dropna().reset_index(drop=True)
    current_slice = raw_df.tail(cfg.drift.current_days)[cols].dropna().reset_index(drop=True)

    if reference_slice.empty or current_slice.empty:
        raise ValueError("Reference or current window is empty after dropping NaNs.")

    report = Report([DataDriftPreset(drift_share=cfg.drift.drift_share_threshold)])
    snapshot = report.run(current_data=current_slice, reference_data=reference_slice)

    cfg.paths.drift_html.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(cfg.paths.drift_html))

    drifted_count, drift_share, column_scores = _extract_drift_summary(
        snapshot.dict(), cfg.drift.drift_share_threshold
    )
    dataset_drift = drift_share >= cfg.drift.drift_share_threshold

    current_dates = pd.to_datetime(raw_df.tail(cfg.drift.current_days)["date"])
    summary: dict[str, Any] = {
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "city": cfg.city.name,
        "reference_rows": len(reference_slice),
        "current_rows": len(current_slice),
        "current_window_start": str(current_dates.min().date()),
        "current_window_end": str(current_dates.max().date()),
        "monitored_columns": len(cols),
        "drifted_columns_count": drifted_count,
        "drift_share": drift_share,
        "drift_share_threshold": cfg.drift.drift_share_threshold,
        "dataset_drift_detected": dataset_drift,
        "column_drift_scores": column_scores,
    }

    cfg.paths.drift_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        "Drift check complete: %d/%d columns drifted (share=%.2f, alert=%s)",
        drifted_count,
        len(cols),
        drift_share,
        dataset_drift,
    )

    upload_json(summary, "reports/drift_summary.json")
    upload_file(cfg.paths.drift_html, "reports/drift_report.html")
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Run Evidently data drift check on Delhi data.")
    parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="Exit with code 1 if dataset drift exceeds threshold.",
    )
    args = parser.parse_args()

    summary = run_drift_check()
    print("\n=== VayuCast Drift Summary ===")
    print(
        f"Window: {summary['current_window_start']} -> {summary['current_window_end']} "
        f"({summary['current_rows']} days vs {summary['reference_rows']} ref days)"
    )
    print(
        f"Drifted columns: {summary['drifted_columns_count']}/{summary['monitored_columns']} "
        f"(share={summary['drift_share']:.2f}, threshold={summary['drift_share_threshold']:.2f})"
    )
    print(f"Dataset drift detected: {summary['dataset_drift_detected']}")

    if args.fail_on_drift and summary["dataset_drift_detected"]:
        sys.exit(1)


if __name__ == "__main__":
    main()