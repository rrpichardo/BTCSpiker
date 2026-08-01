"""
Regenerate the committed handoff/data_sample/ fixture set from its own raw
NDJSON, writing all four artifacts (plus a manifest) atomically so they can
never drift out of sync with each other again.

This fixture exists to exercise schema and pipeline wiring in tests and local
runs — it is NOT a training-distribution proxy (see manifest.json's
`purpose` field and docs/runbook.md). The real training reference is a
784K-row corpus that is not committed to this repo (docs/drift_summary.md).

Usage
-----
    python scripts/regenerate_data_sample.py
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
FEATURES_DIR = PROJECT_ROOT / "features"
if str(FEATURES_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURES_DIR))

import pandas as pd
import yaml

from btcspiker_ml.features import FeatureEngine
from btcspiker_ml.storage import sha256_file
from btcspiker_ml.tick_identity import tick_dedupe_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("regenerate_data_sample")

DATA_SAMPLE_DIR = PROJECT_ROOT / "handoff" / "data_sample"
RAW_NDJSON = DATA_SAMPLE_DIR / "raw_slice.ndjson"
RAW_PARQUET = DATA_SAMPLE_DIR / "raw_slice.parquet"
FEATURES_PARQUET = DATA_SAMPLE_DIR / "features_slice.parquet"
FEATURES_CSV = DATA_SAMPLE_DIR / "features_slice.csv"
MANIFEST_PATH = DATA_SAMPLE_DIR / "manifest.json"

RAW_COLUMNS = ["product_id", "price", "best_bid", "best_ask", "volume_24_h", "timestamp"]
# Matches the committed features_slice.{parquet,csv} column set exactly —
# FeatureEngine rows also carry feature_set_id/feature_schema_version, which
# this fixture has never included and REQUIRED_FEATURE_COLUMNS doesn't need.
FEATURES_COLUMNS = [
    "product_id", "timestamp", "price", "midprice", "log_return", "spread_abs",
    "spread_bps", "vol_60s", "mean_return_60s", "n_ticks_60s",
    "trade_intensity_60s", "spread_mean_60s", "price_range_60s",
    "future_vol_60s", "vol_spike",
]

FEATURE_SET_ID = "core_v1"


def _dedupe_lines(lines: list[str]) -> list[str]:
    """Keep the first line of every duplicate-payload group, order preserved."""
    seen: set = set()
    kept = []
    for line in lines:
        key = tick_dedupe_key(json.loads(line))
        if key in seen:
            continue
        seen.add(key)
        kept.append(line)
    return kept


def _write_raw(lines: list[str]) -> list[dict]:
    RAW_NDJSON.write_text("\n".join(lines) + "\n")
    ticks = [json.loads(line) for line in lines]
    df = pd.DataFrame(ticks)[RAW_COLUMNS]
    df.to_parquet(RAW_PARQUET, index=False)
    return ticks


def _materialize_features(ticks: list[dict], horizon_sec: float, vol_threshold: float) -> pd.DataFrame:
    engine = FeatureEngine(FEATURE_SET_ID, horizon_sec, vol_threshold)
    rows = [row for tick in ticks for row in engine.ingest(tick)]
    rows.extend(engine.drain_remaining())
    df = pd.DataFrame(rows)[FEATURES_COLUMNS]
    return df


def _write_features(df: pd.DataFrame) -> None:
    df.to_parquet(FEATURES_PARQUET, index=False)
    df.to_csv(FEATURES_CSV, index=False)


def _write_manifest(raw_lines: list[str], features_df: pd.DataFrame) -> None:
    raw_df = pd.DataFrame([json.loads(line) for line in raw_lines])
    manifest = {
        "purpose": (
            "Schema/smoke-test fixture for local runs and CI. NOT a training-"
            "distribution proxy — this is a 10-minute slice; the real training "
            "reference (784K rows) is not committed to this repo. Do not use "
            "features_slice.* as a drift reference or a fallback training corpus."
        ),
        "source_sha256": hashlib.sha256(RAW_NDJSON.read_bytes()).hexdigest(),
        "generated_by": "scripts/regenerate_data_sample.py",
        "artifacts": {
            "raw_slice.ndjson": {
                "rows": len(raw_lines),
                "sha256": sha256_file(RAW_NDJSON),
            },
            "raw_slice.parquet": {
                "rows": len(raw_lines),
                "columns": RAW_COLUMNS,
                "sha256": sha256_file(RAW_PARQUET),
            },
            "features_slice.parquet": {
                "rows": len(features_df),
                "columns": FEATURES_COLUMNS,
                "sha256": sha256_file(FEATURES_PARQUET),
            },
            "features_slice.csv": {
                "rows": len(features_df),
                "columns": FEATURES_COLUMNS,
                "sha256": sha256_file(FEATURES_CSV),
            },
        },
        "event_time_bounds": {
            "start": str(raw_df["timestamp"].min()),
            "end": str(raw_df["timestamp"].max()),
        },
        "class_prevalence": {
            "vol_spike": float(features_df["vol_spike"].mean()) if len(features_df) else None,
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    horizon_sec = float(cfg["features"]["label_horizon_sec"])
    vol_threshold = float(cfg["features"]["vol_threshold"])

    raw_lines = [ln.strip() for ln in RAW_NDJSON.read_text().splitlines() if ln.strip()]
    before = len(raw_lines)
    deduped_lines = _dedupe_lines(raw_lines)
    log.info("raw_slice.ndjson: %d lines -> %d after dedupe", before, len(deduped_lines))

    ticks = _write_raw(deduped_lines)
    features_df = _materialize_features(ticks, horizon_sec, vol_threshold)
    log.info(
        "features_slice: %d rows, vol_spike prevalence %.4f",
        len(features_df),
        features_df["vol_spike"].mean() if len(features_df) else float("nan"),
    )
    _write_features(features_df)
    _write_manifest(deduped_lines, features_df)
    log.info("Wrote manifest to %s", MANIFEST_PATH)


if __name__ == "__main__":
    main()
