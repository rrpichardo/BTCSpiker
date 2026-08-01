"""
Regenerate an Evidently train-vs-current drift report on demand.

Compares a reference dataset (the real training feature corpus, or a
held-out split of it) against a current dataset (typically the live
featurizer's Parquet output) and writes a self-contained HTML report.

Usage:
    python scripts/drift_report.py \\
        --reference /path/to/training/features.parquet \\
        --current   data/processed/features.parquet \\
        --out       reports/drift_$(date +%Y%m%d).html
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("drift_report")

# handoff/data_sample/ is a schema/smoke fixture (see its manifest.json), not
# a distribution proxy — a 10-minute slice compared against itself reports
# "no drift" on features that are genuinely miscalibrated. Refuse it outright
# rather than silently producing a report nobody should trust.
SMOKE_FIXTURE_DIR = "handoff/data_sample"

FEATURE_COLS = [
    "log_return",
    "spread_bps",
    "vol_60s",
    "mean_return_60s",
    "trade_intensity_60s",
    "n_ticks_60s",
    "spread_mean_60s",
]


def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix in {".csv", ".tsv"}:
        df = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
    else:
        raise ValueError(f"Unsupported extension: {path.suffix}")
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    return df


def build_report(reference: pd.DataFrame, current: pd.DataFrame, out_path: Path) -> None:
    # Imported lazily so the script can fail fast on bad inputs before paying
    # the ~3s evidently import cost.
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

    cols = FEATURE_COLS + (["vol_spike"] if "vol_spike" in reference.columns and "vol_spike" in current.columns else [])
    ref = reference[cols].copy()
    cur = current[cols].copy()

    log.info(
        "Building drift report — reference rows=%d, current rows=%d, columns=%d",
        len(ref), len(cur), len(cols),
    )

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(out_path))
    log.info("Wrote %s", out_path)


def main():
    p = argparse.ArgumentParser(description="Generate Evidently drift report")
    p.add_argument("--reference", required=True, help="Training-period features (CSV or Parquet)")
    p.add_argument("--current", required=True, help="Current-period features (CSV or Parquet)")
    p.add_argument("--out", required=True, help="Output HTML path")
    args = p.parse_args()

    ref_path = Path(args.reference)
    cur_path = Path(args.current)
    out_path = Path(args.out)

    if not ref_path.exists():
        log.error("reference not found: %s", ref_path); sys.exit(1)
    if SMOKE_FIXTURE_DIR in ref_path.as_posix():
        log.error(
            "refusing to use %s as a drift reference — it is a schema/smoke "
            "fixture, not a training-distribution proxy (see its manifest.json). "
            "Point --reference at the real training corpus.",
            ref_path,
        )
        sys.exit(1)
    if not cur_path.exists():
        log.error("current not found: %s — is the featurizer running?", cur_path); sys.exit(1)

    ref = load_dataset(ref_path)
    cur = load_dataset(cur_path)
    build_report(ref, cur, out_path)


if __name__ == "__main__":
    main()
