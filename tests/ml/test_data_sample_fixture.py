"""Guards the committed handoff/data_sample/ fixture set against the exact
defect this suite exists to catch: every raw tick duplicated, which silently
corrupted both model inputs (6.3 sigma off training) and labels (vol_spike
collapsed to 0%). See scripts/regenerate_data_sample.py."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from btcspiker_ml.storage import sha256_file
from btcspiker_ml.tick_identity import tick_dedupe_key

DATA_SAMPLE_DIR = Path("handoff/data_sample")
RAW_NDJSON = DATA_SAMPLE_DIR / "raw_slice.ndjson"
FEATURES_PARQUET = DATA_SAMPLE_DIR / "features_slice.parquet"
FEATURES_CSV = DATA_SAMPLE_DIR / "features_slice.csv"
MANIFEST_PATH = DATA_SAMPLE_DIR / "manifest.json"


def _raw_lines() -> list[str]:
    return [ln for ln in RAW_NDJSON.read_text().splitlines() if ln.strip()]


def test_raw_slice_has_no_duplicate_payloads():
    lines = _raw_lines()
    keys = [tick_dedupe_key(json.loads(line)) for line in lines]
    # Uniqueness on the canonical dedupe key, NOT on timestamp alone —
    # a timestamp-only check would forbid legitimate same-microsecond,
    # different-price events.
    assert len(keys) == len(set(keys))


def test_features_slice_target_is_not_degenerate():
    df = pd.read_parquet(FEATURES_PARQUET)
    assert set(df["vol_spike"].unique()) == {0, 1}, (
        "vol_spike must contain both classes — a single-class fixture is what "
        "let a 0%-positive corpus silently become a tournament dataset"
    )


def test_manifest_matches_artifacts():
    manifest = json.loads(MANIFEST_PATH.read_text())
    artifacts = manifest["artifacts"]

    raw_lines = _raw_lines()
    assert artifacts["raw_slice.ndjson"]["rows"] == len(raw_lines)
    assert artifacts["raw_slice.ndjson"]["sha256"] == sha256_file(RAW_NDJSON)
    assert manifest["source_sha256"] == hashlib.sha256(RAW_NDJSON.read_bytes()).hexdigest()

    raw_parquet_path = DATA_SAMPLE_DIR / "raw_slice.parquet"
    assert artifacts["raw_slice.parquet"]["sha256"] == sha256_file(raw_parquet_path)
    assert artifacts["raw_slice.parquet"]["rows"] == len(raw_lines)
    assert list(pd.read_parquet(raw_parquet_path).columns) == artifacts["raw_slice.parquet"]["columns"]

    for name, path in (
        ("features_slice.parquet", FEATURES_PARQUET),
        ("features_slice.csv", FEATURES_CSV),
    ):
        assert artifacts[name]["sha256"] == sha256_file(path)
        assert artifacts[name]["rows"] == len(_load(path))
        assert list(_load(path).columns) == artifacts[name]["columns"]

    features_df = pd.read_parquet(FEATURES_PARQUET)
    assert manifest["class_prevalence"]["vol_spike"] == float(features_df["vol_spike"].mean())


def _load(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def test_manifest_documents_the_fixture_is_not_a_training_reference():
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert "not a training-distribution proxy" in manifest["purpose"].lower()
