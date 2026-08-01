from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from btcspiker_ml.manifest import DatasetManifest, manifest_id
from btcspiker_ml.storage import sha256_file


REQUIRED_FEATURE_COLUMNS = {
    "timestamp",
    "log_return",
    "spread_bps",
    "vol_60s",
    "mean_return_60s",
    "trade_intensity_60s",
    "n_ticks_60s",
    "spread_mean_60s",
    "vol_spike",
}

TARGET_COLUMN = "vol_spike"
TIMESTAMP_COLUMN = "timestamp"

# Mirrors evaluation.py's MIN_POSITIVES: below this many examples of a class,
# a temporal split (btcspiker_ml.splits.make_temporal_splits) can't reliably
# keep both classes in every fold's train AND validation slice, so a corpus
# this thin can only fail confusingly several layers downstream rather than
# being rejected here at bind time. This does not replace make_temporal_splits'
# own per-fold class check -- it's a fast pre-check against the corpus as a
# whole, not a guarantee any specific fold plan will succeed.
MIN_CLASS_COUNT = 10


@dataclass(frozen=True)
class ExistingDataset:
    path: Path
    rows: int
    start_time: str
    end_time: str
    columns: tuple[str, ...]
    sha256: str
    parent_dataset_id: str | None = None
    source_manifest_path: str | None = None
    feature_set_id: str | None = None
    feature_engine_git_sha: str | None = None
    excluded_intervals: tuple[dict[str, str], ...] = ()


def resolve_existing_dataset(configured: Path | None) -> Path:
    """Return the absolute path of the collected corpus to bind against.

    Precedence: BTCSPIKER_EXISTING_DATA env var, explicit config, project default,
    checked-in handoff sample. Never fabricates data — raises if none exist.

    The handoff sample fallback resolves to a path, not a guarantee it's
    usable — inspect_existing_dataset is what rejects a corpus too small or
    too class-imbalanced to support a temporal split (MIN_CLASS_COUNT), which
    is what actually matters here: this function only answers "does a file
    exist", never "is it a fit training corpus".
    """
    candidates: list[Path | None] = [
        Path(os.environ["BTCSPIKER_EXISTING_DATA"]) if os.environ.get("BTCSPIKER_EXISTING_DATA") else None,
        configured,
        Path("data/processed/features.parquet"),
        Path("handoff/data_sample/features_slice.parquet"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.expanduser().is_file():
            return candidate.expanduser().resolve()
    raise FileNotFoundError(
        "existing collected dataset not found; set BTCSPIKER_EXISTING_DATA"
    )


def _load_schema_columns(path: Path) -> list[str]:
    schema = pq.read_schema(path)
    return list(schema.names)


def inspect_existing_dataset(path: Path) -> ExistingDataset:
    """Read the collected parquet, validate its shape, and describe it.

    Raises ValueError on any of: empty file, duplicate column names, missing target
    or required features, non-UTC or non-monotonic timestamps, non-binary target.
    Does not mutate the source file.
    """
    if not path.is_file():
        raise ValueError(f"parquet not found or not a file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"parquet is empty: {path}")

    try:
        schema_columns = _load_schema_columns(path)
    except Exception as exc:  # pragma: no cover - defensive parse guard
        raise ValueError(f"failed to read parquet schema at {path}: {exc}") from exc

    if len(schema_columns) != len(set(schema_columns)):
        duplicates = sorted({c for c in schema_columns if schema_columns.count(c) > 1})
        raise ValueError(f"duplicate column names in {path}: {duplicates}")

    if TARGET_COLUMN not in schema_columns:
        raise ValueError(f"missing target column {TARGET_COLUMN!r} in {path}")

    missing_features = REQUIRED_FEATURE_COLUMNS - set(schema_columns)
    if missing_features:
        raise ValueError(
            f"missing required feature columns in {path}: {sorted(missing_features)}"
        )

    # Only read the columns we validate against — parquet's columnar layout keeps
    # this cheap even on the multi-hundred-MB corpus, and no other feature values
    # are needed for the manifest.
    validation_columns = sorted(REQUIRED_FEATURE_COLUMNS)
    df = pd.read_parquet(path, columns=validation_columns)

    if len(df) == 0:
        raise ValueError(f"parquet has zero rows: {path}")

    parsed_timestamps = pd.to_datetime(df[TIMESTAMP_COLUMN], utc=True, errors="coerce")
    if parsed_timestamps.isna().any():
        raise ValueError(f"timestamp column contains unparseable values in {path}")
    if parsed_timestamps.dt.tz is None or str(parsed_timestamps.dt.tz) != "UTC":
        raise ValueError(f"timestamp column is not UTC in {path}")
    if not parsed_timestamps.is_monotonic_increasing:
        raise ValueError(
            f"timestamp column is not monotonic non-decreasing in {path}"
        )

    target = df[TARGET_COLUMN]
    unique_targets = set(pd.unique(target.dropna()))
    if unique_targets - {0, 1}:
        raise ValueError(
            f"target column {TARGET_COLUMN!r} is not binary in {path}: "
            f"observed values {sorted(unique_targets)}"
        )
    if unique_targets != {0, 1}:
        # A corpus corrupted upstream (e.g. duplicated ticks halving realised
        # volatility, see handoff/data_sample/manifest.json) can collapse to
        # a single class while still passing the binary check above -- this
        # is what let a 0%-positive corpus silently become a tournament
        # dataset. Caught here rather than several layers downstream where
        # make_temporal_splits would fail with a fold-shaped error that
        # doesn't point back at the corpus itself.
        raise ValueError(
            f"target column {TARGET_COLUMN!r} in {path} contains only class(es) "
            f"{sorted(unique_targets)} — both 0 and 1 must be present; a "
            "single-class corpus cannot support a temporal validation split"
        )

    class_counts = target.value_counts()
    undersized = {int(k): int(v) for k, v in class_counts.items() if v < MIN_CLASS_COUNT}
    if undersized:
        raise ValueError(
            f"target column {TARGET_COLUMN!r} in {path} has too few examples of "
            f"class(es) {undersized} (minimum {MIN_CLASS_COUNT} per class) — too "
            "small to reliably support a temporal validation split"
        )

    lineage_path = path.with_suffix(path.suffix + ".lineage.json")
    lineage: dict[str, object] = {}
    if lineage_path.is_file():
        try:
            lineage = json.loads(lineage_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid feature lineage sidecar at {lineage_path}: {error}") from error
        if not isinstance(lineage, dict):
            raise ValueError(f"invalid feature lineage sidecar at {lineage_path}")
    exclusions = lineage.get("excluded_intervals", [])
    if not isinstance(exclusions, list) or any(not isinstance(item, dict) for item in exclusions):
        raise ValueError(f"invalid excluded intervals in {lineage_path}")

    return ExistingDataset(
        path=path.resolve(),
        rows=int(len(df)),
        start_time=parsed_timestamps.iloc[0].isoformat(),
        end_time=parsed_timestamps.iloc[-1].isoformat(),
        columns=tuple(schema_columns),
        sha256=sha256_file(path),
        parent_dataset_id=lineage.get("parent_dataset_id") if isinstance(lineage.get("parent_dataset_id"), str) else None,
        source_manifest_path=lineage.get("source_manifest_path") if isinstance(lineage.get("source_manifest_path"), str) else None,
        feature_set_id=lineage.get("feature_set_id") if isinstance(lineage.get("feature_set_id"), str) else None,
        feature_engine_git_sha=lineage.get("feature_engine_git_sha") if isinstance(lineage.get("feature_engine_git_sha"), str) else None,
        excluded_intervals=tuple(exclusions),
    )


def publish_existing_manifest(
    dataset: ExistingDataset,
    artifact_root: Path,
    *,
    parent_dataset_id: str | None = None,
    source_manifest_path: str | None = None,
    feature_set_id: str | None = None,
    feature_engine_git_sha: str | None = None,
    excluded_intervals: list[dict[str, str]] | None = None,
) -> tuple[str, Path]:
    """Write a deterministic manifest describing the bound corpus.

    Returns (dataset_id, manifest_path). Only writes under artifact_root.
    """
    # Re-derive quality stats from the parquet without mutating it; we want the
    # manifest to record exactly what was present at bind time.
    validation_columns = sorted(REQUIRED_FEATURE_COLUMNS)
    df = pd.read_parquet(dataset.path, columns=validation_columns)
    parsed_timestamps = pd.to_datetime(df[TIMESTAMP_COLUMN], utc=True)

    null_counts = {col: int(df[col].isna().sum()) for col in validation_columns}
    duplicate_timestamps = int(parsed_timestamps.duplicated().sum())
    prevalence = float(df[TARGET_COLUMN].mean())

    quality: dict[str, int | float | str | dict[str, int]] = {
        "sha256": dataset.sha256,
        "prevalence": prevalence,
        "null_counts": null_counts,
        "duplicate_timestamps": duplicate_timestamps,
        "input_mode": "existing_collected",
        "absolute_source_path": str(dataset.path.resolve()),
    }

    manifest = DatasetManifest(
        source="local_parquet",
        product="BTC-USD",
        rows=dataset.rows,
        start_time=dataset.start_time,
        end_time=dataset.end_time,
        quality=quality,  # type: ignore[arg-type]
        partitions=[
            {
                "path": str(dataset.path.resolve()),
                "rows": dataset.rows,
                "sha256": dataset.sha256,
            }
        ],
        parent_dataset_id=parent_dataset_id if parent_dataset_id is not None else dataset.parent_dataset_id,
        source_manifest_path=source_manifest_path if source_manifest_path is not None else dataset.source_manifest_path,
        feature_set_id=feature_set_id if feature_set_id is not None else dataset.feature_set_id,
        feature_engine_git_sha=feature_engine_git_sha if feature_engine_git_sha is not None else dataset.feature_engine_git_sha,
        excluded_intervals=list(dataset.excluded_intervals if excluded_intervals is None else excluded_intervals),
    )
    dataset_id = manifest_id(manifest)

    manifests_dir = artifact_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifests_dir / f"existing-{dataset_id}.json"
    manifest_path.write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n"
    )
    return dataset_id, manifest_path.resolve()
