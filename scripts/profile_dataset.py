"""Audit the bound corpus: target prevalence, gaps, drift, autocorrelation.

Reads the parquet resolved via the ExperimentConfig, produces a deterministic
``DataProfile`` plus CSV / PNG diagnostics under ``artifact_root/eda/<dataset_id>/<utc-stamp>/``,
and logs everything to MLflow.  MLflow logging is best-effort: if
``cfg.mlflow.tracking_uri`` is unreachable, falls back to a local file store
under ``artifact_root/mlruns`` and prints a warning.  Never generates data.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Allow `python scripts/profile_dataset.py ...` to import btcspiker_ml without
# requiring PYTHONPATH=. or an editable install. Same shim as
# scripts/bind_existing_dataset.py so the whole scripts/ family behaves
# consistently under bare `python scripts/foo.py` invocation.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from btcspiker_ml.config import load_experiment_config  # noqa: E402
from btcspiker_ml.datasets import (  # noqa: E402
    inspect_existing_dataset,
    publish_existing_manifest,
    resolve_existing_dataset,
)
from btcspiker_ml.eda import profile_dataset, write_profile_artifacts  # noqa: E402


TIMESTAMP_COLUMN = "timestamp"
TARGET_COLUMN = "vol_spike"


def _configure_mlflow(tracking_uri: str, fallback_dir: Path) -> tuple[str, bool]:
    """Return (effective_tracking_uri, used_fallback).

    Attempts to reach the configured tracking server; on any failure, points
    MLflow at a local file store so the CLI stays runnable without a running
    server.  Never raises — logging is best-effort.
    """
    import mlflow  # local import so tests / library consumers don't pay for it

    try:
        mlflow.set_tracking_uri(tracking_uri)
        # search_experiments touches the backend; a failure here means the
        # configured tracking server is not reachable.
        mlflow.search_experiments(max_results=1)
        return tracking_uri, False
    except Exception as exc:  # broad: mlflow raises many exception types
        fallback_dir.mkdir(parents=True, exist_ok=True)
        fallback_uri = fallback_dir.resolve().as_uri()
        print(
            f"warning: MLflow tracking server {tracking_uri!r} unreachable "
            f"({type(exc).__name__}); falling back to local file store at {fallback_uri}",
            file=sys.stderr,
        )
        mlflow.set_tracking_uri(fallback_uri)
        return fallback_uri, True


def _log_to_mlflow(
    experiment_name: str,
    dataset_id: str,
    profile,
    artifact_paths: list[Path],
    dataset_source: Path,
    dataset_sha256: str,
) -> str | None:
    """Log the EDA bundle under a parent run named ``eda-<dataset_id>``.

    Returns the run_id on success, None on any logging failure — the CLI's
    output on disk is the ground truth; MLflow is a mirror.
    """
    import mlflow

    try:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"eda-{dataset_id}") as run:
            mlflow.set_tag("dataset_id", dataset_id)
            mlflow.set_tag("stage", "eda")
            mlflow.set_tag("qualification_data", str(profile.qualification_data).lower())
            mlflow.set_tag("neural_data_eligible", str(profile.neural_data_eligible).lower())
            mlflow.log_param("dataset_source", str(dataset_source))
            mlflow.log_param("dataset_sha256", dataset_sha256)
            mlflow.log_param("rows", profile.rows)
            mlflow.log_param("start_time", profile.start_time)
            mlflow.log_param("end_time", profile.end_time)
            mlflow.log_metric("positive_rate", profile.positive_rate)
            mlflow.log_metric("duplicate_timestamps", profile.duplicate_timestamps)
            mlflow.log_metric(
                "gap_count_over_60s",
                float(profile.gap_summary.get("count_over_threshold", 0)),
            )
            mlflow.log_metric(
                "longest_gap_seconds",
                float(profile.gap_summary.get("longest_gap_seconds", 0.0)),
            )
            for name, value in profile.inter_arrival_ms.items():
                mlflow.log_metric(f"inter_arrival_{name}", value)
            for path in artifact_paths:
                mlflow.log_artifact(str(path))
            return run.info.run_id
    except Exception as exc:
        print(
            f"warning: MLflow logging failed ({type(exc).__name__}: {exc}); "
            "continuing — on-disk artifacts are still authoritative.",
            file=sys.stderr,
        )
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiment.yaml"),
        help="Path to the frozen experiment.yaml",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        required=True,
        help="Dataset id produced by bind_existing_dataset.py",
    )
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    resolved_path = resolve_existing_dataset(cfg.storage.existing_data)
    dataset = inspect_existing_dataset(resolved_path)
    # Re-derive the manifest id to confirm the caller's --dataset-id matches
    # what the bound corpus actually resolves to. This is the audit gate: if
    # someone edited the parquet between bind and profile, the ids diverge and
    # we refuse rather than silently profiling a different file.
    resolved_id, _ = publish_existing_manifest(dataset, cfg.storage.artifact_root)
    if resolved_id != args.dataset_id:
        raise SystemExit(
            f"dataset_id mismatch: --dataset-id {args.dataset_id} does not match "
            f"resolved dataset {resolved_id}. Re-run bind_existing_dataset.py."
        )

    # EDA needs the full frame — every numeric feature contributes to
    # correlations, gaps, and drift plots.
    frame = pd.read_parquet(dataset.path)
    profile = profile_dataset(frame, TARGET_COLUMN, TIMESTAMP_COLUMN)

    utc_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = cfg.storage.artifact_root / "eda" / args.dataset_id / utc_stamp
    artifact_paths = write_profile_artifacts(
        profile, frame, TARGET_COLUMN, TIMESTAMP_COLUMN, output_dir
    )

    fallback_dir = cfg.storage.artifact_root / "mlruns"
    effective_uri, used_fallback = _configure_mlflow(cfg.mlflow.tracking_uri, fallback_dir)
    run_id = _log_to_mlflow(
        cfg.mlflow.experiment_name,
        args.dataset_id,
        profile,
        artifact_paths,
        dataset.path,
        dataset.sha256,
    )

    print(f"dataset_id: {args.dataset_id}")
    print(f"source: {dataset.path}")
    print(f"rows: {profile.rows}")
    print(f"event_time: {profile.start_time} -> {profile.end_time}")
    print(f"positive_rate: {profile.positive_rate:.6f}")
    print(f"coverage_days: {profile.coverage_days:.1f}")
    print(f"qualification_data: {str(profile.qualification_data).lower()}")
    print(f"neural_data_eligible: {str(profile.neural_data_eligible).lower()}")
    print(f"duplicate_timestamps: {profile.duplicate_timestamps}")
    print(
        f"gap_over_60s: {profile.gap_summary.get('count_over_threshold', 0)} "
        f"(longest {profile.gap_summary.get('longest_gap_seconds', 0.0):.1f}s "
        f"at {profile.gap_summary.get('longest_gap_start', '')})"
    )
    print(f"mlflow_tracking_uri: {effective_uri}{' (fallback)' if used_fallback else ''}")
    print(f"mlflow_run_id: {run_id if run_id else '(not logged)'}")
    print(f"output_dir: {output_dir}")
    print("artifacts:")
    for path in artifact_paths:
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
