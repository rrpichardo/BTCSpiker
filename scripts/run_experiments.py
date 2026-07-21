"""Run one resumable model-tournament stage; never promotes models."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btcspiker_ml.config import load_experiment_config
from btcspiker_ml.search import VALID_STAGES, run_stage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--stage", required=True, choices=sorted(VALID_STAGES))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text())
    config = load_experiment_config(args.config)
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    values = {
        "tracking_uri": config.mlflow.tracking_uri, "experiment_name": config.mlflow.experiment_name,
        "search_id": raw.get("search_id", args.dataset_id), "git_sha": git_sha,
        "target_version": raw.get("target_version", config.target.name),
        "validation_version": raw.get("validation_version", "purged_walkforward_v1"),
        "deployable": False, "max_hours": config.search.max_hours,
        "trials": raw.get("trials", []), "labelled_rows": raw.get("labelled_rows", 0),
        "development_fold_positive_events": raw.get("development_fold_positive_events", []),
    }
    result = run_stage(values, args.dataset_id, raw.get("feature_set_id", "unknown"), args.stage)
    print(f"{args.stage}: {result.status} ({result.parent_run_id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
