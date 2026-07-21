"""Run one resumable model-tournament stage; never promotes models."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btcspiker_ml.config import load_experiment_config
from btcspiker_ml.datasets import resolve_existing_dataset
from btcspiker_ml.metrics import evaluate_predictions
from btcspiker_ml.models import build_model
from btcspiker_ml.search import VALID_STAGES, run_stage
from btcspiker_ml.splits import make_temporal_splits


def _development_trial(config, raw: dict, stage: str) -> dict:
    """Evaluate one deterministic candidate only on development folds.

    ``make_temporal_splits`` owns the final holdout boundary.  This function
    intentionally never reads ``plan.final_holdout``: it is not a development
    input and cannot be opened by this CLI.
    """
    path = resolve_existing_dataset(config.storage.existing_data)
    frame = pd.read_parquet(path)
    columns = list(raw.get("feature_columns", []))
    if not columns:
        columns = [
            column for column in frame.select_dtypes(include=["number"]).columns
            if column not in {"vol_spike", "future_vol_60s"}
        ]
    if stage == "ablation":
        removed = raw.get("ablation_remove", "vol_60s")
        columns = [column for column in columns if column != removed]
    if not columns:
        raise ValueError("no usable development feature columns")
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    target = frame["vol_spike"].to_numpy(dtype=int)
    plan = make_temporal_splits(
        timestamps,
        config.validation,
        targets=target,
        event_keys=np.arange(len(frame)),
    )
    # Fill only from the training side of each fold, avoiding a whole-corpus
    # statistic that could leak the future development segment.
    source = frame.loc[:, columns].replace([np.inf, -np.inf], np.nan)
    fold_metrics: dict[str, float] = {}
    scores: list[float] = []
    family = {"linear": "logistic", "trees": "hist_gradient_boosting"}.get(stage)
    for fold_number, fold in enumerate(plan.folds):
        train = np.asarray(fold.train)
        validation = np.asarray(fold.validation)
        train_x = source.iloc[train]
        medians = train_x.median().fillna(0.0)
        validation_x = source.iloc[validation]
        if stage == "baseline":
            prediction = np.full(len(validation), float(target[train].mean()))
            model_name = "development_prevalence"
        else:
            if stage == "ensemble":
                linear = build_model("logistic", dict(raw.get("logistic_params", {})), config.validation.random_seed, 1)
                trees = build_model("hist_gradient_boosting", dict(raw.get("tree_params", {})), config.validation.random_seed, 1)
                fitted = [linear, trees]
                prediction_parts = []
                for model in fitted:
                    model.fit(train_x.fillna(medians), target[train])
                    prediction_parts.append(model.predict_proba(validation_x.fillna(medians))[:, 1])
                prediction = np.mean(prediction_parts, axis=0)
                model_name = "logistic_hist_gradient_mean"
            else:
                params_key = "tree_params" if stage == "trees" else "logistic_params"
                model = build_model(family or "logistic", dict(raw.get(params_key, {})), config.validation.random_seed, 1)
                model.fit(train_x.fillna(medians), target[train])
                prediction = model.predict_proba(validation_x.fillna(medians))[:, 1]
                model_name = family or "logistic"
        metric = evaluate_predictions(target[validation], prediction, timestamps.iloc[validation], threshold=0.5)
        fold_metrics[f"fold_{fold_number}_pr_auc"] = metric.pr_auc
        scores.append(metric.pr_auc)
    return {
        "id": f"{stage}-development-v1",
        "model_family": model_name,
        "outcome": "finished",
        "params": {"feature_columns": columns, "final_holdout": "sealed"},
        "metrics": {"aggregate_pr_auc": float(np.mean(scores)), **fold_metrics},
    }


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
        "resume": args.resume,
    }
    if args.stage in {"baseline", "linear", "trees", "ablation", "ensemble"}:
        values["trials"] = [_development_trial(config, raw, args.stage)]
        values["labelled_rows"] = int(pd.read_parquet(resolve_existing_dataset(config.storage.existing_data), columns=["vol_spike"]).shape[0])
    elif args.stage == "neural":
        values["labelled_rows"] = int(pd.read_parquet(resolve_existing_dataset(config.storage.existing_data), columns=["vol_spike"]).shape[0])
        values["neural_skip_reason"] = (
            "neural stage skipped: boosted-tree progress plateau is not established "
            "from one bounded tree trial"
        )
    result = run_stage(values, args.dataset_id, raw.get("feature_set_id", "unknown"), args.stage)
    print(f"{args.stage}: {result.status} ({result.parent_run_id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
