"""MLflow experiment ledger with mandatory, queryable lineage."""

from __future__ import annotations

import json
import traceback as traceback_module
from typing import Any, Mapping

import mlflow
import mlflow.sklearn
import yaml


REQUIRED_LINEAGE = (
    "dataset_id", "feature_set_id", "target_version", "validation_version",
    "git_sha", "search_id", "model_family", "deployable",
)


class ExperimentTracker:
    """Small stateful wrapper so every run has the same lineage contract."""

    def __init__(self, experiment_name: str, tracking_uri: str | None = None):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        self.experiment_id = experiment.experiment_id if experiment else mlflow.create_experiment(experiment_name)
        self._active_run_id: str | None = None

    def start_run(
        self, config: Mapping[str, Any], *, nested: bool = False,
        run_name: str | None = None, run_id: str | None = None,
    ) -> str:
        missing = [key for key in REQUIRED_LINEAGE if config.get(key) in (None, "")]
        if missing:
            raise ValueError(f"missing required MLflow lineage: {', '.join(missing)}")
        if run_id is None:
            run = mlflow.start_run(experiment_id=self.experiment_id, nested=nested, run_name=run_name)
        else:
            # MLflow requires its fluent active-experiment context to match a
            # resumed run's stored experiment.  New tracker instances are
            # common on process resume, so restore that context explicitly.
            mlflow.set_experiment(experiment_id=self.experiment_id)
            run = mlflow.start_run(run_id=run_id)
        self._active_run_id = run.info.run_id
        tags = {key: str(config[key]).lower() if isinstance(config[key], bool) else str(config[key]) for key in REQUIRED_LINEAGE}
        tags.update({"run_status": "running", "candidate_stage": str(config.get("candidate_stage", "unknown"))})
        mlflow.set_tags(tags)
        return run.info.run_id

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})

    def log_params(self, params: Mapping[str, Any]) -> None:
        mlflow.log_params({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value) for key, value in params.items()})

    def log_tags(self, tags: Mapping[str, Any]) -> None:
        mlflow.set_tags({key: str(value).lower() if isinstance(value, bool) else str(value) for key, value in tags.items()})

    def log_text(self, text: str, artifact_file: str) -> None:
        mlflow.log_text(text, artifact_file)

    def log_model(self, model: Any, *, artifact_path: str = "model") -> None:
        """Log a fitted sklearn-compatible model without registering it."""
        mlflow.sklearn.log_model(model, artifact_path)

    def log_resource_timing(self, measurements: Mapping[str, float]) -> None:
        """Keep queryable resource metrics and the exact measurement artifact."""
        numeric = {key: float(value) for key, value in measurements.items()}
        self.log_metrics(numeric)
        self.log_text(json.dumps(numeric, sort_keys=True, indent=2), "resource-timing.json")

    def run_metric(self, run_id: str, key: str) -> float | None:
        return mlflow.tracking.MlflowClient().get_run(run_id).data.metrics.get(key)

    def find_stage_parent(self, *, search_id: str, stage: str) -> str | None:
        """Recover the earliest parent created before parent IDs were persisted."""
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(
            [self.experiment_id],
            filter_string=f"tags.search_id = '{search_id}' and tags.candidate_stage = '{stage}'",
            order_by=["attributes.start_time ASC"],
        )
        parents = [run for run in runs if not run.data.tags.get("mlflow.parentRunId")]
        return parents[0].info.run_id if parents else None

    def log_lineage_artifacts(
        self, *, config: Mapping[str, Any], dataset_manifest: Any, feature_manifest: Any,
        fold_boundaries: Any, oof_predictions: Any, pr_plot: str | None = None,
        calibration_plot: str | None = None, regime_table: Any | None = None,
    ) -> None:
        """Log immutable run inputs and evaluation evidence without schema guessing."""
        self.log_text(yaml.safe_dump(dict(config), sort_keys=True), "experiment-config.yaml")
        self.log_text(json.dumps(dataset_manifest, sort_keys=True, indent=2, default=str), "dataset-manifest.json")
        self.log_text(json.dumps(feature_manifest, sort_keys=True, indent=2, default=str), "feature-manifest.json")
        self.log_text(json.dumps(fold_boundaries, sort_keys=True, indent=2, default=str), "fold-boundaries.json")
        self.log_text(_as_text(oof_predictions), "oof-predictions.csv")
        if pr_plot is not None:
            self.log_text(pr_plot, "pr-curve.txt")
        if calibration_plot is not None:
            self.log_text(calibration_plot, "calibration-curve.txt")
        if regime_table is not None:
            self.log_text(_as_text(regime_table), "regime-table.json")

    def log_failure(self, error: BaseException | str) -> None:
        if isinstance(error, BaseException):
            detail = "".join(traceback_module.format_exception(error))
            failure_class = type(error).__name__
        else:
            detail, failure_class = str(error), "TrialError"
        mlflow.set_tags({"run_status": "failed", "failure_class": failure_class})
        self.log_text(detail, "traceback.txt")

    def end_run(self, status: str = "FINISHED", *, run_status: str | None = None, extra_tags: Mapping[str, Any] | None = None) -> None:
        normalized = (run_status or {"FINISHED": "finished", "FAILED": "failed", "KILLED": "pruned"}.get(status, status.lower()))
        tags = {"run_status": normalized}
        if extra_tags:
            tags.update({key: str(value) for key, value in extra_tags.items()})
        mlflow.set_tags(tags)
        mlflow.end_run(status=status)
        self._active_run_id = None


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, indent=2, default=str)
