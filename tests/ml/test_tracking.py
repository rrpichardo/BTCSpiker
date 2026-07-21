from pathlib import Path

import mlflow
import pytest

from btcspiker_ml.tracking import ExperimentTracker


def _lineage(**overrides):
    values = {
        "dataset_id": "d1",
        "feature_set_id": "core_v1",
        "target_version": "target_v1",
        "validation_version": "walkforward_v1",
        "git_sha": "abc",
        "search_id": "search-1",
        "model_family": "logistic",
        "deployable": "true",
    }
    values.update(overrides)
    return values


def test_tracker_logs_required_lineage_and_artifacts(tmp_path: Path):
    mlflow.set_tracking_uri(tmp_path.as_uri())
    tracker = ExperimentTracker("test-experiment")
    run_id = tracker.start_run(_lineage())
    tracker.log_metrics({"fold_0_pr_auc": 0.2, "aggregate_pr_auc": 0.2})
    tracker.log_lineage_artifacts(
        config={"seed": 42}, dataset_manifest={"rows": 4}, feature_manifest={"id": "core_v1"},
        fold_boundaries=[{"fold": 0}], oof_predictions="a,b\n1,2\n",
    )
    tracker.end_run("FINISHED")
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    assert run.data.tags["dataset_id"] == "d1"
    assert run.data.tags["run_status"] == "finished"
    assert run.data.metrics["aggregate_pr_auc"] == 0.2
    assert {item.path for item in client.list_artifacts(run_id)} >= {
        "experiment-config.yaml", "dataset-manifest.json", "feature-manifest.json",
        "fold-boundaries.json", "oof-predictions.csv",
    }


def test_tracker_rejects_missing_required_lineage(tmp_path: Path):
    mlflow.set_tracking_uri(tmp_path.as_uri())
    tracker = ExperimentTracker("test-experiment")
    with pytest.raises(ValueError, match="target_version"):
        tracker.start_run(_lineage(target_version=""))
