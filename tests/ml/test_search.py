from pathlib import Path

import mlflow

from btcspiker_ml.search import SearchState, run_stage


def _config(tmp_path: Path):
    return {
        "tracking_uri": tmp_path.as_uri(), "experiment_name": "search-test",
        "state_dir": tmp_path / "state", "search_id": "search-1",
        "feature_set_id": "core_v1", "target_version": "target_v1",
        "validation_version": "walkforward_v1", "git_sha": "abc", "deployable": True,
        "trials": [
            {"id": "ok", "model_family": "logistic", "outcome": "finished", "metrics": {"aggregate_pr_auc": 0.2}},
            {"id": "pruned", "model_family": "logistic", "outcome": "pruned"},
            {"id": "bad", "model_family": "logistic", "outcome": "failed", "exception": "bad trial"},
            {"id": "ok-2", "model_family": "logistic", "outcome": "finished"},
            {"id": "ok-3", "model_family": "logistic", "outcome": "finished"},
        ],
    }


def test_run_stage_logs_finished_pruned_failed_and_resume_does_not_duplicate(tmp_path: Path):
    config = _config(tmp_path)
    first = run_stage(config, "d1", "core_v1", "linear")
    assert first.completed_trial_ids == ("ok", "pruned", "bad", "ok-2", "ok-3")
    second = run_stage(config, "d1", "core_v1", "linear")
    assert second.completed_trial_ids == ()
    state = SearchState.load(tmp_path / "state" / "search-1.json")
    assert state.completed_stages == ["linear"]
    client = mlflow.tracking.MlflowClient(tmp_path.as_uri())
    runs = client.search_runs([client.get_experiment_by_name("search-test").experiment_id])
    children = [run for run in runs if run.data.tags.get("mlflow.parentRunId")]
    statuses = [run.data.tags["run_status"] for run in children]
    assert statuses.count("finished") == 3
    assert statuses.count("pruned") == 1
    assert statuses.count("failed") == 1
    assert state.failure_counts["linear"] == 1


def test_neural_ineligibility_is_logged_as_finished_skipped_parent(tmp_path: Path):
    config = _config(tmp_path)
    config["labelled_rows"] = 99_999
    result = run_stage(config, "d1", "core_v1", "neural")
    assert result.status == "skipped"
    client = mlflow.tracking.MlflowClient(tmp_path.as_uri())
    experiment = client.get_experiment_by_name("search-test")
    run = client.search_runs([experiment.experiment_id])[0]
    assert run.info.status == "FINISHED"
    assert run.data.tags["stage_status"] == "skipped"
    assert "100,000" in run.data.tags["skip_reason"]
