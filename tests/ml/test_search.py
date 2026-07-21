from pathlib import Path
import json

import mlflow
import pytest

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
    second = run_stage({**config, "resume": True}, "d1", "core_v1", "linear")
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


def test_existing_search_state_requires_explicit_resume_and_same_contract(tmp_path: Path):
    config = _config(tmp_path)
    run_stage(config, "d1", "core_v1", "linear")
    with pytest.raises(ValueError, match="--resume"):
        run_stage(config, "d1", "core_v1", "trees")

    changed = {**config, "target_version": "different-target", "resume": True}
    with pytest.raises(ValueError, match="immutable experiment contract"):
        run_stage(changed, "d1", "core_v1", "trees")


def test_partial_resume_reuses_parent_and_keeps_better_persisted_winner(tmp_path: Path):
    config = _config(tmp_path)
    config["trials"] = [
        {"id": "high", "model_family": "logistic", "outcome": "finished", "metrics": {"aggregate_pr_auc": 0.8}},
    ]
    first = run_stage(config, "d1", "core_v1", "linear")
    state_path = tmp_path / "state" / "search-1.json"
    legacy_state = json.loads(state_path.read_text())
    legacy_state.pop("best_scores")
    legacy_state.pop("stage_parent_run_ids")
    state_path.write_text(json.dumps(legacy_state))

    resumed = {
        **config,
        "resume": True,
        "trials": [
            *config["trials"],
            {"id": "low", "model_family": "logistic", "outcome": "finished", "metrics": {"aggregate_pr_auc": 0.2}},
        ],
    }
    second = run_stage(resumed, "d1", "core_v1", "linear")

    state = SearchState.load(state_path)
    assert second.parent_run_id == first.parent_run_id
    assert second.completed_trial_ids == ("low",)
    assert state.stage_parent_run_ids["linear"] == first.parent_run_id
    assert state.best_run_ids["linear"]
    assert state.best_scores["linear"] == pytest.approx(0.8)

    client = mlflow.tracking.MlflowClient(tmp_path.as_uri())
    experiment = client.get_experiment_by_name("search-test")
    runs = client.search_runs([experiment.experiment_id])
    parents = [run for run in runs if run.data.tags.get("candidate_stage") == "linear" and not run.data.tags.get("mlflow.parentRunId")]
    assert len(parents) == 1
