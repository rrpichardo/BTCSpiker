from pathlib import Path
import json
import threading
import time

import mlflow
import pytest

from btcspiker_ml import search
from btcspiker_ml.search import SearchState, run_stage


def _config(tmp_path: Path):
    return {
        "tracking_uri": tmp_path.as_uri(),
        "experiment_name": "search-test",
        "state_dir": tmp_path / "state",
        "search_id": "search-1",
        "feature_set_id": "core_v1",
        "target_version": "target_v1",
        "validation_version": "walkforward_v1",
        "git_sha": "abc",
        "deployable": True,
        "trials": [
            {
                "id": "ok",
                "model_family": "logistic",
                "outcome": "finished",
                "metrics": {"aggregate_pr_auc": 0.2},
            },
            {"id": "pruned", "model_family": "logistic", "outcome": "pruned"},
            {
                "id": "bad",
                "model_family": "logistic",
                "outcome": "failed",
                "exception": "bad trial",
            },
            {"id": "ok-2", "model_family": "logistic", "outcome": "finished"},
            {"id": "ok-3", "model_family": "logistic", "outcome": "finished"},
        ],
    }


def test_assert_thread_budget_capped_raises_when_env_vars_unset(monkeypatch):
    for name in search.THREAD_LIMIT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="max_parallel_jobs"):
        search._assert_thread_budget_capped()


def test_assert_thread_budget_capped_passes_when_env_vars_set(monkeypatch):
    for name in search.THREAD_LIMIT_ENV_VARS:
        monkeypatch.setenv(name, "1")

    search._assert_thread_budget_capped()  # must not raise


def test_run_stage_with_concurrent_trials_requires_capped_thread_budget(
    tmp_path: Path, monkeypatch
):
    """The precondition is enforced at the real call site (run_stage's
    ThreadPoolExecutor path), not just in isolation -- a caller with
    max_parallel_jobs > 1 that forgot to cap native threads gets a clear,
    immediate RuntimeError instead of silently oversubscribing the CPU.
    """
    for name in search.THREAD_LIMIT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    config = _config(tmp_path)
    config.update(
        max_parallel_jobs=2,
        trials=[
            {
                "id": str(number),
                "model_family": "logistic",
                "evaluate": lambda: {"metrics": {}},
            }
            for number in range(2)
        ],
    )

    with pytest.raises(RuntimeError, match="max_parallel_jobs"):
        run_stage(config, "d1", "core_v1", "linear")


def test_run_stage_logs_finished_pruned_failed_and_resume_does_not_duplicate(
    tmp_path: Path,
):
    config = _config(tmp_path)
    first = run_stage(config, "d1", "core_v1", "linear")
    assert first.completed_trial_ids == ("ok", "pruned", "bad", "ok-2", "ok-3")
    second = run_stage({**config, "resume": True}, "d1", "core_v1", "linear")
    assert second.completed_trial_ids == ()
    state = SearchState.load(tmp_path / "state" / "search-1.json")
    assert state.completed_stages == ["linear"]
    client = mlflow.tracking.MlflowClient(tmp_path.as_uri())
    runs = client.search_runs(
        [client.get_experiment_by_name("search-test").experiment_id]
    )
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


def test_existing_search_state_requires_explicit_resume_and_same_contract(
    tmp_path: Path,
):
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
        {
            "id": "high",
            "model_family": "logistic",
            "outcome": "finished",
            "metrics": {"aggregate_pr_auc": 0.8},
        },
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
            {
                "id": "low",
                "model_family": "logistic",
                "outcome": "finished",
                "metrics": {"aggregate_pr_auc": 0.2},
            },
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
    parents = [
        run
        for run in runs
        if run.data.tags.get("candidate_stage") == "linear"
        and not run.data.tags.get("mlflow.parentRunId")
    ]
    assert len(parents) == 1


def test_stage_evaluates_independent_trials_up_to_configured_parallel_limit(
    tmp_path: Path, monkeypatch
):
    # run_stage's ThreadPoolExecutor path asserts the process-wide native
    # thread budget is capped before running trials concurrently (see
    # btcspiker_ml.search._assert_thread_budget_capped) -- this test's
    # `evaluate` is synthetic and doesn't touch numpy/sklearn, but the
    # precondition is checked unconditionally whenever max_parallel_jobs > 1,
    # so it must be satisfied here too, same as any real caller would.
    for name in search.THREAD_LIMIT_ENV_VARS:
        monkeypatch.setenv(name, "1")

    config = _config(tmp_path)
    active = 0
    peak = 0
    lock = threading.Lock()

    def evaluate():
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return {"metrics": {"aggregate_pr_auc": 0.2}}

    config.update(
        max_parallel_jobs=2,
        trials=[
            {"id": str(number), "model_family": "logistic", "evaluate": evaluate}
            for number in range(4)
        ],
    )

    run_stage(config, "d1", "core_v1", "linear")

    assert peak == 2


def test_neural_stage_requires_measured_boosted_tree_plateau(tmp_path: Path):
    config = _config(tmp_path)
    state = SearchState.new("search-1", "d1", wall_clock_seconds=3600)
    state.completed_stages = ["trees", "ablation"]
    state.stage_score_history = {"trees": [0.10, 0.11, 0.12, 0.13]}
    state.save(tmp_path / "state" / "search-1.json")
    config.update(
        labelled_rows=100_000,
        development_fold_positive_events=[100, 100, 100, 100, 100],
        trials=[{"id": "neural", "outcome": "skipped", "skip_reason": "environment"}],
        resume=True,
    )

    result = run_stage(config, "d1", "core_v1", "neural")

    assert result.status == "skipped"
    assert "boosted-tree plateau" in result.skipped_reason


def test_miscalibrated_trial_does_not_win_despite_a_higher_ranking_score(
    tmp_path: Path,
):
    config = _config(tmp_path)
    config.update(
        max_development_brier_ratio=1.05,
        trials=[
            {
                "id": "calibrated",
                "model_family": "logistic",
                "outcome": "finished",
                "metrics": {
                    "aggregate_pr_auc": 0.20,
                    "development_brier_ratio": 0.98,
                },
            },
            {
                "id": "miscalibrated",
                "model_family": "logistic",
                "outcome": "finished",
                "metrics": {
                    "aggregate_pr_auc": 0.40,
                    "development_brier_ratio": 1.83,
                },
            },
        ],
    )

    run_stage(config, "d1", "core_v1", "linear")
    state = SearchState.load(tmp_path / "state" / "search-1.json")

    # The miscalibrated trial ranks higher (0.40) but is disqualified, so the
    # calibrated trial holds the stage win.
    assert state.best_scores["linear"] == pytest.approx(0.20)
    assert state.stage_score_history["linear"] == [0.20, 0.40]


def test_one_lucky_fold_cannot_win_a_stage_for_an_inconsistent_trial(tmp_path: Path):
    config = _config(tmp_path)
    config.update(
        min_development_folds_won=4,
        trials=[
            {
                "id": "consistent",
                "model_family": "logistic",
                "outcome": "finished",
                "metrics": {
                    "aggregate_pr_auc": 0.20,
                    "development_folds_won": 5,
                },
            },
            {
                "id": "one-lucky-fold",
                "model_family": "logistic",
                "outcome": "finished",
                "metrics": {
                    "aggregate_pr_auc": 0.40,
                    "development_folds_won": 1,
                },
            },
        ],
    )

    run_stage(config, "d1", "core_v1", "linear")
    state = SearchState.load(tmp_path / "state" / "search-1.json")

    assert state.best_scores["linear"] == pytest.approx(0.20)


def test_stage_completes_and_records_no_winner_when_no_trial_clears_the_bar(
    tmp_path: Path,
):
    config = _config(tmp_path)
    config.update(
        min_development_folds_won=4,
        trials=[
            {
                "id": "inconsistent",
                "model_family": "logistic",
                "outcome": "finished",
                "metrics": {
                    "aggregate_pr_auc": 0.40,
                    "development_folds_won": 1,
                },
            },
        ],
    )

    result = run_stage(config, "d1", "core_v1", "linear")
    state = SearchState.load(tmp_path / "state" / "search-1.json")

    assert result.status == "completed"
    assert "linear" in state.completed_stages
    assert "linear" not in state.best_run_ids


def test_baseline_stage_records_its_reference_winner_despite_candidate_bars(
    tmp_path: Path,
):
    """The baseline scores exactly prevalence, so it can never beat prevalence.

    Gating it like a candidate leaves qualification with no reference at all.
    """
    config = _config(tmp_path)
    config.update(
        min_development_folds_won=4,
        max_development_brier_ratio=1.05,
        trials=[
            {
                "id": "prevalence",
                "model_family": "development_prevalence",
                "outcome": "finished",
                "metrics": {
                    "aggregate_pr_auc": 0.08,
                    "development_folds_won": 0,
                    "development_brier_ratio": 1.0,
                },
            },
        ],
    )

    run_stage(config, "d1", "core_v1", "baseline")
    state = SearchState.load(tmp_path / "state" / "search-1.json")

    assert state.best_run_ids["baseline"]
