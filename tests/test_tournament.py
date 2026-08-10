"""Tests for the read-only tournament projection over MLflow's file store.

Fixtures here write a real MLflow file-store layout to a tmp_path rather than
mocking, because the whole point of the module is that it parses that on-disk
format correctly.
"""

import json
import sys
from pathlib import Path

import pytest

# Same path convention as tests/test_evaluation.py: the materializer's modules
# import each other flatly (`import evaluation`), because the container runs
# with that directory as its working directory. Importing them as
# `materializer.<mod>` instead would bind the name `materializer` to the
# directory namespace package and shadow sibling tests' `import materializer`,
# which expects materializer/materializer.py.
MATERIALIZER_DIR = Path(__file__).resolve().parent.parent / "materializer"
sys.path.insert(0, str(MATERIALIZER_DIR))

from tournament import (  # noqa: E402
    TournamentUnavailable,
    get_run,
    list_runs,
)


def _write_run(
    experiment_dir,
    run_id,
    *,
    run_name,
    status=3,
    start_time=1784699164481,
    end_time=1784699166659,
    params=None,
    metrics=None,
    tags=None,
    artifacts=(),
):
    run_dir = experiment_dir / run_id
    (run_dir / "params").mkdir(parents=True)
    (run_dir / "metrics").mkdir()
    (run_dir / "tags").mkdir()
    (run_dir / "artifacts").mkdir()

    # meta.yaml is flat `key: value`; artifact_uri carries colons in its value,
    # which is exactly what a naive split(":") would corrupt.
    (run_dir / "meta.yaml").write_text(
        "\n".join(
            [
                f"artifact_uri: file:///tmp/{run_id}/artifacts",
                f"end_time: {end_time}",
                "entry_point_name: ''",
                f"experiment_id: '{experiment_dir.name}'",
                "lifecycle_stage: active",
                f"run_id: {run_id}",
                f"run_name: {run_name}",
                f"start_time: {start_time}",
                f"status: {status}",
                "tags: []",
                "user_id: ricopichardo",
                "",
            ]
        )
    )
    for key, value in (params or {}).items():
        (run_dir / "params" / key).write_text(str(value))
    for key, value in (metrics or {}).items():
        # MLflow appends one "<timestamp_ms> <value> <step>" line per record.
        (run_dir / "metrics" / key).write_text(f"1784699164489 {value} 0\n")
    for key, value in (tags or {}).items():
        (run_dir / "tags" / key).write_text(str(value))
    for name in artifacts:
        (run_dir / "artifacts" / name).write_text("x")
    return run_dir


@pytest.fixture()
def mlruns(tmp_path):
    root = tmp_path / "mlruns"
    experiment = root / "435324055989446079"
    experiment.mkdir(parents=True)
    (experiment / "meta.yaml").write_text(
        "artifact_location: file:///tmp/mlruns\n"
        "experiment_id: '435324055989446079'\n"
        "lifecycle_stage: active\n"
        "name: btc-volatility-tournament\n"
    )

    _write_run(
        experiment,
        "aaaa1111",
        run_name="trees-trial-trees-0002-hist_gradient_boosting",
        params={
            "model_family": "hist_gradient_boosting",
            "model_params": json.dumps({"learning_rate": 0.0567, "max_iter": 180}),
            "tau": "0.7015",
            "feature_cols": "log_return,spread_bps,vol_60s",
        },
        metrics={
            "aggregate_pr_auc": 0.3102,
            "fold_0_pr_auc": 0.28,
            "fold_1_pr_auc": 0.34,
            "development_folds_won": 4,
        },
        tags={"candidate_stage": "trees", "deployable": "true"},
        artifacts=("pr-curve.txt", "regime-table.json"),
    )
    _write_run(
        experiment,
        "bbbb2222",
        run_name="linear-trial-linear-0001-logistic_regression",
        params={"model_family": "logistic_regression"},
        metrics={"aggregate_pr_auc": 0.1459, "development_folds_won": 1},
        tags={"candidate_stage": "linear", "deployable": "false"},
    )
    return root


def test_list_runs_ranks_by_pr_auc_descending(mlruns):
    runs = list_runs(mlruns)

    assert [r["run_id"] for r in runs] == ["aaaa1111", "bbbb2222"]
    assert runs[0]["aggregate_pr_auc"] == pytest.approx(0.3102)
    assert runs[0]["model_family"] == "hist_gradient_boosting"
    assert runs[0]["stage"] == "trees"
    assert runs[0]["deployable"] is True
    assert runs[1]["deployable"] is False


def test_list_runs_reports_status_and_duration(mlruns):
    runs = list_runs(mlruns)

    assert runs[0]["status"] == "FINISHED"
    # 1784699166659 - 1784699164481 = 2178 ms
    assert runs[0]["duration_seconds"] == pytest.approx(2.178)


def test_run_detail_parses_model_params_json_and_folds(mlruns):
    detail = get_run(mlruns, "aaaa1111")

    assert detail["model_params"] == {"learning_rate": 0.0567, "max_iter": 180}
    assert detail["fold_pr_aucs"] == [
        {"fold": 0, "pr_auc": pytest.approx(0.28)},
        {"fold": 1, "pr_auc": pytest.approx(0.34)},
    ]
    assert detail["params"]["tau"] == "0.7015"
    assert sorted(detail["artifacts"]) == ["pr-curve.txt", "regime-table.json"]


def test_run_detail_rejects_path_traversal(mlruns):
    # run_id reaches this from the URL, so it must never escape the root.
    with pytest.raises(KeyError):
        get_run(mlruns, "../../etc")


def test_unknown_run_raises_keyerror(mlruns):
    with pytest.raises(KeyError):
        get_run(mlruns, "nope")


def test_missing_root_raises_tournament_unavailable(tmp_path):
    with pytest.raises(TournamentUnavailable):
        list_runs(tmp_path / "does-not-exist")


def test_empty_existing_root_raises_tournament_unavailable(tmp_path):
    # Docker Compose auto-creates a bind-mount source directory that doesn't
    # exist on the host as an empty directory before the container starts --
    # so a fresh clone (tournament never run) mounts an empty, EXISTING
    # directory, not a missing one. Both must report the same signal, or a
    # fresh clone would misleadingly say "ran, found nothing" instead of
    # "not wired up, run scripts/run_experiments.py".
    root = tmp_path / "mlruns"
    root.mkdir()

    with pytest.raises(TournamentUnavailable):
        list_runs(root)
    with pytest.raises(TournamentUnavailable):
        get_run(root, "any-run-id")


def test_run_missing_optional_metrics_is_not_fatal(tmp_path):
    root = tmp_path / "mlruns"
    experiment = root / "1"
    experiment.mkdir(parents=True)
    _write_run(experiment, "cccc3333", run_name="baseline-run", status=4)

    runs = list_runs(root)

    assert len(runs) == 1
    assert runs[0]["aggregate_pr_auc"] is None
    assert runs[0]["status"] == "FAILED"
    assert runs[0]["model_family"] is None
