from pathlib import Path
import json
from types import SimpleNamespace

import mlflow
import numpy as np
import pytest
import pandas as pd
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.metrics import brier_score_loss

from btcspiker_ml.config import load_experiment_config
from btcspiker_ml.search import SearchState, run_stage
from scripts.run_experiments import (
    _TemporalCalibrationSplit,
    _candidate_model,
    _roundtrip_feature_parity,
    build_stage_trials,
)


def _write_config(
    tmp_path: Path, *, linear_trials: int = 4, drifting_prevalence: bool = False
) -> tuple[Path, dict]:
    dataset = tmp_path / "features.parquet"
    rows = 1_600
    rng = np.random.default_rng(42)
    if drifting_prevalence:
        # Real corpora do not hold a constant event rate across folds; a uniform
        # target hides reference errors that only appear when prevalence moves.
        target = rng.binomial(1, np.linspace(0.15, 0.55, rows))
    else:
        target = np.arange(rows) % 2
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="s", tz="UTC"),
            "vol_spike": target,
            "future_vol_60s": rng.random(rows),
            "log_return": rng.normal(size=rows),
            "spread_bps": rng.random(rows),
        }
    )
    frame.to_parquet(dataset)
    raw = {
        "storage": {
            "existing_data": str(dataset),
            "artifact_root": str(tmp_path / "artifacts"),
            "local_cache": str(tmp_path / "cache"),
            "local_cache_max_gib": 1,
        },
        "target": {
            "name": "vol_spike_v1",
            "horizon_seconds": 60,
            "volatility_threshold": 0.1,
            "price_field": "price",
        },
        "validation": {
            "folds": 2,
            "final_holdout_fraction": 0.2,
            "max_feature_lookback_seconds": 1,
            "bootstrap_block_minutes": 1,
            "bootstrap_resamples": 10,
            "random_seed": 42,
        },
        "search": {
            "max_hours": 1,
            "max_parallel_jobs": 2,
            "stage_trials": {
                "linear": linear_trials,
                "trees": 7,
                "ablation": 3,
                "ensemble": 2,
                "neural": 1,
            },
        },
        "feature_set_id": "core_v1",
        "feature_columns": ["log_return", "spread_bps"],
        "logistic_params": {"max_iter": 50},
        "mlflow": {
            "tracking_uri": tmp_path.as_uri(),
            "experiment_name": "runner-test",
            "registered_model_name": "unused",
        },
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path, raw


def test_stage_trial_plans_consume_configured_budgets_families_and_search_params(
    tmp_path: Path,
):
    path, raw = _write_config(tmp_path)
    config = load_experiment_config(path)

    linear = build_stage_trials(config, raw, "linear")
    trees = build_stage_trials(config, raw, "trees")
    ablation = build_stage_trials(config, raw, "ablation")
    ensemble = build_stage_trials(config, raw, "ensemble")

    assert len(linear) == 4
    assert {trial["model_family"] for trial in linear} == {"logistic", "sgd_logistic"}
    assert len(trees) == 7
    assert {trial["model_family"] for trial in trees} >= {
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
    }
    assert len(ablation) == 3
    assert {trial["removed_feature"] for trial in ablation} <= set(
        raw["feature_columns"]
    )
    assert len(ensemble) == 2
    assert all(0.0 < trial["params"]["tree_weight"] < 1.0 for trial in ensemble)
    assert all(trial["params"] for trial in linear + trees)


def test_neural_trials_run_when_the_isolated_neural_environment_is_available(
    tmp_path: Path, monkeypatch
):
    path, raw = _write_config(tmp_path)
    config = load_experiment_config(path)
    import scripts.run_experiments as runner

    original_find_spec = runner.importlib.util.find_spec
    monkeypatch.setattr(
        runner.importlib.util,
        "find_spec",
        lambda name: object() if name == "torch" else original_find_spec(name),
    )

    trial = build_stage_trials(config, raw, "neural")[0]

    assert "outcome" not in trial
    assert callable(trial["evaluate"])


def test_cli_passes_configured_parallel_trial_limit_to_search(
    tmp_path: Path, monkeypatch
):
    path, _raw = _write_config(tmp_path, linear_trials=1)
    import scripts.run_experiments as runner

    received: dict[str, object] = {}
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_experiments.py",
            "--config",
            str(path),
            "--dataset-id",
            "d1",
            "--stage",
            "linear",
        ],
    )
    monkeypatch.setattr(
        runner.subprocess, "check_output", lambda *_args, **_kwargs: "abc\n"
    )
    monkeypatch.setattr(
        runner,
        "run_stage",
        lambda values, *_args: received.update(values)
        or SimpleNamespace(status="completed", parent_run_id="parent"),
    )

    assert runner.main() == 0
    assert received["max_parallel_jobs"] == 2


def test_parallel_trials_do_not_start_nested_estimator_pools(tmp_path: Path):
    path, raw = _write_config(tmp_path)
    config = load_experiment_config(path)

    model = _candidate_model(
        {"model_family": "random_forest", "params": {"n_estimators": 10}},
        config,
    )

    calibrator = model.named_steps["model"]
    assert calibrator.n_jobs == 1
    assert calibrator.estimator.n_jobs == 1


def test_feature_parity_compares_refitted_model_with_serialized_runtime_model():
    from sklearn.dummy import DummyClassifier

    model = DummyClassifier(strategy="prior").fit([[0.0], [1.0]], [0, 1])
    assert _roundtrip_feature_parity(model, pd.DataFrame({"signal": [0.25, 0.75]}))


def test_development_winner_is_registration_ready_without_opening_holdout(
    tmp_path: Path,
):
    path, raw = _write_config(tmp_path, linear_trials=1)
    config = load_experiment_config(path)
    trials = build_stage_trials(config, raw, "linear")
    values = {
        "tracking_uri": config.mlflow.tracking_uri,
        "experiment_name": config.mlflow.experiment_name,
        "state_dir": tmp_path / "state",
        "search_id": "search-registration-ready",
        "git_sha": "abc",
        "target_version": config.target.name,
        "validation_version": "purged_walkforward_v1",
        "deployable": False,
        "max_hours": config.search.max_hours,
        "trials": trials,
        "experiment_config": raw,
    }

    run_stage(values, "dataset-1", "core_v1", "linear")
    state = SearchState.load(tmp_path / "state" / "search-registration-ready.json")
    winner_id = state.best_run_ids["linear"]
    client = mlflow.tracking.MlflowClient(tmp_path.as_uri())
    winner = client.get_run(winner_id)

    assert winner.data.tags["deployable"] == "true"
    assert winner.data.tags["feature_parity_passed"] == "true"
    assert winner.data.params["feature_cols"] == "log_return,spread_bps"
    assert json.loads(winner.data.params["feature_columns"]) == [
        "log_return",
        "spread_bps",
    ]
    assert winner.data.params["feature_set_id"] == "core_v1"
    assert winner.data.params["feature_schema_version"] == "1"
    assert 0.0 <= float(winner.data.params["tau"]) <= 1.0
    artifact_paths = {item.path for item in client.list_artifacts(winner_id)}
    assert {
        "model",
        "dataset-manifest.json",
        "feature-manifest.json",
        "fold-boundaries.json",
        "oof-predictions.csv",
        "resource-timing.json",
    } <= artifact_paths
    model = mlflow.sklearn.load_model(f"runs:/{winner_id}/model")
    assert model.predict_proba(
        pd.DataFrame({"log_return": [0.0], "spread_bps": [0.1]})
    ).shape == (1, 2)
    assert state.final_holdout_opened is False


def _regime_shift_frame(
    rows: int = 6_000, seed: int = 7
) -> tuple[pd.DataFrame, np.ndarray, int]:
    """Rare events whose driver weakens in the later window.

    Reproduces the rehearsal's calibration failure: the features drift and the
    signal-to-target coupling decays, so a model fitted on the earlier window
    stays confident while it stops being right.
    """
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=rows)
    cut = int(rows * 0.75)
    signal[cut:] += 1.2
    probability = 1.0 / (1.0 + np.exp(-(0.9 * signal - 2.4)))
    probability[cut:] = 1.0 / (1.0 + np.exp(-(0.3 * signal[cut:] - 2.4)))
    target = rng.binomial(1, probability)
    frame = pd.DataFrame({"log_return": signal, "spread_bps": rng.normal(size=rows)})
    return frame, target, cut


def test_calibration_split_trains_before_the_rows_it_calibrates_on():
    split = _TemporalCalibrationSplit(0.25)

    fit_rows, calibration_rows = next(split.split(np.zeros((100, 2))))

    assert fit_rows.max() < calibration_rows.min()
    assert len(calibration_rows) == 25
    assert split.get_n_splits() == 1


def test_deployable_candidate_probabilities_do_not_regress_brier_against_prevalence(
    tmp_path: Path,
):
    path, _raw = _write_config(tmp_path)
    config = load_experiment_config(path)
    frame, target, cut = _regime_shift_frame()
    train_x, train_y = frame.iloc[:cut], target[:cut]
    test_x, test_y = frame.iloc[cut:], target[cut:]

    candidate = _candidate_model(
        {
            "model_family": "extra_trees",
            "params": {"n_estimators": 5, "min_samples_leaf": 1},
        },
        config,
    )
    candidate.fit(train_x, train_y)
    candidate_brier = brier_score_loss(test_y, candidate.predict_proba(test_x)[:, 1])
    prevalence_brier = brier_score_loss(
        test_y, np.full(len(test_y), float(train_y.mean()))
    )

    assert candidate_brier / prevalence_brier <= 1.05


def test_development_trials_report_calibration_against_the_prevalence_baseline(
    tmp_path: Path,
):
    path, raw = _write_config(tmp_path, linear_trials=1)
    config = load_experiment_config(path)
    trial = build_stage_trials(config, raw, "linear")[0]

    metrics = trial["evaluate"]()["metrics"]

    assert metrics["development_brier_ratio"] > 0.0


def test_prevalence_baseline_scores_a_neutral_calibration_ratio(tmp_path: Path):
    """The baseline is the reference, so it must measure as neither better nor worse."""
    path, raw = _write_config(tmp_path, linear_trials=1, drifting_prevalence=True)
    config = load_experiment_config(path)
    trial = build_stage_trials(config, raw, "baseline")[0]

    metrics = trial["evaluate"]()["metrics"]

    assert metrics["development_brier_ratio"] == pytest.approx(1.0, abs=0.01)


def test_development_trials_report_how_many_folds_beat_prevalence(tmp_path: Path):
    path, raw = _write_config(tmp_path, linear_trials=1)
    config = load_experiment_config(path)
    trial = build_stage_trials(config, raw, "linear")[0]

    metrics = trial["evaluate"]()["metrics"]

    assert 0 <= metrics["development_folds_won"] <= config.validation.folds


def test_prevalence_baseline_is_never_wrapped_in_calibration(tmp_path: Path):
    path, _raw = _write_config(tmp_path)
    config = load_experiment_config(path)

    baseline = _candidate_model({"model_family": "development_prevalence"}, config)

    assert isinstance(baseline.named_steps["model"], DummyClassifier)
