from pathlib import Path
import json

import mlflow
import numpy as np
import pandas as pd
import yaml

from btcspiker_ml.config import load_experiment_config
from btcspiker_ml.search import SearchState, run_stage
from scripts.run_experiments import _roundtrip_feature_parity, build_stage_trials


def _write_config(tmp_path: Path, *, linear_trials: int = 4) -> tuple[Path, dict]:
    dataset = tmp_path / "features.parquet"
    rows = 1_600
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=rows, freq="s", tz="UTC"),
        "vol_spike": np.arange(rows) % 2,
        "future_vol_60s": rng.random(rows),
        "log_return": rng.normal(size=rows),
        "spread_bps": rng.random(rows),
    })
    frame.to_parquet(dataset)
    raw = {
        "storage": {
            "existing_data": str(dataset), "artifact_root": str(tmp_path / "artifacts"),
            "local_cache": str(tmp_path / "cache"), "local_cache_max_gib": 1,
        },
        "target": {"name": "vol_spike_v1", "horizon_seconds": 60, "volatility_threshold": 0.1, "price_field": "price"},
        "validation": {
            "folds": 2, "final_holdout_fraction": 0.2, "max_feature_lookback_seconds": 1,
            "bootstrap_block_minutes": 1, "bootstrap_resamples": 10, "random_seed": 42,
        },
        "search": {"max_hours": 1, "max_parallel_jobs": 2, "stage_trials": {"linear": linear_trials, "trees": 7, "ablation": 3, "ensemble": 2, "neural": 1}},
        "feature_set_id": "core_v1", "feature_columns": ["log_return", "spread_bps"],
        "logistic_params": {"max_iter": 50},
        "mlflow": {"tracking_uri": tmp_path.as_uri(), "experiment_name": "runner-test", "registered_model_name": "unused"},
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path, raw


def test_stage_trial_plans_consume_configured_budgets_families_and_search_params(tmp_path: Path):
    path, raw = _write_config(tmp_path)
    config = load_experiment_config(path)

    linear = build_stage_trials(config, raw, "linear")
    trees = build_stage_trials(config, raw, "trees")
    ablation = build_stage_trials(config, raw, "ablation")
    ensemble = build_stage_trials(config, raw, "ensemble")

    assert len(linear) == 4
    assert {trial["model_family"] for trial in linear} == {"logistic", "sgd_logistic"}
    assert len(trees) == 7
    assert {trial["model_family"] for trial in trees} >= {"random_forest", "extra_trees", "hist_gradient_boosting"}
    assert len(ablation) == 3
    assert {trial["removed_feature"] for trial in ablation} <= set(raw["feature_columns"])
    assert len(ensemble) == 2
    assert all(0.0 < trial["params"]["tree_weight"] < 1.0 for trial in ensemble)
    assert all(trial["params"] for trial in linear + trees)


def test_feature_parity_compares_refitted_model_with_serialized_runtime_model():
    from sklearn.dummy import DummyClassifier

    model = DummyClassifier(strategy="prior").fit([[0.0], [1.0]], [0, 1])
    assert _roundtrip_feature_parity(model, pd.DataFrame({"signal": [0.25, 0.75]}))


def test_development_winner_is_registration_ready_without_opening_holdout(tmp_path: Path):
    path, raw = _write_config(tmp_path, linear_trials=1)
    config = load_experiment_config(path)
    trials = build_stage_trials(config, raw, "linear")
    values = {
        "tracking_uri": config.mlflow.tracking_uri, "experiment_name": config.mlflow.experiment_name,
        "state_dir": tmp_path / "state", "search_id": "search-registration-ready", "git_sha": "abc",
        "target_version": config.target.name, "validation_version": "purged_walkforward_v1",
        "deployable": False, "max_hours": config.search.max_hours, "trials": trials,
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
    assert json.loads(winner.data.params["feature_columns"]) == ["log_return", "spread_bps"]
    assert winner.data.params["feature_set_id"] == "core_v1"
    assert winner.data.params["feature_schema_version"] == "1"
    assert 0.0 <= float(winner.data.params["tau"]) <= 1.0
    artifact_paths = {item.path for item in client.list_artifacts(winner_id)}
    assert {"model", "dataset-manifest.json", "feature-manifest.json", "fold-boundaries.json", "oof-predictions.csv", "resource-timing.json"} <= artifact_paths
    model = mlflow.sklearn.load_model(f"runs:/{winner_id}/model")
    assert model.predict_proba(pd.DataFrame({"log_return": [0.0], "spread_bps": [0.1]})).shape == (1, 2)
    assert state.final_holdout_opened is False
