from pathlib import Path

from btcspiker_ml.config import load_experiment_config


def test_loads_frozen_target_and_validation_contract():
    cfg = load_experiment_config(Path("experiment.yaml"))
    assert cfg.target.horizon_seconds == 60
    assert cfg.target.volatility_threshold == 0.000048
    assert cfg.target.price_field == "price"
    assert cfg.validation.folds == 5
    assert cfg.validation.final_holdout_fraction == 0.20
    assert cfg.validation.max_feature_lookback_seconds == 300
    assert cfg.search.max_hours == 24
    assert cfg.storage.local_cache_max_gib == 4
    assert cfg.storage.existing_data == Path("data/processed/features.parquet")
