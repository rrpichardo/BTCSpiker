import pandas as pd
import pytest

from btcspiker_ml.config import ValidationConfig
from btcspiker_ml.splits import make_temporal_splits


def _binary_targets(length):
    return [index % 2 for index in range(length)]


def test_every_fold_has_required_embargo_and_isolated_holdout():
    timestamps = pd.date_range("2026-01-01", periods=10_000, freq="s", tz="UTC")
    plan = make_temporal_splits(
        timestamps,
        folds=5,
        final_holdout_fraction=0.20,
        embargo_seconds=360,
        targets=_binary_targets(len(timestamps)),
    )

    assert len(plan.folds) == 5
    for fold in plan.folds:
        assert timestamps[fold.validation[0]] - timestamps[
            fold.train[-1]
        ] >= pd.Timedelta(seconds=360)
    used = {index for fold in plan.folds for index in fold.train + fold.validation}
    assert set(plan.final_holdout).isdisjoint(used)


def test_config_derives_embargo_from_lookback_plus_target_horizon():
    timestamps = pd.date_range("2026-01-01", periods=1_000, freq="s", tz="UTC")
    config = ValidationConfig(2, 0.20, 300, 30, 20, 42)

    plan = make_temporal_splits(
        timestamps, config, targets=_binary_targets(len(timestamps))
    )

    for fold in plan.folds:
        assert timestamps[fold.validation[0]] - timestamps[
            fold.train[-1]
        ] >= pd.Timedelta(seconds=360)


def test_duplicate_timestamps_require_stable_event_keys():
    timestamps = pd.to_datetime(
        ["2026-01-01T00:00:00Z"] * 2
        + [f"2026-01-01T00:{minute:02d}:00Z" for minute in range(1, 21)]
    )

    with pytest.raises(ValueError, match="duplicate"):
        make_temporal_splits(
            timestamps,
            folds=2,
            final_holdout_fraction=0.2,
            embargo_seconds=1,
            targets=_binary_targets(len(timestamps)),
        )


def test_stable_event_keys_allow_duplicate_timestamps():
    timestamps = pd.to_datetime(
        ["2026-01-01T00:00:00Z"] * 2
        + [f"2026-01-01T00:{minute:02d}:00Z" for minute in range(1, 21)]
    )

    plan = make_temporal_splits(
        timestamps,
        folds=2,
        final_holdout_fraction=0.2,
        embargo_seconds=1,
        event_keys=list(range(len(timestamps))),
        targets=_binary_targets(len(timestamps)),
    )

    assert len(plan.folds) == 2


def test_splits_require_targets_to_validate_development_folds():
    timestamps = pd.date_range("2026-01-01", periods=100, freq="s", tz="UTC")

    with pytest.raises(ValueError, match="targets are required"):
        make_temporal_splits(
            timestamps, folds=2, final_holdout_fraction=0.2, embargo_seconds=1
        )


def test_splits_reject_single_class_folds():
    timestamps = pd.date_range("2026-01-01", periods=100, freq="s", tz="UTC")

    with pytest.raises(ValueError, match="both target classes"):
        make_temporal_splits(
            timestamps,
            folds=2,
            final_holdout_fraction=0.2,
            embargo_seconds=1,
            targets=[0] * len(timestamps),
        )


def test_first_fold_is_not_starved_of_training_data():
    """An expanding window must start from a usable base.

    Anchoring the first validation window to a bare embargo interval left
    fold 0 training on minutes of history while predicting the rest of the
    corpus, so it lost to a constant predictor no matter how much data the
    run was given.
    """
    timestamps = pd.date_range("2026-01-01", periods=60_000, freq="s", tz="UTC")

    plan = make_temporal_splits(
        timestamps,
        folds=5,
        final_holdout_fraction=0.20,
        embargo_seconds=360,
        targets=_binary_targets(len(timestamps)),
    )

    development_rows = len(timestamps) - len(plan.final_holdout)
    assert len(plan.folds[0].train) >= development_rows * 0.10
