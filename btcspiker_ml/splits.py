"""Leakage-resistant temporal validation splits."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from btcspiker_ml.config import ValidationConfig


@dataclass(frozen=True)
class TemporalFold:
    train: list[int]
    validation: list[int]


@dataclass(frozen=True)
class SplitPlan:
    folds: tuple[TemporalFold, ...]
    final_holdout: list[int]


def make_temporal_splits(
    timestamps: Sequence[object],
    config: ValidationConfig | None = None,
    *,
    folds: int | None = None,
    final_holdout_fraction: float | None = None,
    embargo_seconds: int | None = None,
    event_keys: Sequence[object] | None = None,
    targets: Sequence[int] | None = None,
) -> SplitPlan:
    """Return expanding-window folds and a final, never-used time holdout."""
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    n_rows = len(ts)
    if n_rows == 0:
        raise ValueError("timestamps cannot be empty")
    if event_keys is not None and len(event_keys) != n_rows:
        raise ValueError("event_keys must have one value per timestamp")
    if ts.has_duplicates and event_keys is None:
        raise ValueError("duplicate timestamps require stable event_keys")

    if config is not None:
        folds = config.folds if folds is None else folds
        final_holdout_fraction = (
            config.final_holdout_fraction if final_holdout_fraction is None else final_holdout_fraction
        )
        embargo_seconds = config.max_feature_lookback_seconds + 60 if embargo_seconds is None else embargo_seconds
    if folds is None or final_holdout_fraction is None or embargo_seconds is None:
        raise TypeError("provide config or folds, final_holdout_fraction, and embargo_seconds")
    if folds < 1 or not 0 < final_holdout_fraction < 1 or embargo_seconds < 0:
        raise ValueError("invalid temporal split configuration")

    keys = np.arange(n_rows) if event_keys is None else np.asarray(event_keys)
    order = np.lexsort((keys, ts.asi8))
    ordered_ts = ts[order]
    development_end = n_rows - int(np.ceil(n_rows * final_holdout_fraction))
    embargo = pd.Timedelta(seconds=embargo_seconds)
    first_validation_start = int(np.searchsorted(ordered_ts.asi8, (ordered_ts[0] + embargo).value, side="left"))
    available_validation_rows = development_end - first_validation_start
    chunk = available_validation_rows // folds
    if first_validation_start == 0 or chunk == 0:
        raise ValueError("not enough development rows for requested folds")

    ordered_targets = None if targets is None else np.asarray(targets)[order]
    if ordered_targets is not None and len(ordered_targets) != n_rows:
        raise ValueError("targets must have one value per timestamp")
    result: list[TemporalFold] = []
    for fold_number in range(folds):
        validation_start = first_validation_start + fold_number * chunk
        validation_end = development_end if fold_number == folds - 1 else validation_start + chunk
        boundary = ordered_ts[validation_start] - embargo
        train_positions = np.flatnonzero(ordered_ts[:validation_start] <= boundary)
        validation_positions = np.arange(validation_start, validation_end)
        if not len(train_positions) or not len(validation_positions):
            raise ValueError("fold has no training or validation rows after embargo")
        if ordered_targets is not None:
            if len(np.unique(ordered_targets[train_positions])) < 2 or len(np.unique(ordered_targets[validation_positions])) < 2:
                raise ValueError("each fold must contain both target classes")
        result.append(TemporalFold(train=order[train_positions].tolist(), validation=order[validation_positions].tolist()))

    return SplitPlan(folds=tuple(result), final_holdout=order[development_end:].tolist())
