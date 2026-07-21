"""Row-level, alert-level, and paired temporal evaluation metrics."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class EventMetrics:
    alerts: int
    true_positive_alerts: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class MetricBundle:
    pr_auc: float
    prevalence: float
    pr_auc_lift: float
    roc_auc: float
    log_loss: float
    brier: float
    ece: float
    f1: float
    precision: float
    recall: float
    event_f1: float
    alerts_per_hour: float
    per_day: pd.DataFrame
    per_regime: pd.DataFrame


@dataclass(frozen=True)
class ConfidenceInterval:
    lower: float
    estimate: float
    upper: float


def _arrays(y: Sequence[int], scores: Sequence[float], timestamps: Sequence[object]) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    target = np.asarray(y, dtype=int)
    probability = np.asarray(scores, dtype=float)
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    if not (len(target) == len(probability) == len(ts)) or not len(target):
        raise ValueError("y, scores, and timestamps must be non-empty and aligned")
    if not np.isin(target, [0, 1]).all() or len(np.unique(target)) < 2:
        raise ValueError("y must contain both binary target classes")
    if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
        raise ValueError("scores must be finite probabilities in [0, 1]")
    return target, probability, ts


def event_metrics(y: Sequence[int], scores: Sequence[float], timestamps: Sequence[object], threshold: float, cooldown_seconds: int = 60) -> EventMetrics:
    target, probability, ts = _arrays(y, scores, timestamps)
    if cooldown_seconds < 0:
        raise ValueError("cooldown_seconds must be non-negative")
    ordered = np.argsort(ts.asi8, kind="stable")
    last_alert: pd.Timestamp | None = None
    alerts: list[int] = []
    cooldown = pd.Timedelta(seconds=cooldown_seconds)
    for position in ordered:
        if probability[position] >= threshold and (last_alert is None or ts[position] - last_alert >= cooldown):
            alerts.append(position)
            last_alert = ts[position]
    true_positive_alerts = int(target[alerts].sum()) if alerts else 0
    precision = true_positive_alerts / len(alerts) if alerts else 0.0
    recall = true_positive_alerts / int(target.sum()) if target.sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return EventMetrics(len(alerts), true_positive_alerts, precision, recall, f1)


def _ece(y: np.ndarray, scores: np.ndarray) -> float:
    bins = np.linspace(0.0, 1.0, 11)
    bin_ids = np.digitize(scores, bins[1:-1], right=True)
    return float(sum(np.mean(bin_ids == index) * abs(scores[bin_ids == index].mean() - y[bin_ids == index].mean()) for index in range(10) if np.any(bin_ids == index)))


def evaluate_predictions(y: Sequence[int], scores: Sequence[float], timestamps: Sequence[object], threshold: float, regimes: Sequence[object] | None = None) -> MetricBundle:
    target, probability, ts = _arrays(y, scores, timestamps)
    if regimes is not None and len(regimes) != len(target):
        raise ValueError("regimes must have one value per prediction")
    predicted = probability >= threshold
    events = event_metrics(target, probability, ts, threshold)
    prevalence = float(target.mean())
    frame = pd.DataFrame({"y": target, "score": probability}, index=ts)
    per_day = frame.groupby(frame.index.normalize()).apply(lambda group: pd.Series({"prevalence": group.y.mean(), "pr_auc": average_precision_score(group.y, group.score) if group.y.nunique() == 2 else np.nan}), include_groups=False)
    if regimes is None:
        per_regime = pd.DataFrame({"prevalence": [prevalence], "pr_auc": [average_precision_score(target, probability)]}, index=pd.Index(["all"], name="regime"))
    else:
        frame["regime"] = list(regimes)
        per_regime = frame.groupby("regime").apply(lambda group: pd.Series({"prevalence": group.y.mean(), "pr_auc": average_precision_score(group.y, group.score) if group.y.nunique() == 2 else np.nan}), include_groups=False)
    hours = max((ts.max() - ts.min()).total_seconds() / 3600, 1 / 3600)
    pr_auc = float(average_precision_score(target, probability))
    return MetricBundle(pr_auc, prevalence, pr_auc / prevalence, float(roc_auc_score(target, probability)), float(log_loss(target, probability, labels=[0, 1])), float(brier_score_loss(target, probability)), _ece(target, probability), float(f1_score(target, predicted, zero_division=0)), float(precision_score(target, predicted, zero_division=0)), float(recall_score(target, predicted, zero_division=0)), events.f1, events.alerts / hours, per_day, per_regime)


def paired_block_bootstrap(y_true: Sequence[int], candidate_scores: Sequence[float], baseline_scores: Sequence[float], timestamps: Sequence[object], block_minutes: int = 30, resamples: int = 2000, seed: int = 42) -> ConfidenceInterval:
    target, candidate, ts = _arrays(y_true, candidate_scores, timestamps)
    baseline = np.asarray(baseline_scores, dtype=float)
    if len(baseline) != len(target) or not np.isfinite(baseline).all() or (baseline < 0).any() or (baseline > 1).any():
        raise ValueError("baseline_scores must be aligned probabilities in [0, 1]")
    if block_minutes <= 0 or resamples <= 0:
        raise ValueError("block_minutes and resamples must be positive")
    blocks = pd.Series(np.arange(len(target))).groupby(ts.floor(f"{block_minutes}min"), sort=True).apply(np.asarray).tolist()
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(resamples):
        sampled_block_indices = rng.integers(0, len(blocks), size=len(blocks))
        indices = np.concatenate([blocks[index] for index in sampled_block_indices])
        deltas.append(average_precision_score(target[indices], candidate[indices]) - average_precision_score(target[indices], baseline[indices]))
    return ConfidenceInterval(float(np.percentile(deltas, 2.5)), float(np.mean(deltas)), float(np.percentile(deltas, 97.5)))
