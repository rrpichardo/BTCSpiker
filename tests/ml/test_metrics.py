import numpy as np
import pandas as pd
import pytest

from btcspiker_ml.metrics import event_metrics, evaluate_predictions, paired_block_bootstrap


def test_event_metrics_apply_sixty_second_cooldown():
    ts = pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:00:10Z", "2026-01-01T00:02:00Z"])

    result = event_metrics(np.array([1, 1, 0]), np.array([0.9, 0.8, 0.9]), ts, threshold=0.5, cooldown_seconds=60)

    assert result.alerts == 2
    assert result.true_positive_alerts == 1


def test_evaluate_predictions_reports_calibration_and_time_tables():
    timestamps = pd.date_range("2026-01-01", periods=8, freq="12h", tz="UTC")
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.4, 0.7, 0.3, 0.6])

    result = evaluate_predictions(y, scores, timestamps, threshold=0.5)

    assert result.pr_auc == pytest.approx(1.0)
    assert result.prevalence == pytest.approx(0.5)
    assert result.pr_auc_lift == pytest.approx(2.0)
    assert result.ece == pytest.approx(0.25)
    assert set(result.per_day.index) == {pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-01-02", tz="UTC"), pd.Timestamp("2026-01-03", tz="UTC"), pd.Timestamp("2026-01-04", tz="UTC")}
    assert "all" in result.per_regime.index


def test_paired_block_bootstrap_is_deterministic_and_reports_improvement():
    timestamps = pd.date_range("2026-01-01", periods=12, freq="10min", tz="UTC")
    y = np.array([0, 1] * 6)
    candidate = np.array([0.2, 0.9] * 6)
    baseline = np.array([0.6, 0.5] * 6)

    first = paired_block_bootstrap(y, candidate, baseline, timestamps, 30, 100, 42)
    second = paired_block_bootstrap(y, candidate, baseline, timestamps, 30, 100, 42)

    assert first == second
    assert first.lower > 0
    assert first.lower <= first.estimate <= first.upper
