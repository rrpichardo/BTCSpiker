import importlib.util

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from btcspiker_ml.models import build_model


def _regime_windows(n_rows: int, *, period: int, noise: float, seed: int):
    """Rows whose label depends on a slow regime visible only across a window.

    Each row's own feature value is dominated by noise, so a row-independent
    classifier cannot beat chance. The regime sign only becomes visible once
    several consecutive rows are averaged together.
    """
    rng = np.random.default_rng(seed)
    block = rng.choice([-1.0, 1.0], size=n_rows // period + 1)
    regime = np.repeat(block, period)[:n_rows]
    x = regime * 0.2 + rng.normal(scale=noise, size=n_rows)
    y = (regime > 0).astype(int)
    return x.reshape(-1, 1), y


def test_sequence_classifier_returns_calibrated_shaped_probabilities():
    x, y = _regime_windows(400, period=10, noise=1.0, seed=1)

    model = build_model("neural", {"hidden_width": 8, "epochs": 3}, seed=42, n_jobs=1)
    model.fit(x, y)
    probabilities = model.predict_proba(x)[:, 1]

    assert probabilities.shape == (400,)
    assert np.all(np.isfinite(probabilities))
    assert np.all((0 <= probabilities) & (probabilities <= 1))


def test_sequence_classifier_handles_history_shorter_than_window():
    x, y = _regime_windows(3, period=10, noise=1.0, seed=1)

    model = build_model("neural", {"hidden_width": 4, "epochs": 2, "window": 20}, seed=42, n_jobs=1)
    model.fit(x, y)
    probabilities = model.predict_proba(x)[:, 1]

    assert probabilities.shape == (3,)
    assert np.all((0 <= probabilities) & (probabilities <= 1))


def test_sequence_classifier_beats_row_independent_model_on_window_only_signal():
    """The whole point of a sequence model: use history a single row can't show."""
    x_train, y_train = _regime_windows(1200, period=15, noise=1.2, seed=7)
    x_test, y_test = _regime_windows(400, period=15, noise=1.2, seed=8)

    sequence_model = build_model("neural", {"hidden_width": 16, "epochs": 12, "window": 15}, seed=42, n_jobs=1)
    sequence_model.fit(x_train, y_train)
    sequence_auc = roc_auc_score(y_test, sequence_model.predict_proba(x_test)[:, 1])

    row_baseline = LogisticRegression().fit(x_train, y_train)
    baseline_auc = roc_auc_score(y_test, row_baseline.predict_proba(x_test)[:, 1])

    assert baseline_auc < 0.62, f"baseline should be near chance on window-only signal, got {baseline_auc}"
    assert sequence_auc > baseline_auc + 0.15, (
        f"sequence model ({sequence_auc}) should clearly beat the row-independent baseline ({baseline_auc})"
    )


def test_sequence_classifier_respects_configured_thread_budget():
    import torch

    x, y = _regime_windows(60, period=10, noise=1.0, seed=1)
    model = build_model("neural", {"hidden_width": 4, "epochs": 1}, seed=42, n_jobs=1)

    model.fit(x, y)

    assert torch.get_num_threads() == 1


def test_build_model_neural_requires_torch(monkeypatch):
    import btcspiker_ml.models as models

    monkeypatch.setattr(models.importlib.util, "find_spec", lambda name: None)

    with pytest.raises(ImportError, match="neural"):
        build_model("neural", {}, seed=42, n_jobs=1)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_build_model_neural_uses_sequence_classifier_not_mlp():
    from btcspiker_ml.neural import SequenceWindowClassifier

    model = build_model("neural", {}, seed=42, n_jobs=1)

    assert isinstance(model.named_steps["model"], SequenceWindowClassifier)
