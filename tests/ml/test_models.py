import importlib.util

import numpy as np
import pytest

from btcspiker_ml.models import build_model, model_families, suggest_params


@pytest.mark.parametrize("family", model_families("all"))
def test_every_tabular_model_returns_probabilities(family):
    if family in {"lightgbm", "xgboost", "catboost"} and importlib.util.find_spec(family) is None:
        pytest.skip(f"optional {family} package is not installed")

    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    model = build_model(family, {}, seed=42, n_jobs=1)
    model.fit(X, y)
    probabilities = model.predict_proba(X)[:, 1]

    assert probabilities.shape == (4,)
    assert np.all(np.isfinite(probabilities))
    assert np.all((0 <= probabilities) & (probabilities <= 1))


@pytest.mark.parametrize("family", ["lightgbm", "xgboost", "catboost"])
def test_unavailable_optional_model_has_installation_message(monkeypatch, family):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    with pytest.raises(ImportError, match=family):
        build_model(family, {}, seed=42, n_jobs=1)


class _CapturingTrial:
    def __init__(self):
        self.calls = []

    def suggest_float(self, name, low, high, **kwargs):
        self.calls.append(("float", name, low, high, kwargs))
        return low

    def suggest_int(self, name, low, high, **kwargs):
        self.calls.append(("int", name, low, high, kwargs))
        return low


@pytest.mark.parametrize("family", model_families("all"))
def test_search_spaces_are_bounded(family):
    trial = _CapturingTrial()

    params = suggest_params(trial, family)

    assert params
    for kind, _name, low, high, _kwargs in trial.calls:
        assert kind in {"float", "int"}
        assert low <= high
