"""Regression coverage for the legacy MLflow registry bootstrap."""

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "log_model_to_mlflow.py"


def _bootstrap_module():
    spec = importlib.util.spec_from_file_location("mlflow_bootstrap", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Client:
    def __init__(self):
        self.transitions = []

    def transition_model_version_stage(self, *args, **kwargs):
        self.transitions.append((args, kwargs))


def test_legacy_bootstrap_promotes_its_registered_version_to_production(monkeypatch):
    bootstrap = _bootstrap_module()
    client = _Client()
    monkeypatch.setattr(bootstrap, "client", client)

    bootstrap._ensure_legacy_bootstrap_production("btc-volatility-lr", "7")

    assert client.transitions == [
        (("btc-volatility-lr", "7", "Production"), {"archive_existing_versions": True})
    ]


def test_bootstrap_never_promotes_a_candidate_model(monkeypatch):
    bootstrap = _bootstrap_module()
    client = _Client()
    monkeypatch.setattr(bootstrap, "client", client)

    bootstrap._ensure_legacy_bootstrap_production("btc-volatility-candidate", "7")

    assert client.transitions == []
