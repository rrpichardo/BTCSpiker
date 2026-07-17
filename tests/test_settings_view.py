"""Unit tests for api/settings_view.py.

Calls the route handler function directly rather than spinning up the app,
and never imports api.main (that triggers model loading). File-backed
sources are pointed at tmp_path via SETTINGS_ENV_FILE / SETTINGS_CONFIG_FILE.
"""

import pytest

from api import settings_view


@pytest.fixture(autouse=True)
def _stub_model_source(monkeypatch):
    # Default seam value for every test in this file, so tests that don't
    # care about the MODEL_STAGE source cross-check never trigger the real
    # `from api import main` fallback (which would load the model).
    # Tests that do care override it again within the test body.
    monkeypatch.setattr(settings_view, "_get_model_source", lambda: "mlflow")


def _settings_by_key(body: dict) -> dict:
    return {row["key"]: row for row in body["settings"]}


def test_duplicate_env_key_last_occurrence_wins(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("MODEL_VARIANT=ml\nMODEL_VARIANT=baseline\n")
    monkeypatch.setenv("SETTINGS_ENV_FILE", str(env_file))
    monkeypatch.setenv("SETTINGS_CONFIG_FILE", str(tmp_path / "missing.yaml"))

    body = settings_view.get_settings()

    assert _settings_by_key(body)["MODEL_VARIANT"]["saved_value"] == "baseline"


def test_apply_state_restart_required_and_applied(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("MODEL_VARIANT=baseline\n")
    monkeypatch.setenv("SETTINGS_ENV_FILE", str(env_file))
    monkeypatch.setenv("SETTINGS_CONFIG_FILE", str(tmp_path / "missing.yaml"))
    monkeypatch.delenv("MODEL_VARIANT", raising=False)

    # Active default is "ml", saved is "baseline" -> mismatch.
    body = settings_view.get_settings()
    row = _settings_by_key(body)["MODEL_VARIANT"]
    assert row["active_value"] == "ml"
    assert row["saved_value"] == "baseline"
    assert row["apply_state"] == "restart_required"

    # Make active match saved -> applied.
    monkeypatch.setenv("MODEL_VARIANT", "baseline")
    body = settings_view.get_settings()
    row = _settings_by_key(body)["MODEL_VARIANT"]
    assert row["active_value"] == "baseline"
    assert row["apply_state"] == "applied"


def test_model_variant_active_value_matches_main_normalization(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("MODEL_VARIANT=ml\n")
    monkeypatch.setenv("SETTINGS_ENV_FILE", str(env_file))
    monkeypatch.setenv("SETTINGS_CONFIG_FILE", str(tmp_path / "missing.yaml"))
    monkeypatch.setenv("MODEL_VARIANT", "ML")

    row = _settings_by_key(settings_view.get_settings())["MODEL_VARIANT"]

    assert row["active_value"] == "ml"
    assert row["apply_state"] == "applied"


def test_missing_files_return_nulls_and_unknown(tmp_path, monkeypatch):
    monkeypatch.setenv("SETTINGS_ENV_FILE", str(tmp_path / "nope.env"))
    monkeypatch.setenv("SETTINGS_CONFIG_FILE", str(tmp_path / "nope.yaml"))
    monkeypatch.delenv("REPLAY_SPEED", raising=False)

    body = settings_view.get_settings()
    by_key = _settings_by_key(body)

    for key in ("MODEL_VARIANT", "BASELINE_VOL_THRESHOLD", "MODEL_STAGE", "MODEL_NAME"):
        assert by_key[key]["saved_value"] is None
        assert by_key[key]["apply_state"] == "unknown"

    assert by_key["REPLAY_SPEED"]["saved_value"] is None
    assert by_key["REPLAY_SPEED"]["active_value"] is None
    assert by_key["REPLAY_SPEED"]["apply_state"] == "unknown"

    for key in ("features.window_seconds", "features.vol_threshold"):
        assert by_key[key]["saved_value"] is None
        assert by_key[key]["active_value"] is None
        assert by_key[key]["apply_state"] == "unknown"


def test_quoted_env_values_are_unquoted(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text('MODEL_NAME="btc-volatility-lr"\n' "MODEL_STAGE='Production'\n")
    monkeypatch.setenv("SETTINGS_ENV_FILE", str(env_file))
    monkeypatch.setenv("SETTINGS_CONFIG_FILE", str(tmp_path / "missing.yaml"))

    body = settings_view.get_settings()
    by_key = _settings_by_key(body)

    assert by_key["MODEL_NAME"]["saved_value"] == "btc-volatility-lr"
    assert by_key["MODEL_STAGE"]["saved_value"] == "Production"


def test_danger_flags_only_on_the_two_config_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("SETTINGS_ENV_FILE", str(tmp_path / "nope.env"))
    monkeypatch.setenv("SETTINGS_CONFIG_FILE", str(tmp_path / "nope.yaml"))

    body = settings_view.get_settings()

    danger_keys = {row["key"] for row in body["settings"] if row["danger"] is not None}
    assert danger_keys == {"features.window_seconds", "features.vol_threshold"}
    for row in body["settings"]:
        if row["key"] in danger_keys:
            assert row["danger"] == "requires_retraining"
        else:
            assert row["danger"] is None


def test_model_stage_is_inactive_when_pickle_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("SETTINGS_ENV_FILE", str(tmp_path / "nope.env"))
    monkeypatch.setenv("SETTINGS_CONFIG_FILE", str(tmp_path / "nope.yaml"))
    monkeypatch.delenv("MODEL_VARIANT", raising=False)  # active default "ml"
    monkeypatch.setattr(settings_view, "_get_model_source", lambda: "pickle")

    body = settings_view.get_settings()
    row = _settings_by_key(body)["MODEL_STAGE"]

    assert row["active_value"] is None
    assert row["apply_state"] == "unknown"


def test_model_stage_is_active_when_mlflow_source(tmp_path, monkeypatch):
    monkeypatch.setenv("SETTINGS_ENV_FILE", str(tmp_path / "nope.env"))
    monkeypatch.setenv("SETTINGS_CONFIG_FILE", str(tmp_path / "nope.yaml"))
    monkeypatch.delenv("MODEL_VARIANT", raising=False)
    monkeypatch.setattr(settings_view, "_get_model_source", lambda: "mlflow")

    body = settings_view.get_settings()
    row = _settings_by_key(body)["MODEL_STAGE"]

    assert row["active_value"] == "Production"


def test_each_setting_has_the_documented_response_fields():
    body = settings_view.get_settings()

    expected_fields = {
        "key",
        "active_value",
        "saved_value",
        "apply_state",
        "source",
        "editable_via",
        "description",
        "danger",
    }
    for row in body["settings"]:
        assert set(row) == expected_fields


def test_config_yaml_values_loaded_as_strings(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "features:\n  window_seconds: 60\n  vol_threshold: 0.000048\n"
    )
    monkeypatch.setenv("SETTINGS_ENV_FILE", str(tmp_path / "nope.env"))
    monkeypatch.setenv("SETTINGS_CONFIG_FILE", str(config_file))

    body = settings_view.get_settings()
    by_key = _settings_by_key(body)

    assert by_key["features.window_seconds"]["saved_value"] == "60"
    assert by_key["features.vol_threshold"]["saved_value"] == "4.8e-05"
    # Not observable from the api process regardless of file contents.
    assert by_key["features.window_seconds"]["active_value"] is None
    assert by_key["features.vol_threshold"]["apply_state"] == "unknown"


def test_registry_order_and_completeness():
    body = settings_view.get_settings()
    keys = [row["key"] for row in body["settings"]]
    assert keys == [
        "MODEL_VARIANT",
        "BASELINE_VOL_THRESHOLD",
        "MODEL_STAGE",
        "MODEL_NAME",
        "REPLAY_SPEED",
        "features.window_seconds",
        "features.vol_threshold",
    ]
