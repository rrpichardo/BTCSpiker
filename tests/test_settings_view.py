"""Unit tests for api/settings_view.py.

Calls the route handler function directly rather than spinning up the app,
and never imports api.main (that triggers model loading). File-backed
sources are pointed at tmp_path via SETTINGS_ENV_FILE / SETTINGS_CONFIG_FILE.
"""

from types import SimpleNamespace

import api
import pytest
from fastapi import HTTPException

from api import settings_view

_REAL_GET_ACTIVE_API_SETTINGS = settings_view._get_active_api_settings


@pytest.fixture(autouse=True)
def _stub_active_api_settings(monkeypatch):
    # Avoid importing api.main (and loading a model) in unit tests. Individual
    # tests replace this snapshot when they exercise a different startup state.
    monkeypatch.setattr(
        settings_view,
        "_get_active_api_settings",
        lambda: {
            "MODEL_VARIANT": "ml",
            "BASELINE_VOL_THRESHOLD": "4.8e-05",
            "MODEL_STAGE": "Production",
            "MODEL_NAME": "btc-volatility-lr",
        },
    )


def _settings_by_key(body: dict) -> dict:
    return {row["key"]: row for row in body["settings"]}


def test_active_api_settings_come_from_loaded_main_constants(monkeypatch):
    loaded_main = SimpleNamespace(
        MODEL_VARIANT="baseline",
        BASELINE_VOL_THRESHOLD=0.25,
        MODEL_STAGE="Staging",
        MODEL_NAME="loaded-model",
        model_source="pickle",
    )
    monkeypatch.setattr(api, "main", loaded_main, raising=False)
    monkeypatch.setenv("MODEL_VARIANT", "ml")
    monkeypatch.setenv("MODEL_STAGE", "Production")

    snapshot = _REAL_GET_ACTIVE_API_SETTINGS()

    assert snapshot == {
        "MODEL_VARIANT": "baseline",
        "BASELINE_VOL_THRESHOLD": "0.25",
        "MODEL_STAGE": None,
        "MODEL_NAME": "loaded-model",
    }


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

    # Startup snapshot is "ml", saved is "baseline" -> mismatch.
    body = settings_view.get_settings()
    row = _settings_by_key(body)["MODEL_VARIANT"]
    assert row["active_value"] == "ml"
    assert row["saved_value"] == "baseline"
    assert row["apply_state"] == "restart_required"

    # Replace the startup snapshot with one that matches saved -> applied.
    monkeypatch.setattr(
        settings_view,
        "_get_active_api_settings",
        lambda: {
            "MODEL_VARIANT": "baseline",
            "BASELINE_VOL_THRESHOLD": "4.8e-05",
            "MODEL_STAGE": None,
            "MODEL_NAME": "btc-volatility-lr",
        },
    )
    body = settings_view.get_settings()
    row = _settings_by_key(body)["MODEL_VARIANT"]
    assert row["active_value"] == "baseline"
    assert row["apply_state"] == "applied"


def test_environment_mutation_does_not_change_active_startup_value(
    tmp_path, monkeypatch
):
    env_file = tmp_path / ".env"
    env_file.write_text("MODEL_VARIANT=ml\n")
    monkeypatch.setenv("SETTINGS_ENV_FILE", str(env_file))
    monkeypatch.setenv("SETTINGS_CONFIG_FILE", str(tmp_path / "missing.yaml"))
    monkeypatch.setenv("MODEL_VARIANT", "baseline")

    row = _settings_by_key(settings_view.get_settings())["MODEL_VARIANT"]

    assert row["active_value"] == "ml"
    assert row["apply_state"] == "applied"


def test_saved_model_variant_is_compared_canonically(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("MODEL_VARIANT=ML\n")
    monkeypatch.setenv("SETTINGS_ENV_FILE", str(env_file))
    monkeypatch.setenv("SETTINGS_CONFIG_FILE", str(tmp_path / "missing.yaml"))

    row = _settings_by_key(settings_view.get_settings())["MODEL_VARIANT"]

    assert row["saved_value"] == "ML"
    assert row["active_value"] == "ml"
    assert row["apply_state"] == "applied"


def test_saved_numeric_threshold_is_compared_canonically(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("BASELINE_VOL_THRESHOLD=0.000048\n")
    monkeypatch.setenv("SETTINGS_ENV_FILE", str(env_file))
    monkeypatch.setenv("SETTINGS_CONFIG_FILE", str(tmp_path / "missing.yaml"))

    row = _settings_by_key(settings_view.get_settings())["BASELINE_VOL_THRESHOLD"]

    assert row["saved_value"] == "0.000048"
    assert row["active_value"] == "4.8e-05"
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


def test_env_path_resolving_to_a_directory_raises_500_not_reported_as_missing(
    tmp_path, monkeypatch
):
    # The exact trap: a Docker bind mount for a missing host .env resolves
    # to an empty directory rather than failing the mount, so path.exists()
    # is True but reading it as a file raises IsADirectoryError. This must
    # surface as a distinct, structured 500 -- not the same "file missing"
    # behavior tested above (which returns 200 with nulls).
    env_dir = tmp_path / ".env"
    env_dir.mkdir()
    monkeypatch.setenv("SETTINGS_ENV_FILE", str(env_dir))
    monkeypatch.setenv("SETTINGS_CONFIG_FILE", str(tmp_path / "nope.yaml"))

    with pytest.raises(HTTPException) as exc_info:
        settings_view.get_settings()

    assert exc_info.value.status_code == 500
    assert str(env_dir) in exc_info.value.detail


def test_config_path_resolving_to_a_directory_raises_500_not_reported_as_missing(
    tmp_path, monkeypatch
):
    config_dir = tmp_path / "config.yaml"
    config_dir.mkdir()
    monkeypatch.setenv("SETTINGS_ENV_FILE", str(tmp_path / "nope.env"))
    monkeypatch.setenv("SETTINGS_CONFIG_FILE", str(config_dir))

    with pytest.raises(HTTPException) as exc_info:
        settings_view.get_settings()

    assert exc_info.value.status_code == 500
    assert str(config_dir) in exc_info.value.detail


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
    monkeypatch.setattr(
        settings_view,
        "_get_active_api_settings",
        lambda: {
            "MODEL_VARIANT": "ml",
            "BASELINE_VOL_THRESHOLD": "4.8e-05",
            "MODEL_STAGE": None,
            "MODEL_NAME": "btc-volatility-lr",
        },
    )

    body = settings_view.get_settings()
    row = _settings_by_key(body)["MODEL_STAGE"]

    assert row["active_value"] is None
    assert row["apply_state"] == "unknown"


def test_model_stage_is_active_when_mlflow_source(tmp_path, monkeypatch):
    monkeypatch.setenv("SETTINGS_ENV_FILE", str(tmp_path / "nope.env"))
    monkeypatch.setenv("SETTINGS_CONFIG_FILE", str(tmp_path / "nope.yaml"))
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


@pytest.mark.parametrize("yaml_text", ["- not\n- a\n- mapping\n", "plain scalar\n"])
def test_non_mapping_config_yaml_is_treated_as_missing(
    yaml_text, tmp_path, monkeypatch
):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_text)
    monkeypatch.setenv("SETTINGS_ENV_FILE", str(tmp_path / "nope.env"))
    monkeypatch.setenv("SETTINGS_CONFIG_FILE", str(config_file))

    by_key = _settings_by_key(settings_view.get_settings())

    assert by_key["features.window_seconds"]["saved_value"] is None
    assert by_key["features.vol_threshold"]["saved_value"] is None


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
