"""
GET /settings — read-only view of the settings the web UI needs to display.

Shows, per setting, the value saved on disk (.env / config.yaml) next to the
value the running process actually has ("active"), plus how to apply a
changed saved value. This module must not import `api.main` at module load
time: `api/main.py` imports this module's router, so a top-level import here
would be circular. The one place we need data from `api.main` (the
MODEL_STAGE cross-check) goes through `_get_model_source()`, which imports
lazily inside the function so tests can monkeypatch it instead of triggering
api.main's module-level model loading.
"""

import os
from pathlib import Path

import yaml
from fastapi import APIRouter

router = APIRouter()

ENV_FILE_DEFAULT = ".env"
CONFIG_FILE_DEFAULT = "config.yaml"

# Defaults must mirror api/main.py's own os.getenv(...) defaults exactly, so
# "active_value" reflects what that process actually resolved.
_ACTIVE_ENV_DEFAULTS = {
    "MODEL_VARIANT": "ml",
    "BASELINE_VOL_THRESHOLD": "0.000048",
    "MODEL_STAGE": "Production",
    "MODEL_NAME": "btc-volatility-lr",
}

_EDIT_API = "edit .env, then: docker compose up -d api"
_EDIT_INGESTOR = "edit .env, then: docker compose up -d ingestor"
_EDIT_CONFIG = "edit config.yaml, then: docker compose restart featurizer"

# Fixed registry, in display order. "config_key" is only set for keys backed
# by config.yaml (nested under `features:`).
REGISTRY = [
    {
        "key": "MODEL_VARIANT",
        "source": ".env",
        "editable_via": _EDIT_API,
        "danger": None,
        "description": (
            "Selects the active scoring backend: 'ml' (the trained "
            "logistic-regression pipeline) or 'baseline' (a deterministic "
            "rule on vol_60s vs BASELINE_VOL_THRESHOLD)."
        ),
    },
    {
        "key": "BASELINE_VOL_THRESHOLD",
        "source": ".env",
        "editable_via": _EDIT_API,
        "danger": None,
        "description": (
            "Threshold on 60-second realised volatility that drives live "
            "baseline-variant scoring: a prediction scores 1.0 when vol_60s "
            "exceeds this value, 0.0 otherwise. Only used when "
            "MODEL_VARIANT=baseline."
        ),
    },
    {
        "key": "MODEL_STAGE",
        "source": ".env",
        "editable_via": _EDIT_API,
        "danger": None,
        "description": (
            "MLflow registry stage the API loads the 'ml' model from at "
            "startup (e.g. 'Production'). Only takes effect when the model "
            "actually loads from MLflow; if MLflow is unavailable the API "
            "silently falls back to a local pickle file and this stage is "
            "not in effect."
        ),
    },
    {
        "key": "MODEL_NAME",
        "source": ".env",
        "editable_via": _EDIT_API,
        "danger": None,
        "description": (
            "Registered model name the API looks up in the MLflow model "
            "registry at startup."
        ),
    },
    {
        "key": "REPLAY_SPEED",
        "source": ".env",
        "editable_via": _EDIT_INGESTOR,
        "danger": None,
        "description": (
            "Playback speed multiplier for the historical-data replay "
            "ingestor (1.0 = real-time). Consumed by the ingestor "
            "container, not the api service."
        ),
    },
    {
        "key": "features.window_seconds",
        "source": "config.yaml",
        "config_key": "window_seconds",
        "editable_via": _EDIT_CONFIG,
        "danger": "requires_retraining",
        "description": (
            "Rolling window length, in seconds, used to compute volatility "
            "and return features. Changing it requires model retraining — "
            "do not change casually."
        ),
    },
    {
        "key": "features.vol_threshold",
        "source": "config.yaml",
        "config_key": "vol_threshold",
        "editable_via": _EDIT_CONFIG,
        "danger": "requires_retraining",
        "description": (
            "Threshold used only to generate the vol_spike training label "
            "during feature engineering; it has no effect on live "
            "predictions. Changing it requires model retraining — do not "
            "change casually."
        ),
    },
]


def _parse_env_file(path: Path) -> dict[str, str]:
    """Parse a .env file: skip blanks/comments, split on first '=', strip
    whitespace and matching surrounding quotes. If a key appears more than
    once, the last occurrence wins (docker compose semantics)."""
    values: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key] = value
    return values


def _load_saved_env() -> dict[str, str]:
    path = Path(os.environ.get("SETTINGS_ENV_FILE", ENV_FILE_DEFAULT))
    if not path.exists():
        return {}
    return _parse_env_file(path)


def _load_saved_features() -> dict:
    path = Path(os.environ.get("SETTINGS_CONFIG_FILE", CONFIG_FILE_DEFAULT))
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data.get("features") or {}


def _active_value(key: str) -> str | None:
    if key in _ACTIVE_ENV_DEFAULTS:
        value = os.environ.get(key, _ACTIVE_ENV_DEFAULTS[key])
        return value.lower() if key == "MODEL_VARIANT" else value
    if key == "REPLAY_SPEED":
        # Consumed by the ingestor container, not this process — unset here
        # simply means "not observable from the api".
        return os.environ.get("REPLAY_SPEED")
    return None  # config.yaml keys: the api process cannot introspect them


def _apply_state(saved_value: str | None, active_value: str | None) -> str:
    if saved_value is None or active_value is None:
        return "unknown"
    return "applied" if saved_value == active_value else "restart_required"


def _get_model_source():
    """Return what /version would report as model source ('mlflow' or
    'pickle'), or None if it can't be determined. Imports api.main lazily —
    only inside this function — so importing this module never triggers
    api.main's module-level model loading. Tests monkeypatch this function
    directly instead of exercising the import."""
    try:
        from api import main

        return main.model_source
    except Exception:
        return None


def _build_entry(reg: dict, saved_env: dict[str, str], saved_features: dict) -> dict:
    key = reg["key"]

    if "config_key" in reg:
        raw = saved_features.get(reg["config_key"])
        saved_value = None if raw is None else str(raw)
    else:
        saved_value = saved_env.get(key)

    active_value = _active_value(key)
    if key == "MODEL_STAGE" and _get_model_source() != "mlflow":
        # Mirrors /version: a requested registry stage is not active when
        # startup fell back to the local pickle (or source is unknown).
        active_value = None
    apply_state = _apply_state(saved_value, active_value)

    return {
        "key": key,
        "active_value": active_value,
        "saved_value": saved_value,
        "apply_state": apply_state,
        "source": reg["source"],
        "editable_via": reg["editable_via"],
        "description": reg["description"],
        "danger": reg["danger"],
    }


@router.get("/settings")
def get_settings():
    """Return the fixed settings registry with saved vs active values."""
    saved_env = _load_saved_env()
    saved_features = _load_saved_features()
    settings = [_build_entry(reg, saved_env, saved_features) for reg in REGISTRY]
    return {"settings": settings}
