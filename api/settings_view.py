"""
GET /settings — read-only view of the settings the web UI needs to display.

Shows, per setting, the value saved on disk (.env / config.yaml) next to the
value the running process actually has ("active"), plus how to apply a
changed saved value. This module must not import `api.main` at module load
time: `api/main.py` imports this module's router, so a top-level import here
would be circular. Active API values are read lazily from `api.main`'s
startup constants.
"""

import os
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException

router = APIRouter()


class SettingsConfigError(RuntimeError):
    """A settings source path exists but could not be read as the file it's
    supposed to be — e.g. a Docker bind mount for a missing host `.env`
    resolves to an empty directory rather than failing the mount, so
    `path.exists()` is True but `path.read_text()` raises IsADirectoryError.
    Deliberately distinct from "file missing" (which returns {} below):
    a broken mount is a real, fixable configuration problem and must not be
    silently reported the same as "nothing configured yet"."""

ENV_FILE_DEFAULT = ".env"
CONFIG_FILE_DEFAULT = "config.yaml"

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
    try:
        return _parse_env_file(path)
    except OSError as exc:
        raise SettingsConfigError(f"could not read {path}: {exc}") from exc


def _load_saved_features() -> dict:
    path = Path(os.environ.get("SETTINGS_CONFIG_FILE", CONFIG_FILE_DEFAULT))
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    except OSError as exc:
        raise SettingsConfigError(f"could not read {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        return {}
    features = data.get("features") or {}
    return dict(features) if isinstance(features, Mapping) else {}


def _get_active_api_settings() -> dict[str, str | None]:
    """Snapshot constants already loaded by api.main, without re-reading env."""
    try:
        from api import main

        return {
            "MODEL_VARIANT": str(main.MODEL_VARIANT),
            "BASELINE_VOL_THRESHOLD": str(main.BASELINE_VOL_THRESHOLD),
            "MODEL_STAGE": main.MODEL_STAGE if main.model_source == "mlflow" else None,
            "MODEL_NAME": str(main.MODEL_NAME),
        }
    except Exception:
        # Outside the FastAPI app (or after failed startup), these values are
        # genuinely unknown. Re-reading mutable env would misreport them.
        return {
            "MODEL_VARIANT": None,
            "BASELINE_VOL_THRESHOLD": None,
            "MODEL_STAGE": None,
            "MODEL_NAME": None,
        }


def _canonical_value(key: str, value: str | None):
    if value is None:
        return None
    if key == "MODEL_VARIANT":
        return value.lower()
    if key in {"BASELINE_VOL_THRESHOLD", "REPLAY_SPEED"}:
        try:
            return Decimal(value)
        except InvalidOperation:
            return value
    return value


def _apply_state(key: str, saved_value: str | None, active_value: str | None) -> str:
    if saved_value is None or active_value is None:
        return "unknown"
    saved = _canonical_value(key, saved_value)
    active = _canonical_value(key, active_value)
    return "applied" if saved == active else "restart_required"


def _build_entry(
    reg: dict,
    saved_env: dict[str, str],
    saved_features: dict,
    active_api_settings: dict[str, str | None],
) -> dict:
    key = reg["key"]

    if "config_key" in reg:
        raw = saved_features.get(reg["config_key"])
        saved_value = None if raw is None else str(raw)
    else:
        saved_value = saved_env.get(key)

    active_value = active_api_settings.get(key)
    apply_state = _apply_state(key, saved_value, active_value)

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
    try:
        saved_env = _load_saved_env()
        saved_features = _load_saved_features()
    except SettingsConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    active_api_settings = _get_active_api_settings()
    settings = [
        _build_entry(reg, saved_env, saved_features, active_api_settings)
        for reg in REGISTRY
    ]
    return {"settings": settings}
