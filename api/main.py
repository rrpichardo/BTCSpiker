"""
BTC Volatility Spike Detector — FastAPI Service

Loads the trained Logistic Regression pipeline and serves predictions
via a REST API with /health, /predict, /version, and /metrics endpoints.

Supports a rollback toggle via MODEL_VARIANT=ml|baseline:
  - ml       (default) — sklearn LR pipeline; score = predict_proba[:, 1]
  - baseline           — deterministic z-style rule on vol_60s vs threshold

Model loading priority (ml variant only):
  1. MLflow model registry (models:/<MODEL_NAME>/<MODEL_STAGE>)
  2. Local pickle fallback at MODEL_PATH with a warning log
"""

import logging
import math
import os
import pickle
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from mlflow.tracking import MlflowClient
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "models/artifacts/lr_pipeline.pkl")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0")
MODEL_VARIANT = os.getenv("MODEL_VARIANT", "ml").lower()
BASELINE_VOL_THRESHOLD = float(os.getenv("BASELINE_VOL_THRESHOLD", "0.000048"))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MODEL_NAME = os.getenv("MODEL_NAME", "btc-volatility-lr")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

if MODEL_VARIANT not in {"ml", "baseline"}:
    raise ValueError(f"MODEL_VARIANT must be 'ml' or 'baseline', got {MODEL_VARIANT!r}")

# ---------------------------------------------------------------------------
# Model loading — MLflow first, local pickle fallback
# Baseline variant skips model loading entirely; the fallback path must work
# even when the ML artifact is missing, since that is the exact failure mode
# MODEL_VARIANT=baseline exists to handle.
# ---------------------------------------------------------------------------
FEATURE_COLS_DEFAULT = [
    "log_return",
    "spread_bps",
    "vol_60s",
    "mean_return_60s",
    "trade_intensity_60s",
    "n_ticks_60s",
    "spread_mean_60s",
]
FEATURE_SET_ID_DEFAULT = "core_v1"
FEATURE_SCHEMA_VERSION_DEFAULT = "1"

PIPELINE = None
FEATURE_COLS = FEATURE_COLS_DEFAULT
FEATURE_SET_ID = FEATURE_SET_ID_DEFAULT
FEATURE_SCHEMA_VERSION = FEATURE_SCHEMA_VERSION_DEFAULT
TAU: float | None = None

# Module-level variables set by whichever load path succeeds
model_source: str = "pickle"
mlflow_run_id: str | None = None

if MODEL_VARIANT == "ml":
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        # Resolve run_id and feature metadata from the registry
        _client = MlflowClient(MLFLOW_TRACKING_URI)
        _versions = _client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        if not _versions:
            raise LookupError(
                f"No {MODEL_STAGE} version found for registered model '{MODEL_NAME}'"
            )
        mlflow_run_id = _versions[0].run_id

        # Load the sklearn pipeline via the sklearn flavor (gives predict_proba)
        PIPELINE = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")

        # Retrieve the complete feature contract stored with the registered run.
        _run = _client.get_run(mlflow_run_id)
        FEATURE_COLS = _run.data.params["feature_cols"].split(",")
        FEATURE_SET_ID = _run.data.params.get(
            "feature_set_id", FEATURE_SET_ID_DEFAULT
        )
        FEATURE_SCHEMA_VERSION = _run.data.params.get(
            "feature_schema_version", FEATURE_SCHEMA_VERSION_DEFAULT
        )
        TAU = float(_run.data.params["tau"])

        model_source = "mlflow"
        logger.info("Loaded model from MLflow run %s", mlflow_run_id)

    except Exception as _mlflow_exc:
        logger.warning("MLflow unavailable (%s), falling back to pickle", _mlflow_exc)
        # Fall back to local pickle bundle
        _model_path = Path(MODEL_PATH)
        if not _model_path.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

        with open(_model_path, "rb") as _f:
            _bundle = pickle.load(_f)

        PIPELINE = _bundle["pipeline"]
        FEATURE_COLS = _bundle["feature_cols"]
        FEATURE_SET_ID = _bundle.get("feature_set_id", FEATURE_SET_ID_DEFAULT)
        FEATURE_SCHEMA_VERSION = _bundle.get(
            "feature_schema_version", FEATURE_SCHEMA_VERSION_DEFAULT
        )
        TAU = _bundle["tau"]
        model_source = "pickle"
        mlflow_run_id = None

# ---------------------------------------------------------------------------
# Git SHA (resolved once at startup)
# ---------------------------------------------------------------------------
try:
    GIT_SHA = (
        subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        .decode()
        .strip()
    )
except Exception:
    GIT_SHA = os.getenv("GIT_SHA", "unknown")

# ---------------------------------------------------------------------------
# Prometheus metrics — labelled by model_variant so panels can split / overlay
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "predict_requests_total",
    "Total prediction requests",
    ["model_variant"],
)
REQUEST_ERRORS = Counter(
    "predict_errors_total",
    "Total failed prediction requests",
    ["model_variant"],
)
REQUEST_LATENCY = Histogram(
    "predict_latency_seconds",
    "Prediction request latency in seconds",
    ["model_variant"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.8, 1.0],
)
ACTIVE_VARIANT = Gauge(
    "model_variant_active",
    "Currently active model variant (1 = active)",
    ["model_variant"],
)
ACTIVE_VARIANT.labels(model_variant=MODEL_VARIANT).set(1)
# Tracks how old the feature data is (seconds between feature timestamp and now).
# Older payloads may omit `ts`, in which case the service falls back to request
# processing lag as a degraded proxy rather than true feature freshness.
FEATURE_FRESHNESS = Gauge(
    "feature_freshness_seconds",
    "Seconds between the incoming feature row's timestamp and now",
)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class TickRow(BaseModel):
    """One already-featurized observation sent to `/predict`.

    The API boundary is post-featurization: raw Coinbase ticks stay upstream,
    and the prediction service accepts the engineered 60-second features below.
    """

    model_config = ConfigDict(extra="allow")

    log_return: float = Field(
        description="Instantaneous log-return vs the previous tick."
    )
    spread_bps: float = Field(description="Current bid-ask spread in basis points.")
    vol_60s: float = Field(
        description="Rolling 60-second standard deviation of log-returns."
    )
    mean_return_60s: float = Field(description="Rolling 60-second mean log-return.")
    trade_intensity_60s: float = Field(
        description="Ticks per second over the trailing 60-second window."
    )
    n_ticks_60s: float = Field(
        description="Raw tick count over the trailing 60-second window."
    )
    spread_mean_60s: float = Field(
        description="Rolling 60-second mean absolute spread."
    )
    ts: str | None = Field(
        default=None,
        description=(
            "Optional ISO-8601 feature timestamp used to compute "
            "feature_freshness_seconds."
        ),
    )
    feature_set_id: str | None = Field(
        default=None,
        description="Optional identifier for the deployed feature set.",
    )
    feature_schema_version: str | None = Field(
        default=None,
        description="Optional schema version for the deployed feature set.",
    )

    @model_validator(mode="before")
    @classmethod
    def numeric_features_are_finite(cls, value):
        if not isinstance(value, dict):
            return value
        metadata_fields = {"ts", "feature_set_id", "feature_schema_version"}
        invalid = [
            name
            for name, feature_value in value.items()
            if name not in metadata_fields
            and (
                isinstance(feature_value, bool)
                or not isinstance(feature_value, (int, float))
                or not math.isfinite(feature_value)
            )
        ]
        if invalid:
            raise ValueError(
                "feature values must be finite numeric values; invalid fields: "
                + ", ".join(sorted(invalid))
            )
        return value


class PredictRequest(BaseModel):
    """Batch request for already-engineered feature rows."""

    rows: list[TickRow]


class PredictResponse(BaseModel):
    scores: list[float] = Field(description="Predicted positive-class scores per row.")
    model_variant: str = Field(
        description="Active scoring backend: 'ml' or 'baseline'."
    )
    version: str = Field(description="Human-readable model bundle version.")
    ts: str = Field(description="UTC wall-clock timestamp when scoring completed.")


class VersionResponse(BaseModel):
    """Metadata for the model artifact loaded by the API at startup."""

    model: str = Field(description="Registered model name.")
    version: str = Field(description="Human-readable model bundle version.")
    stage: str | None = Field(
        default=None,
        description="MLflow stage when loaded from the registry; null on pickle fallback.",
    )
    source: str = Field(
        description="Artifact source used at startup: 'mlflow' or 'pickle'."
    )
    run_id: str | None = Field(
        default=None,
        description="MLflow run ID for the loaded model; null on pickle fallback.",
    )
    sha: str = Field(description="Short Git commit SHA for the running service.")


# ---------------------------------------------------------------------------
# Scoring backends
# ---------------------------------------------------------------------------


def _score_ml(rows: list[TickRow]) -> list[float]:
    matrices = []
    for row in rows:
        payload = row.model_dump()
        missing = [column for column in FEATURE_COLS if column not in payload]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"feature row missing required columns: {missing}",
            )
        matrices.append([payload[column] for column in FEATURE_COLS])
    X = np.array(matrices)
    y_prob = PIPELINE.predict_proba(X)[:, 1]
    return [round(float(p), 6) for p in y_prob]


def _score_baseline(rows: list[TickRow]) -> list[float]:
    # Mirrors the labeling rule: a tick is "spiking" when its 60s realised
    # volatility exceeds the same threshold used to generate training labels.
    # Returns 1.0 / 0.0 so the response shape stays compatible with /predict.
    return [1.0 if row.vol_60s > BASELINE_VOL_THRESHOLD else 0.0 for row in rows]


SCORERS = {"ml": _score_ml, "baseline": _score_baseline}


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="BTC Volatility Spike Detector")


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, BaseException):
        return str(value)
    return value


@app.exception_handler(RequestValidationError)
async def request_validation_error(_request: Request, exc: RequestValidationError):
    """Keep malformed non-finite JSON payloads on the public 422 path."""
    return JSONResponse(status_code=422, content={"detail": _json_safe(exc.errors())})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/version", response_model=VersionResponse)
def version():
    """Return the exact metadata shape documented for the deployed model."""
    # source/stage/run_id reflect whichever load path was taken at startup
    return VersionResponse(
        model=MODEL_NAME,
        version=MODEL_VERSION,
        stage=MODEL_STAGE if model_source == "mlflow" else None,
        source=model_source,
        run_id=mlflow_run_id,
        sha=GIT_SHA,
    )


@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Score engineered feature rows; raw tick ingestion happens upstream."""
    REQUEST_COUNT.labels(model_variant=MODEL_VARIANT).inc()
    start = time.perf_counter()

    for row in req.rows:
        if row.feature_set_id is not None and row.feature_set_id != FEATURE_SET_ID:
            raise HTTPException(
                status_code=422,
                detail=(
                    "feature_set_id does not match the registered model: "
                    f"expected {FEATURE_SET_ID!r}, got {row.feature_set_id!r}"
                ),
            )
        if (
            row.feature_schema_version is not None
            and row.feature_schema_version != FEATURE_SCHEMA_VERSION
        ):
            raise HTTPException(
                status_code=422,
                detail=(
                    "feature_schema_version does not match the registered model: "
                    f"expected {FEATURE_SCHEMA_VERSION!r}, "
                    f"got {row.feature_schema_version!r}"
                ),
            )

    # Update feature freshness gauge when the client supplies a feature timestamp.
    # Without `ts`, the API falls back to request processing lag as a degraded proxy.
    if req.rows and hasattr(req.rows[0], "ts") and req.rows[0].ts:
        row_ts = datetime.fromisoformat(req.rows[0].ts.replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - row_ts).total_seconds()
        FEATURE_FRESHNESS.set(age)
    else:
        logger.warning(
            "feature_freshness_seconds: request row omitted ts — "
            "setting degraded fallback (request processing lag)"
        )
        FEATURE_FRESHNESS.set(time.perf_counter() - start)

    try:
        scores = SCORERS[MODEL_VARIANT](req.rows)
        return PredictResponse(
            scores=scores,
            model_variant=MODEL_VARIANT,
            version=MODEL_VERSION,
            ts=datetime.now(timezone.utc).isoformat(),
        )
    except HTTPException:
        REQUEST_ERRORS.labels(model_variant=MODEL_VARIANT).inc()
        raise
    except Exception as exc:
        REQUEST_ERRORS.labels(model_variant=MODEL_VARIANT).inc()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        REQUEST_LATENCY.labels(model_variant=MODEL_VARIANT).observe(
            time.perf_counter() - start
        )
