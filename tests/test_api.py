"""Smoke tests for the FastAPI prediction service.

Run from the repo root with either:

    pytest tests/test_api.py -q
    python tests/test_api.py
"""

import json
import pickle
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

from scripts.feature_to_predict_bridge import _build_row

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOST = "127.0.0.1"
MODEL_PATH = PROJECT_ROOT / "handoff" / "models" / "artifacts" / "lr_pipeline.pkl"

SAMPLE_ROW = {
    "log_return": 0.0001,
    "spread_bps": 1.5,
    "vol_60s": 0.00005,
    "mean_return_60s": 0.0,
    "trade_intensity_60s": 10.0,
    "n_ticks_60s": 50,
    "spread_mean_60s": 1.2,
}
FEATURE_COLS = list(SAMPLE_ROW)


def _reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _wait_for_api(
    base_url: str, process: subprocess.Popen[str], timeout: int = 30
) -> None:
    deadline = time.monotonic() + timeout
    last_error = None

    while time.monotonic() < deadline:
        if process.poll() is not None:
            output, _ = process.communicate(timeout=1)
            raise RuntimeError(f"API process exited before becoming ready:\n{output}")

        try:
            response = requests.get(f"{base_url}/health", timeout=1)
            if response.status_code == 200 and response.json() == {"status": "ok"}:
                return
        except requests.RequestException as exc:
            last_error = exc

        time.sleep(0.5)

    process.terminate()
    output, _ = process.communicate(timeout=10)
    raise RuntimeError(
        f"Timed out waiting for {base_url}/health after {timeout}s. "
        f"Last error: {last_error}\n{output}"
    )


@pytest.fixture(scope="module")
def base_url():
    port = _reserve_port()
    env = os.environ.copy()
    env["MODEL_PATH"] = str(MODEL_PATH)
    env["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:99999"

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api.main:app",
            "--host",
            HOST,
            "--port",
            str(port),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    url = f"http://{HOST}:{port}"
    _wait_for_api(url, process)

    try:
        yield url
    finally:
        process.terminate()
        try:
            process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=10)


def test_health(base_url):
    r = requests.get(f"{base_url}/health", timeout=5)
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_version(base_url):
    r = requests.get(f"{base_url}/version", timeout=5)
    assert r.status_code == 200
    body = r.json()
    # Required fields always present in the new shape
    assert "model" in body
    assert "sha" in body
    assert "source" in body
    assert "run_id" in body
    assert "stage" in body


def test_version_source(base_url):
    r = requests.get(f"{base_url}/version", timeout=5)
    assert r.status_code == 200
    body = r.json()
    # source must be one of the two known load paths — never empty
    assert body["source"] in ("mlflow", "pickle")
    # run_id is allowed to be null when MLflow is not available in CI
    assert "run_id" in body
    assert "stage" in body


def test_nondefault_registered_candidate_never_silently_uses_the_legacy_pickle():
    """A Staging candidate must fail closed when its registry is unavailable."""
    env = os.environ.copy()
    env.update(
        {
            "MODEL_PATH": str(MODEL_PATH),
            "MODEL_NAME": "btc-volatility-candidate",
            "MODEL_STAGE": "Staging",
            "MLFLOW_TRACKING_URI": "http://127.0.0.1:99999",
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", "import api.main"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
    )

    assert result.returncode != 0
    assert "candidate" in (result.stdout + result.stderr).lower()


def test_registered_candidate_requires_feature_identity_on_every_request(tmp_path):
    """Legacy payload compatibility must not bypass a candidate's schema gate."""
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient

    tracking_uri = tmp_path.as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment("candidate-contract-test")
    with open(MODEL_PATH, "rb") as artifact:
        pipeline = pickle.load(artifact)["pipeline"]
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        mlflow.log_params(
            {
                "feature_cols": ",".join(FEATURE_COLS),
                "feature_set_id": "multi_window_v1",
                "feature_schema_version": "2",
                "tau": "0.5",
            }
        )
        mlflow.sklearn.log_model(pipeline, "model")
        run_id = run.info.run_id
    version = mlflow.register_model(f"runs:/{run_id}/model", "btc-volatility-candidate")
    MlflowClient(tracking_uri).transition_model_version_stage(
        "btc-volatility-candidate", version.version, "Staging"
    )

    port = _reserve_port()
    env = os.environ.copy()
    env.update(
        {
            "MODEL_NAME": "btc-volatility-candidate",
            "MODEL_STAGE": "Staging",
            "MLFLOW_TRACKING_URI": tracking_uri,
        }
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api.main:app",
            "--host",
            HOST,
            "--port",
            str(port),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    url = f"http://{HOST}:{port}"
    try:
        _wait_for_api(url, process)
        response = requests.post(
            url + "/predict", json={"rows": [SAMPLE_ROW]}, timeout=5
        )
        assert response.status_code == 422
        assert "feature_set_id is required" in response.text
    finally:
        process.terminate()
        process.communicate(timeout=10)


def test_predict_single(base_url):
    r = requests.post(f"{base_url}/predict", json={"rows": [SAMPLE_ROW]}, timeout=5)
    assert r.status_code == 200
    body = r.json()
    assert len(body["scores"]) == 1
    assert 0.0 <= body["scores"][0] <= 1.0
    assert body["model_variant"] == "ml"
    # Model provenance: pickle fallback in this fixture ships a tau but no
    # MLflow run_id.
    assert "tau" in body
    assert isinstance(body["tau"], float)
    assert "run_id" in body
    assert body["run_id"] is None


def test_predict_accepts_legacy_v1_payload(base_url):
    response = requests.post(
        f"{base_url}/predict", json={"rows": [SAMPLE_ROW]}, timeout=5
    )
    assert response.status_code == 200


def test_predict_accepts_registered_numeric_extra_features(base_url):
    payload = {
        "rows": [
            {
                **SAMPLE_ROW,
                "feature_set_id": "core_v1",
                "feature_schema_version": "1",
                "price_range_60s": 0.0002,
            }
        ]
    }
    response = requests.post(f"{base_url}/predict", json=payload, timeout=5)
    assert response.status_code == 200


def test_predict_rejects_registered_feature_version_mismatch(base_url):
    payload = {"rows": [{**SAMPLE_ROW, "feature_schema_version": "wrong"}]}
    response = requests.post(f"{base_url}/predict", json=payload, timeout=5)
    assert response.status_code == 422
    assert "feature_schema_version" in response.text


def test_bridge_row_preserves_registered_versions_through_api_validation(base_url):
    """The real bridge row must retain the API's registered feature contract."""
    feature_message = {
        **SAMPLE_ROW,
        "timestamp": "2026-07-21T12:00:00Z",
        "feature_set_id": "core_v1",
        "feature_schema_version": "1",
    }

    bridge_row = _build_row(feature_message, kafka_timestamp_ms=None)

    assert bridge_row["feature_set_id"] == "core_v1"
    assert bridge_row["feature_schema_version"] == "1"
    accepted = requests.post(
        f"{base_url}/predict", json={"rows": [bridge_row]}, timeout=5
    )
    assert accepted.status_code == 200

    bridge_row["feature_schema_version"] = "altered"
    rejected = requests.post(
        f"{base_url}/predict", json={"rows": [bridge_row]}, timeout=5
    )
    assert rejected.status_code == 422
    assert "feature_schema_version" in rejected.text


@pytest.mark.parametrize("value", [True, "not-a-number", float("nan"), float("inf")])
def test_predict_rejects_invalid_extra_feature_values(base_url, value):
    payload = {"rows": [{**SAMPLE_ROW, "price_range_60s": value}]}
    response = requests.post(
        f"{base_url}/predict",
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
        timeout=5,
    )
    assert response.status_code == 422


def test_predict_batch(base_url):
    r = requests.post(f"{base_url}/predict", json={"rows": [SAMPLE_ROW] * 5}, timeout=5)
    assert r.status_code == 200
    assert len(r.json()["scores"]) == 5


def test_predict_missing_field(base_url):
    bad_row = {"log_return": 0.0001}  # missing 6 fields
    r = requests.post(f"{base_url}/predict", json={"rows": [bad_row]}, timeout=5)
    assert r.status_code == 422


def test_metrics(base_url):
    r = requests.get(f"{base_url}/metrics", timeout=5)
    assert r.status_code == 200
    assert "predict_requests_total" in r.text


def test_sample_json_payload(base_url):
    sample_path = PROJECT_ROOT / "handoff" / "data_sample" / "sample.json"
    with open(sample_path) as f:
        payload = json.load(f)

    response = requests.post(f"{base_url}/predict", json=payload, timeout=5)
    assert response.status_code == 200

    data = response.json()
    assert "scores" in data
    assert "model_variant" in data
    assert "version" in data
    assert "ts" in data
    assert data["version"] == "v1.0"
    assert isinstance(data["scores"], list)
    assert len(data["scores"]) == 1
    assert 0 <= data["scores"][0] <= 1


# ---------------------------------------------------------------------------
# Train/serve skew telemetry (feature_zscore_abs / feature_skew_rows_total)
# ---------------------------------------------------------------------------

_SKEW_METRIC_PATTERN = (
    r'^{metric}\{{(?=[^}}]*feature="{feature}")(?=[^}}]*model_version=)[^}}]*\}} ([0-9.eE+\-]+)$'
)


def _metric_value(text: str, metric: str, feature: str) -> float:
    import re

    pattern = re.compile(
        _SKEW_METRIC_PATTERN.format(metric=re.escape(metric), feature=re.escape(feature)),
        re.MULTILINE,
    )
    match = pattern.search(text)
    return float(match.group(1)) if match else 0.0


def test_predict_exports_feature_zscore_for_skewed_row(base_url):
    # The exact defect observed in the 2026-04 duplicated-tick incident:
    # trade_intensity_60s and n_ticks_60s both ~6 sigma off the training
    # scaler's mean, for an entire run, with nothing surfacing it.
    skewed_row = {**SAMPLE_ROW, "trade_intensity_60s": 12.0, "n_ticks_60s": 720}

    before = requests.get(f"{base_url}/metrics", timeout=5).text
    count_before = _metric_value(before, "feature_zscore_abs_count", "trade_intensity_60s")
    skew_before = _metric_value(before, "feature_skew_rows_total", "trade_intensity_60s")

    r = requests.post(f"{base_url}/predict", json={"rows": [skewed_row]}, timeout=5)
    assert r.status_code == 200

    after = requests.get(f"{base_url}/metrics", timeout=5).text
    count_after = _metric_value(after, "feature_zscore_abs_count", "trade_intensity_60s")
    skew_after = _metric_value(after, "feature_skew_rows_total", "trade_intensity_60s")

    assert count_after == count_before + 1
    assert skew_after >= skew_before + 1


def test_normal_row_does_not_trip_skew_counter(base_url):
    # Near the real scaler's training mean for both features (see
    # handoff/models/artifacts/lr_pipeline.pkl: trade_intensity_60s
    # mean=3.842, n_ticks_60s mean=230.5) -- must NOT cross |z| >= 4.
    normal_row = {**SAMPLE_ROW, "trade_intensity_60s": 3.8, "n_ticks_60s": 230}

    before = requests.get(f"{base_url}/metrics", timeout=5).text
    skew_before = _metric_value(before, "feature_skew_rows_total", "trade_intensity_60s")

    r = requests.post(f"{base_url}/predict", json={"rows": [normal_row]}, timeout=5)
    assert r.status_code == 200

    after = requests.get(f"{base_url}/metrics", timeout=5).text
    skew_after = _metric_value(after, "feature_skew_rows_total", "trade_intensity_60s")

    assert skew_after == skew_before


def test_scaler_extraction_degrades_to_none_instead_of_raising():
    # _extract_scaler touches only its own arguments (no module globals), so
    # it's tested via a fresh subprocess import rather than importing
    # api.main in-process -- this test suite never does that (see
    # tests/test_system.py's docstring), since api/main.py's top-level model
    # load has real side effects (an MLflow connection attempt, a pickle
    # read) that must stay isolated per test process.
    script = """
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from api.main import _extract_scaler

no_scaler = Pipeline([("clf", LogisticRegression())])
assert _extract_scaler(no_scaler, ["a", "b"]) is None, "expected None for a pipeline with no scaler step"

scaler = StandardScaler()
scaler.mean_ = np.array([1.0, 2.0, 3.0])
scaler.scale_ = np.array([1.0, 1.0, 1.0])
mismatched = Pipeline([("scaler", scaler), ("clf", LogisticRegression())])
assert _extract_scaler(mismatched, ["a", "b"]) is None, "expected None for a dimensionality mismatch"

matched = _extract_scaler(mismatched, ["a", "b", "c"])
assert matched is not None
mean, scale = matched
assert list(mean) == [1.0, 2.0, 3.0]
assert list(scale) == [1.0, 1.0, 1.0]

print("SCALER_EXTRACTION_OK")
"""
    env = os.environ.copy()
    env["MODEL_PATH"] = str(MODEL_PATH)
    env["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:99999"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=30,
    )
    assert "SCALER_EXTRACTION_OK" in result.stdout, result.stdout + result.stderr


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__)), "-q"]))
