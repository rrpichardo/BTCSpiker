"""Smoke tests for the FastAPI prediction service.

Run from the repo root with either:

    pytest tests/test_api.py -q
    python tests/test_api.py
"""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

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


def test_predict_single(base_url):
    r = requests.post(f"{base_url}/predict", json={"rows": [SAMPLE_ROW]}, timeout=5)
    assert r.status_code == 200
    body = r.json()
    assert len(body["scores"]) == 1
    assert 0.0 <= body["scores"][0] <= 1.0
    assert body["model_variant"] == "ml"


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__)), "-q"]))
