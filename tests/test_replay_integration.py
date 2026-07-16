"""
End-to-end replay integration test.

Verifies the live runtime path:
    ingestor -> ticks.raw -> featurizer -> ticks.features
    -> predict-bridge -> /predict -> Prometheus

Run locally:
    pytest tests/test_replay_integration.py -v

Run locally with auto-start:
    REPLAY_TEST_AUTOSTART=1 REPLAY_SPEED=50 pytest tests/test_replay_integration.py -v
"""

import json
import os
import subprocess
import time
import uuid

import pytest
import requests
from confluent_kafka import Consumer, KafkaException

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
RAW_TOPIC = "ticks.raw"
FEATURES_TOPIC = "ticks.features"
FEATURIZER_GROUP = "ticks-featurizer"
BRIDGE_GROUP = "predict-bridge"
AUTO_START_STACK = os.getenv("REPLAY_TEST_AUTOSTART", "0") == "1"

FEATURE_COLS = [
    "log_return",
    "spread_bps",
    "vol_60s",
    "mean_return_60s",
    "trade_intensity_60s",
    "n_ticks_60s",
    "spread_mean_60s",
]


def _api_healthy() -> bool:
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=2)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except Exception:
        return False


def _prometheus_ready() -> bool:
    try:
        r = requests.get(f"{PROMETHEUS_URL}/-/ready", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _wait_for_api(timeout: int = 60) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _api_healthy():
            return
        time.sleep(2)
    raise RuntimeError(
        f"API at {BASE_URL} did not become healthy within {timeout}s. "
        "Make sure the full stack is running: docker compose up -d --build"
    )


def _wait_for_prometheus(timeout: int = 60) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _prometheus_ready():
            return
        time.sleep(2)
    raise RuntimeError(
        f"Prometheus at {PROMETHEUS_URL} did not become ready within {timeout}s. "
        "Make sure the full stack is running: docker compose up -d --build"
    )


def _prom_query(expr: str) -> list[dict]:
    r = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query",
        params={"query": expr},
        timeout=5,
    )
    r.raise_for_status()
    payload = r.json()
    assert payload["status"] == "success", payload
    return payload["data"]["result"]


def _prom_scalar(expr: str) -> float | None:
    result = _prom_query(expr)
    if not result:
        return None
    return float(result[0]["value"][1])


def _wait_for_scalar(expr: str, predicate, timeout: int, description: str) -> float:
    deadline = time.monotonic() + timeout
    last_value = None
    while time.monotonic() < deadline:
        value = _prom_scalar(expr)
        last_value = value
        if value is not None and predicate(value):
            return value
        time.sleep(2)
    raise AssertionError(
        f"Timed out waiting for {description}; last value from {expr!r} was {last_value!r}. "
        "If you have an older stack running, rebuild it with docker compose up -d --build."
    )


def _wait_for_series(expr: str, timeout: int, description: str) -> list[dict]:
    deadline = time.monotonic() + timeout
    last_result = []
    while time.monotonic() < deadline:
        result = _prom_query(expr)
        last_result = result
        if result:
            return result
        time.sleep(2)
    raise AssertionError(
        f"Timed out waiting for {description}; query {expr!r} returned {last_result!r}"
    )


def _wait_for_topic_message(
    topic: str, timeout: int, offset_reset: str = "latest"
) -> dict:
    consumer = Consumer(
        {
            "bootstrap.servers": KAFKA_BOOTSTRAP,
            "group.id": f"test-replay-{topic}-{uuid.uuid4()}",
            "auto.offset.reset": offset_reset,
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([topic])

    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            msg = consumer.poll(timeout=2.0)
            if msg is None:
                continue
            if msg.error():
                raise KafkaException(msg.error())
            return json.loads(msg.value())
    finally:
        consumer.close()

    raise AssertionError(f"No new messages arrived on {topic!r} within {timeout}s")


@pytest.fixture(scope="module", autouse=True)
def stack():
    try:
        _wait_for_api(timeout=120)
        _wait_for_prometheus(timeout=120)
    except RuntimeError:
        if not AUTO_START_STACK:
            raise
        subprocess.run(
            ["docker", "compose", "up", "-d", "--build"],
            check=True,
        )
        _wait_for_api(timeout=120)
        _wait_for_prometheus(timeout=120)
    yield


def test_replay_runtime_drives_prediction_metrics():
    """
    Assert the replay stack produces Kafka traffic on both topics, that the
    runtime bridge drives /predict without test-side POSTs, and that Prometheus
    observes the live prediction path.
    """
    baseline_requests = _prom_scalar("sum(predict_requests_total)") or 0.0

    raw_msg = _wait_for_topic_message(RAW_TOPIC, timeout=30)
    assert {"timestamp", "price", "best_bid", "best_ask"} <= raw_msg.keys(), raw_msg

    feature_msg = _wait_for_topic_message(
        FEATURES_TOPIC,
        timeout=120,
        offset_reset="earliest",
    )
    missing = [col for col in FEATURE_COLS if col not in feature_msg]
    assert not missing, (
        f"Feature message is missing required fields: {missing}. "
        f"Message keys: {list(feature_msg.keys())}"
    )
    assert feature_msg.get(
        "timestamp"
    ), "Feature messages should carry the timestamp that feeds API freshness."

    _wait_for_scalar(
        "sum(predict_requests_total)",
        lambda value: value > baseline_requests,
        timeout=90,
        description="prediction traffic to reach the API from replay mode",
    )
    _wait_for_scalar(
        "feature_freshness_seconds",
        lambda value: value >= 0.0,
        timeout=60,
        description="feature freshness to be populated by the runtime bridge",
    )
    _wait_for_series(
        f'kafka_consumergroup_lag{{topic="{RAW_TOPIC}",consumergroup="{FEATURIZER_GROUP}"}}',
        timeout=60,
        description="featurizer consumer lag series",
    )
    _wait_for_series(
        f'kafka_consumergroup_lag{{topic="{FEATURES_TOPIC}",consumergroup="{BRIDGE_GROUP}"}}',
        timeout=60,
        description="predict-bridge consumer lag series",
    )
