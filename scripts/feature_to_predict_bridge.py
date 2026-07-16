"""
Consume engineered feature rows from Kafka and POST them to the prediction API.

This is the runtime bridge that closes the replay loop:
    ticks.raw -> ticks.features -> /predict -> Prometheus metrics

Usage:
    python scripts/feature_to_predict_bridge.py
"""

import json
import logging
import os
import signal
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from confluent_kafka import Consumer, KafkaError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("feature_to_predict_bridge")

KAFKA_BOOTSTRAP = (
    os.getenv("KAFKA_BOOTSTRAP")
    or os.getenv("KAFKA_BOOTSTRAP_SERVERS")
    or "localhost:9092"
)
FEATURES_TOPIC = os.getenv("TOPIC_FEATURES", "ticks.features")
GROUP_ID = os.getenv("KAFKA_GROUP_ID", "predict-bridge")
API_URL = os.getenv("PREDICT_API_URL", "http://localhost:8000/predict")
API_HEALTH_URL = os.getenv("PREDICT_API_HEALTH_URL", "http://localhost:8000/health")
API_TIMEOUT = float(os.getenv("PREDICT_API_TIMEOUT", "10"))
STARTUP_TIMEOUT = float(os.getenv("STARTUP_TIMEOUT", "30"))
RETRY_BACKOFF = float(os.getenv("RETRY_BACKOFF", "2"))

FEATURE_COLS = [
    "log_return",
    "spread_bps",
    "vol_60s",
    "mean_return_60s",
    "trade_intensity_60s",
    "n_ticks_60s",
    "spread_mean_60s",
]


def _wait_for_kafka(consumer: Consumer, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        try:
            consumer.list_topics(timeout=1.0)
            return
        except Exception as exc:  # pragma: no cover - integration behavior
            last_exc = exc
            time.sleep(1.0)
    raise RuntimeError(
        f"Kafka bootstrap {KAFKA_BOOTSTRAP!r} was not reachable within {timeout:.0f}s"
    ) from last_exc


def _wait_for_api(timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(API_HEALTH_URL, timeout=API_TIMEOUT) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:  # pragma: no cover - integration behavior
            last_exc = exc
            time.sleep(1.0)
    raise RuntimeError(
        f"Prediction API {API_HEALTH_URL!r} was not reachable within {timeout:.0f}s"
    ) from last_exc


def _isoformat_utc(timestamp_ms: int) -> str:
    return (
        datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _build_row(message: dict, kafka_timestamp_ms: int | None) -> dict:
    missing = [col for col in FEATURE_COLS if col not in message]
    if missing:
        raise KeyError(f"feature row missing required fields: {missing}")

    row = {col: message[col] for col in FEATURE_COLS}
    # Prefer the Kafka publish timestamp so API freshness reflects the real
    # feature-to-predict hop rather than the archived market capture time.
    if kafka_timestamp_ms and kafka_timestamp_ms > 0:
        row["ts"] = _isoformat_utc(kafka_timestamp_ms)
    elif "timestamp" in message and message["timestamp"]:
        row["ts"] = message["timestamp"]
    return row


def _post_prediction(row: dict) -> tuple[bool, str]:
    payload = json.dumps({"rows": [row]}).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:
            if 200 <= resp.status < 300:
                return True, ""
            return False, f"unexpected status {resp.status}"
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if 400 <= exc.code < 500:
            return True, f"non-retriable API error {exc.code}: {body}"
        return False, f"retriable API error {exc.code}: {body}"
    except urllib.error.URLError as exc:
        return False, f"API request failed: {exc}"


def main() -> None:
    consumer = Consumer(
        {
            "bootstrap.servers": KAFKA_BOOTSTRAP,
            "group.id": GROUP_ID,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([FEATURES_TOPIC])

    _wait_for_kafka(consumer, STARTUP_TIMEOUT)
    _wait_for_api(STARTUP_TIMEOUT)
    log.info(
        "Bridge started | %s -> %s | bootstrap=%s | group=%s",
        FEATURES_TOPIC,
        API_URL,
        KAFKA_BOOTSTRAP,
        GROUP_ID,
    )

    stop = False
    sent = 0

    def _shutdown(*_args) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        while not stop:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    log.error("Kafka error: %s", msg.error())
                continue

            try:
                feature_message = json.loads(msg.value())
                _, kafka_timestamp_ms = msg.timestamp()
                row = _build_row(feature_message, kafka_timestamp_ms)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                log.warning("Skipping malformed feature row: %s", exc)
                consumer.commit(message=msg)
                continue

            success, detail = _post_prediction(row)
            if not success:
                log.warning("Prediction POST failed; retrying after backoff: %s", detail)
                time.sleep(RETRY_BACKOFF)
                continue

            consumer.commit(message=msg)
            sent += 1
            if detail:
                log.warning("Prediction row skipped after client error: %s", detail)
            elif sent % 100 == 0:
                log.info("Sent %d prediction requests", sent)
    finally:
        consumer.close()
        log.info("Bridge stopped.")


if __name__ == "__main__":
    main()
