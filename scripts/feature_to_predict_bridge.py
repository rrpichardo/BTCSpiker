"""
Consume engineered feature rows from Kafka and POST them to the prediction API.

This is the runtime bridge that closes the replay loop:
    ticks.raw -> ticks.features -> /predict -> Prometheus metrics

Usage:
    python scripts/feature_to_predict_bridge.py
"""

import json
import logging
import math
import os
import signal
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from numbers import Real

from confluent_kafka import Consumer, KafkaError, Message, Producer

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
TOPIC_PREDICTIONS = os.getenv("TOPIC_PREDICTIONS", "ticks.predictions")
GROUP_ID = os.getenv("KAFKA_GROUP_ID", "predict-bridge")
API_URL = os.getenv("PREDICT_API_URL", "http://localhost:8000/predict")
API_HEALTH_URL = os.getenv("PREDICT_API_HEALTH_URL", "http://localhost:8000/health")
API_TIMEOUT = float(os.getenv("PREDICT_API_TIMEOUT", "10"))
STARTUP_TIMEOUT = float(os.getenv("STARTUP_TIMEOUT", "30"))
RETRY_BACKOFF = float(os.getenv("RETRY_BACKOFF", "2"))

# feature_id/stream_epoch are read separately (see feature_message.get(...)
# below) to stamp provenance on outgoing predictions; forwarding them into
# the /predict row would make the API 422 on every request, since it treats
# any key outside {"ts", "feature_set_id", "feature_schema_version"} as a
# numeric model feature.
NON_MODEL_FIELDS = {
    "product_id",
    "timestamp",
    "future_vol_60s",
    "vol_spike",
    "feature_id",
    "stream_epoch",
}


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
    row = {key: value for key, value in message.items() if key not in NON_MODEL_FIELDS}
    # Prefer the Kafka publish timestamp so API freshness reflects the real
    # feature-to-predict hop rather than the archived market capture time.
    if kafka_timestamp_ms and kafka_timestamp_ms > 0:
        row["ts"] = _isoformat_utc(kafka_timestamp_ms)
    else:
        row["ts"] = message.get("timestamp")
    return row


def _post_prediction(row: dict) -> tuple[bool, str, dict | None]:
    """POST a feature row to the prediction API.

    Returns (success, detail, response_payload). response_payload is only
    populated when the API returned 2xx with a parseable scored body — a
    non-retriable 4xx (row skipped) still counts as "success" but carries
    no payload, matching the existing skip semantics.
    """
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
                body = resp.read().decode("utf-8")
                try:
                    response_payload = json.loads(body)
                    if not isinstance(response_payload, dict):
                        raise ValueError("body must be a mapping")

                    scores = response_payload.get("scores")
                    if not isinstance(scores, list) or len(scores) != 1:
                        raise ValueError("scores must contain exactly one value")

                    score = scores[0]
                    if (
                        isinstance(score, bool)
                        or not isinstance(score, Real)
                        or (isinstance(score, float) and not math.isfinite(score))
                    ):
                        raise ValueError("score must be a finite number")

                    for field in ("ts", "model_variant", "version"):
                        value = response_payload.get(field)
                        if not isinstance(value, str) or not value.strip():
                            raise ValueError(f"{field} must be a nonempty string")
                except (json.JSONDecodeError, ValueError) as exc:
                    return False, f"malformed API response: {exc}", None
                return True, "", response_payload
            return False, f"unexpected status {resp.status}", None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if 400 <= exc.code < 500:
            return True, f"non-retriable API error {exc.code}: {body}", None
        return False, f"retriable API error {exc.code}: {body}", None
    except urllib.error.URLError as exc:
        return False, f"API request failed: {exc}", None


def _build_prediction_event(
    msg: Message,
    row: dict,
    response_payload: dict,
    *,
    feature_ts: str | None,
    feature_id: str | None,
    stream_epoch: int | None,
) -> dict:
    """Assemble the PredictionEvent for the consumed features message.

    event_id is derived from the consumed message's own topic/partition/offset
    so retries of the same message produce the same id (downstream dedup key).

    feature_id/stream_epoch come from the raw consumed feature message (old-
    format backlog messages lack them, so they fall back to null rather than
    crashing). tau/run_id come from the /predict response payload.
    """
    return {
        "event_id": f"{msg.topic()}:{msg.partition()}:{msg.offset()}",
        "source_partition": msg.partition(),
        "source_offset": msg.offset(),
        "feature_ts": feature_ts,
        "feature_id": feature_id,
        "stream_epoch": stream_epoch,
        "api_ts": response_payload.get("ts"),
        "score": response_payload["scores"][0],
        "model_variant": response_payload.get("model_variant"),
        "model_version": response_payload.get("version"),
        "tau": response_payload.get("tau"),
        "run_id": response_payload.get("run_id"),
        "vol_60s": row.get("vol_60s"),
        "spread_bps": row.get("spread_bps"),
        "log_return": row.get("log_return"),
        "trade_intensity_60s": row.get("trade_intensity_60s"),
    }


def _publish_prediction(producer: Producer, event: dict) -> bool:
    """Produce the event and block until its delivery outcome is known."""
    delivered = {"ok": False}

    def _on_delivery(err, _msg) -> None:
        if err is not None:
            log.warning(
                "Prediction event delivery failed for %s: %s", event["event_id"], err
            )
        else:
            delivered["ok"] = True

    producer.produce(
        TOPIC_PREDICTIONS,
        key=event["event_id"],
        value=json.dumps(event).encode("utf-8"),
        callback=_on_delivery,
    )
    producer.poll(0)
    remaining = producer.flush(API_TIMEOUT)
    if remaining:
        log.warning(
            "Prediction event publish timed out for %s (%d message(s) still queued)",
            event["event_id"],
            remaining,
        )
    return delivered["ok"]


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
    producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP})

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

            while not stop:
                success, detail, response_payload = _post_prediction(row)
                if success:
                    break
                log.warning(
                    "Prediction POST failed; retrying after backoff: %s", detail
                )
                time.sleep(RETRY_BACKOFF)

            if stop:
                continue

            if response_payload is None:
                # Non-retriable 4xx: row skipped, no score to publish.
                consumer.commit(message=msg)
                sent += 1
                log.warning("Prediction row skipped after client error: %s", detail)
                continue

            event = _build_prediction_event(
                msg,
                row,
                response_payload,
                feature_ts=feature_message.get("timestamp"),
                feature_id=feature_message.get("feature_id"),
                stream_epoch=feature_message.get("stream_epoch"),
            )
            while not stop and not _publish_prediction(producer, event):
                log.warning(
                    "Prediction event publish unconfirmed; retrying after backoff: %s",
                    event["event_id"],
                )
                time.sleep(RETRY_BACKOFF)

            if stop:
                continue

            consumer.commit(message=msg)
            sent += 1
            if sent % 100 == 0:
                log.info("Sent %d prediction requests", sent)
    finally:
        producer.flush(API_TIMEOUT)
        consumer.close()
        log.info("Bridge stopped.")


if __name__ == "__main__":
    main()
