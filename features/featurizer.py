"""
Featurizer — live Kafka consumer.

Each product is processed by the shared causal ``FeatureEngine``. It produces
core_v1 features and only releases a row once its lookahead label has closed.
Labelled rows are emitted to Kafka and Parquet in bounded batches.

Parquet schema includes future_vol_60s and vol_spike label columns.

Usage
-----
    python features/featurizer.py \\
        [--config config.yaml] \\
        [--topic_in  ticks.raw] \\
        [--topic_out ticks.features] \\
        [--output_parquet data/processed/features.parquet]
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
import uuid
from pathlib import Path

# Direct invocation (``python features/featurizer.py``) otherwise puts only
# ``features/`` on sys.path, not the project root containing btcspiker_ml.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pyarrow as pa
import yaml
from confluent_kafka import Consumer, KafkaError, Producer

from btcspiker_ml.features import FeatureEngine
from parquet_sink import AtomicParquetSink

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

FLUSH_ROWS     = 500   # Parquet row-group size
FEATURE_SET_ID = "core_v1"

PARQUET_SCHEMA = pa.schema([
    ("product_id",          pa.string()),
    ("timestamp",           pa.string()),
    ("price",               pa.float64()),
    ("midprice",            pa.float64()),
    ("log_return",          pa.float64()),
    ("spread_abs",          pa.float64()),
    ("spread_bps",          pa.float64()),
    ("vol_60s",             pa.float64()),
    ("mean_return_60s",     pa.float64()),
    ("n_ticks_60s",         pa.int64()),
    ("trade_intensity_60s", pa.float64()),
    ("spread_mean_60s",     pa.float64()),
    ("price_range_60s",     pa.float64()),
    ("feature_set_id",      pa.string()),
    ("feature_schema_version", pa.string()),
    ("future_vol_60s",      pa.float64()),
    ("vol_spike",           pa.int64()),
])


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Parquet sink
# ---------------------------------------------------------------------------

class ParquetSink(AtomicParquetSink):
    """Buffers labelled feature rows and commits the output atomically."""

    def __init__(self, path: Path, flush_rows: int = FLUSH_ROWS):
        super().__init__(path=path, schema=PARQUET_SCHEMA, flush_rows=flush_rows)


def _delivery_report(err, _msg):
    if err:
        log.error("Delivery failed: %s", err)


def _wait_for_kafka(client, bootstrap: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        try:
            client.list_topics(timeout=1.0)
            return
        except Exception as exc:  # pragma: no cover - exercised in integration
            last_exc = exc
            time.sleep(1.0)
    raise RuntimeError(
        f"Kafka bootstrap {bootstrap!r} was not reachable within {timeout:.0f}s"
    ) from last_exc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tick featurizer with 60s label delay")
    parser.add_argument("--config",         default="config.yaml")
    parser.add_argument("--topic_in",       default=None, help="Override ticks.raw topic")
    parser.add_argument("--topic_out",      default=None, help="Override ticks.features topic")
    parser.add_argument("--output_parquet", default=None, help="Override Parquet output path")
    parser.add_argument("--group-id",       default=None,
                        help="Kafka consumer group id. Default creates a throwaway rerunnable group.")
    parser.add_argument("--latest",         action="store_true",
                        help="Only consume new ticks arriving after startup.")
    parser.add_argument("--startup-timeout", type=float, default=10.0,
                        help="Seconds to wait for Kafka before failing (default: 10)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    bootstrap     = os.getenv("KAFKA_BOOTSTRAP", cfg["kafka"]["bootstrap_servers"])
    topic_in      = args.topic_in      or cfg["kafka"]["topic_raw"]
    topic_out     = args.topic_out     or cfg["kafka"]["topic_features"]
    group_id      = args.group_id or f"{cfg['kafka']['group_id']}-{uuid.uuid4().hex[:8]}"
    horizon_sec   = float(cfg["features"]["label_horizon_sec"])
    vol_threshold = float(cfg["features"]["vol_threshold"])
    parquet_path  = Path(args.output_parquet or cfg["data"]["features_file"])

    consumer = Consumer({
        "bootstrap.servers": bootstrap,
        "group.id":          group_id,
        "auto.offset.reset": "latest" if args.latest else "earliest",
        "enable.auto.commit": True,
    })
    consumer.subscribe([topic_in])

    producer = Producer({"bootstrap.servers": bootstrap})
    _wait_for_kafka(consumer, bootstrap, args.startup_timeout)
    sink     = ParquetSink(parquet_path)

    states: dict[str, FeatureEngine] = {}
    stop = False

    def _shutdown(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    log.info(
        "Featurizer started | %s → %s | parquet=%s | group=%s | offset=%s | feature_set=%s horizon=%.0fs",
        topic_in, topic_out, parquet_path, group_id,
        "latest" if args.latest else "earliest", FEATURE_SET_ID, horizon_sec,
    )

    while not stop:
        kmsg = consumer.poll(timeout=1.0)

        if kmsg is None:
            continue
        if kmsg.error():
            if kmsg.error().code() != KafkaError._PARTITION_EOF:
                log.error("Kafka error: %s", kmsg.error())
            continue

        try:
            tick = json.loads(kmsg.value())
        except json.JSONDecodeError as exc:
            log.warning("Bad JSON: %s", exc)
            continue

        pid = tick.get("product_id", "unknown")
        if pid not in states:
            states[pid] = FeatureEngine(FEATURE_SET_ID, horizon_sec, vol_threshold)

        try:
            labelled_rows = states[pid].ingest(tick)
        except (KeyError, ValueError, TypeError) as exc:
            log.warning("Featurize error %s: %s | tick=%s", pid, exc, tick)
            continue

        for row in labelled_rows:
            value = json.dumps(row)
            # Attempt Kafka publish; on failure log and continue so Parquet sink still gets the row
            try:
                producer.produce(topic_out, key=pid, value=value, callback=_delivery_report)
                producer.poll(0)
            except Exception as e:
                log.warning("Kafka publish failed, continuing: %s", e)
            sink.write(row)
            log.debug("Emitted: %s", row)

    # Drain any pending rows on shutdown
    for pid, state in states.items():
        for row in state.drain_remaining():
            value = json.dumps(row)
            # Same graceful fallback on shutdown drain — don't lose Parquet rows if Kafka is down
            try:
                producer.produce(topic_out, key=pid, value=value, callback=_delivery_report)
                producer.poll(0)
            except Exception as e:
                log.warning("Kafka publish failed, continuing: %s", e)
            sink.write(row)

    producer.flush()
    consumer.close()
    sink.close()
    log.info("Featurizer stopped.")


if __name__ == "__main__":
    main()
