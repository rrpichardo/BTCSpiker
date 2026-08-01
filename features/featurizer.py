"""
Featurizer — live Kafka consumer.

Each product is processed by the shared causal ``FeatureEngine`` (core_v1),
wrapped per-product by ``ProductStream`` for feature_id/epoch bookkeeping.

Two output streams per tick
----------------------------
1. ``FeatureEngine.ingest_with_tag`` computes this tick's features and
   returns them immediately, tagged with a ``feature_id`` so the delayed
   label below can be correlated back to it.
2. The unlabeled row is emitted immediately to ticks.features (real-time
   scoring — no delay, no label attached).
3. Once a pending entry's lookahead window has closed
   (current_ts - pending_ts >= horizon_sec), ``FeatureEngine`` emits its
   labelled row paired with the original tag:
     a. Slice price history over [pending_ts, pending_ts + horizon_sec].
     b. Call compute_future_vol() on that slice.
     c. Assign vol_spike label (1 if future_vol > threshold else 0).
     d. Write the labelled row to the Parquet batch (training sink, unchanged
        shape) and publish an OutcomeEvent (feature_id + label only) to
        ticks.outcomes (Kafka).
4. Flush Parquet batch every FLUSH_ROWS rows; also flush on clean shutdown.

Parquet schema includes future_vol_60s and vol_spike label columns.

Usage
-----
    python features/featurizer.py \\
        [--config config.yaml] \\
        [--topic_in  ticks.raw] \\
        [--topic_out ticks.features] \\
        [--topic_outcomes ticks.outcomes] \\
        [--output_parquet data/processed/features.parquet]
"""

import argparse
import json
import logging
import os
import re
import signal
import sys
import time
import uuid
from collections import deque
from datetime import datetime
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
from btcspiker_ml.tick_identity import tick_dedupe_key
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


# ---------------------------------------------------------------------------
# Per-product stream identity
# ---------------------------------------------------------------------------

class ProductStream:
    """Wraps the shared ``FeatureEngine`` with feature_id/epoch bookkeeping
    for one product_id.

    Rolling-buffer correctness (resetting on timestamp regression) stays
    entirely inside ``FeatureEngine`` — this class never reaches into it.
    It tracks only the identity concerns the training/serving core has no
    reason to know about: a monotonically increasing ``feature_id`` per
    tick, and an ``epoch`` counter it bumps on its own regression check so a
    replay restart's early feature_ids can't collide with the prior run's.
    """

    # Bounded window of recent (product_id, timestamp, price, best_bid,
    # best_ask) keys used to catch tick redelivery — a websocket reconnect
    # replaying its last few messages, or a corrupt capture with the same
    # tick recorded twice (see the 2026-04-06 duplicated-fixture incident,
    # which put trade_intensity_60s/n_ticks_60s ~6 sigma off training).
    # Bounded rather than the immediately-preceding tick only, since
    # redelivery after a reconnect is not always adjacent; bounded rather
    # than unbounded, since a looping replay's *next pass* legitimately
    # re-emits the same ticks and must not be swallowed — the epoch bump on
    # timestamp regression (below) already fires at exactly that boundary
    # and clears this window.
    _DEDUPE_WINDOW = 256

    def __init__(self, feature_set_id: str, horizon_sec: float, vol_threshold: float, boot_id: str):
        self.engine        = FeatureEngine(feature_set_id, horizon_sec, vol_threshold)
        self.horizon_sec    = horizon_sec
        self.vol_threshold  = vol_threshold
        self.boot_id        = boot_id   # unique per process start, keeps feature_id
                                         # collision-free across restarts of the
                                         # stable, resuming consumer group
        self.epoch: int = 0
        self.seq:   int = 0
        self._last_ts: float | None = None
        self._recent_keys: deque = deque(maxlen=self._DEDUPE_WINDOW)
        self._recent_keys_set: set = set()
        self.duplicates_dropped: int = 0

    def _is_redelivery(self, tick: dict) -> bool:
        """True if `tick` exactly matches one already seen within the
        current dedupe window; remembers it either way."""
        key = tick_dedupe_key(tick)
        if key in self._recent_keys_set:
            return True
        if len(self._recent_keys) == self._recent_keys.maxlen:
            self._recent_keys_set.discard(self._recent_keys[0])
        self._recent_keys.append(key)
        self._recent_keys_set.add(key)
        return False

    def ingest(self, tick: dict) -> tuple[dict | None, list[tuple[dict, dict]]]:
        """
        Process one tick.

        Returns
        -------
        (feature_row, drained) where:
          feature_row : this tick's unlabeled feature dict, emitted
                        immediately, stamped with feature_id/stream_epoch —
                        or None if `tick` is an exact redelivery of one
                        already seen (see `_is_redelivery`), in which case
                        no feature_id/seq is consumed and the engine is not
                        touched.
          drained     : list of (labelled_row, outcome_event) pairs for
                        pending entries whose lookahead window has just
                        closed. labelled_row is the Parquet training row
                        (unchanged shape); outcome_event is the delayed
                        label event for ticks.outcomes.
        """
        ts = _parse_ts(tick["timestamp"])
        if self._last_ts is not None and ts < self._last_ts:
            log.warning(
                "Timestamp regression for %s: bumping stream epoch (%s -> %s)",
                tick.get("product_id", "unknown"), self._last_ts, ts,
            )
            self.epoch += 1
            self.seq = 0
            self._recent_keys.clear()
            self._recent_keys_set.clear()
        self._last_ts = ts

        if self._is_redelivery(tick):
            self.duplicates_dropped += 1
            if self.duplicates_dropped == 1:
                log.warning(
                    "Dropping redelivered tick for %s (exact duplicate within "
                    "the last %d ticks); further drops this boot are counted, "
                    "not logged individually",
                    tick.get("product_id", "unknown"), self._DEDUPE_WINDOW,
                )
            return None, []

        feature_id = f"{tick['product_id']}:{self.boot_id}:{self.epoch}:{self.seq}"
        stream_id = f"{self.boot_id}:{self.epoch}"
        row, drained = self.engine.ingest_with_tag(tick, tag=(feature_id, self.epoch, stream_id))
        self.seq += 1

        feature_row = {
            **row,
            "feature_id": feature_id,
            "stream_epoch": self.epoch,
            "stream_id": stream_id,
            "ingest_mode": tick.get("ingest_mode"),
        }
        return feature_row, [self._to_outcome(labelled, tag) for labelled, tag in drained]

    def drain_remaining(self) -> list[tuple[dict, dict]]:
        """On shutdown: flush pending rows that can still be labelled."""
        drained = self.engine.drain_remaining_tagged()
        return [self._to_outcome(labelled, tag) for labelled, tag in drained]

    def _to_outcome(self, labelled: dict, tag: object) -> tuple[dict, dict]:
        feature_id, epoch, stream_id = tag
        outcome_event = {
            "feature_id":     feature_id,
            "stream_epoch":   epoch,
            "stream_id":      stream_id,
            "product_id":     labelled["product_id"],
            "feature_ts":     labelled["timestamp"],
            "future_vol_60s": labelled["future_vol_60s"],
            "vol_spike":      labelled["vol_spike"],
            "label_schema":   f"p85-{int(self.horizon_sec)}s-{self.vol_threshold}-v1",
        }
        return labelled, outcome_event


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_ts(ts_str: str) -> float:
    # Truncate nanosecond precision to microseconds; fromisoformat supports up to 6 digits
    ts_str = re.sub(r"(\.\d{6})\d+", r"\1", ts_str)
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()


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
    parser = argparse.ArgumentParser(
        description="Tick featurizer: immediate feature stream + 60s-delayed outcome events"
    )
    parser.add_argument("--config",         default="config.yaml")
    parser.add_argument("--topic_in",       default=None, help="Override ticks.raw topic")
    parser.add_argument("--topic_out",      default=None, help="Override ticks.features topic")
    parser.add_argument("--topic_outcomes", default=None, help="Override ticks.outcomes topic")
    parser.add_argument("--output_parquet", default=None, help="Override Parquet output path")
    parser.add_argument("--group-id",       default=None,
                        help="Kafka consumer group id. Default creates a throwaway rerunnable group.")
    parser.add_argument("--latest",         action="store_true",
                        help="Only consume new ticks arriving after startup.")
    parser.add_argument("--startup-timeout", type=float, default=10.0,
                        help="Seconds to wait for Kafka before failing (default: 10)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    bootstrap      = os.getenv("KAFKA_BOOTSTRAP", cfg["kafka"]["bootstrap_servers"])
    topic_in       = args.topic_in       or cfg["kafka"]["topic_raw"]
    topic_out      = args.topic_out      or cfg["kafka"]["topic_features"]
    topic_outcomes = args.topic_outcomes or os.getenv("TOPIC_OUTCOMES", "ticks.outcomes")
    group_id       = args.group_id or f"{cfg['kafka']['group_id']}-{uuid.uuid4().hex[:8]}"
    horizon_sec    = float(cfg["features"]["label_horizon_sec"])
    vol_threshold  = float(cfg["features"]["vol_threshold"])
    parquet_path   = Path(args.output_parquet or cfg["data"]["features_file"])

    # Unconditional, independent of group_id: the deployed group_id is stable
    # (ticks-featurizer) so a restart resumes mid-stream while ProductStream's
    # epoch/seq would otherwise restart at 0 — boot_id keeps every feature_id
    # collision-free across restarts (see ProductStream.ingest).
    boot_id = uuid.uuid4().hex[:8]

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

    states: dict[str, ProductStream] = {}
    stop = False

    def _shutdown(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    log.info(
        "Featurizer started | boot_id=%s | %s → %s (immediate) + %s (outcomes, %.0fs delay) | "
        "parquet=%s | group=%s | offset=%s | feature_set=%s",
        boot_id, topic_in, topic_out, topic_outcomes, horizon_sec,
        parquet_path, group_id,
        "latest" if args.latest else "earliest", FEATURE_SET_ID,
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
            states[pid] = ProductStream(FEATURE_SET_ID, horizon_sec, vol_threshold, boot_id)

        try:
            feature_row, drained = states[pid].ingest(tick)
        except (KeyError, ValueError, TypeError) as exc:
            log.warning("Featurize error %s: %s | tick=%s", pid, exc, tick)
            continue

        if feature_row is not None:
            # Immediate, unlabeled feature row — published as soon as its tick arrives.
            value = json.dumps(feature_row)
            try:
                producer.produce(topic_out, key=pid, value=value, callback=_delivery_report)
                producer.poll(0)
            except Exception as e:
                log.warning("Kafka publish failed, continuing: %s", e)
            log.debug("Emitted feature: %s", feature_row)

        for labelled_row, outcome_event in drained:
            sink.write(labelled_row)
            # Delayed label event; on failure log and continue so Parquet sink still gets the row
            outcome_value = json.dumps(outcome_event)
            try:
                producer.produce(
                    topic_outcomes,
                    key=outcome_event["feature_id"],
                    value=outcome_value,
                    callback=_delivery_report,
                )
                producer.poll(0)
            except Exception as e:
                log.warning("Kafka publish failed, continuing: %s", e)
            log.debug("Emitted outcome: %s", outcome_event)

    # Drain any pending rows on shutdown
    for state in states.values():
        for labelled_row, outcome_event in state.drain_remaining():
            sink.write(labelled_row)
            outcome_value = json.dumps(outcome_event)
            # Same graceful fallback on shutdown drain — don't lose Parquet rows if Kafka is down
            try:
                producer.produce(
                    topic_outcomes,
                    key=outcome_event["feature_id"],
                    value=outcome_value,
                    callback=_delivery_report,
                )
                producer.poll(0)
            except Exception as e:
                log.warning("Kafka publish failed, continuing: %s", e)

    producer.flush()
    consumer.close()
    sink.close()
    total_duplicates = sum(state.duplicates_dropped for state in states.values())
    if total_duplicates:
        log.warning("Dropped %d redelivered tick(s) this run.", total_duplicates)
    log.info("Featurizer stopped.")


if __name__ == "__main__":
    main()
