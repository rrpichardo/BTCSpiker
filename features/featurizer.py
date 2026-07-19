"""
Featurizer — live Kafka consumer.

Two output streams per tick
----------------------------
1. Compute features (midprice, spread, log-return, rolling vol/intensity)
   using the current price_buf / ts_buf deques.
2. Emit the CURRENT tick's unlabeled feature row immediately to
   ticks.features (real-time scoring — no delay, no label attached).
3. Push a pending entry {features, ts} onto the label delay buffer.
4. Drain pending entries whose lookahead window has closed
   (current_ts - pending_ts >= horizon_sec):
     a. Slice price_buf over [pending_ts, pending_ts + horizon_sec].
     b. Call compute_future_vol() on that slice.
     c. Assign vol_spike label (1 if future_vol > threshold else 0).
     d. Emit a labelled row to the Parquet batch (training sink, unchanged
        shape) and an OutcomeEvent (feature_id + label only) to
        ticks.outcomes (Kafka).
5. Flush Parquet batch every FLUSH_ROWS rows; also flush on clean shutdown.

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
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import yaml
from confluent_kafka import Consumer, KafkaError, Producer

from feature_funcs import (
    compute_future_vol,
    compute_midprice,
    compute_return,
    compute_rolling_stats,
    compute_spread,
    compute_spread_mean,
    compute_trade_intensity,
)
from parquet_sink import AtomicParquetSink

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

BUFFER_MAX_AGE = 600   # seconds of tick history kept in memory (≥ 2 × horizon)
FLUSH_ROWS     = 500   # Parquet row-group size

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
# Per-product state
# ---------------------------------------------------------------------------

class ProductState:
    """All mutable state for a single product_id."""

    def __init__(self, window_sec: float, horizon_sec: float, vol_threshold: float):
        self.window_sec    = window_sec
        self.horizon_sec   = horizon_sec
        self.vol_threshold = vol_threshold
        self.price_buf:  deque = deque()   # {"price": float, "ts": float}
        self.spread_buf: deque = deque()  # {"spread_abs": float, "ts": float}
        self.ts_buf:     deque = deque()  # float timestamps
        self.pending:    deque = deque()  # {"row": dict, "ts": float, "feature_id": str, "epoch": int}
        self.epoch:      int = 0          # bumps on timestamp regression (replay restarts)
        self.seq:        int = 0          # per-epoch feature counter, resets with epoch

    # ------------------------------------------------------------------
    def ingest(self, tick: dict) -> tuple[dict | None, list[tuple[dict, dict]]]:
        """
        Process one tick.

        Returns
        -------
        (feature_row, drained) where:
          feature_row : this tick's unlabeled feature dict, emitted
                        immediately (None only if feature computation
                        legitimately fails — see exception behavior below).
          drained     : list of (labelled_row, outcome_event) pairs for
                        pending entries whose lookahead window has just
                        closed. labelled_row is the Parquet training row
                        (unchanged shape); outcome_event is the delayed
                        label event for ticks.outcomes.
        """
        bid   = float(tick["best_bid"])
        ask   = float(tick["best_ask"])
        price = float(tick["price"])
        ts    = _parse_ts(tick["timestamp"])

        # Replay mode can restart from the beginning of the capture while Kafka
        # still holds newer historical ticks from a prior run. When event time
        # moves backwards, reset rolling state so feature emission can resume
        # cleanly instead of stalling behind an impossible lookahead window.
        if self.ts_buf and ts < self.ts_buf[-1]:
            log.warning(
                "Timestamp regression for %s: resetting rolling state (%s -> %s)",
                tick.get("product_id", "unknown"),
                self.ts_buf[-1],
                ts,
            )
            self.price_buf.clear()
            self.spread_buf.clear()
            self.ts_buf.clear()
            self.pending.clear()
            self.epoch += 1
            self.seq = 0

        mid    = compute_midprice(bid, ask)
        spread = compute_spread(bid, ask, mid)

        prev_price = self.price_buf[-1]["price"] if self.price_buf else price
        log_ret    = compute_return(price, prev_price)

        # Append before rolling stats so this tick is included
        self.price_buf.append({"price": price, "ts": ts})
        self.spread_buf.append({"spread_abs": spread["spread_abs"], "ts": ts})
        self.ts_buf.append(ts)

        # Prune entries older than BUFFER_MAX_AGE
        cutoff = ts - BUFFER_MAX_AGE
        while self.price_buf and self.price_buf[0]["ts"] < cutoff:
            self.price_buf.popleft()
        while self.spread_buf and self.spread_buf[0]["ts"] < cutoff:
            self.spread_buf.popleft()
        while self.ts_buf and self.ts_buf[0] < cutoff:
            self.ts_buf.popleft()

        rolling     = compute_rolling_stats(self.price_buf, self.window_sec)
        intensity   = compute_trade_intensity(self.ts_buf, self.window_sec)
        spread_mean = compute_spread_mean(self.spread_buf, self.window_sec)

        features = {
            "product_id":          tick["product_id"],
            "timestamp":           tick["timestamp"],
            "price":               price,
            "midprice":            mid,
            "log_return":          log_ret,
            "spread_abs":          spread["spread_abs"],
            "spread_bps":          spread["spread_bps"],
            "vol_60s":             rolling["vol"],
            "mean_return_60s":     rolling["mean_return"],
            "n_ticks_60s":         rolling["n_ticks"],
            "trade_intensity_60s": intensity,
            "spread_mean_60s":     spread_mean,
            "price_range_60s":     rolling["price_range"],
        }

        feature_id = f"{tick['product_id']}:{self.epoch}:{self.seq}"
        feature_row = {**features, "feature_id": feature_id, "stream_epoch": self.epoch}

        self.pending.append(
            {"row": features, "ts": ts, "feature_id": feature_id, "epoch": self.epoch}
        )
        self.seq += 1

        return feature_row, self._drain_pending(ts)

    # ------------------------------------------------------------------
    def drain_remaining(self) -> list[tuple[dict, dict]]:
        """On shutdown: flush pending rows that can still be labelled."""
        if not self.price_buf:
            return []
        ts_now = self.price_buf[-1]["ts"]
        return self._drain_pending(ts_now, force=True)

    # ------------------------------------------------------------------
    def _drain_pending(self, ts_now: float, force: bool = False) -> list[tuple[dict, dict]]:
        ready = []
        while self.pending:
            entry = self.pending[0]
            age   = ts_now - entry["ts"]

            if not force and age < self.horizon_sec:
                break  # nothing older is ready either

            self.pending.popleft()

            # Build a future-window slice from price_buf
            t_start = entry["ts"]
            t_end   = t_start + self.horizon_sec
            future_slice = deque(
                e for e in self.price_buf if t_start <= e["ts"] <= t_end
            )

            future_vol = compute_future_vol(future_slice, self.horizon_sec)

            if future_vol is None:
                # Never emit partially labelled rows; they contaminate training targets.
                continue

            vol_spike = int(future_vol > self.vol_threshold)

            # Parquet training row — shape is a pinned contract, never add columns here.
            labelled = {
                **entry["row"],
                "future_vol_60s": future_vol,
                "vol_spike":      vol_spike,
            }

            outcome_event = {
                "feature_id":     entry["feature_id"],
                "stream_epoch":   entry["epoch"],
                "product_id":     entry["row"]["product_id"],
                "feature_ts":     entry["row"]["timestamp"],
                "future_vol_60s": future_vol,
                "vol_spike":      vol_spike,
                "label_schema":   f"p85-{int(self.horizon_sec)}s-{self.vol_threshold}-v1",
            }

            ready.append((labelled, outcome_event))

        return ready


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
    window_sec     = float(cfg["features"]["window_seconds"])
    horizon_sec    = float(cfg["features"]["label_horizon_sec"])
    vol_threshold  = float(cfg["features"]["vol_threshold"])
    parquet_path   = Path(args.output_parquet or cfg["data"]["features_file"])

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

    states: dict[str, ProductState] = {}
    stop = False

    def _shutdown(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    log.info(
        "Featurizer started | %s → %s (immediate) + %s (outcomes, %.0fs delay) | "
        "parquet=%s | group=%s | offset=%s | window=%.0fs horizon=%.0fs",
        topic_in, topic_out, topic_outcomes, horizon_sec,
        parquet_path, group_id,
        "latest" if args.latest else "earliest", window_sec, horizon_sec,
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
            states[pid] = ProductState(window_sec, horizon_sec, vol_threshold)

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
    log.info("Featurizer stopped.")


if __name__ == "__main__":
    main()
