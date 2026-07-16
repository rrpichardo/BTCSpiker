"""
WebSocket ingestor — streams Coinbase ticker data into Kafka topic ticks.raw.

Two concurrent tasks run per invocation:
  • ticker    — price/volume updates → Kafka + optional file mirror
  • heartbeats — keeps the connection alive; tracks counter to detect gaps

Usage:
    python scripts/ws_ingest.py [--pair BTC-USD] [--minutes 0] [--no-mirror]
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

from confluent_kafka import Producer
from dotenv import load_dotenv
import websockets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

WS_URL = "wss://advanced-trade-ws.coinbase.com"
load_dotenv()

KAFKA_BOOTSTRAP = (
    os.getenv("KAFKA_BOOTSTRAP")
    or os.getenv("KAFKA_BOOTSTRAP_SERVERS")
    or "localhost:9092"
)
TOPIC = os.getenv("TOPIC_RAW", "ticks.raw")
BACKOFF_MIN = 0.5
BACKOFF_MAX = 60.0
# Circuit breaker — exit after this many consecutive reconnect failures
MAX_CONSECUTIVE_FAILURES = 10
MAX_QUEUED_MESSAGES = 50_000


# ---------------------------------------------------------------------------
# Kafka helpers
# ---------------------------------------------------------------------------

def make_producer() -> Producer:
    return Producer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        # --- Batching: accumulate messages before sending to reduce overhead ---
        "linger.ms": 50,               # wait up to 50 ms to fill a batch
        "batch.num.messages": 500,      # cap each batch at 500 messages
        "batch.size": 65536,            # cap each batch at 64 KB
        # --- Reliability ---
        "acks": "all",                  # wait for all in-sync replicas to ack
        "retries": 5,                   # retry transient broker failures
        "retry.backoff.ms": 200,        # pause between retries
        # --- Throughput ---
        "compression.type": "lz4",      # compress batches; lz4 = fast + decent ratio
        "queue.buffering.max.messages": 100000,  # internal buffer before back-pressure
    })


def delivery_report(err, msg):
    if err:
        log.error("Delivery failed for %s: %s", msg.key(), err)


def wait_for_kafka(producer: Producer, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        try:
            producer.list_topics(timeout=1.0)
            return
        except Exception as exc:  # pragma: no cover - exercised in integration
            last_exc = exc
            time.sleep(1.0)
    raise RuntimeError(
        f"Kafka bootstrap {KAFKA_BOOTSTRAP!r} was not reachable within {timeout:.0f}s"
    ) from last_exc


# ---------------------------------------------------------------------------
# File mirror helper — buffered to reduce syscalls
# ---------------------------------------------------------------------------

MIRROR_FLUSH_SIZE = 100  # flush to disk every N lines

def mirror_dir() -> Path:
    path = Path("data/raw")
    path.mkdir(parents=True, exist_ok=True)
    return path


class MirrorWriter:
    """Buffers NDJSON chunks and commits each flush atomically as its own segment."""

    def __init__(self, pair: str):
        self._pair = pair
        self._buf: list[str] = []
        self._segment_counter = 0

    def write(self, value: str) -> None:
        self._buf.append(value + "\n")
        if len(self._buf) >= MIRROR_FLUSH_SIZE:
            self.flush()

    def _next_segment_path(self) -> Path:
        now = datetime.now(timezone.utc)
        stamp = now.strftime("%Y%m%d_%H%M%S_%f")
        self._segment_counter += 1
        return mirror_dir() / f"{self._pair}_{stamp}_{self._segment_counter:06d}.ndjson"

    def flush(self) -> None:
        if not self._buf:
            return
        chunk = "".join(self._buf)
        target = self._next_segment_path()
        with NamedTemporaryFile("w", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False) as fh:
            fh.write(chunk)
            fh.flush()
            os.fsync(fh.fileno())
            tmp_path = Path(fh.name)
        tmp_path.replace(target)
        self._buf.clear()


# ---------------------------------------------------------------------------
# Heartbeat task
# ---------------------------------------------------------------------------

async def heartbeat_task(pair: str, stop_event: asyncio.Event, circuit_broken: asyncio.Event):
    """Subscribe to the heartbeats channel and track sequence gaps."""
    channel = "heartbeats"
    subscribe_msg = json.dumps({
        "type": "subscribe",
        "product_ids": [pair],
        "channel": channel,
    })

    backoff = BACKOFF_MIN
    last_seq: int | None = None
    # Running total of missed messages across the session for observability
    total_gaps = 0
    # Track consecutive connection failures for circuit breaker
    consecutive_failures = 0

    try:
        while not stop_event.is_set():
            try:
                async for ws in websockets.connect(WS_URL):
                    try:
                        await ws.send(subscribe_msg)
                        log.info("[heartbeats] subscribed for %s", pair)
                        # Successful connection resets backoff and failure counter
                        backoff = BACKOFF_MIN
                        consecutive_failures = 0
                        # Reset sequence on new connection — numbers restart per session
                        last_seq = None

                        async for raw in ws:
                            if stop_event.is_set():
                                return

                            msg = json.loads(raw)
                            if msg.get("channel") != channel:
                                continue

                            seq = msg.get("sequence_num")
                            if seq is not None:
                                if last_seq is not None and seq != last_seq + 1:
                                    # Calculate how many messages we missed
                                    missed = seq - last_seq - 1
                                    total_gaps += missed
                                    log.warning(
                                        "[heartbeats] gap detected: expected %d, got %d "
                                        "(missed %d msg(s), %d total gaps this session)",
                                        last_seq + 1, seq, missed, total_gaps,
                                    )
                                last_seq = seq

                    except websockets.ConnectionClosed as exc:
                        consecutive_failures += 1
                        # Trip circuit breaker if too many consecutive failures
                        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                            log.error("[heartbeats] circuit breaker tripped after %d consecutive failures", consecutive_failures)
                            circuit_broken.set()
                            stop_event.set()
                            return
                        log.warning("[heartbeats] connection closed (%d/%d): %s — reconnecting in %.1fs",
                                    consecutive_failures, MAX_CONSECUTIVE_FAILURES, exc, backoff)
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, BACKOFF_MAX)

            except Exception as exc:
                consecutive_failures += 1
                # Trip circuit breaker if too many consecutive failures
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    log.error("[heartbeats] circuit breaker tripped after %d consecutive failures", consecutive_failures)
                    circuit_broken.set()
                    stop_event.set()
                    return
                log.error("[heartbeats] unexpected error (%d/%d): %s — reconnecting in %.1fs",
                          consecutive_failures, MAX_CONSECUTIVE_FAILURES, exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, BACKOFF_MAX)
    except asyncio.CancelledError:
        return


# ---------------------------------------------------------------------------
# Ticker task
# ---------------------------------------------------------------------------

async def ticker_task(
    pair: str,
    producer: Producer,
    stop_event: asyncio.Event,
    mirror: bool,
    circuit_broken: asyncio.Event,
):
    """Subscribe to the ticker channel; produce each tick to Kafka."""
    channel = "ticker"
    subscribe_msg = json.dumps({
        "type": "subscribe",
        "product_ids": [pair],
        "channel": channel,
    })

    backoff = BACKOFF_MIN
    # Track messages since last poll so we drain callbacks periodically
    poll_interval = 50  # poll every N messages instead of every message
    msg_count = 0
    # Buffered mirror writer — flushes every MIRROR_FLUSH_SIZE lines
    mirror_writer = MirrorWriter(pair) if mirror else None
    # Track consecutive connection failures for circuit breaker
    consecutive_failures = 0
    # Sequence gap tracking for feed integrity
    last_seq: int | None = None
    total_gaps = 0

    try:
        while not stop_event.is_set():
            try:
                async for ws in websockets.connect(WS_URL):
                    try:
                        await ws.send(subscribe_msg)
                        log.info("[ticker] subscribed for %s → topic '%s'", pair, TOPIC)
                        # Successful connection resets backoff and failure counter
                        backoff = BACKOFF_MIN
                        consecutive_failures = 0
                        # Reset sequence on new connection — numbers restart per session
                        last_seq = None

                        async for raw in ws:
                            if stop_event.is_set():
                                return

                            msg = json.loads(raw)
                            if msg.get("channel") != channel:
                                continue

                            # Check for sequence gaps before processing the tick
                            seq = msg.get("sequence_num")
                            if seq is not None:
                                if last_seq is not None and seq != last_seq + 1:
                                    missed = seq - last_seq - 1
                                    total_gaps += missed
                                    log.warning(
                                        "[ticker] sequence gap: expected %d, got %d "
                                        "(missed %d msg(s), %d total gaps this session)",
                                        last_seq + 1, seq, missed, total_gaps,
                                    )
                                last_seq = seq

                            ts = msg.get("timestamp", "")
                            for event in msg.get("events", []):
                                for tick in event.get("tickers", []):
                                    # Validate required fields before producing — reject nulls early
                                    # so poisoned ticks never reach the featurizer downstream
                                    price    = tick.get("price")
                                    best_bid = tick.get("best_bid")
                                    best_ask = tick.get("best_ask")
                                    if price is None or best_bid is None or best_ask is None or not ts:
                                        log.warning("[ticker] skipped tick with null field(s): price=%s bid=%s ask=%s ts=%s",
                                                    price, best_bid, best_ask, ts)
                                        continue

                                    payload = {
                                        "product_id":  tick.get("product_id", pair),
                                        "price":        price,
                                        "best_bid":     best_bid,
                                        "best_ask":     best_ask,
                                        "volume_24_h":  tick.get("volume_24_h"),
                                        "timestamp":    ts,
                                    }
                                    value = json.dumps(payload)

                                    producer.produce(
                                        TOPIC,
                                        key=payload["product_id"],
                                        value=value,
                                        callback=delivery_report,
                                    )
                                    # Poll periodically instead of every message to let batching work
                                    msg_count += 1
                                    if msg_count % poll_interval == 0:
                                        producer.poll(0)
                                        queued = len(producer)
                                        if queued > MAX_QUEUED_MESSAGES:
                                            raise RuntimeError(
                                                f"Kafka producer backlog grew to {queued} queued message(s)"
                                            )
                                    log.debug("[ticker] produced: %s", payload)

                                    if mirror_writer:
                                        mirror_writer.write(value)

                    except websockets.ConnectionClosed as exc:
                        consecutive_failures += 1
                        if mirror_writer:
                            mirror_writer.flush()
                        # Trip circuit breaker if too many consecutive failures
                        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                            log.error("[ticker] circuit breaker tripped after %d consecutive failures", consecutive_failures)
                            circuit_broken.set()
                            stop_event.set()
                            return
                        log.warning("[ticker] connection closed (%d/%d): %s — reconnecting in %.1fs",
                                    consecutive_failures, MAX_CONSECUTIVE_FAILURES, exc, backoff)
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, BACKOFF_MAX)

            except Exception as exc:
                consecutive_failures += 1
                if mirror_writer:
                    mirror_writer.flush()
                # Trip circuit breaker if too many consecutive failures
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    log.error("[ticker] circuit breaker tripped after %d consecutive failures", consecutive_failures)
                    circuit_broken.set()
                    stop_event.set()
                    return
                log.error("[ticker] unexpected error (%d/%d): %s — reconnecting in %.1fs",
                          consecutive_failures, MAX_CONSECUTIVE_FAILURES, exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, BACKOFF_MAX)
    except asyncio.CancelledError:
        if mirror_writer:
            mirror_writer.flush()
        return


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Coinbase WebSocket → Kafka ingestor")
    parser.add_argument("--pair", default="BTC-USD", help="Trading pair (default: BTC-USD)")
    parser.add_argument("--minutes", type=float, default=0,
                        help="Run duration in minutes; 0 = run forever (default: 0)")
    parser.add_argument("--no-mirror", dest="mirror", action="store_false",
                        help="Disable NDJSON file mirror under data/raw/")
    parser.add_argument("--startup-timeout", type=float, default=10.0,
                        help="Seconds to wait for Kafka before failing (default: 10)")
    args = parser.parse_args()

    producer = make_producer()
    try:
        wait_for_kafka(producer, args.startup_timeout)
    except RuntimeError as exc:
        log.error("%s", exc)
        sys.exit(1)
    stop_event = asyncio.Event()
    # Shared flag — set by any task that trips the circuit breaker
    circuit_broken = asyncio.Event()

    async def run():
        loop = asyncio.get_running_loop()

        def _shutdown(*_):
            log.info("Shutdown signal received")
            loop.call_soon_threadsafe(stop_event.set)

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        tasks = [
            asyncio.create_task(ticker_task(args.pair, producer, stop_event, args.mirror, circuit_broken)),
            asyncio.create_task(heartbeat_task(args.pair, stop_event, circuit_broken)),
        ]
        try:
            if args.minutes > 0:
                log.info("Will run for %.1f minute(s)", args.minutes)
                await asyncio.wait_for(stop_event.wait(), timeout=args.minutes * 60)
            else:
                await stop_event.wait()
        except asyncio.TimeoutError:
            stop_event.set()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            remaining = producer.flush(10.0)
            if remaining:
                log.error("Kafka producer stopped with %d undelivered message(s)", remaining)
                circuit_broken.set()
            log.info("Ingestor stopped — Kafka buffer flushed.")

    asyncio.run(run())
    # Exit with error code 1 if circuit breaker tripped, 0 for clean shutdown
    sys.exit(1 if circuit_broken.is_set() else 0)


if __name__ == "__main__":
    main()
