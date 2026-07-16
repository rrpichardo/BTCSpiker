"""
Replay NDJSON tick file → Kafka topic ticks.raw, paced by tick timestamps.

Used for Week 4 replay-mode demo: reads handoff/data_sample/raw_slice.ndjson
(or any NDJSON file) and publishes each tick to Kafka at real-time pace.

Usage:
    python scripts/replay_to_kafka.py \\
        [--file handoff/data_sample/raw_slice.ndjson] \\
        [--speed 1.0] [--loop]
"""

import argparse
import json
import logging
import os
import re
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

from confluent_kafka import Producer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("replay_to_kafka")

KAFKA_BOOTSTRAP = (
    os.getenv("KAFKA_BOOTSTRAP")
    or os.getenv("KAFKA_BOOTSTRAP_SERVERS")
    or "localhost:9092"
)
TOPIC = os.getenv("TOPIC_RAW", "ticks.raw")
# Env-var speed multiplier so CI can fast-forward replay without touching CLI args
REPLAY_SPEED = float(os.getenv("REPLAY_SPEED", "1.0"))


def parse_ts(s: str) -> float:
    s = re.sub(r"(\.\d{6})\d+", r"\1", s)
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def make_producer() -> Producer:
    return Producer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "linger.ms": 50,
        "compression.type": "lz4",
        "acks": "all",
    })


def wait_for_kafka(producer: Producer, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        try:
            producer.list_topics(timeout=1.0)
            return
        except Exception as exc:
            last_exc = exc
            time.sleep(1.0)
    raise RuntimeError(
        f"Kafka bootstrap {KAFKA_BOOTSTRAP!r} not reachable within {timeout:.0f}s"
    ) from last_exc


def _delivery(err, _msg):
    if err:
        log.error("delivery failed: %s", err)


def replay(path: Path, speed: float, loop: bool, stop: dict) -> int:
    producer = make_producer()
    wait_for_kafka(producer)

    total = 0
    pass_num = 0
    while not stop["flag"]:
        pass_num += 1
        with open(path) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            log.error("no ticks in %s", path)
            return 0

        first_ts = parse_ts(json.loads(lines[0])["timestamp"])
        wall_start = time.monotonic()

        for line in lines:
            if stop["flag"]:
                break
            tick = json.loads(line)
            target = (parse_ts(tick["timestamp"]) - first_ts) / max(speed, 0.001)
            sleep_for = target - (time.monotonic() - wall_start)
            if sleep_for > 0:
                time.sleep(sleep_for)

            producer.produce(
                TOPIC,
                key=tick.get("product_id", "BTC-USD"),
                value=line,
                callback=_delivery,
            )
            producer.poll(0)
            total += 1
            if total % 500 == 0:
                log.info("replayed %d ticks (pass %d)", total, pass_num)

        log.info("pass %d complete (%d ticks cumulative)", pass_num, total)
        if not loop:
            break

    remaining = producer.flush(10.0)
    if remaining:
        log.warning("flush left %d undelivered messages", remaining)
    return total


def main():
    parser = argparse.ArgumentParser(description="Replay NDJSON → Kafka ticks.raw")
    parser.add_argument(
        "--file",
        default="handoff/data_sample/raw_slice.ndjson",
        help="NDJSON tick file to replay",
    )
    parser.add_argument(
        "--speed",
        type=float,
        # CLI arg falls back to env var so CI can set REPLAY_SPEED=50 without
        # changing the docker-compose command line
        default=REPLAY_SPEED,
        help="Playback speed multiplier (1.0 = real-time, overrides REPLAY_SPEED env var)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop indefinitely (useful for long-running demo)",
    )
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        log.error("file not found: %s", path)
        sys.exit(1)

    stop = {"flag": False}

    def _shutdown(*_):
        log.info("shutdown signal received")
        stop["flag"] = True

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    log.info(
        "Replaying %s → topic=%s bootstrap=%s speed=%.2fx loop=%s",
        path, TOPIC, KAFKA_BOOTSTRAP, args.speed, args.loop,
    )
    total = replay(path, args.speed, args.loop, stop)
    log.info("done (%d ticks total)", total)


if __name__ == "__main__":
    main()
