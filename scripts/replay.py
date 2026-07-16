"""
Replay — batch feature builder from NDJSON mirror files.

Reads all NDJSON tick files matching a glob pattern, sorts lines by
timestamp across files, then feeds each tick through the same feature_funcs
calls used by the live featurizer.  Outputs a single Parquet file with
future-vol labels (60s lookahead by default).

State management mirrors featurizer.ProductState exactly so that feature
values are numerically identical to what the live pipeline would produce
given the same tick stream.

Usage
-----
    python scripts/replay.py
    python scripts/replay.py --raw "data/raw/**/*.ndjson" --out data/processed/features.parquet
    python scripts/replay.py --raw "data/raw/BTC-USD/*.ndjson" --config config.yaml
"""

import argparse
import glob
import heapq
import json
import logging
import re
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "features"))

import pyarrow as pa
import yaml

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

BUFFER_MAX_AGE = 600
FLUSH_ROWS     = 500

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


def _parse_ts(ts_str: str) -> float:
    # Truncate nanosecond precision to microseconds; fromisoformat supports up to 6 digits
    ts_str = re.sub(r"(\.\d{6})\d+", r"\1", ts_str)
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()


# ---------------------------------------------------------------------------
# Parquet sink
# ---------------------------------------------------------------------------

class ParquetSink(AtomicParquetSink):
    def __init__(self, path: Path):
        super().__init__(path=path, schema=PARQUET_SCHEMA, flush_rows=FLUSH_ROWS)


# ---------------------------------------------------------------------------
# Per-product state  (mirrors featurizer.ProductState)
# ---------------------------------------------------------------------------

class ProductState:
    def __init__(self, window_sec: float, horizon_sec: float, vol_threshold: float):
        self.window_sec    = window_sec
        self.horizon_sec   = horizon_sec
        self.vol_threshold = vol_threshold
        self.price_buf:  deque = deque()
        self.spread_buf: deque = deque()
        self.ts_buf:     deque = deque()
        self.pending:    deque = deque()   # {"row": dict, "ts": float}

    def ingest(self, tick: dict) -> list[dict]:
        bid   = float(tick["best_bid"])
        ask   = float(tick["best_ask"])
        price = float(tick["price"])
        ts    = _parse_ts(tick["timestamp"])

        mid    = compute_midprice(bid, ask)
        spread = compute_spread(bid, ask, mid)

        prev_price = self.price_buf[-1]["price"] if self.price_buf else price
        log_ret    = compute_return(price, prev_price)

        self.price_buf.append({"price": price, "ts": ts})
        self.spread_buf.append({"spread_abs": spread["spread_abs"], "ts": ts})
        self.ts_buf.append(ts)

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

        self.pending.append({"row": features, "ts": ts})
        return self._drain(ts)

    def drain_remaining(self) -> list[dict]:
        if not self.price_buf:
            return []
        return self._drain(self.price_buf[-1]["ts"], force=True)

    def _drain(self, ts_now: float, force: bool = False) -> list[dict]:
        ready = []
        while self.pending:
            entry = self.pending[0]
            if not force and ts_now - entry["ts"] < self.horizon_sec:
                break

            self.pending.popleft()
            t_start = entry["ts"]
            t_end   = t_start + self.horizon_sec
            future_slice = deque(
                e for e in self.price_buf if t_start <= e["ts"] <= t_end
            )

            future_vol = compute_future_vol(future_slice, self.horizon_sec)

            if future_vol is None:
                continue

            ready.append({
                **entry["row"],
                "future_vol_60s": future_vol,
                "vol_spike": int(future_vol > self.vol_threshold),
            })
        return ready


# ---------------------------------------------------------------------------
# NDJSON loading
# ---------------------------------------------------------------------------

def _discover_files(raw_inputs: list[str]) -> list[str]:
    files: list[str] = []
    for raw_input in raw_inputs:
        matches = sorted(glob.glob(raw_input, recursive=True))
        if matches:
            files.extend(matches)
        elif Path(raw_input).is_file():
            files.append(raw_input)

    files = [
        path for path in sorted(dict.fromkeys(files))
        if path.endswith(".ndjson") and not Path(path).name.startswith(".")
    ]
    if not files:
        raise FileNotFoundError(f"No files matched: {raw_inputs!r}")
    return files


def _iter_file_ticks(path: str):
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                tick = json.loads(line)
                yield _parse_ts(tick["timestamp"]), tick
            except (json.JSONDecodeError, KeyError) as exc:
                log.warning("%s:%d — skipped (%s)", path, lineno, exc)


def iter_ticks(raw_inputs: list[str]):
    """
    Yield (ts_float, tick_dict) for every valid line across all matching files,
    sorted by timestamp.
    """
    files = _discover_files(raw_inputs)
    log.info("Loading %d file(s) from %r", len(files), raw_inputs)

    iterators = [_iter_file_ticks(path) for path in files]
    heap: list[tuple[float, int, dict]] = []
    for idx, iterator in enumerate(iterators):
        try:
            ts, tick = next(iterator)
        except StopIteration:
            continue
        heapq.heappush(heap, (ts, idx, tick))

    duplicate_count = 0
    emitted = 0
    current_ts = None
    seen_for_ts: set[tuple] = set()

    while heap:
        ts, idx, tick = heapq.heappop(heap)
        dedupe_key = (
            tick.get("product_id"),
            tick.get("timestamp"),
            tick.get("price"),
            tick.get("best_bid"),
            tick.get("best_ask"),
            tick.get("volume_24_h"),
        )
        if current_ts != ts:
            current_ts = ts
            seen_for_ts.clear()

        if dedupe_key in seen_for_ts:
            duplicate_count += 1
        else:
            seen_for_ts.add(dedupe_key)
            emitted += 1
            yield ts, tick

        iterator = iterators[idx]
        try:
            next_ts, next_tick = next(iterator)
        except StopIteration:
            continue
        heapq.heappush(heap, (next_ts, idx, next_tick))

    log.info(
        "Loaded %d ticks across %d file(s) (%d duplicate lines skipped)",
        emitted,
        len(files),
        duplicate_count,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Replay NDJSON ticks → Parquet features")
    parser.add_argument(
        "--raw",
        nargs="+",
        default=["data/raw/**/*.ndjson"],
        help='One or more globs/files for NDJSON ticks (default: "data/raw/**/*.ndjson")',
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output Parquet path (default: data.features_file from config)",
    )
    parser.add_argument("--config", default="config.yaml")
    # Optional wall-clock cap: stop after N minutes of tick-timestamp elapsed
    parser.add_argument(
        "--minutes",
        type=float,
        default=None,
        help="Only replay the first N minutes of ticks (by tick timestamp). Default: no cap.",
    )
    args = parser.parse_args()

    cfg           = load_config(args.config)
    window_sec    = float(cfg["features"]["window_seconds"])
    horizon_sec   = float(cfg["features"]["label_horizon_sec"])
    vol_threshold = float(cfg["features"]["vol_threshold"])
    out_path      = Path(args.out or cfg["data"]["features_file"])

    sink   = ParquetSink(out_path)
    states: dict[str, ProductState] = {}
    emitted = 0
    pending_count = 0

    # Convert --minutes to a wall-clock cut-off in seconds; None means no cap
    max_seconds = args.minutes * 60.0 if args.minutes else None
    # Anchor is set from the first tick's timestamp so the cap is relative, not absolute
    t_anchor: float | None = None

    for _ts, tick in iter_ticks(args.raw):
        # Establish the replay start time from the very first tick we see
        if t_anchor is None:
            t_anchor = _ts
        # Stop once we've walked past the --minutes window in tick time
        if max_seconds is not None and (_ts - t_anchor) > max_seconds:
            log.info(
                "Reached --minutes=%.2f cap at tick ts offset %.1fs; stopping.",
                args.minutes, _ts - t_anchor,
            )
            break
        pid = tick.get("product_id", "unknown")
        if pid not in states:
            states[pid] = ProductState(window_sec, horizon_sec, vol_threshold)

        try:
            rows = states[pid].ingest(tick)
        except (KeyError, ValueError, TypeError) as exc:
            log.warning("Featurize error %s: %s", pid, exc)
            continue

        for row in rows:
            sink.write(row)
            emitted += 1

        pending_count += 1

    # Drain pending rows (last horizon_sec worth of ticks won't have full labels)
    drained = 0
    for state in states.values():
        for row in state.drain_remaining():
            sink.write(row)
            emitted += 1
            drained += 1

    sink.close()

    log.info(
        "Done. %d ticks processed → %d rows emitted (%d from drain) → %s",
        pending_count, emitted, drained, out_path,
    )
if __name__ == "__main__":
    main()
