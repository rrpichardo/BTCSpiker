"""
Replay — batch feature builder from NDJSON mirror files.

Reads all NDJSON tick files matching a glob pattern, sorts lines by
timestamp across files, then feeds each tick through the shared
``btcspiker_ml.features.FeatureEngine`` used by the live featurizer. Outputs
a single Parquet file with future-vol labels (60s lookahead by default).

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
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
FEATURES_DIR = PROJECT_ROOT / "features"
if str(FEATURES_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURES_DIR))

import pyarrow as pa
import yaml

from btcspiker_ml.features import FeatureEngine, parse_timestamp
from parquet_sink import AtomicParquetSink

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

FLUSH_ROWS     = 500
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
    def __init__(self, path: Path):
        super().__init__(path=path, schema=PARQUET_SCHEMA, flush_rows=FLUSH_ROWS)


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
                yield parse_timestamp(tick["timestamp"]), tick
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
    horizon_sec   = float(cfg["features"]["label_horizon_sec"])
    vol_threshold = float(cfg["features"]["vol_threshold"])
    out_path      = Path(args.out or cfg["data"]["features_file"])

    sink   = ParquetSink(out_path)
    states: dict[str, FeatureEngine] = {}
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
            states[pid] = FeatureEngine(FEATURE_SET_ID, horizon_sec, vol_threshold)

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
