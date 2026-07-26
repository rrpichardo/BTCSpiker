"""
Materializer — Kafka read-model service.

Consumes PredictionEvent messages from `ticks.predictions` AND OutcomeEvent
messages from `ticks.outcomes` (two independent consumer threads) into a
local SQLite read model (WAL mode) and serves it over a small FastAPI
surface:

    GET /predictions/recent?limit=200          -> newest predictions, newest-first
    GET /predictions/performance?window_minutes=30
                                                -> graded model-vs-baseline snapshot
                                                   (predictions LEFT JOIN outcomes,
                                                   evaluated by evaluation.py)
    GET /health                                -> consumer + DB health
    GET /predictions/health                    -> alias of /health (nginx proxies
                                                   only /api/predictions/* to this
                                                   service, so health must live
                                                   under that prefix too)

Kafka is the source of truth; the SQLite file is a disposable projection.
When a projection is missing or rebuilt, assigned partitions are explicitly
rewound to the beginning even if the stable consumer group has committed
offsets. Delivery on both topics is at-least-once, so every insert is
`INSERT OR IGNORE` keyed on `event_id` (predictions) / `feature_id`
(outcomes).

Timestamp normalization: `feature_ts` arrives from upstream in either of two
ISO-8601 spellings — trailing "Z" or an explicit "+00:00" offset — which do
NOT sort identically as raw strings. Every `feature_ts` is normalized at
insert time (`_normalize_ts`) to a single canonical form (UTC, "+00:00"
suffix, fixed microsecond precision) before it's written to either table, so
`WHERE feature_ts >= ?` range queries sort correctly regardless of which
spelling a given upstream service used.

Importing this module has NO side effects: no Kafka connection, no thread,
no DB file creation. The consumer threads are started from the FastAPI
lifespan hook at app startup, not at import time — see `consume_loop` /
`consume_outcomes_loop` / `lifespan`.

Usage
-----
    uvicorn materializer:app --host 0.0.0.0 --port 8090
"""

import asyncio
import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from confluent_kafka import Consumer, KafkaError, OFFSET_BEGINNING, TopicPartition
from confluent_kafka.admin import AdminClient
from fastapi import FastAPI, HTTPException, Query

import evaluation
import timeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC_PREDICTIONS = os.getenv("TOPIC_PREDICTIONS", "ticks.predictions")
TOPIC_OUTCOMES = os.getenv("TOPIC_OUTCOMES", "ticks.outcomes")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "materializer")
KAFKA_OUTCOMES_GROUP_ID = os.getenv("KAFKA_OUTCOMES_GROUP_ID", "materializer-outcomes")
PREDICTIONS_DB_PATH = os.getenv("PREDICTIONS_DB_PATH", "/data/predictions.db")

# Evaluation-layer knobs (see evaluation.compute_performance). Same env var
# name/default as api/main.py's BASELINE_VOL_THRESHOLD for consistency.
MIN_POSITIVES = int(os.getenv("MIN_POSITIVES", "10"))
MIN_NOTE_SAMPLE_N = int(os.getenv("MIN_NOTE_SAMPLE_N", "1000"))
DRIFT_PR_AUC_RATIO = float(os.getenv("DRIFT_PR_AUC_RATIO", "0.7"))
BASELINE_VOL_THRESHOLD = float(os.getenv("BASELINE_VOL_THRESHOLD", "0.000048"))
ADAPTIVE_PERCENTILE = 85

# Same env var name/default as api/system.py's _materializer_stale_seconds,
# so /health's own `ok` and the /system/status rollup agree on what "stale"
# means. A consumer thread can stay alive while wedged retrying a Kafka
# commit (e.g. UNKNOWN_MEMBER_ID after a rebalance) without ever polling a
# new message — this catches that by requiring last_event_ts to keep moving,
# which lets the container healthcheck's `restart: on-failure` recover it.
STALE_SECONDS = float(os.getenv("SYSTEM_MATERIALIZER_STALE_SECONDS", "60"))

# Plan decisions — module constants, not configurable.
BATCH_MAX_SIZE = 200
BATCH_WINDOW_SEC = 1.0
BUSY_TIMEOUT_MS = 5000
PRUNE_EVERY_N_INSERTS = 1000
PRUNE_KEEP_ROWS = 100_000
LIMIT_MIN = 1
LIMIT_MAX = 2000
WINDOW_MINUTES_MIN = 1
WINDOW_MINUTES_MAX = 120
PERF_CACHE_SECONDS = 5.0
TIMELINE_RANGE_MAX_SECONDS = 24 * 60 * 60 + 60  # 24h + a minute of slack
STARTUP_TIMEOUT = 30.0
STARTUP_READY_TIMEOUT = STARTUP_TIMEOUT + 5.0
KAFKA_SOCKET_TIMEOUT_MS = 3000
THREAD_JOIN_TIMEOUT = 10.0
COMMIT_RETRY_BACKOFF_SEC = 0.1
CONSUMER_RESTART_BACKOFF_SEC = 1.0
BROKER_HEALTHCHECK_INTERVAL_SEC = 5.0
BROKER_HEALTHCHECK_TIMEOUT_SEC = 1.0

# Pinned event_id / column ordering shared by parsing, inserts, and reads.
# feature_id/stream_epoch/tau/run_id/market_price are newer, nullable fields
# (added via ALTER TABLE migration below) — old rows simply carry null for
# them, which makes them ungradeable (or price-less) rather than wrong.
EVENT_FIELDS = [
    "event_id",
    "source_partition",
    "source_offset",
    "feature_ts",
    "api_ts",
    "model_variant",
    "model_version",
    "score",
    "vol_60s",
    "spread_bps",
    "log_return",
    "trade_intensity_60s",
    "feature_id",
    "stream_epoch",
    "tau",
    "run_id",
    "market_price",
]

# Columns added after the original predictions table shipped; migrated in
# via idempotent ALTER TABLE ADD COLUMN (see _ensure_columns).
PREDICTIONS_MIGRATED_COLUMNS = [
    ("feature_id", "TEXT"),
    ("stream_epoch", "INT"),
    ("tau", "REAL"),
    ("run_id", "TEXT"),
    ("market_price", "REAL"),
]

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    event_id TEXT PRIMARY KEY,
    source_partition INT,
    source_offset INT,
    feature_ts TEXT,
    api_ts TEXT,
    model_variant TEXT,
    model_version TEXT,
    score REAL,
    vol_60s REAL,
    spread_bps REAL,
    log_return REAL,
    trade_intensity_60s REAL
)
"""

INSERT_SQL = (
    f"INSERT OR IGNORE INTO predictions ({', '.join(EVENT_FIELDS)}) "
    f"VALUES ({', '.join('?' for _ in EVENT_FIELDS)})"
)

NEWEST_ORDER_SQL = """
ORDER BY
    julianday(COALESCE(api_ts, feature_ts)) DESC,
    source_partition DESC,
    source_offset DESC,
    event_id ASC
"""

RECENT_SQL = (
    f"SELECT {', '.join(EVENT_FIELDS)} FROM predictions " f"{NEWEST_ORDER_SQL} LIMIT ?"
)

# ticks.outcomes -> outcomes table. written_at is NOT part of the wire event —
# it's stamped by the materializer at insert time (see insert_outcomes), which
# is what the online-grading rule in evaluation.py hinges on.
OUTCOME_FIELDS = [
    "feature_id",
    "stream_epoch",
    "product_id",
    "feature_ts",
    "future_vol_60s",
    "vol_spike",
    "label_schema",
]

OUTCOMES_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS outcomes (
    feature_id TEXT PRIMARY KEY,
    stream_epoch INT,
    product_id TEXT,
    feature_ts TEXT,
    future_vol_60s REAL,
    vol_spike INT,
    label_schema TEXT,
    written_at TEXT
)
"""

INSERT_OUTCOME_SQL = (
    f"INSERT OR IGNORE INTO outcomes ({', '.join(OUTCOME_FIELDS)}, written_at) "
    f"VALUES ({', '.join('?' for _ in OUTCOME_FIELDS)}, ?)"
)

INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_pred_feature_ts ON predictions(feature_ts)",
    "CREATE INDEX IF NOT EXISTS idx_outcomes_feature_ts ON outcomes(feature_ts)",
]

# predictions LEFT JOIN outcomes, scoped by the predictions side's feature_ts
# window. Outcome-side feature_id/stream_epoch/feature_ts are never selected:
# they duplicate the prediction's own columns by construction (the featurizer
# derives both from the same feature row), so selecting them would just
# collide names in the result dict for no informational gain.
PERFORMANCE_JOIN_FIELDS = [f"p.{f}" for f in EVENT_FIELDS] + [
    "o.product_id",
    "o.future_vol_60s",
    "o.vol_spike",
    "o.label_schema",
    "o.written_at",
]

PERFORMANCE_JOIN_SQL = f"""
SELECT {', '.join(PERFORMANCE_JOIN_FIELDS)}
FROM predictions p
LEFT JOIN outcomes o ON p.feature_id = o.feature_id
WHERE p.feature_ts >= ?
ORDER BY p.feature_ts ASC
"""

# Same join as PERFORMANCE_JOIN_SQL, bounded on both sides for /predictions/timeline.
TIMELINE_JOIN_SQL = f"""
SELECT {', '.join(PERFORMANCE_JOIN_FIELDS)}
FROM predictions p
LEFT JOIN outcomes o ON p.feature_id = o.feature_id
WHERE p.feature_ts >= ? AND p.feature_ts < ?
ORDER BY p.feature_ts ASC
"""


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------


def _normalize_ts(raw: str | None) -> str | None:
    """Canonicalize an ISO-8601 timestamp to UTC, "+00:00" suffix, fixed
    microsecond precision — so raw-string comparisons/sorts (`feature_ts >=
    ?`) agree with real chronological order regardless of whether the
    producer wrote a trailing "Z" or an explicit "+00:00" offset.

    Malformed or empty input is passed through unchanged rather than
    dropped — better to keep an ungradeable-but-present row than lose data
    on a parse hiccup.
    """
    if not raw:
        return raw
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return raw
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _ensure_columns(
    conn: sqlite3.Connection, table: str, columns: list[tuple[str, str]]
) -> None:
    """Idempotent ALTER TABLE ADD COLUMN migration: add any of `columns`
    (name, sqlite type) not already present on `table`. Existing rows keep
    null for newly-added columns."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, coltype in columns:
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {coltype}")


def _delete_db_files(path: Path) -> None:
    """Delete the DB file and its -wal/-shm siblings, if present."""
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(path) + suffix)
        if candidate.exists():
            candidate.unlink()


def _open_verified(path: Path) -> sqlite3.Connection | None:
    """Open `path` and confirm it's a readable, non-corrupt SQLite file.

    Returns None (closing any partial handle) if the file exists but is
    corrupt or not a SQLite database at all.
    """
    conn = None
    try:
        conn = sqlite3.connect(str(path))
        conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        row = conn.execute("PRAGMA integrity_check").fetchone()
        if row is None or row[0] != "ok":
            conn.close()
            return None
        return conn
    except sqlite3.DatabaseError:
        if conn is not None:
            conn.close()
        return None


def _init_db_with_recovery_status(
    path: str | Path,
) -> tuple[sqlite3.Connection, bool]:
    """Open the read model and report whether Kafka replay is required."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()

    conn = _open_verified(path)
    corrupt = conn is None
    if corrupt:
        log.error(
            "Predictions DB at %s is corrupt/unreadable; deleting and rebuilding "
            "from Kafka replay",
            path,
        )
        _delete_db_files(path)
        conn = sqlite3.connect(str(path))
        conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")

    schema_existed = (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'predictions'"
        ).fetchone()
        is not None
    )
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute(SCHEMA_SQL)
    _ensure_columns(conn, "predictions", PREDICTIONS_MIGRATED_COLUMNS)
    conn.execute(OUTCOMES_SCHEMA_SQL)
    for index_sql in INDEX_SQL:
        conn.execute(index_sql)
    conn.commit()
    return conn, corrupt or not existed or not schema_existed


def _outcomes_table_missing(path: str | Path) -> bool:
    """Best-effort pre-check for the outcomes consumer's own replay decision:
    does the outcomes table not exist yet in the DB file at `path`?

    Read via a throwaway connection BEFORE `_init_db_with_recovery_status`
    (which would create the table) runs on this thread. There's a narrow
    race with the predictions thread's own schema-ensure call at simultaneous
    fresh boot; accepted as a known limitation — worst case that race is lost
    is a skipped explicit replay-from-beginning, and `auto.offset.reset:
    earliest` on a brand-new consumer group still starts from the beginning
    of the topic regardless.
    """
    path = Path(path)
    if not path.exists():
        return True
    try:
        conn = sqlite3.connect(str(path))
        try:
            return (
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'outcomes'"
                ).fetchone()
                is None
            )
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return True


def init_db(path: str | Path) -> sqlite3.Connection:
    """Open (or create) the predictions read-model DB: WAL mode, schema applied.

    A corrupt file is deleted (with its -wal/-shm siblings) and recreated —
    Kafka is the source of truth, so the consumer explicitly replays assigned
    partitions when this function had to create or rebuild the projection.
    """
    conn, _ = _init_db_with_recovery_status(path)
    return conn


def insert_events(conn: sqlite3.Connection, events: list[dict]) -> int:
    """INSERT OR IGNORE a batch of event dicts in one transaction.

    Returns the number of rows actually inserted — duplicates ignored via the
    event_id primary key (at-least-once delivery) don't count. feature_ts is
    normalized (see `_normalize_ts`) so the stored value sorts/compares
    correctly regardless of the upstream "Z" vs "+00:00" spelling.
    """
    if not events:
        return 0
    inserted = 0
    with conn:
        cur = conn.cursor()
        for event in events:
            row = dict(event)
            row["feature_ts"] = _normalize_ts(row.get("feature_ts"))
            cur.execute(INSERT_SQL, tuple(row.get(f) for f in EVENT_FIELDS))
            inserted += cur.rowcount
    return inserted


def insert_outcomes(conn: sqlite3.Connection, events: list[dict]) -> int:
    """INSERT OR IGNORE a batch of OutcomeEvent dicts in one transaction.

    `written_at` is stamped here (once per batch, at insert time) rather than
    taken from the event — it's the wall-clock anchor the online-grading rule
    in evaluation.py compares each prediction's api_ts against.
    """
    if not events:
        return 0
    written_at = _normalize_ts(datetime.now(timezone.utc).isoformat())
    inserted = 0
    with conn:
        cur = conn.cursor()
        for event in events:
            row = dict(event)
            row["feature_ts"] = _normalize_ts(row.get("feature_ts"))
            values = tuple(row.get(f) for f in OUTCOME_FIELDS) + (written_at,)
            cur.execute(INSERT_OUTCOME_SQL, values)
            inserted += cur.rowcount
    return inserted


def maybe_prune(conn: sqlite3.Connection, inserted_since_prune: int) -> bool:
    """Delete all but the newest PRUNE_KEEP_ROWS rows once
    PRUNE_EVERY_N_INSERTS inserts have accumulated since the last prune.

    Returns True if a prune ran.
    """
    if inserted_since_prune < PRUNE_EVERY_N_INSERTS:
        return False
    with conn:
        conn.execute(
            """
            DELETE FROM predictions
            WHERE event_id NOT IN (
                SELECT event_id FROM predictions
                {NEWEST_ORDER_SQL}
                LIMIT ?
            )
            """.format(
                NEWEST_ORDER_SQL=NEWEST_ORDER_SQL
            ),
            (PRUNE_KEEP_ROWS,),
        )
    return True


def maybe_prune_outcomes(conn: sqlite3.Connection, inserted_since_prune: int) -> bool:
    """Same retention policy as `maybe_prune`, applied to the outcomes table,
    ordered by feature_ts (outcomes have no partition/offset columns to
    tie-break on). Returns True if a prune ran.
    """
    if inserted_since_prune < PRUNE_EVERY_N_INSERTS:
        return False
    with conn:
        conn.execute(
            """
            DELETE FROM outcomes
            WHERE feature_id NOT IN (
                SELECT feature_id FROM outcomes
                ORDER BY feature_ts DESC
                LIMIT ?
            )
            """,
            (PRUNE_KEEP_ROWS,),
        )
    return True


def _recent_rows(conn: sqlite3.Connection, limit: int) -> list[dict]:
    cur = conn.execute(RECENT_SQL, (limit,))
    columns = [d[0] for d in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]


def _open_readonly(path: str | Path) -> sqlite3.Connection:
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    return conn


def recent(path_or_conn: str | Path | sqlite3.Connection, limit: int) -> list[dict]:
    """Return up to `limit` predictions, newest timestamp first.

    Given a path, opens a short-lived read connection — the HTTP-handler
    pattern, safe across threads since the consumer thread owns the single
    writer connection. Given an existing connection, queries it directly
    (used by the consumer thread and by tests).
    """
    if isinstance(path_or_conn, sqlite3.Connection):
        return _recent_rows(path_or_conn, limit)

    conn = _open_readonly(path_or_conn)
    try:
        return _recent_rows(conn, limit)
    finally:
        conn.close()


def _rows_total(path: str | Path) -> tuple[int, bool]:
    """Return (count, db_ok) via a short-lived read connection."""
    try:
        conn = _open_readonly(path)
        try:
            count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            return count, True
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return 0, False


def performance_window(
    path_or_conn: str | Path | sqlite3.Connection, cutoff: str
) -> dict:
    """Gather everything the /predictions/performance handler needs for
    predictions with feature_ts >= `cutoff` (already `_normalize_ts`-d):

        joined_rows            -- predictions LEFT JOIN outcomes, for
                                   evaluation.compute_performance
        oldest_feature_ts       -- oldest prediction feature_ts in the whole
                                   table (None if empty) -- used to report
                                   `complete`
        n_outcomes_unmatched    -- outcomes in-window with no matching
                                   prediction row at all; a plain LEFT JOIN
                                   rooted at predictions can never surface
                                   these, so it's a separate query
    """

    def _query(conn: sqlite3.Connection) -> dict:
        cur = conn.execute(PERFORMANCE_JOIN_SQL, (cutoff,))
        columns = [d[0] for d in cur.description]
        joined_rows = [dict(zip(columns, row)) for row in cur.fetchall()]

        oldest = conn.execute("SELECT MIN(feature_ts) FROM predictions").fetchone()[0]

        unmatched = conn.execute(
            """
            SELECT COUNT(*) FROM outcomes o
            WHERE o.feature_ts >= ?
            AND NOT EXISTS (
                SELECT 1 FROM predictions p WHERE p.feature_id = o.feature_id
            )
            """,
            (cutoff,),
        ).fetchone()[0]

        return {
            "joined_rows": joined_rows,
            "oldest_feature_ts": oldest,
            "n_outcomes_unmatched": unmatched,
        }

    if isinstance(path_or_conn, sqlite3.Connection):
        return _query(path_or_conn)

    conn = _open_readonly(path_or_conn)
    try:
        return _query(conn)
    finally:
        conn.close()


def timeline_window(
    path_or_conn: str | Path | sqlite3.Connection, from_ts: str, to_ts: str
) -> dict:
    """Gather everything /predictions/timeline needs for predictions with
    feature_ts in [from_ts, to_ts) (both already `_normalize_ts`-d):

        joined_rows      -- predictions LEFT JOIN outcomes, ordered by
                             feature_ts ascending, for timeline.build_timeline
        available_from   -- oldest feature_ts in the WHOLE table (None if
                             empty) -- used to report retained-history coverage
        available_to     -- newest feature_ts in the whole table -- also the
                             maturity anchor for pending-vs-unavailable
    """

    def _query(conn: sqlite3.Connection) -> dict:
        cur = conn.execute(TIMELINE_JOIN_SQL, (from_ts, to_ts))
        columns = [d[0] for d in cur.description]
        joined_rows = [dict(zip(columns, row)) for row in cur.fetchall()]

        bounds = conn.execute(
            "SELECT MIN(feature_ts), MAX(feature_ts) FROM predictions"
        ).fetchone()

        return {
            "joined_rows": joined_rows,
            "available_from": bounds[0],
            "available_to": bounds[1],
        }

    if isinstance(path_or_conn, sqlite3.Connection):
        return _query(path_or_conn)

    conn = _open_readonly(path_or_conn)
    try:
        return _query(conn)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Event parsing
# ---------------------------------------------------------------------------


def parse_event(raw: bytes | str) -> dict:
    """Parse one Kafka message value into a predictions-table row dict.

    Raises ValueError on malformed JSON or a missing/empty event_id; callers
    treat that as a consume error: log it, skip the message, commit past it.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc

    if not isinstance(data, dict) or not data.get("event_id"):
        raise ValueError("missing event_id")

    return {f: data.get(f) for f in EVENT_FIELDS}


def parse_outcome_event(raw: bytes | str) -> dict:
    """Parse one Kafka message value into an outcomes-table row dict.

    Raises ValueError on malformed JSON or a missing/empty feature_id;
    callers treat that as a consume error: log it, skip the message, commit
    past it — mirrors `parse_event`'s contract exactly.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc

    if not isinstance(data, dict) or not data.get("feature_id"):
        raise ValueError("missing feature_id")

    return {f: data.get(f) for f in OUTCOME_FIELDS}


# ---------------------------------------------------------------------------
# Consumer state + health + request validation
# ---------------------------------------------------------------------------


@dataclass
class ConsumerState:
    """Mutable counters shared between the consumer threads and HTTP handlers.

    `alive`/`ready`/`startup_error` describe the predictions consumer, which
    is what gates overall app readiness in `lifespan`. `outcomes_alive`
    describes the second (outcomes) consumer thread; it doesn't gate startup
    (the outcomes consumer is best-effort — see `lifespan`), but it does fold
    into `health_snapshot`'s `ok`, same as `alive` already does.
    """

    last_event_ts: str | None = None
    last_write_ts: str | None = None
    consume_errors: int = 0
    write_errors: int = 0
    alive: bool = False
    outcomes_alive: bool = False
    broker_ok: bool = False
    ready: bool = False
    startup_error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


def _reset_state(state: ConsumerState) -> None:
    with state.lock:
        state.last_event_ts = None
        state.last_write_ts = None
        state.consume_errors = 0
        state.write_errors = 0
        state.alive = False
        state.outcomes_alive = False
        state.broker_ok = False
        state.ready = False
        state.startup_error = None


def _is_fresh(last_event_ts: str | None, stale_seconds: float = STALE_SECONDS) -> bool:
    """True if last_event_ts is a valid tz-aware timestamp within stale_seconds of now."""
    if last_event_ts is None:
        return False
    try:
        parsed = datetime.fromisoformat(last_event_ts.replace("Z", "+00:00"))
    except ValueError:
        return False
    if parsed.tzinfo is None:
        return False
    return (datetime.now(timezone.utc) - parsed).total_seconds() <= stale_seconds


def health_snapshot(state: ConsumerState, db_path: str | Path) -> dict:
    """Build the /health (and /predictions/health) response body."""
    rows_total, db_ok = _rows_total(db_path)
    with state.lock:
        last_event_ts = state.last_event_ts
        base_ok = state.alive and state.outcomes_alive and state.broker_ok and db_ok
        result = {
            "last_event_ts": last_event_ts,
            "last_write_ts": state.last_write_ts,
            "rows_total": rows_total,
            "consume_errors": state.consume_errors,
            "write_errors": state.write_errors,
        }
    result["ok"] = base_ok and _is_fresh(last_event_ts)
    return result


def validate_limit(limit: int) -> int:
    """Validate the /predictions/recent `limit` query param (1..2000)."""
    if not (LIMIT_MIN <= limit <= LIMIT_MAX):
        raise ValueError(
            f"limit must be between {LIMIT_MIN} and {LIMIT_MAX}, got {limit}"
        )
    return limit


def validate_window_minutes(window_minutes: int) -> int:
    """Validate the /predictions/performance `window_minutes` query param."""
    if not (WINDOW_MINUTES_MIN <= window_minutes <= WINDOW_MINUTES_MAX):
        raise ValueError(
            f"window_minutes must be between {WINDOW_MINUTES_MIN} and "
            f"{WINDOW_MINUTES_MAX}, got {window_minutes}"
        )
    return window_minutes


def validate_timeline_range(
    from_raw: str, to_raw: str, resolution: int | None
) -> tuple[datetime, datetime, int | None]:
    """Validate the /predictions/timeline `from`/`to`/`resolution` query params.

    Returns (from_dt, to_dt, resolution) as tz-aware datetimes / int-or-None.
    """
    from_dt = _parse_iso(from_raw)
    to_dt = _parse_iso(to_raw)
    if from_dt is None:
        raise ValueError(f"from must be a valid ISO-8601 timestamp, got {from_raw!r}")
    if to_dt is None:
        raise ValueError(f"to must be a valid ISO-8601 timestamp, got {to_raw!r}")
    if to_dt <= from_dt:
        raise ValueError("to must be after from")

    range_seconds = (to_dt - from_dt).total_seconds()
    if range_seconds > TIMELINE_RANGE_MAX_SECONDS:
        raise ValueError(
            f"requested range ({range_seconds:.0f}s) exceeds the "
            f"{TIMELINE_RANGE_MAX_SECONDS:.0f}s (24h) maximum"
        )

    if resolution is not None:
        if resolution < 1:
            raise ValueError(
                f"resolution must be a positive number of seconds, got {resolution}"
            )
        # An explicit resolution bypasses build_timeline's own auto-bucketing
        # (which only kicks in when resolution is omitted), so a too-fine
        # resolution over a wide range could otherwise force an unbounded
        # SQLite read/response despite MAX_TIMELINE_POINTS existing.
        implied_points = range_seconds / resolution
        if implied_points > timeline.MAX_TIMELINE_POINTS:
            raise ValueError(
                f"resolution={resolution}s over a {range_seconds:.0f}s range "
                f"implies {implied_points:.0f} points, exceeding the "
                f"{timeline.MAX_TIMELINE_POINTS}-point limit"
            )

    return from_dt, to_dt, resolution


# ---------------------------------------------------------------------------
# Kafka consumer loop (background thread in production; focused tests inject a
# deterministic fake consumer so write/commit ordering is covered without a broker)
# ---------------------------------------------------------------------------


def _probe_broker(bootstrap: str, timeout: float) -> bool:
    """Fresh-client metadata probe of broker connectivity.

    A NEW AdminClient per call is deliberate: librdkafka serves cached
    metadata to long-lived clients, so `list_topics` on the long-lived
    consumer keeps "succeeding" during a broker outage. A fresh client has
    no cache, so success/failure here reflects real connectivity — bounding
    both outage detection and recovery to one probe interval.
    """
    try:
        AdminClient({"bootstrap.servers": bootstrap}).list_topics(timeout=timeout)
        return True
    except Exception:
        return False


def _wait_for_kafka(consumer: Consumer, bootstrap: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        try:
            consumer.list_topics(timeout=1.0)
            return
        except Exception as exc:  # pragma: no cover - exercised in integration
            last_exc = exc
            time.sleep(1.0)
    raise RuntimeError(
        f"Kafka bootstrap {bootstrap!r} was not reachable within {timeout:.0f}s"
    ) from last_exc


def _subscribe_for_projection(
    consumer: Consumer, replay_from_beginning: bool, topic: str | None = None
) -> set[tuple[str, int]]:
    """Subscribe and return recovery partitions with a successful commit."""
    topic = topic or TOPIC_PREDICTIONS
    if not replay_from_beginning:
        consumer.subscribe([topic])
        return set()

    committed_recovery_partitions: set[tuple[str, int]] = set()

    def _on_assign(assigned_consumer: Consumer, partitions: list[TopicPartition]):
        rewound = 0
        for partition in partitions:
            key = (partition.topic, partition.partition)
            if key not in committed_recovery_partitions:
                partition.offset = OFFSET_BEGINNING
                rewound += 1
        if rewound:
            log.warning(
                "%s projection is new/rebuilt; replaying %d assigned "
                "partition(s) from the beginning",
                topic,
                rewound,
            )
        assigned_consumer.assign(partitions)

    consumer.subscribe([topic], on_assign=_on_assign)
    return committed_recovery_partitions


def _offsets_after(messages) -> list[TopicPartition]:
    """Return explicit next offsets for every partition represented in a batch."""
    next_offsets: dict[tuple[str, int], int] = {}
    for message in messages:
        key = (message.topic(), message.partition())
        next_offsets[key] = max(next_offsets.get(key, 0), message.offset() + 1)
    return [
        TopicPartition(topic, partition, offset)
        for (topic, partition), offset in sorted(next_offsets.items())
    ]


def probe_loop(
    state: ConsumerState,
    stop_event: threading.Event,
    probe_broker=None,
) -> None:
    """Dedicated health-probe thread: sole owner of `state.broker_ok`.

    Runs a fresh-client metadata probe every BROKER_HEALTHCHECK_INTERVAL_SEC
    in its OWN thread, so broker health tracks reality within one probe
    interval no matter what the consumer loop is doing — its synchronous
    offset commit can block for the full group session timeout (~45s) when
    the coordinator disappears, which starved an earlier in-loop probe.
    Poll/commit errors in the consumer never touch the flag, so queued
    transport errors can't hold health false after the broker recovers;
    both transitions (up->down, down->up) are detected without restarting
    anything.
    """
    if probe_broker is None:

        def probe_broker():
            return _probe_broker(KAFKA_BOOTSTRAP, BROKER_HEALTHCHECK_TIMEOUT_SEC)

    while not stop_event.is_set():
        broker_up = probe_broker()
        with state.lock:
            transition = state.broker_ok != broker_up
            state.broker_ok = broker_up
        if transition:
            if broker_up:
                log.warning(
                    "Kafka broker probe recovered (bootstrap=%s)", KAFKA_BOOTSTRAP
                )
            else:
                log.error("Kafka broker probe failed (bootstrap=%s)", KAFKA_BOOTSTRAP)
        stop_event.wait(BROKER_HEALTHCHECK_INTERVAL_SEC)


def consume_loop(
    state: ConsumerState,
    stop_event: threading.Event,
    ready_event: threading.Event | None = None,
) -> None:
    """Background thread entrypoint: ticks.predictions -> SQLite read model.

    Batches up to BATCH_MAX_SIZE messages or a BATCH_WINDOW_SEC window,
    INSERT OR IGNOREs the batch in one transaction, then commits Kafka
    offsets — at-least-once delivery + the event_id primary key make replays
    harmless.

    Never writes `state.broker_ok` — that flag is owned by `probe_loop`.
    """
    conn = None
    consumer = None
    try:
        conn, replay_from_beginning = _init_db_with_recovery_status(PREDICTIONS_DB_PATH)
        consumer = Consumer(
            {
                "bootstrap.servers": KAFKA_BOOTSTRAP,
                "group.id": KAFKA_GROUP_ID,
                "auto.offset.reset": "earliest",
                "enable.auto.commit": False,
                "socket.timeout.ms": KAFKA_SOCKET_TIMEOUT_MS,
            }
        )
        committed_recovery_partitions = _subscribe_for_projection(
            consumer, replay_from_beginning
        )
        _wait_for_kafka(consumer, KAFKA_BOOTSTRAP, STARTUP_TIMEOUT)
        log.info(
            "Materializer consumer started | %s -> %s | group=%s",
            TOPIC_PREDICTIONS,
            PREDICTIONS_DB_PATH,
            KAFKA_GROUP_ID,
        )

        with state.lock:
            state.alive = True
            state.ready = True
            state.startup_error = None
        if ready_event is not None:
            ready_event.set()

        inserted_since_prune = 0
        pending_batch = None
        pending_events = None
        pending_persisted = False

        while not stop_event.is_set():
            if pending_batch is None:
                batch = []
                deadline = time.monotonic() + BATCH_WINDOW_SEC
                while len(batch) < BATCH_MAX_SIZE and time.monotonic() < deadline:
                    msg = consumer.poll(timeout=0.2)
                    if msg is None:
                        continue
                    if msg.error():
                        if msg.error().code() != KafkaError._PARTITION_EOF:
                            with state.lock:
                                state.consume_errors += 1
                            log.error("Kafka error: %s", msg.error())
                        continue
                    batch.append(msg)

                if not batch:
                    continue

                events = []
                for msg in batch:
                    try:
                        event = parse_event(msg.value())
                    except ValueError as exc:
                        with state.lock:
                            state.consume_errors += 1
                        log.warning("Skipping malformed prediction event: %s", exc)
                        continue
                    events.append(event)
                pending_batch = batch
                pending_events = events
                pending_persisted = False

            if not pending_persisted:
                try:
                    inserted = insert_events(conn, pending_events)
                    inserted_since_prune += inserted
                    if inserted:
                        newest = _recent_rows(conn, 1)[0]
                        with state.lock:
                            state.last_event_ts = (
                                newest["api_ts"] or newest["feature_ts"]
                            )
                            state.last_write_ts = datetime.now(timezone.utc).isoformat()
                    if maybe_prune(conn, inserted_since_prune):
                        inserted_since_prune %= PRUNE_EVERY_N_INSERTS
                    pending_persisted = True
                except sqlite3.DatabaseError as exc:
                    with state.lock:
                        state.write_errors += 1
                    log.error("Batch write failed, will retry: %s", exc)
                    stop_event.wait(0.1)
                    continue

            offsets = _offsets_after(pending_batch)
            try:
                committed = consumer.commit(offsets=offsets, asynchronous=False)
            except Exception as exc:
                with state.lock:
                    state.consume_errors += 1
                log.error("Kafka offset commit failed, will retry: %s", exc)
                stop_event.wait(COMMIT_RETRY_BACKOFF_SEC)
                continue

            commit_errors = [
                partition
                for partition in committed or []
                if partition.error is not None
            ]
            successful_partitions = [
                partition for partition in committed or [] if partition.error is None
            ]
            committed_recovery_partitions.update(
                (partition.topic, partition.partition)
                for partition in successful_partitions
            )
            if commit_errors:
                with state.lock:
                    state.consume_errors += len(commit_errors)
                for partition in commit_errors:
                    log.error(
                        "Kafka offset commit failed for %s[%d], will retry: %s",
                        partition.topic,
                        partition.partition,
                        partition.error,
                    )
                stop_event.wait(COMMIT_RETRY_BACKOFF_SEC)
                continue
            if not committed:
                with state.lock:
                    state.consume_errors += 1
                log.error("Kafka offset commit returned no results; will retry")
                stop_event.wait(COMMIT_RETRY_BACKOFF_SEC)
                continue

            pending_batch = None
            pending_events = None
            pending_persisted = False
    except Exception as exc:
        with state.lock:
            if not state.ready:
                state.startup_error = str(exc)
        if not stop_event.is_set():
            log.exception("Materializer consumer failed: %s", exc)
    finally:
        with state.lock:
            state.alive = False
            state.ready = False
        if ready_event is not None:
            ready_event.set()
        if consumer is not None:
            try:
                consumer.close()
            except Exception:
                log.exception("Failed to close Kafka consumer")
        if conn is not None:
            conn.close()
        log.info("Materializer consumer stopped.")


def consume_outcomes_loop(
    state: ConsumerState,
    stop_event: threading.Event,
    ready_event: threading.Event | None = None,
) -> None:
    """Background thread entrypoint: ticks.outcomes -> SQLite read model.

    Mirrors `consume_loop`'s batching/insert/commit discipline exactly
    (batch of BATCH_MAX_SIZE or BATCH_WINDOW_SEC, one INSERT OR IGNORE
    transaction per batch keyed on feature_id, offsets committed only after
    the write succeeds, malformed messages counted as an error and skipped).

    Best-effort relative to `consume_loop`: it never touches `state.ready` /
    `state.startup_error` (those gate overall app readiness in `lifespan`
    and stay owned by the predictions consumer) or `state.broker_ok` (owned
    by `probe_loop`). Its own liveness lives in `state.outcomes_alive`, which
    folds into `health_snapshot`'s `ok` the same way `state.alive` does.
    """
    conn = None
    consumer = None
    try:
        replay_from_beginning = _outcomes_table_missing(PREDICTIONS_DB_PATH)
        conn, _ = _init_db_with_recovery_status(PREDICTIONS_DB_PATH)
        consumer = Consumer(
            {
                "bootstrap.servers": KAFKA_BOOTSTRAP,
                "group.id": KAFKA_OUTCOMES_GROUP_ID,
                "auto.offset.reset": "earliest",
                "enable.auto.commit": False,
                "socket.timeout.ms": KAFKA_SOCKET_TIMEOUT_MS,
            }
        )
        committed_recovery_partitions = _subscribe_for_projection(
            consumer, replay_from_beginning, topic=TOPIC_OUTCOMES
        )
        _wait_for_kafka(consumer, KAFKA_BOOTSTRAP, STARTUP_TIMEOUT)
        log.info(
            "Materializer outcomes consumer started | %s -> %s | group=%s",
            TOPIC_OUTCOMES,
            PREDICTIONS_DB_PATH,
            KAFKA_OUTCOMES_GROUP_ID,
        )

        with state.lock:
            state.outcomes_alive = True
        if ready_event is not None:
            ready_event.set()

        inserted_since_prune = 0
        pending_batch = None
        pending_events = None
        pending_persisted = False

        while not stop_event.is_set():
            if pending_batch is None:
                batch = []
                deadline = time.monotonic() + BATCH_WINDOW_SEC
                while len(batch) < BATCH_MAX_SIZE and time.monotonic() < deadline:
                    msg = consumer.poll(timeout=0.2)
                    if msg is None:
                        continue
                    if msg.error():
                        if msg.error().code() != KafkaError._PARTITION_EOF:
                            with state.lock:
                                state.consume_errors += 1
                            log.error("Kafka error: %s", msg.error())
                        continue
                    batch.append(msg)

                if not batch:
                    continue

                events = []
                for msg in batch:
                    try:
                        event = parse_outcome_event(msg.value())
                    except ValueError as exc:
                        with state.lock:
                            state.consume_errors += 1
                        log.warning("Skipping malformed outcome event: %s", exc)
                        continue
                    events.append(event)
                pending_batch = batch
                pending_events = events
                pending_persisted = False

            if not pending_persisted:
                try:
                    inserted = insert_outcomes(conn, pending_events)
                    inserted_since_prune += inserted
                    if maybe_prune_outcomes(conn, inserted_since_prune):
                        inserted_since_prune %= PRUNE_EVERY_N_INSERTS
                    pending_persisted = True
                except sqlite3.DatabaseError as exc:
                    with state.lock:
                        state.write_errors += 1
                    log.error("Outcomes batch write failed, will retry: %s", exc)
                    stop_event.wait(0.1)
                    continue

            offsets = _offsets_after(pending_batch)
            try:
                committed = consumer.commit(offsets=offsets, asynchronous=False)
            except Exception as exc:
                with state.lock:
                    state.consume_errors += 1
                log.error("Kafka offset commit failed, will retry: %s", exc)
                stop_event.wait(COMMIT_RETRY_BACKOFF_SEC)
                continue

            commit_errors = [
                partition
                for partition in committed or []
                if partition.error is not None
            ]
            successful_partitions = [
                partition for partition in committed or [] if partition.error is None
            ]
            committed_recovery_partitions.update(
                (partition.topic, partition.partition)
                for partition in successful_partitions
            )
            if commit_errors:
                with state.lock:
                    state.consume_errors += len(commit_errors)
                for partition in commit_errors:
                    log.error(
                        "Kafka offset commit failed for %s[%d], will retry: %s",
                        partition.topic,
                        partition.partition,
                        partition.error,
                    )
                stop_event.wait(COMMIT_RETRY_BACKOFF_SEC)
                continue
            if not committed:
                with state.lock:
                    state.consume_errors += 1
                log.error("Kafka offset commit returned no results; will retry")
                stop_event.wait(COMMIT_RETRY_BACKOFF_SEC)
                continue

            pending_batch = None
            pending_events = None
            pending_persisted = False
    except Exception as exc:
        if not stop_event.is_set():
            log.exception("Materializer outcomes consumer failed: %s", exc)
    finally:
        with state.lock:
            state.outcomes_alive = False
        if ready_event is not None:
            ready_event.set()
        if consumer is not None:
            try:
                consumer.close()
            except Exception:
                log.exception("Failed to close Kafka outcomes consumer")
        if conn is not None:
            conn.close()
        log.info("Materializer outcomes consumer stopped.")


def supervise_consumer(
    state: ConsumerState,
    stop_event: threading.Event,
    ready_event: threading.Event,
    loop_fn=None,
) -> None:
    """Restart the consumer loop when it fails after initial readiness.

    `loop_fn` defaults to `consume_loop` (the predictions consumer); the
    outcomes consumer supervisor is started with `loop_fn=consume_outcomes_loop`
    from the same `lifespan` — both share this restart-on-failure discipline.
    """
    if loop_fn is None:
        loop_fn = consume_loop
    first_attempt = True
    while not stop_event.is_set():
        loop_fn(
            state,
            stop_event,
            ready_event if first_attempt else None,
        )
        first_attempt = False
        if stop_event.is_set():
            break
        log.error(
            "Materializer consumer stopped unexpectedly; restarting in %.1fs",
            CONSUMER_RESTART_BACKOFF_SEC,
        )
        stop_event.wait(CONSUMER_RESTART_BACKOFF_SEC)


# ---------------------------------------------------------------------------
# /predictions/performance — snapshot cache + payload assembly
# ---------------------------------------------------------------------------

_perf_cache: dict[int, tuple[float, dict]] = {}
_perf_cache_lock = threading.Lock()


def _reset_perf_cache() -> None:
    """Test seam: clear the cached /predictions/performance responses."""
    with _perf_cache_lock:
        _perf_cache.clear()


def _newest_predictions_feature_ts(path: str | Path) -> str | None:
    conn = _open_readonly(path)
    try:
        row = conn.execute("SELECT MAX(feature_ts) FROM predictions").fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _build_performance_payload(window_minutes: int) -> dict:
    """Assemble the full /predictions/performance response for `window_minutes`.

    The window is anchored to the newest available prediction's feature_ts
    (not wall-clock now) so replay/backtest runs — where feature_ts can lag
    far behind real time — still see a populated window.
    """
    now = datetime.now(timezone.utc)
    newest_feature_ts = _newest_predictions_feature_ts(PREDICTIONS_DB_PATH)
    anchor = _parse_iso(newest_feature_ts) or now
    cutoff = _normalize_ts((anchor - timedelta(minutes=window_minutes)).isoformat())

    window_data = performance_window(PREDICTIONS_DB_PATH, cutoff)
    oldest_feature_ts = window_data["oldest_feature_ts"]
    # "complete" = the DB's retained history actually reaches back past the
    # cutoff, i.e. pruning/short history hasn't silently truncated the window.
    complete = oldest_feature_ts is None or oldest_feature_ts <= cutoff

    result = evaluation.compute_performance(
        window_data["joined_rows"],
        min_positives=MIN_POSITIVES,
        min_note_sample_n=MIN_NOTE_SAMPLE_N,
        drift_pr_auc_ratio=DRIFT_PR_AUC_RATIO,
        baseline_vol_threshold=BASELINE_VOL_THRESHOLD,
        adaptive_percentile=ADAPTIVE_PERCENTILE,
    )
    result["window"]["from_feature_ts"] = cutoff
    result["window"]["to_feature_ts"] = newest_feature_ts
    result["window"]["n_outcomes_unmatched"] = window_data["n_outcomes_unmatched"]

    return {
        "as_of": now.isoformat(),
        "window_minutes": window_minutes,
        "complete": complete,
        **result,
    }


def _parse_iso(raw: str | None):
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _build_timeline_payload(
    from_dt: datetime, to_dt: datetime, resolution: int | None
) -> dict:
    """Assemble the full /predictions/timeline response for [from_dt, to_dt).

    `complete` reports whether retained history actually reaches back to
    from_dt (not a row-count-based guess -- Coinbase can exceed 1 event/s,
    so a fixed row cap does not reliably imply a fixed hours-of-history
    figure). The maturity anchor for pending-vs-unavailable classification
    is available_to (the newest feature_ts in the whole table), never
    wall-clock now, so replay/backtest data classifies identically to live.
    """
    from_ts = _normalize_ts(from_dt.isoformat())
    to_ts = _normalize_ts(to_dt.isoformat())

    window_data = timeline_window(PREDICTIONS_DB_PATH, from_ts, to_ts)
    available_from = window_data["available_from"]
    available_to = window_data["available_to"]
    complete = available_from is None or available_from <= from_ts

    anchor_dt = _parse_iso(available_to)
    points, aggregated, resolution_used = timeline.build_timeline(
        window_data["joined_rows"], from_dt, to_dt, anchor_dt, resolution
    )

    return {
        "from": from_ts,
        "to": to_ts,
        "resolution": resolution_used,
        "aggregated": aggregated,
        "complete": complete,
        "available_from": available_from,
        "available_to": available_to,
        "points": points,
    }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

_state = ConsumerState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _reset_state(_state)
    _reset_perf_cache()
    stop_event = threading.Event()
    ready_event = threading.Event()
    outcomes_ready_event = threading.Event()
    probe_thread = threading.Thread(
        target=probe_loop,
        args=(_state, stop_event),
        name="materializer-broker-probe",
        daemon=True,
    )
    probe_thread.start()
    thread = threading.Thread(
        target=supervise_consumer,
        args=(_state, stop_event, ready_event),
        name="materializer-consumer-supervisor",
        daemon=True,
    )
    thread.start()
    # The outcomes consumer shares the same stop_event (one shutdown signal
    # stops both) but its own readiness is best-effort: overall app startup
    # only waits on the predictions consumer below, matching the plan's
    # "folds into /health ok, doesn't gate startup" requirement.
    outcomes_thread = threading.Thread(
        target=supervise_consumer,
        args=(_state, stop_event, outcomes_ready_event),
        kwargs={"loop_fn": consume_outcomes_loop},
        name="materializer-outcomes-consumer-supervisor",
        daemon=True,
    )
    outcomes_thread.start()
    signaled = await asyncio.to_thread(ready_event.wait, STARTUP_READY_TIMEOUT)
    with _state.lock:
        ready = _state.ready
        startup_error = _state.startup_error
    if not signaled or not ready:
        stop_event.set()
        await asyncio.to_thread(thread.join, THREAD_JOIN_TIMEOUT)
        if thread.is_alive():
            log.error("Materializer consumer did not stop after startup failure")
        await asyncio.to_thread(outcomes_thread.join, THREAD_JOIN_TIMEOUT)
        await asyncio.to_thread(probe_thread.join, THREAD_JOIN_TIMEOUT)
        detail = startup_error or "consumer readiness timed out"
        raise RuntimeError(f"Materializer startup failed: {detail}")

    try:
        yield
    finally:
        stop_event.set()
        await asyncio.to_thread(thread.join, THREAD_JOIN_TIMEOUT)
        if thread.is_alive():
            log.error(
                "Materializer consumer did not stop within %.1fs",
                THREAD_JOIN_TIMEOUT,
            )
        await asyncio.to_thread(outcomes_thread.join, THREAD_JOIN_TIMEOUT)
        if outcomes_thread.is_alive():
            log.error(
                "Materializer outcomes consumer did not stop within %.1fs",
                THREAD_JOIN_TIMEOUT,
            )
        await asyncio.to_thread(probe_thread.join, THREAD_JOIN_TIMEOUT)
        if probe_thread.is_alive():
            log.error(
                "Materializer broker probe did not stop within %.1fs",
                THREAD_JOIN_TIMEOUT,
            )


app = FastAPI(title="BTCSpiker Materializer", lifespan=lifespan)


@app.get("/predictions/recent")
def get_recent(limit: int = 200):
    try:
        limit = validate_limit(limit)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    with _state.lock:
        ready = _state.ready
    if not ready:
        raise HTTPException(status_code=503, detail="materializer is not ready")
    try:
        rows = recent(PREDICTIONS_DB_PATH, limit)
    except sqlite3.DatabaseError as exc:
        raise HTTPException(
            status_code=503, detail="predictions database unavailable"
        ) from exc
    return {"predictions": rows, "count": len(rows)}


@app.get("/predictions/performance")
def get_performance(window_minutes: int = 30):
    try:
        window_minutes = validate_window_minutes(window_minutes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    with _state.lock:
        ready = _state.ready
    if not ready:
        raise HTTPException(status_code=503, detail="materializer is not ready")

    now = time.monotonic()
    with _perf_cache_lock:
        cached = _perf_cache.get(window_minutes)
        if cached is not None and (now - cached[0]) < PERF_CACHE_SECONDS:
            return cached[1]

    try:
        payload = _build_performance_payload(window_minutes)
    except sqlite3.DatabaseError as exc:
        raise HTTPException(
            status_code=503, detail="predictions database unavailable"
        ) from exc

    with _perf_cache_lock:
        _perf_cache[window_minutes] = (time.monotonic(), payload)
    return payload


@app.get("/predictions/timeline")
def get_timeline(
    from_: str = Query(alias="from"), to: str = Query(), resolution: int | None = None
):
    try:
        from_dt, to_dt, resolution = validate_timeline_range(from_, to, resolution)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    with _state.lock:
        ready = _state.ready
    if not ready:
        raise HTTPException(status_code=503, detail="materializer is not ready")

    try:
        return _build_timeline_payload(from_dt, to_dt, resolution)
    except sqlite3.DatabaseError as exc:
        raise HTTPException(
            status_code=503, detail="predictions database unavailable"
        ) from exc


@app.get("/health")
def get_health():
    return health_snapshot(_state, PREDICTIONS_DB_PATH)


@app.get("/predictions/health")
def get_predictions_health():
    return health_snapshot(_state, PREDICTIONS_DB_PATH)
