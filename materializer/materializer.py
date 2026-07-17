"""
Materializer — Kafka read-model service.

Consumes PredictionEvent messages from `ticks.predictions` into a local
SQLite read model (WAL mode) and serves it over a small FastAPI surface:

    GET /predictions/recent?limit=200   -> newest predictions, newest-first
    GET /health                         -> consumer + DB health
    GET /predictions/health             -> alias of /health (nginx proxies
                                            only /api/predictions/* to this
                                            service, so health must live
                                            under that prefix too)

Kafka is the source of truth; the SQLite file is a disposable projection —
`docker volume rm` + a topic replay (fresh consumer group, auto.offset.reset
= earliest) rebuilds it from scratch. Delivery on ticks.predictions is
at-least-once, so every insert is `INSERT OR IGNORE` keyed on `event_id`.

Importing this module has NO side effects: no Kafka connection, no thread,
no DB file creation. The consumer thread is started from the FastAPI
lifespan hook at app startup, not at import time — see `consume_loop` /
`lifespan`.

Usage
-----
    uvicorn materializer:app --host 0.0.0.0 --port 8090
"""

import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from confluent_kafka import Consumer, KafkaError
from fastapi import FastAPI, HTTPException

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
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "materializer")
PREDICTIONS_DB_PATH = os.getenv("PREDICTIONS_DB_PATH", "/data/predictions.db")

# Plan decisions — module constants, not configurable.
BATCH_MAX_SIZE = 200
BATCH_WINDOW_SEC = 1.0
BUSY_TIMEOUT_MS = 5000
PRUNE_EVERY_N_INSERTS = 1000
PRUNE_KEEP_ROWS = 100_000
LIMIT_MIN = 1
LIMIT_MAX = 2000
STARTUP_TIMEOUT = 30.0

# Pinned event_id / column ordering shared by parsing, inserts, and reads.
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

RECENT_SQL = (
    f"SELECT {', '.join(EVENT_FIELDS)} FROM predictions "
    "ORDER BY source_offset DESC, event_id ASC LIMIT ?"
)


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------


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
    try:
        conn = sqlite3.connect(str(path))
        conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        row = conn.execute("PRAGMA integrity_check").fetchone()
        if row is None or row[0] != "ok":
            conn.close()
            return None
        return conn
    except sqlite3.DatabaseError:
        return None


def init_db(path: str | Path) -> sqlite3.Connection:
    """Open (or create) the predictions read-model DB: WAL mode, schema applied.

    A corrupt file is deleted (with its -wal/-shm siblings) and recreated —
    Kafka is the source of truth, so the read model can always be rebuilt by
    replaying `ticks.predictions` from a fresh consumer group.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = _open_verified(path)
    if conn is None:
        log.error(
            "Predictions DB at %s is corrupt/unreadable; deleting and rebuilding "
            "from Kafka replay",
            path,
        )
        _delete_db_files(path)
        conn = sqlite3.connect(str(path))
        conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")

    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute(SCHEMA_SQL)
    conn.commit()
    return conn


def insert_events(conn: sqlite3.Connection, events: list[dict]) -> int:
    """INSERT OR IGNORE a batch of event dicts in one transaction.

    Returns the number of rows actually inserted — duplicates ignored via the
    event_id primary key (at-least-once delivery) don't count.
    """
    if not events:
        return 0
    inserted = 0
    with conn:
        cur = conn.cursor()
        for event in events:
            cur.execute(INSERT_SQL, tuple(event.get(f) for f in EVENT_FIELDS))
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
                ORDER BY source_offset DESC, event_id ASC
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


def recent(path_or_conn: str | Path | sqlite3.Connection, limit: int) -> list[dict]:
    """Return up to `limit` predictions, newest-first by source_offset.

    Given a path, opens a short-lived read connection — the HTTP-handler
    pattern, safe across threads since the consumer thread owns the single
    writer connection. Given an existing connection, queries it directly
    (used by the consumer thread and by tests).
    """
    if isinstance(path_or_conn, sqlite3.Connection):
        return _recent_rows(path_or_conn, limit)

    conn = sqlite3.connect(str(path_or_conn))
    try:
        return _recent_rows(conn, limit)
    finally:
        conn.close()


def _rows_total(path: str | Path) -> tuple[int, bool]:
    """Return (count, db_ok) via a short-lived read connection."""
    try:
        conn = sqlite3.connect(str(path))
        try:
            count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            return count, True
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return 0, False


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


# ---------------------------------------------------------------------------
# Consumer state + health + request validation
# ---------------------------------------------------------------------------


@dataclass
class ConsumerState:
    """Mutable counters shared between the consumer thread and HTTP handlers."""

    last_event_ts: str | None = None
    last_write_ts: str | None = None
    consume_errors: int = 0
    write_errors: int = 0
    alive: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


def health_snapshot(state: ConsumerState, db_path: str | Path) -> dict:
    """Build the /health (and /predictions/health) response body."""
    rows_total, db_ok = _rows_total(db_path)
    with state.lock:
        return {
            "ok": state.alive and db_ok,
            "last_event_ts": state.last_event_ts,
            "last_write_ts": state.last_write_ts,
            "rows_total": rows_total,
            "consume_errors": state.consume_errors,
            "write_errors": state.write_errors,
        }


def validate_limit(limit: int) -> int:
    """Validate the /predictions/recent `limit` query param (1..2000)."""
    if not (LIMIT_MIN <= limit <= LIMIT_MAX):
        raise ValueError(
            f"limit must be between {LIMIT_MIN} and {LIMIT_MAX}, got {limit}"
        )
    return limit


# ---------------------------------------------------------------------------
# Kafka consumer loop (background thread; requires a live broker, so it is
# not exercised by unit tests — those cover the pure functions above)
# ---------------------------------------------------------------------------


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


def consume_loop(state: ConsumerState, stop_event: threading.Event) -> None:
    """Background thread entrypoint: ticks.predictions -> SQLite read model.

    Batches up to BATCH_MAX_SIZE messages or a BATCH_WINDOW_SEC window,
    INSERT OR IGNOREs the batch in one transaction, then commits Kafka
    offsets — at-least-once delivery + the event_id primary key make replays
    harmless.
    """
    conn = init_db(PREDICTIONS_DB_PATH)
    consumer = Consumer(
        {
            "bootstrap.servers": KAFKA_BOOTSTRAP,
            "group.id": KAFKA_GROUP_ID,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([TOPIC_PREDICTIONS])
    _wait_for_kafka(consumer, KAFKA_BOOTSTRAP, STARTUP_TIMEOUT)
    log.info(
        "Materializer consumer started | %s -> %s | group=%s",
        TOPIC_PREDICTIONS,
        PREDICTIONS_DB_PATH,
        KAFKA_GROUP_ID,
    )

    with state.lock:
        state.alive = True
    inserted_since_prune = 0

    try:
        while not stop_event.is_set():
            batch = []
            deadline = time.monotonic() + BATCH_WINDOW_SEC
            while len(batch) < BATCH_MAX_SIZE and time.monotonic() < deadline:
                msg = consumer.poll(timeout=0.2)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() != KafkaError._PARTITION_EOF:
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
                with state.lock:
                    if event["api_ts"] and (
                        state.last_event_ts is None
                        or event["api_ts"] > state.last_event_ts
                    ):
                        state.last_event_ts = event["api_ts"]

            try:
                inserted = insert_events(conn, events)
                inserted_since_prune += inserted
                if maybe_prune(conn, inserted_since_prune):
                    inserted_since_prune = 0
                with state.lock:
                    state.last_write_ts = datetime.now(timezone.utc).isoformat()
            except sqlite3.DatabaseError as exc:
                with state.lock:
                    state.write_errors += 1
                log.error("Batch write failed, will retry: %s", exc)
                continue  # don't commit offsets; retry this batch next loop

            consumer.commit(asynchronous=False)
    finally:
        with state.lock:
            state.alive = False
        consumer.close()
        conn.close()
        log.info("Materializer consumer stopped.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

_state = ConsumerState()
_stop_event = threading.Event()


@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(
        target=consume_loop, args=(_state, _stop_event), daemon=True
    )
    thread.start()
    yield
    _stop_event.set()


app = FastAPI(title="BTCSpiker Materializer", lifespan=lifespan)


@app.get("/predictions/recent")
def get_recent(limit: int = 200):
    try:
        limit = validate_limit(limit)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    rows = recent(PREDICTIONS_DB_PATH, limit)
    return {"predictions": rows, "count": len(rows)}


@app.get("/health")
def get_health():
    return health_snapshot(_state, PREDICTIONS_DB_PATH)


@app.get("/predictions/health")
def get_predictions_health():
    return health_snapshot(_state, PREDICTIONS_DB_PATH)
