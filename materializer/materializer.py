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

Kafka is the source of truth; the SQLite file is a disposable projection.
When the projection is missing or rebuilt, assigned partitions are explicitly
rewound to the beginning even if the stable consumer group has committed
offsets. Delivery on ticks.predictions is at-least-once, so every insert is
`INSERT OR IGNORE` keyed on `event_id`.

Importing this module has NO side effects: no Kafka connection, no thread,
no DB file creation. The consumer thread is started from the FastAPI
lifespan hook at app startup, not at import time — see `consume_loop` /
`lifespan`.

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
from datetime import datetime, timezone
from pathlib import Path

from confluent_kafka import Consumer, KafkaError, OFFSET_BEGINNING, TopicPartition
from confluent_kafka.admin import AdminClient
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
STARTUP_READY_TIMEOUT = STARTUP_TIMEOUT + 5.0
KAFKA_SOCKET_TIMEOUT_MS = 3000
THREAD_JOIN_TIMEOUT = 10.0
COMMIT_RETRY_BACKOFF_SEC = 0.1
CONSUMER_RESTART_BACKOFF_SEC = 1.0
BROKER_HEALTHCHECK_INTERVAL_SEC = 5.0
BROKER_HEALTHCHECK_TIMEOUT_SEC = 1.0

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
    conn.commit()
    return conn, corrupt or not existed or not schema_existed


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
                {NEWEST_ORDER_SQL}
                LIMIT ?
            )
            """.format(
                NEWEST_ORDER_SQL=NEWEST_ORDER_SQL
            ),
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
        state.broker_ok = False
        state.ready = False
        state.startup_error = None


def health_snapshot(state: ConsumerState, db_path: str | Path) -> dict:
    """Build the /health (and /predictions/health) response body."""
    rows_total, db_ok = _rows_total(db_path)
    with state.lock:
        return {
            "ok": state.alive and state.broker_ok and db_ok,
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
    consumer: Consumer, replay_from_beginning: bool
) -> set[tuple[str, int]]:
    """Subscribe and return recovery partitions with a successful commit."""
    if not replay_from_beginning:
        consumer.subscribe([TOPIC_PREDICTIONS])
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
                "Predictions projection is new/rebuilt; replaying %d assigned "
                "partition(s) from the beginning",
                rewound,
            )
        assigned_consumer.assign(partitions)

    consumer.subscribe([TOPIC_PREDICTIONS], on_assign=_on_assign)
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


def consume_loop(
    state: ConsumerState,
    stop_event: threading.Event,
    ready_event: threading.Event | None = None,
    probe_broker=None,
) -> None:
    """Background thread entrypoint: ticks.predictions -> SQLite read model.

    Batches up to BATCH_MAX_SIZE messages or a BATCH_WINDOW_SEC window,
    INSERT OR IGNOREs the batch in one transaction, then commits Kafka
    offsets — at-least-once delivery + the event_id primary key make replays
    harmless.

    `state.broker_ok` is owned EXCLUSIVELY by the periodic fresh-client
    probe (`probe_broker`, every BROKER_HEALTHCHECK_INTERVAL_SEC): poll and
    commit errors are logged and counted but never touch the flag, so
    queued transport errors can't hold health false after the broker has
    recovered, and both transitions (up->down, down->up) are detected
    within one probe interval without restarting the process.
    """
    if probe_broker is None:

        def probe_broker():
            return _probe_broker(KAFKA_BOOTSTRAP, BROKER_HEALTHCHECK_TIMEOUT_SEC)

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
            state.broker_ok = True
            state.ready = True
            state.startup_error = None
        if ready_event is not None:
            ready_event.set()

        inserted_since_prune = 0
        pending_batch = None
        pending_events = None
        pending_persisted = False
        next_broker_healthcheck = time.monotonic() + BROKER_HEALTHCHECK_INTERVAL_SEC

        while not stop_event.is_set():
            if time.monotonic() >= next_broker_healthcheck:
                broker_up = probe_broker()
                with state.lock:
                    state.broker_ok = broker_up
                if not broker_up:
                    log.error(
                        "Kafka broker probe failed (bootstrap=%s)", KAFKA_BOOTSTRAP
                    )
                next_broker_healthcheck = (
                    time.monotonic() + BROKER_HEALTHCHECK_INTERVAL_SEC
                )

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
            state.broker_ok = False
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


def supervise_consumer(
    state: ConsumerState,
    stop_event: threading.Event,
    ready_event: threading.Event,
) -> None:
    """Restart the consumer loop when it fails after initial readiness."""
    first_attempt = True
    while not stop_event.is_set():
        consume_loop(
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
# FastAPI app
# ---------------------------------------------------------------------------

_state = ConsumerState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _reset_state(_state)
    stop_event = threading.Event()
    ready_event = threading.Event()
    thread = threading.Thread(
        target=supervise_consumer,
        args=(_state, stop_event, ready_event),
        name="materializer-consumer-supervisor",
        daemon=True,
    )
    thread.start()
    signaled = await asyncio.to_thread(ready_event.wait, STARTUP_READY_TIMEOUT)
    with _state.lock:
        ready = _state.ready
        startup_error = _state.startup_error
    if not signaled or not ready:
        stop_event.set()
        await asyncio.to_thread(thread.join, THREAD_JOIN_TIMEOUT)
        if thread.is_alive():
            log.error("Materializer consumer did not stop after startup failure")
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


@app.get("/health")
def get_health():
    return health_snapshot(_state, PREDICTIONS_DB_PATH)


@app.get("/predictions/health")
def get_predictions_health():
    return health_snapshot(_state, PREDICTIONS_DB_PATH)
