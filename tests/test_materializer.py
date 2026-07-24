"""Unit tests for the materializer read-model service.

Exercises SQLite helpers against tmp_path databases, the consumer loop through
a deterministic fake Kafka consumer, and the actual FastAPI route contract via
TestClient. No broker or listening network socket is required.

Run from the repo root with:

    python3 -m pytest tests/test_materializer.py -q
"""

import json
import logging
import sqlite3
import sys
import threading
import time
from pathlib import Path

import pytest
from confluent_kafka import OFFSET_BEGINNING, TopicPartition
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATERIALIZER_DIR = PROJECT_ROOT / "materializer"
sys.path.insert(0, str(MATERIALIZER_DIR))

import materializer  # noqa: E402


class _FakeMessage:
    def __init__(self, event: dict, offset: int, partition: int = 0):
        self._value = json.dumps(event).encode()
        self._offset = offset
        self._partition = partition

    def value(self):
        return self._value

    def offset(self):
        return self._offset

    def partition(self):
        return self._partition

    def topic(self):
        return materializer.TOPIC_PREDICTIONS

    def error(self):
        return None


class _FakeKafkaError:
    def __init__(self, code):
        self._code = code

    def code(self):
        return self._code

    def __str__(self):
        return f"fake Kafka error {self._code}"


class _FakeErrorMessage:
    def __init__(self, code):
        self._error = _FakeKafkaError(code)

    def error(self):
        return self._error


class _FakeCommittedPartition:
    def __init__(self, topic, partition, offset, error=None):
        self.topic = topic
        self.partition = partition
        self.offset = offset
        self.error = error


class _FakeConsumer:
    def __init__(
        self,
        messages,
        stop_event,
        *,
        committed_offset=0,
        replay_messages=None,
        commits_until_stop=1,
        commit_outcomes=None,
        pre_commit_rebalance_offset=None,
    ):
        self.messages = list(messages)
        self.replay_messages = list(replay_messages or [])
        self.stop_event = stop_event
        self.committed_offset = committed_offset
        self.commits_until_stop = commits_until_stop
        self.commit_outcomes = list(commit_outcomes or [])
        self.pre_commit_rebalance_offset = pre_commit_rebalance_offset
        self.assigned = []
        self.assignment_history = []
        self.commits = []
        self.successful_commits = 0
        self.closed = False

    def list_topics(self, timeout):
        return object()

    def subscribe(self, topics, on_assign=None):
        self.topics = topics
        self.on_assign = on_assign
        if on_assign is not None:
            on_assign(
                self,
                [TopicPartition(topics[0], 0, self.committed_offset)],
            )

    def assign(self, partitions):
        self.assigned = list(partitions)
        self.assignment_history.append(list(partitions))
        if any(partition.offset == OFFSET_BEGINNING for partition in partitions):
            self.messages = list(self.replay_messages)

    def poll(self, timeout):
        if self.messages:
            return self.messages.pop(0)
        return None

    def commit(self, message=None, offsets=None, asynchronous=True):
        self.commits.append(offsets)
        if (
            len(self.commits) == 1
            and self.pre_commit_rebalance_offset is not None
            and self.on_assign is not None
        ):
            self.on_assign(
                self,
                [TopicPartition(self.topics[0], 0, self.pre_commit_rebalance_offset)],
            )
        outcome = self.commit_outcomes.pop(0) if self.commit_outcomes else offsets
        if isinstance(outcome, Exception):
            raise outcome
        if all(partition.error is None for partition in outcome or []):
            self.successful_commits += 1
        if self.successful_commits >= self.commits_until_stop:
            self.stop_event.set()
        return outcome

    def close(self):
        self.closed = True


def _event(event_id: str, source_offset: int, **overrides) -> dict:
    """Build a full PredictionEvent dict matching the pinned ticks.predictions
    contract, with sensible defaults for every field."""
    row = {
        "event_id": event_id,
        "source_partition": 0,
        "source_offset": source_offset,
        "feature_ts": "2026-07-16T19:00:00Z",
        "api_ts": "2026-07-16T19:00:00.123456+00:00",
        "score": 0.42,
        "model_variant": "ml",
        "model_version": "v1.0",
        "vol_60s": 0.00005,
        "spread_bps": 1.5,
        "log_return": 0.0001,
        "trade_intensity_60s": 10.0,
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# init_db / dedup / corruption
# ---------------------------------------------------------------------------


def test_dedup_insert_twice_yields_one_row(tmp_path):
    conn = materializer.init_db(tmp_path / "test.db")
    event = _event("ticks.features:0:1", 1)

    first = materializer.insert_events(conn, [event])
    second = materializer.insert_events(conn, [event])

    assert first == 1
    assert second == 0
    rows = materializer.recent(conn, 10)
    assert len(rows) == 1
    assert rows[0]["event_id"] == "ticks.features:0:1"


def test_corrupt_db_is_deleted_and_recreated(tmp_path, caplog):
    db_path = tmp_path / "corrupt.db"
    db_path.write_bytes(b"this is not a sqlite file, just garbage bytes")

    with caplog.at_level(logging.ERROR):
        conn = materializer.init_db(db_path)

    assert "corrupt" in caplog.text.lower()

    # File was recreated as a working, writable database.
    event = _event("ticks.features:0:1", 1)
    inserted = materializer.insert_events(conn, [event])
    assert inserted == 1
    assert len(materializer.recent(conn, 10)) == 1


def test_init_db_recreates_wal_and_shm_siblings(tmp_path):
    db_path = tmp_path / "corrupt.db"
    db_path.write_bytes(b"garbage")
    (tmp_path / "corrupt.db-wal").write_bytes(b"stale wal bytes")
    (tmp_path / "corrupt.db-shm").write_bytes(b"stale shm bytes")

    conn = materializer.init_db(db_path)

    # Recreated DB is usable; stale -wal/-shm garbage didn't wedge it.
    inserted = materializer.insert_events(conn, [_event("e1", 1)])
    assert inserted == 1


def test_consumer_retries_failed_batch_before_polling_or_committing_past_it(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "predictions.db"
    materializer.init_db(db_path).close()
    stop_event = threading.Event()
    first = _FakeMessage(_event("e1", 1), 0)
    second = _FakeMessage(_event("e2", 2), 1)
    consumer = _FakeConsumer([first, second], stop_event)
    real_insert_events = materializer.insert_events
    attempted_event_ids = []

    def _fail_first_write(conn, events):
        attempted_event_ids.append([event["event_id"] for event in events])
        if len(attempted_event_ids) == 1:
            raise sqlite3.OperationalError("transient write failure")
        return real_insert_events(conn, events)

    monkeypatch.setattr(materializer, "PREDICTIONS_DB_PATH", str(db_path))
    monkeypatch.setattr(materializer, "BATCH_MAX_SIZE", 1)
    monkeypatch.setattr(materializer, "BATCH_WINDOW_SEC", 0.01)
    monkeypatch.setattr(materializer, "Consumer", lambda config: consumer)
    monkeypatch.setattr(materializer, "insert_events", _fail_first_write)

    state = materializer.ConsumerState()
    materializer.consume_loop(state, stop_event)

    rows = materializer.recent(db_path, 10)
    assert attempted_event_ids == [["e1"], ["e1"]]
    assert [row["event_id"] for row in rows] == ["e1"]
    assert consumer.commits[0][0].offset == 1
    assert consumer.closed is True


def test_corrupt_db_recovery_replays_before_committed_group_offset(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "predictions.db"
    db_path.write_bytes(b"not a sqlite database")
    stop_event = threading.Event()
    historical = _FakeMessage(_event("historical", 1), 0)
    after_committed_offset = _FakeMessage(_event("future", 43), 42)
    consumer = _FakeConsumer(
        [after_committed_offset],
        stop_event,
        committed_offset=42,
        replay_messages=[historical],
    )

    monkeypatch.setattr(materializer, "PREDICTIONS_DB_PATH", str(db_path))
    monkeypatch.setattr(materializer, "BATCH_MAX_SIZE", 1)
    monkeypatch.setattr(materializer, "BATCH_WINDOW_SEC", 0.01)
    monkeypatch.setattr(materializer, "Consumer", lambda config: consumer)

    materializer.consume_loop(materializer.ConsumerState(), stop_event)

    rows = materializer.recent(db_path, 10)
    assert consumer.assigned[0].offset == OFFSET_BEGINNING
    assert [row["event_id"] for row in rows] == ["historical"]


def test_recovery_rewinds_again_on_rebalance_until_first_commit(tmp_path, monkeypatch):
    db_path = tmp_path / "predictions.db"
    db_path.write_bytes(b"not a sqlite database")
    stop_event = threading.Event()
    historical = _FakeMessage(_event("historical", 1), 0)
    consumer = _FakeConsumer(
        [],
        stop_event,
        committed_offset=42,
        replay_messages=[historical],
        pre_commit_rebalance_offset=7,
    )
    monkeypatch.setattr(materializer, "PREDICTIONS_DB_PATH", str(db_path))
    monkeypatch.setattr(materializer, "BATCH_MAX_SIZE", 1)
    monkeypatch.setattr(materializer, "BATCH_WINDOW_SEC", 0.01)
    monkeypatch.setattr(materializer, "Consumer", lambda config: consumer)

    materializer.consume_loop(materializer.ConsumerState(), stop_event)

    assert consumer.assignment_history[0][0].offset == OFFSET_BEGINNING
    assert consumer.assignment_history[1][0].offset == OFFSET_BEGINNING
    consumer.on_assign(
        consumer,
        [TopicPartition(materializer.TOPIC_PREDICTIONS, 0, 8)],
    )
    assert consumer.assigned[0].offset == 8


def test_commit_exception_retries_same_persisted_batch(tmp_path, monkeypatch):
    db_path = tmp_path / "predictions.db"
    materializer.init_db(db_path).close()
    stop_event = threading.Event()
    consumer = _FakeConsumer(
        [_FakeMessage(_event("e1", 1), 0)],
        stop_event,
        commit_outcomes=[RuntimeError("commit unavailable")],
    )
    real_insert_events = materializer.insert_events
    insert_calls = 0

    def _count_inserts(conn, events):
        nonlocal insert_calls
        insert_calls += 1
        return real_insert_events(conn, events)

    monkeypatch.setattr(materializer, "PREDICTIONS_DB_PATH", str(db_path))
    monkeypatch.setattr(materializer, "BATCH_MAX_SIZE", 1)
    monkeypatch.setattr(materializer, "BATCH_WINDOW_SEC", 0.01)
    monkeypatch.setattr(materializer, "Consumer", lambda config: consumer)
    monkeypatch.setattr(materializer, "insert_events", _count_inserts)

    state = materializer.ConsumerState()
    materializer.consume_loop(state, stop_event)

    assert len(consumer.commits) == 2
    assert insert_calls == 1
    assert state.consume_errors == 1
    assert [row["event_id"] for row in materializer.recent(db_path, 10)] == ["e1"]


def test_partition_commit_error_retries_same_persisted_batch(tmp_path, monkeypatch):
    db_path = tmp_path / "predictions.db"
    materializer.init_db(db_path).close()
    stop_event = threading.Event()
    commit_error = _FakeKafkaError(materializer.KafkaError._TIMED_OUT)
    consumer = _FakeConsumer(
        [_FakeMessage(_event("e1", 1), 0)],
        stop_event,
        commit_outcomes=[
            [
                _FakeCommittedPartition(
                    materializer.TOPIC_PREDICTIONS,
                    0,
                    1,
                    error=commit_error,
                )
            ]
        ],
    )
    real_insert_events = materializer.insert_events
    insert_calls = 0

    def _count_inserts(conn, events):
        nonlocal insert_calls
        insert_calls += 1
        return real_insert_events(conn, events)

    monkeypatch.setattr(materializer, "PREDICTIONS_DB_PATH", str(db_path))
    monkeypatch.setattr(materializer, "BATCH_MAX_SIZE", 1)
    monkeypatch.setattr(materializer, "BATCH_WINDOW_SEC", 0.01)
    monkeypatch.setattr(materializer, "Consumer", lambda config: consumer)
    monkeypatch.setattr(materializer, "insert_events", _count_inserts)

    state = materializer.ConsumerState()
    materializer.consume_loop(state, stop_event)

    assert len(consumer.commits) == 2
    assert insert_calls == 1
    assert state.consume_errors == 1
    assert [row["event_id"] for row in materializer.recent(db_path, 10)] == ["e1"]


def _run_probe_loop(monkeypatch, probe_results, initial_broker_ok=True):
    """Run probe_loop with an injected probe returning the given sequence,
    recording broker_ok as observed at each probe call (i.e. the state left
    by the PREVIOUS probe). Stops after the sequence is exhausted.
    """
    stop_event = threading.Event()
    state = materializer.ConsumerState()
    state.broker_ok = initial_broker_ok
    results = list(probe_results)
    observed = []

    def fake_probe():
        with state.lock:
            observed.append(state.broker_ok)
        result = results.pop(0)
        if not results:
            stop_event.set()
        return result

    monkeypatch.setattr(materializer, "BROKER_HEALTHCHECK_INTERVAL_SEC", 0.001)
    materializer.probe_loop(state, stop_event, probe_broker=fake_probe)
    return state, observed


def test_broker_outage_marks_health_not_ok_within_one_probe_interval(monkeypatch):
    # Probe call 1 fails (broker down); probe call 2 observes the state the
    # failure left behind: broker_ok must already be False.
    _, observed = _run_probe_loop(monkeypatch, [False, True])
    assert observed == [True, False]


def test_broker_recovery_restores_health_without_restart(monkeypatch):
    # Down (False), then recovered (True); the final probe call observes the
    # state the successful probe left behind: broker_ok must be True again,
    # within the SAME probe_loop invocation — no process or thread restart.
    _, observed = _run_probe_loop(monkeypatch, [False, True, False])
    assert observed == [True, False, True]


def test_probe_thread_is_sole_owner_of_broker_health(tmp_path, monkeypatch):
    # The consume loop must never touch broker_ok: queued transport errors
    # from the long-lived consumer used to flip it false (sticky-false after
    # recovery), and its blocking synchronous commit used to starve an
    # in-loop probe (slow outage detection). Consuming error events must
    # leave broker_ok exactly as the probe thread last set it.
    stop_event = threading.Event()
    error_messages = [_FakeErrorMessage(code=-195) for _ in range(3)]  # _TRANSPORT
    good = _FakeMessage(_event("e1", 1), 1)
    messages = error_messages + [good]
    # Fresh DB triggers the replay-from-beginning rewind, which swaps the
    # fake consumer's queue to replay_messages — supply both.
    consumer = _FakeConsumer(messages, stop_event, replay_messages=messages)
    state = materializer.ConsumerState()
    monkeypatch.setattr(materializer, "PREDICTIONS_DB_PATH", str(tmp_path / "p.db"))
    monkeypatch.setattr(materializer, "BATCH_MAX_SIZE", 1)
    monkeypatch.setattr(materializer, "BATCH_WINDOW_SEC", 0.01)
    monkeypatch.setattr(materializer, "Consumer", lambda config: consumer)

    with state.lock:
        state.broker_ok = True
    materializer.consume_loop(state, stop_event)

    assert state.broker_ok is True  # untouched by errors, commits, or shutdown
    assert state.consume_errors == 3  # errors still counted, just not owning ok


def test_offsets_after_returns_maximum_next_offset_per_partition():
    messages = [
        _FakeMessage(_event("p1-low", 2), 2, partition=1),
        _FakeMessage(_event("p0-high", 9), 9, partition=0),
        _FakeMessage(_event("p1-high", 5), 5, partition=1),
        _FakeMessage(_event("p0-low", 4), 4, partition=0),
    ]

    offsets = materializer._offsets_after(messages)

    assert [
        (partition.topic, partition.partition, partition.offset)
        for partition in offsets
    ] == [
        (materializer.TOPIC_PREDICTIONS, 0, 10),
        (materializer.TOPIC_PREDICTIONS, 1, 6),
    ]


# ---------------------------------------------------------------------------
# recent()
# ---------------------------------------------------------------------------


def test_recent_returns_newest_first_by_timestamp(tmp_path):
    conn = materializer.init_db(tmp_path / "test.db")
    events = [_event(f"e{i}", i) for i in [3, 1, 5, 2, 4]]
    materializer.insert_events(conn, events)

    rows = materializer.recent(conn, 10)

    assert [r["source_offset"] for r in rows] == [5, 4, 3, 2, 1]


def test_recent_uses_global_timestamp_then_stable_partition_offset_ties(tmp_path):
    conn = materializer.init_db(tmp_path / "test.db")
    materializer.insert_events(
        conn,
        [
            _event("old-high-offset", 999, api_ts="2026-07-16T19:00:00+00:00"),
            _event(
                "p0-o9",
                9,
                source_partition=0,
                api_ts="2026-07-16T19:01:00+00:00",
            ),
            _event(
                "p1-o2",
                2,
                source_partition=1,
                api_ts="2026-07-16T19:01:00+00:00",
            ),
            _event(
                "p1-o3",
                3,
                source_partition=1,
                api_ts="2026-07-16T19:01:00+00:00",
            ),
        ],
    )

    rows = materializer.recent(conn, 10)

    assert [row["event_id"] for row in rows] == [
        "p1-o3",
        "p1-o2",
        "p0-o9",
        "old-high-offset",
    ]


def test_recent_via_path_matches_recent_via_connection(tmp_path):
    db_path = tmp_path / "test.db"
    conn = materializer.init_db(db_path)
    materializer.insert_events(conn, [_event("e1", 1), _event("e2", 2)])

    via_conn = materializer.recent(conn, 10)
    via_path = materializer.recent(db_path, 10)

    assert via_conn == via_path


def test_recent_respects_limit(tmp_path):
    conn = materializer.init_db(tmp_path / "test.db")
    materializer.insert_events(conn, [_event(f"e{i}", i) for i in range(5)])

    rows = materializer.recent(conn, 2)

    assert len(rows) == 2
    assert [r["source_offset"] for r in rows] == [4, 3]


def test_concurrent_read_connections_are_safe_during_writes(tmp_path):
    db_path = tmp_path / "test.db"
    materializer.init_db(db_path).close()
    writer_done = threading.Event()
    errors = []

    def _write_rows():
        conn = materializer.init_db(db_path)
        try:
            for index in range(50):
                materializer.insert_events(
                    conn,
                    [
                        _event(
                            f"e{index}",
                            index,
                            api_ts=f"2026-07-16T19:00:{index:02d}+00:00",
                        )
                    ],
                )
        except Exception as exc:  # pragma: no cover - asserted through errors
            errors.append(exc)
        finally:
            conn.close()
            writer_done.set()

    writer = threading.Thread(target=_write_rows)
    writer.start()
    while not writer_done.wait(0.001):
        try:
            materializer.recent(db_path, 10)
        except Exception as exc:  # pragma: no cover - asserted through errors
            errors.append(exc)
            break
    writer.join()

    assert errors == []
    assert len(materializer.recent(db_path, 100)) == 50


def test_fastapi_recent_endpoint_enforces_actual_limit_contract(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    conn = materializer.init_db(db_path)
    materializer.insert_events(conn, [_event(f"e{i}", i) for i in range(3)])
    conn.close()
    monkeypatch.setattr(materializer, "PREDICTIONS_DB_PATH", str(db_path))
    materializer._state.ready = True
    client = TestClient(materializer.app)

    assert client.get("/predictions/recent").json()["count"] == 3
    assert client.get("/predictions/recent?limit=1").json()["count"] == 1
    for query in ("limit=0", "limit=2001", "limit=not-an-int"):
        assert client.get(f"/predictions/recent?{query}").status_code == 422


def test_fastapi_recent_returns_503_without_creating_db_before_readiness(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "not-ready.db"
    monkeypatch.setattr(materializer, "PREDICTIONS_DB_PATH", str(db_path))
    materializer._state.ready = False
    client = TestClient(materializer.app, raise_server_exceptions=False)

    response = client.get("/predictions/recent?limit=1")

    assert response.status_code == 503
    assert db_path.exists() is False


# ---------------------------------------------------------------------------
# validate_limit()
# ---------------------------------------------------------------------------


def test_validate_limit_accepts_bounds():
    assert materializer.validate_limit(1) == 1
    assert materializer.validate_limit(2000) == 2000


@pytest.mark.parametrize("bad_limit", [0, -1, 2001, 10_000])
def test_validate_limit_rejects_out_of_range(bad_limit):
    with pytest.raises(ValueError):
        materializer.validate_limit(bad_limit)


# ---------------------------------------------------------------------------
# parse_event()
# ---------------------------------------------------------------------------


def test_parse_event_valid_json():
    raw = json.dumps(_event("e1", 1))
    parsed = materializer.parse_event(raw)
    assert parsed["event_id"] == "e1"
    assert parsed["source_offset"] == 1
    assert parsed["score"] == 0.42


def test_parse_event_null_feature_ts_is_allowed():
    raw = json.dumps(_event("e1", 1, feature_ts=None))
    parsed = materializer.parse_event(raw)
    assert parsed["feature_ts"] is None


def test_parse_event_malformed_json_raises():
    with pytest.raises(ValueError):
        materializer.parse_event(b"{not valid json")


def test_parse_event_missing_event_id_raises():
    with pytest.raises(ValueError):
        materializer.parse_event(json.dumps({"score": 0.5}))


def test_parse_event_empty_event_id_raises():
    with pytest.raises(ValueError):
        materializer.parse_event(json.dumps(_event("", 1)))


# ---------------------------------------------------------------------------
# maybe_prune()
# ---------------------------------------------------------------------------


def test_maybe_prune_noop_below_threshold(tmp_path):
    conn = materializer.init_db(tmp_path / "test.db")
    materializer.insert_events(conn, [_event(f"e{i}", i) for i in range(5)])

    pruned = materializer.maybe_prune(conn, materializer.PRUNE_EVERY_N_INSERTS - 1)

    assert pruned is False
    assert len(materializer.recent(conn, 100)) == 5


def test_maybe_prune_keeps_only_newest_rows(tmp_path, monkeypatch):
    conn = materializer.init_db(tmp_path / "test.db")
    monkeypatch.setattr(materializer, "PRUNE_KEEP_ROWS", 5)
    materializer.insert_events(conn, [_event(f"e{i}", i) for i in range(10)])

    pruned = materializer.maybe_prune(conn, materializer.PRUNE_EVERY_N_INSERTS)

    assert pruned is True
    rows = materializer.recent(conn, 100)
    assert len(rows) == 5
    assert {r["source_offset"] for r in rows} == {5, 6, 7, 8, 9}


def test_maybe_prune_uses_same_global_order_as_recent(tmp_path, monkeypatch):
    conn = materializer.init_db(tmp_path / "test.db")
    monkeypatch.setattr(materializer, "PRUNE_KEEP_ROWS", 2)
    materializer.insert_events(
        conn,
        [
            _event("old-high-offset", 999, api_ts="2026-07-16T19:00:00+00:00"),
            _event(
                "new-p0",
                1,
                source_partition=0,
                api_ts="2026-07-16T19:01:00+00:00",
            ),
            _event(
                "new-p1",
                1,
                source_partition=1,
                api_ts="2026-07-16T19:01:00+00:00",
            ),
        ],
    )

    materializer.maybe_prune(conn, materializer.PRUNE_EVERY_N_INSERTS)

    assert [row["event_id"] for row in materializer.recent(conn, 10)] == [
        "new-p1",
        "new-p0",
    ]


def test_consumer_preserves_prune_counter_remainder(tmp_path, monkeypatch):
    db_path = tmp_path / "predictions.db"
    materializer.init_db(db_path).close()
    stop_event = threading.Event()
    messages = [_FakeMessage(_event(f"e{i}", i), i) for i in range(6)]
    consumer = _FakeConsumer(messages, stop_event, commits_until_stop=3)
    observed_counts = []

    def _record_prune_count(conn, inserted_since_prune):
        observed_counts.append(inserted_since_prune)
        return inserted_since_prune >= 3

    monkeypatch.setattr(materializer, "PREDICTIONS_DB_PATH", str(db_path))
    monkeypatch.setattr(materializer, "BATCH_MAX_SIZE", 2)
    monkeypatch.setattr(materializer, "BATCH_WINDOW_SEC", 0.01)
    monkeypatch.setattr(materializer, "PRUNE_EVERY_N_INSERTS", 3)
    monkeypatch.setattr(materializer, "Consumer", lambda config: consumer)
    monkeypatch.setattr(materializer, "maybe_prune", _record_prune_count)

    materializer.consume_loop(materializer.ConsumerState(), stop_event)

    assert observed_counts == [2, 4, 3]


# ---------------------------------------------------------------------------
# health_snapshot()
# ---------------------------------------------------------------------------


def test_health_snapshot_reflects_rows_and_counters(tmp_path):
    db_path = tmp_path / "test.db"
    conn = materializer.init_db(db_path)
    materializer.insert_events(conn, [_event(f"e{i}", i) for i in range(3)])

    state = materializer.ConsumerState(
        last_event_ts="2026-07-16T19:00:00+00:00",
        last_write_ts="2026-07-16T19:00:01+00:00",
        consume_errors=2,
        write_errors=1,
        alive=True,
        outcomes_alive=True,
        broker_ok=True,
    )

    snapshot = materializer.health_snapshot(state, db_path)

    assert snapshot == {
        "ok": True,
        "last_event_ts": "2026-07-16T19:00:00+00:00",
        "last_write_ts": "2026-07-16T19:00:01+00:00",
        "rows_total": 3,
        "consume_errors": 2,
        "write_errors": 1,
    }


def test_health_snapshot_ok_false_when_consumer_not_alive(tmp_path):
    db_path = tmp_path / "test.db"
    materializer.init_db(db_path)
    state = materializer.ConsumerState(alive=False)

    snapshot = materializer.health_snapshot(state, db_path)

    assert snapshot["ok"] is False


def test_health_snapshot_ok_false_when_broker_probe_failed(tmp_path):
    db_path = tmp_path / "test.db"
    materializer.init_db(db_path)
    state = materializer.ConsumerState(alive=True, broker_ok=False)

    snapshot = materializer.health_snapshot(state, db_path)

    assert snapshot["ok"] is False


def test_failed_write_does_not_advance_durable_health_timestamps(tmp_path, monkeypatch):
    db_path = tmp_path / "predictions.db"
    materializer.init_db(db_path).close()
    stop_event = threading.Event()
    consumer = _FakeConsumer(
        [_FakeMessage(_event("e1", 1), 0)],
        stop_event,
    )

    def _fail_and_stop(conn, events):
        stop_event.set()
        raise sqlite3.OperationalError("write failed")

    monkeypatch.setattr(materializer, "PREDICTIONS_DB_PATH", str(db_path))
    monkeypatch.setattr(materializer, "BATCH_MAX_SIZE", 1)
    monkeypatch.setattr(materializer, "BATCH_WINDOW_SEC", 0.01)
    monkeypatch.setattr(materializer, "Consumer", lambda config: consumer)
    monkeypatch.setattr(materializer, "insert_events", _fail_and_stop)
    state = materializer.ConsumerState()

    materializer.consume_loop(state, stop_event)

    assert state.last_event_ts is None
    assert state.last_write_ts is None
    assert state.write_errors == 1
    assert consumer.commits == []


def test_duplicate_and_malformed_only_batch_does_not_claim_a_write(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "predictions.db"
    conn = materializer.init_db(db_path)
    materializer.insert_events(conn, [_event("duplicate", 1)])
    conn.close()
    stop_event = threading.Event()
    consumer = _FakeConsumer(
        [
            _FakeMessage(_event("duplicate", 1), 0),
            _FakeMessage({"score": 0.5}, 1),
        ],
        stop_event,
    )
    monkeypatch.setattr(materializer, "PREDICTIONS_DB_PATH", str(db_path))
    monkeypatch.setattr(materializer, "BATCH_MAX_SIZE", 2)
    monkeypatch.setattr(materializer, "BATCH_WINDOW_SEC", 0.01)
    monkeypatch.setattr(materializer, "Consumer", lambda config: consumer)
    state = materializer.ConsumerState()

    materializer.consume_loop(state, stop_event)

    assert state.last_event_ts is None
    assert state.last_write_ts is None
    assert state.consume_errors == 1
    assert consumer.commits[0][0].offset == 2


def test_non_eof_kafka_error_increments_consume_errors(tmp_path, monkeypatch):
    db_path = tmp_path / "predictions.db"
    materializer.init_db(db_path).close()
    stop_event = threading.Event()
    consumer = _FakeConsumer(
        [
            _FakeErrorMessage(materializer.KafkaError._ALL_BROKERS_DOWN),
            _FakeMessage(_event("e1", 1), 0),
        ],
        stop_event,
    )
    monkeypatch.setattr(materializer, "PREDICTIONS_DB_PATH", str(db_path))
    monkeypatch.setattr(materializer, "BATCH_MAX_SIZE", 1)
    monkeypatch.setattr(materializer, "BATCH_WINDOW_SEC", 0.01)
    monkeypatch.setattr(materializer, "Consumer", lambda config: consumer)
    state = materializer.ConsumerState()

    materializer.consume_loop(state, stop_event)

    assert state.consume_errors == 1


def test_consumer_initialization_failure_signals_ready_and_cleans_up(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "predictions.db"
    conn = materializer.init_db(db_path)
    stop_event = threading.Event()
    ready_event = threading.Event()
    consumer = _FakeConsumer([], stop_event)
    consumer_config = {}

    def _fail_startup(consumer_arg, bootstrap, timeout):
        raise RuntimeError("Kafka unavailable")

    monkeypatch.setattr(materializer, "PREDICTIONS_DB_PATH", str(db_path))
    monkeypatch.setattr(
        materializer,
        "_init_db_with_recovery_status",
        lambda path: (conn, False),
    )

    def _make_consumer(config):
        consumer_config.update(config)
        return consumer

    monkeypatch.setattr(materializer, "Consumer", _make_consumer)
    monkeypatch.setattr(materializer, "_wait_for_kafka", _fail_startup)
    state = materializer.ConsumerState()

    materializer.consume_loop(state, stop_event, ready_event)

    assert ready_event.is_set()
    assert state.ready is False
    assert "Kafka unavailable" in state.startup_error
    assert consumer.closed is True
    assert (
        consumer_config["socket.timeout.ms"] < materializer.THREAD_JOIN_TIMEOUT * 1000
    )
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


def test_lifespan_waits_for_consumer_readiness_and_joins_on_shutdown(monkeypatch):
    started = threading.Event()
    stopped = threading.Event()

    def _controlled_loop(state, stop_event, ready_event=None):
        started.set()
        time.sleep(0.05)
        with state.lock:
            state.ready = True
            state.alive = True
        if ready_event is not None:
            ready_event.set()
        stop_event.wait()
        time.sleep(0.05)
        with state.lock:
            state.ready = False
            state.alive = False
        stopped.set()

    monkeypatch.setattr(materializer, "consume_loop", _controlled_loop)
    # The outcomes consumer thread runs alongside the predictions one; stub
    # it too so this test doesn't reach out to a real (nonexistent) broker.
    monkeypatch.setattr(materializer, "consume_outcomes_loop", _controlled_loop)
    materializer._state.ready = False

    with TestClient(materializer.app):
        assert started.is_set()
        assert materializer._state.ready is True

    assert stopped.is_set()


def test_supervisor_restarts_consumer_after_post_readiness_failure(monkeypatch):
    stop_event = threading.Event()
    ready_event = threading.Event()
    restarted = threading.Event()
    calls = []

    def _crash_then_wait(state, loop_stop_event, loop_ready_event=None):
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            with state.lock:
                state.ready = True
                state.alive = True
                state.broker_ok = True
            loop_ready_event.set()
            return
        restarted.set()
        loop_stop_event.wait(1.0)

    monkeypatch.setattr(materializer, "consume_loop", _crash_then_wait)
    monkeypatch.setattr(materializer, "CONSUMER_RESTART_BACKOFF_SEC", 0)
    state = materializer.ConsumerState()
    thread = threading.Thread(
        target=materializer.supervise_consumer,
        args=(state, stop_event, ready_event),
    )

    thread.start()
    assert ready_event.wait(1.0)
    assert restarted.wait(1.0)
    stop_event.set()
    thread.join(1.0)

    assert calls == [1, 2]
    assert thread.is_alive() is False


def test_compose_healthcheck_requires_materializer_ok_true():
    compose = (PROJECT_ROOT / "docker-compose.yaml").read_text()

    materializer_section = compose.split("  materializer:", 1)[1].split("  ui:", 1)[0]
    assert "json.load" in materializer_section
    assert ".get('ok') is True" in materializer_section


# ---------------------------------------------------------------------------
# outcomes table / ticks.outcomes consumer (Perf Task C2)
# ---------------------------------------------------------------------------


class _FakeOutcomeMessage(_FakeMessage):
    def topic(self):
        return materializer.TOPIC_OUTCOMES


def _outcome_event(feature_id: str, **overrides) -> dict:
    """Build a full OutcomeEvent dict matching the pinned ticks.outcomes
    contract, with sensible defaults for every field."""
    row = {
        "feature_id": feature_id,
        "stream_epoch": 1,
        "product_id": "BTC-USD",
        "feature_ts": "2026-07-16T19:00:00Z",
        "future_vol_60s": 2.9e-5,
        "vol_spike": 0,
        "label_schema": "p85-60s-4.8e-05-v1",
    }
    row.update(overrides)
    return row


def test_outcomes_dedup_insert_twice_yields_one_row(tmp_path):
    conn = materializer.init_db(tmp_path / "test.db")
    event = _outcome_event("BTC-USD:1:1")

    first = materializer.insert_outcomes(conn, [event])
    second = materializer.insert_outcomes(conn, [event])

    assert first == 1
    assert second == 0
    rows = conn.execute("SELECT feature_id FROM outcomes").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "BTC-USD:1:1"


def test_predictions_alter_migration_adds_columns_and_preserves_data(tmp_path):
    db_path = tmp_path / "old_schema.db"
    # Build a DB with the OLD predictions schema (no feature_id/stream_epoch/
    # tau/run_id) to exercise the ALTER TABLE migration path.
    raw = sqlite3.connect(str(db_path))
    raw.execute(
        """
        CREATE TABLE predictions (
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
    )
    raw.execute(
        "INSERT INTO predictions (event_id, source_partition, source_offset, "
        "feature_ts, api_ts, model_variant, model_version, score, vol_60s, "
        "spread_bps, log_return, trade_intensity_60s) VALUES "
        "('old1', 0, 1, '2026-07-16T19:00:00Z', '2026-07-16T19:00:00Z', "
        "'ml', 'v1.0', 0.4, 0.00003, 1.0, 0.0001, 5.0)"
    )
    raw.commit()
    raw.close()

    conn = materializer.init_db(db_path)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(predictions)")}
    assert {"feature_id", "stream_epoch", "tau", "run_id"} <= columns

    row = conn.execute(
        "SELECT event_id, feature_id, tau FROM predictions WHERE event_id = 'old1'"
    ).fetchone()
    assert row == ("old1", None, None)
    conn.close()

    # A second init_db call on the now-migrated DB must not error.
    conn2 = materializer.init_db(db_path)
    columns2 = {row[1] for row in conn2.execute("PRAGMA table_info(predictions)")}
    assert {"feature_id", "stream_epoch", "tau", "run_id"} <= columns2
    conn2.close()


def test_performance_unmatched_outcomes_query_uses_index_not_full_scan(tmp_path):
    """The /predictions/performance handler's "outcomes with no matching
    prediction" query correlates on predictions.feature_id. Without an index
    on that column, SQLite has to full-scan `predictions` once per row of
    `outcomes` in the window — fine at dev-sized row counts, but a real
    production incident once the table holds ~100k rows (e.g. after a
    historical backfill): the query never returns, the request handler
    leaks its thread and DB connection forever, and CPU pegs until the
    container is killed and restarted.
    """
    conn = materializer.init_db(tmp_path / "perf.db")

    plan = conn.execute(
        "EXPLAIN QUERY PLAN "
        "SELECT COUNT(*) FROM outcomes o "
        "WHERE o.feature_ts >= ? "
        "AND NOT EXISTS ("
        "  SELECT 1 FROM predictions p WHERE p.feature_id = o.feature_id"
        ")",
        ("2026-01-01T00:00:00+00:00",),
    ).fetchall()
    conn.close()

    detail = "\n".join(row[-1] for row in plan)
    assert "SCAN p" not in detail, (
        f"unmatched-outcomes query full-scans predictions per outcome row "
        f"(no index on predictions.feature_id) -- query plan:\n{detail}"
    )


def test_outcomes_consumer_skips_malformed_message_and_commits_past_it(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "predictions.db"
    materializer.init_db(db_path).close()
    stop_event = threading.Event()
    consumer = _FakeConsumer(
        [
            _FakeOutcomeMessage({"vol_spike": 0}, 0),  # missing feature_id
            _FakeOutcomeMessage(_outcome_event("BTC-USD:1:1"), 1),
        ],
        stop_event,
    )
    monkeypatch.setattr(materializer, "PREDICTIONS_DB_PATH", str(db_path))
    monkeypatch.setattr(materializer, "BATCH_MAX_SIZE", 2)
    monkeypatch.setattr(materializer, "BATCH_WINDOW_SEC", 0.01)
    monkeypatch.setattr(materializer, "Consumer", lambda config: consumer)
    state = materializer.ConsumerState()

    materializer.consume_outcomes_loop(state, stop_event)

    conn = sqlite3.connect(str(db_path))
    feature_ids = [row[0] for row in conn.execute("SELECT feature_id FROM outcomes")]
    conn.close()

    assert feature_ids == ["BTC-USD:1:1"]
    assert state.consume_errors == 1
    assert consumer.commits[0][0].offset == 2
    assert consumer.closed is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__)), "-q"]))
