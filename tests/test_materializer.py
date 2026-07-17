"""Unit tests for the materializer read-model service.

Exercises the pure functions directly (init_db, insert_events, maybe_prune,
recent, health_snapshot, parse_event, validate_limit) against a tmp_path
SQLite file. No Kafka broker and no HTTP client involved — importing
materializer.py must have zero side effects, so these tests never start the
consumer thread or the FastAPI app.

Run from the repo root with:

    python3 -m pytest tests/test_materializer.py -q
"""

import json
import logging
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATERIALIZER_DIR = PROJECT_ROOT / "materializer"
sys.path.insert(0, str(MATERIALIZER_DIR))

import materializer  # noqa: E402


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


# ---------------------------------------------------------------------------
# recent()
# ---------------------------------------------------------------------------


def test_recent_returns_newest_first_by_source_offset(tmp_path):
    conn = materializer.init_db(tmp_path / "test.db")
    events = [_event(f"e{i}", i) for i in [3, 1, 5, 2, 4]]
    materializer.insert_events(conn, events)

    rows = materializer.recent(conn, 10)

    assert [r["source_offset"] for r in rows] == [5, 4, 3, 2, 1]


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__)), "-q"]))
