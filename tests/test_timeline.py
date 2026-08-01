"""Unit tests for /predictions/timeline: classification, aggregation, and the
FastAPI route contract, against tmp_path SQLite databases.

Run from the repo root with:

    python3 -m pytest tests/test_timeline.py -q
"""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATERIALIZER_DIR = PROJECT_ROOT / "materializer"
sys.path.insert(0, str(MATERIALIZER_DIR))

import materializer  # noqa: E402
import timeline  # noqa: E402


def _event(event_id: str, source_offset: int, **overrides) -> dict:
    """Same PredictionEvent shape as test_materializer.py's helper, with
    market_price included since that's the whole point of this feature."""
    row = {
        "event_id": event_id,
        "source_partition": 0,
        "source_offset": source_offset,
        "feature_ts": "2026-07-16T19:00:00Z",
        "api_ts": "2026-07-16T19:00:00.500000+00:00",
        "score": 0.5,
        "model_variant": "ml",
        "model_version": "v1.0",
        "vol_60s": 0.00005,
        "spread_bps": 1.5,
        "log_return": 0.0001,
        "trade_intensity_60s": 10.0,
        "tau": 0.7,
        "market_price": 65000.0,
    }
    row.update(overrides)
    return row


def _insert_outcome_with_written_at(
    conn, feature_id: str, written_at: str, **overrides
):
    """Bypasses insert_outcomes (which stamps written_at from wall-clock now())
    so tests can pin an exact written_at and exercise the on-time/late
    grading boundary deterministically."""
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
    values = tuple(row.get(f) for f in materializer.OUTCOME_FIELDS) + (written_at,)
    conn.execute(materializer.INSERT_OUTCOME_SQL, values)
    conn.commit()


# ---------------------------------------------------------------------------
# Alignment: feature_ts drives windowing/ordering, api_ts never does
# ---------------------------------------------------------------------------


def test_timeline_windows_and_orders_by_feature_ts_not_api_ts(tmp_path):
    conn = materializer.init_db(tmp_path / "test.db")
    materializer.insert_events(
        conn,
        [
            # api_ts values are deliberately reversed vs feature_ts, and one
            # api_ts sits far outside the requested window -- if the query
            # accidentally filtered/ordered on api_ts, this would fail.
            _event(
                "e1",
                1,
                feature_ts="2026-07-16T19:00:00Z",
                api_ts="2026-07-16T20:00:00+00:00",
            ),
            _event(
                "e2",
                2,
                feature_ts="2026-07-16T19:00:05Z",
                api_ts="2026-07-16T18:00:00+00:00",
            ),
            _event(
                "e3",
                3,
                feature_ts="2026-07-16T19:00:10Z",
                api_ts="2026-07-16T19:00:00+00:00",
            ),
        ],
    )

    window = materializer.timeline_window(
        conn, "2026-07-16T19:00:00+00:00", "2026-07-16T19:00:11+00:00", None
    )

    feature_order = [row["feature_ts"] for row in window["joined_rows"]]
    assert feature_order == sorted(feature_order)
    assert len(window["joined_rows"]) == 3


# ---------------------------------------------------------------------------
# Classification: all six states
# ---------------------------------------------------------------------------


def test_timeline_classifies_all_six_states(tmp_path):
    conn = materializer.init_db(tmp_path / "test.db")
    anchor_feature_ts = "2026-07-16T19:10:00Z"  # newest feature_ts -> maturity anchor

    materializer.insert_events(
        conn,
        [
            _event(
                "correct_call",
                1,
                feature_id="correct_call",
                feature_ts="2026-07-16T19:00:00Z",
                score=0.9,
                tau=0.7,
            ),
            _event(
                "false_alarm",
                2,
                feature_id="false_alarm",
                feature_ts="2026-07-16T19:01:00Z",
                score=0.9,
                tau=0.7,
            ),
            _event(
                "missed_spike",
                3,
                feature_id="missed_spike",
                feature_ts="2026-07-16T19:02:00Z",
                score=0.2,
                tau=0.7,
            ),
            _event(
                "correct_quiet",
                4,
                feature_id="correct_quiet",
                feature_ts="2026-07-16T19:03:00Z",
                score=0.2,
                tau=0.7,
            ),
            _event(
                "late_score",
                5,
                feature_id="late_score",
                feature_ts="2026-07-16T19:04:00Z",
                api_ts="2026-07-16T19:04:05+00:00",
                score=0.9,
                tau=0.7,
            ),
            _event(
                "no_outcome_old",
                6,
                feature_ts="2026-07-16T19:05:00Z",  # 5 min before anchor -- matured, no outcome
            ),
            _event(
                "no_outcome_recent",
                7,
                feature_ts="2026-07-16T19:09:30Z",  # 30s before anchor -- still maturing
            ),
            _event("anchor_row", 8, feature_ts=anchor_feature_ts),
        ],
    )

    _insert_outcome_with_written_at(
        conn,
        "correct_call",
        "2026-07-16T19:00:05+00:00",
        vol_spike=1,
        feature_ts="2026-07-16T19:00:00Z",
    )
    _insert_outcome_with_written_at(
        conn,
        "false_alarm",
        "2026-07-16T19:01:05+00:00",
        vol_spike=0,
        feature_ts="2026-07-16T19:01:00Z",
    )
    _insert_outcome_with_written_at(
        conn,
        "missed_spike",
        "2026-07-16T19:02:05+00:00",
        vol_spike=1,
        feature_ts="2026-07-16T19:02:00Z",
    )
    _insert_outcome_with_written_at(
        conn,
        "correct_quiet",
        "2026-07-16T19:03:05+00:00",
        vol_spike=0,
        feature_ts="2026-07-16T19:03:00Z",
    )
    # written BEFORE the score's own api_ts -- the model already "knew".
    _insert_outcome_with_written_at(
        conn,
        "late_score",
        "2026-07-16T19:04:00+00:00",
        vol_spike=1,
        feature_ts="2026-07-16T19:04:00Z",
    )

    window = materializer.timeline_window(
        conn, "2026-07-16T18:59:00+00:00", "2026-07-16T19:11:00+00:00", None
    )
    anchor_dt = materializer._parse_iso(window["available_to"])
    points, aggregated, _ = timeline.build_timeline(
        window["joined_rows"],
        materializer._parse_iso("2026-07-16T18:59:00+00:00"),
        materializer._parse_iso("2026-07-16T19:11:00+00:00"),
        anchor_dt,
    )

    assert aggregated is False
    # feature_ts is stored normalized (microsecond precision); build the
    # lookup keys the same way rather than hardcoding the normalized spelling.
    classes = {p["feature_ts"]: p["class"] for p in points}

    def _norm(ts):
        return materializer._normalize_ts(ts)

    assert classes[_norm("2026-07-16T19:00:00Z")] == "correct_call"
    assert classes[_norm("2026-07-16T19:01:00Z")] == "false_alarm"
    assert classes[_norm("2026-07-16T19:02:00Z")] == "missed_spike"
    assert classes[_norm("2026-07-16T19:03:00Z")] == "correct_quiet"
    assert classes[_norm("2026-07-16T19:04:00Z")] == "unavailable"  # late score
    assert (
        classes[_norm("2026-07-16T19:05:00Z")] == "unavailable"
    )  # outcome-pipeline gap
    assert classes[_norm("2026-07-16T19:09:30Z")] == "pending"  # within 60s of anchor
    assert classes[_norm(anchor_feature_ts)] == "pending"


def test_classify_row_treats_exact_horizon_boundary_as_unavailable_not_pending():
    # btcspiker_ml/features.py gates its skip on `age < horizon_seconds`, i.e.
    # it emits a label once age >= horizon_seconds. So a still-missing outcome
    # exactly at the 60s boundary is a pipeline gap, not "still maturing".
    anchor = materializer._parse_iso("2026-07-16T19:01:00+00:00")
    row_at_boundary = {"feature_ts": "2026-07-16T19:00:00+00:00", "written_at": None}
    row_just_under = {
        "feature_ts": "2026-07-16T19:00:00.500000+00:00",
        "written_at": None,
    }

    assert timeline.classify_row(row_at_boundary, anchor) == "unavailable"
    assert timeline.classify_row(row_just_under, anchor) == "pending"


# ---------------------------------------------------------------------------
# Paired aggregation: a bucket never lets one row's prediction pair with
# another row's outcome
# ---------------------------------------------------------------------------


def test_timeline_bucket_keeps_predictions_paired_with_their_own_outcome(tmp_path):
    conn = materializer.init_db(tmp_path / "test.db")

    # Same bucket (both within one 5-minute window): one missed spike, one
    # false alarm. If aggregation computed "any predicted" and "any spike"
    # independently, this could misreport as a correct_call.
    materializer.insert_events(
        conn,
        [
            _event(
                "missed",
                1,
                feature_id="missed",
                feature_ts="2026-07-16T19:00:00Z",
                score=0.2,
                tau=0.7,
            ),
            _event(
                "false_alarm",
                2,
                feature_id="false_alarm",
                feature_ts="2026-07-16T19:02:00Z",
                score=0.9,
                tau=0.7,
            ),
        ],
    )
    _insert_outcome_with_written_at(
        conn,
        "missed",
        "2026-07-16T19:00:05+00:00",
        vol_spike=1,
        feature_ts="2026-07-16T19:00:00Z",
    )
    _insert_outcome_with_written_at(
        conn,
        "false_alarm",
        "2026-07-16T19:02:05+00:00",
        vol_spike=0,
        feature_ts="2026-07-16T19:02:00Z",
    )

    from_dt = materializer._parse_iso("2026-07-16T19:00:00+00:00")
    to_dt = materializer._parse_iso("2026-07-16T19:05:00+00:00")
    window = materializer.timeline_window(
        conn, "2026-07-16T19:00:00+00:00", "2026-07-16T19:05:00+00:00", None
    )
    anchor_dt = materializer._parse_iso(window["available_to"])

    points, aggregated, _ = timeline.build_timeline(
        window["joined_rows"], from_dt, to_dt, anchor_dt, resolution=300
    )

    assert aggregated is True
    assert len(points) == 1
    classes = points[0]["classes"]
    assert classes.get("missed_spike") == 1
    assert classes.get("false_alarm") == 1
    assert classes.get("correct_call", 0) == 0


def test_timeline_bucket_preserves_max_score(tmp_path):
    conn = materializer.init_db(tmp_path / "test.db")
    materializer.insert_events(
        conn,
        [
            _event("calm1", 1, feature_ts="2026-07-16T19:00:00Z", score=0.1),
            _event("spike", 2, feature_ts="2026-07-16T19:01:00Z", score=0.95, tau=0.7),
            _event("calm2", 3, feature_ts="2026-07-16T19:02:00Z", score=0.15),
        ],
    )

    from_dt = materializer._parse_iso("2026-07-16T19:00:00+00:00")
    to_dt = materializer._parse_iso("2026-07-16T19:05:00+00:00")
    window = materializer.timeline_window(
        conn, "2026-07-16T19:00:00+00:00", "2026-07-16T19:05:00+00:00", None
    )
    anchor_dt = materializer._parse_iso(window["available_to"])

    points, aggregated, _ = timeline.build_timeline(
        window["joined_rows"], from_dt, to_dt, anchor_dt, resolution=300
    )

    assert aggregated is True
    assert len(points) == 1
    assert points[0]["score"] == 0.95
    assert points[0]["tau"] == 0.7


# ---------------------------------------------------------------------------
# Coverage honesty
# ---------------------------------------------------------------------------


def test_timeline_reports_incomplete_when_range_exceeds_retained_history(tmp_path):
    conn = materializer.init_db(tmp_path / "test.db")
    materializer.insert_events(
        conn,
        [_event("e1", 1, feature_ts="2026-07-16T19:00:00Z")],
    )
    monkeypatch_db = tmp_path / "test.db"
    import sqlite3 as _sqlite3

    _conn2 = _sqlite3.connect(str(monkeypatch_db))
    window = materializer.timeline_window(
        _conn2, "2026-07-16T00:00:00+00:00", "2026-07-16T23:59:00+00:00", None
    )
    assert window["available_from"] == "2026-07-16T19:00:00.000000+00:00"
    assert window["available_to"] == "2026-07-16T19:00:00.000000+00:00"


# ---------------------------------------------------------------------------
# FastAPI route: validation + coverage in the response
# ---------------------------------------------------------------------------


def test_fastapi_timeline_rejects_bad_params(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    materializer.init_db(db_path).close()
    monkeypatch.setattr(materializer, "PREDICTIONS_DB_PATH", str(db_path))
    materializer._state.ready = True
    client = TestClient(materializer.app)

    # to <= from
    resp = client.get(
        "/predictions/timeline",
        params={"from": "2026-07-16T19:00:00Z", "to": "2026-07-16T18:00:00Z"},
    )
    assert resp.status_code == 422

    # range > 24h
    resp = client.get(
        "/predictions/timeline",
        params={"from": "2026-07-15T00:00:00Z", "to": "2026-07-16T19:00:00Z"},
    )
    assert resp.status_code == 422

    # bad resolution
    resp = client.get(
        "/predictions/timeline",
        params={
            "from": "2026-07-16T18:00:00Z",
            "to": "2026-07-16T19:00:00Z",
            "resolution": 0,
        },
    )
    assert resp.status_code == 422

    # malformed timestamp
    resp = client.get(
        "/predictions/timeline",
        params={"from": "not-a-timestamp", "to": "2026-07-16T19:00:00Z"},
    )
    assert resp.status_code == 422


def test_fastapi_timeline_rejects_explicit_resolution_that_implies_too_many_points(
    tmp_path, monkeypatch
):
    # An explicit resolution bypasses build_timeline's own auto-bucketing (it
    # only kicks in when resolution is omitted), so without this guard a
    # client could force an unbounded response, e.g. resolution=1 over a full
    # 24h range implies 86,400 points against a 1,000-point cap.
    db_path = tmp_path / "test.db"
    materializer.init_db(db_path).close()
    monkeypatch.setattr(materializer, "PREDICTIONS_DB_PATH", str(db_path))
    materializer._state.ready = True
    client = TestClient(materializer.app)

    resp = client.get(
        "/predictions/timeline",
        params={
            "from": "2026-07-15T19:00:00Z",
            "to": "2026-07-16T19:00:00Z",  # 24h, within the range cap
            "resolution": 1,
        },
    )
    assert resp.status_code == 422

    # A resolution that keeps bucket count within the cap is still accepted.
    resp = client.get(
        "/predictions/timeline",
        params={
            "from": "2026-07-15T19:00:00Z",
            "to": "2026-07-16T19:00:00Z",
            "resolution": 300,  # 24h / 300s = 288 buckets, within the cap
        },
    )
    assert resp.status_code == 200


def test_fastapi_timeline_returns_points_and_coverage(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    conn = materializer.init_db(db_path)
    materializer.insert_events(
        conn,
        [_event("e1", 1, feature_ts="2026-07-16T19:00:00Z", market_price=65432.1)],
    )
    conn.close()
    monkeypatch.setattr(materializer, "PREDICTIONS_DB_PATH", str(db_path))
    materializer._state.ready = True
    client = TestClient(materializer.app)

    resp = client.get(
        "/predictions/timeline",
        params={"from": "2026-07-16T19:00:00Z", "to": "2026-07-16T19:01:00Z"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["aggregated"] is False
    assert body["complete"] is True
    assert len(body["points"]) == 1
    assert body["points"][0]["market_price"] == 65432.1
    assert body["points"][0]["class"] == "pending"


def test_fastapi_timeline_returns_503_before_readiness(tmp_path, monkeypatch):
    db_path = tmp_path / "not-ready.db"
    monkeypatch.setattr(materializer, "PREDICTIONS_DB_PATH", str(db_path))
    materializer._state.ready = False
    client = TestClient(materializer.app, raise_server_exceptions=False)

    resp = client.get(
        "/predictions/timeline",
        params={"from": "2026-07-16T18:00:00Z", "to": "2026-07-16T19:00:00Z"},
    )

    assert resp.status_code == 503
    assert db_path.exists() is False


# ---------------------------------------------------------------------------
# stream_id segment scoping: a looping replay's earlier pass must not bucket
# into the same timeline points as the current pass
# ---------------------------------------------------------------------------


def test_fastapi_timeline_excludes_prior_segment(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    conn = materializer.init_db(db_path)

    # Both passes cover the identical feature_ts range -- exactly what a
    # `--loop` replay does. boot2 has the newer api_ts, so it's the active
    # segment; boot1's rows must not double the per-bucket totals.
    materializer.insert_events(
        conn,
        [
            _event(
                f"boot1-{i}", i,
                feature_ts=f"2026-07-16T19:00:{i:02d}Z",
                api_ts="2026-07-16T18:00:00+00:00",
                feature_id=f"BTC-USD:boot1:0:{i}",
                stream_id="boot1:0",
            )
            for i in range(3)
        ]
        + [
            _event(
                f"boot2-{i}", 100 + i,
                feature_ts=f"2026-07-16T19:00:{i:02d}Z",
                api_ts="2026-07-16T20:00:00+00:00",
                feature_id=f"BTC-USD:boot2:0:{i}",
                stream_id="boot2:0",
            )
            for i in range(3)
        ],
    )
    conn.close()

    monkeypatch.setattr(materializer, "PREDICTIONS_DB_PATH", str(db_path))
    materializer._state.ready = True
    client = TestClient(materializer.app)

    resp = client.get(
        "/predictions/timeline",
        params={
            "from": "2026-07-16T19:00:00Z",
            "to": "2026-07-16T19:01:00Z",
            "resolution": 60,  # force one aggregated bucket covering all 3+3 rows
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["aggregated"] is True
    assert len(body["points"]) == 1
    total_in_bucket = sum(body["points"][0]["classes"].values())
    assert total_in_bucket == 3  # boot2 only, not 6
