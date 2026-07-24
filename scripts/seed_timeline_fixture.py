"""Seed a standalone predictions.db with a fixed, known dataset for
deterministic /predictions/timeline verification.

Writes ONLY to the path given on the command line -- never the live named
volume -- so this is safe to run against a bind-mounted temp file under an
isolated Compose project.

Usage:
    python3 scripts/seed_timeline_fixture.py /path/to/predictions.db

Fixture covers, anchored so the newest row is "now" for maturity purposes:
    - a confirmed spike run (correct_call)
    - a false alarm (predicted, no spike)
    - a missed spike (not predicted, spike happened)
    - a correct-quiet row (not predicted, no spike)
    - a late-scored row (api_ts >= written_at -- must render unavailable,
      never counted as a correct call)
    - an old outcome-less row (past the 60s maturity horizon -- unavailable,
      an outcome-pipeline gap, not "pending")
    - a pending tail (within 60s of the anchor -- outcome hasn't matured yet)
    - rows with api_ts deliberately lagging feature_ts by minutes, to prove
      the timeline aligns on feature_ts, never api_ts
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "materializer"))

import materializer  # noqa: E402

ANCHOR = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _event(event_id: str, offset: int, feature_ts: datetime, **overrides) -> dict:
    row = {
        "event_id": event_id,
        "source_partition": 0,
        "source_offset": offset,
        "feature_ts": _iso(feature_ts),
        "api_ts": _iso(feature_ts + timedelta(milliseconds=500)),
        "score": 0.2,
        "model_variant": "ml",
        "model_version": "v1.0",
        "vol_60s": 0.00005,
        "spread_bps": 1.5,
        "log_return": 0.0001,
        "trade_intensity_60s": 10.0,
        "tau": 0.7,
        "market_price": 65000.0,
        "feature_id": event_id,
        "stream_epoch": 0,
    }
    row.update(overrides)
    return row


def _insert_outcome(
    conn, feature_id: str, feature_ts: datetime, written_at: datetime, vol_spike: int
):
    row = {
        "feature_id": feature_id,
        "stream_epoch": 0,
        "product_id": "BTC-USD",
        "feature_ts": _iso(feature_ts),
        "future_vol_60s": 6.0e-5 if vol_spike else 2.0e-5,
        "vol_spike": vol_spike,
        "label_schema": "p85-60s-4.8e-05-v1",
    }
    values = tuple(row.get(f) for f in materializer.OUTCOME_FIELDS) + (
        _iso(written_at),
    )
    conn.execute(materializer.INSERT_OUTCOME_SQL, values)


def build_fixture(db_path: Path) -> dict:
    if db_path.exists():
        raise FileExistsError(f"{db_path} already exists -- refusing to overwrite")

    conn = materializer.init_db(db_path)
    events = []
    outcomes = []  # (feature_id, feature_ts, written_at, vol_spike)

    # Price walk so the top chart has visible movement.
    price = 64000.0

    def next_price(delta):
        nonlocal price
        price += delta
        return price

    t = ANCHOR - timedelta(minutes=10)

    # 1) Correct-call run: three consecutive predicted+confirmed spike points.
    for i in range(3):
        ts = t + timedelta(seconds=i * 5)
        fid = f"correct_call_{i}"
        events.append(
            _event(
                fid, len(events), ts, score=0.92, tau=0.7, market_price=next_price(15)
            )
        )
        outcomes.append((fid, ts, ts + timedelta(seconds=5), 1))
    t += timedelta(seconds=30)

    # 2) False alarm: predicted, but no real spike.
    fid = "false_alarm"
    events.append(
        _event(fid, len(events), t, score=0.88, tau=0.7, market_price=next_price(2))
    )
    outcomes.append((fid, t, t + timedelta(seconds=5), 0))
    t += timedelta(seconds=30)

    # 3) Missed spike: not predicted, but a real spike happened.
    fid = "missed_spike"
    events.append(
        _event(fid, len(events), t, score=0.3, tau=0.7, market_price=next_price(20))
    )
    outcomes.append((fid, t, t + timedelta(seconds=5), 1))
    t += timedelta(seconds=30)

    # 4) Correct quiet: not predicted, no spike -- the calm baseline case.
    fid = "correct_quiet"
    events.append(
        _event(fid, len(events), t, score=0.1, tau=0.7, market_price=next_price(-5))
    )
    outcomes.append((fid, t, t + timedelta(seconds=5), 0))
    t += timedelta(seconds=30)

    # 5) Late-scored: outcome written BEFORE api_ts -- the model already
    # "knew". Must render unavailable, never a correct call, even though
    # score >= tau and vol_spike == 1 would otherwise read as correct_call.
    fid = "late_score"
    late_event = _event(
        fid,
        len(events),
        t,
        score=0.95,
        tau=0.7,
        api_ts=_iso(t + timedelta(seconds=10)),
        market_price=next_price(3),
    )
    events.append(late_event)
    outcomes.append((fid, t, t + timedelta(seconds=2), 1))  # written before api_ts
    t += timedelta(seconds=30)

    # 6) Old outcome-less row: past the 60s maturity horizon relative to the
    # anchor -- an outcome-pipeline gap, must render unavailable, not pending.
    fid = "outcome_gap"
    events.append(
        _event(fid, len(events), t, score=0.4, tau=0.7, market_price=next_price(1))
    )
    t += timedelta(seconds=30)

    # 7) api_ts deliberately lags feature_ts by minutes, on an otherwise
    # ordinary graded row -- proves the timeline aligns on feature_ts.
    fid = "lagging_api_ts"
    events.append(
        _event(
            fid,
            len(events),
            t,
            score=0.2,
            tau=0.7,
            api_ts=_iso(t + timedelta(minutes=4)),
            market_price=next_price(-2),
        )
    )
    outcomes.append((fid, t, t + timedelta(minutes=4, seconds=5), 0))
    t += timedelta(seconds=30)

    # 8) Pending tail: within 60s of the anchor -- outcome hasn't matured yet.
    for i in range(3):
        ts = ANCHOR - timedelta(seconds=45 - i * 15)
        fid = f"pending_{i}"
        events.append(
            _event(fid, len(events), ts, score=0.5, tau=0.7, market_price=next_price(1))
        )
    # Row exactly at the anchor.
    events.append(
        _event(
            "anchor_row",
            len(events),
            ANCHOR,
            score=0.55,
            tau=0.7,
            market_price=next_price(1),
        )
    )

    materializer.insert_events(conn, events)
    for feature_id, feature_ts, written_at, vol_spike in outcomes:
        _insert_outcome(conn, feature_id, feature_ts, written_at, vol_spike)
    conn.commit()
    conn.close()

    window_from = ANCHOR - timedelta(minutes=15)
    window_to = ANCHOR + timedelta(seconds=1)
    return {
        "db_path": str(db_path),
        "anchor": _iso(ANCHOR),
        "recommended_from": _iso(window_from),
        "recommended_to": _iso(window_to),
        "n_events": len(events),
        "n_outcomes": len(outcomes),
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    summary = build_fixture(Path(sys.argv[1]))
    import json

    print(json.dumps(summary, indent=2))
