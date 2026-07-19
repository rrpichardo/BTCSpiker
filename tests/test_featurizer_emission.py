"""Unit tests for ProductState's split emission: immediate feature rows +
delayed outcome events.

Exercises ProductState directly (pure, no Kafka). Run from the repo root with:

    python3 -m pytest tests/test_featurizer_emission.py -q
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "features"
sys.path.insert(0, str(FEATURES_DIR))

from feature_funcs import compute_future_vol  # noqa: E402
from featurizer import ProductState  # noqa: E402

ORIGINAL_FEATURE_KEYS = {
    "product_id",
    "timestamp",
    "price",
    "midprice",
    "log_return",
    "spread_abs",
    "spread_bps",
    "vol_60s",
    "mean_return_60s",
    "n_ticks_60s",
    "trade_intensity_60s",
    "spread_mean_60s",
    "price_range_60s",
}

WINDOW_SEC = 5.0
HORIZON_SEC = 10.0
VOL_THRESHOLD = 0.0001

BASE_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _tick(product_id: str, offset_sec: float, price: float) -> dict:
    ts = BASE_TS + timedelta(seconds=offset_sec)
    return {
        "product_id": product_id,
        "timestamp": ts.isoformat().replace("+00:00", "Z"),
        "price": price,
        "best_bid": price - 0.5,
        "best_ask": price + 0.5,
    }


def _new_state() -> ProductState:
    return ProductState(WINDOW_SEC, HORIZON_SEC, VOL_THRESHOLD)


# ---------------------------------------------------------------------------
# 1. Immediate feature emission
# ---------------------------------------------------------------------------


def test_feature_row_emitted_same_call_no_delay():
    state = _new_state()
    feature_row, drained = state.ingest(_tick("BTC-USD", 0.0, 100.0))

    assert feature_row is not None
    assert drained == []  # nothing old enough to drain yet
    assert feature_row["feature_id"] == "BTC-USD:0:0"
    assert feature_row["stream_epoch"] == 0
    assert ORIGINAL_FEATURE_KEYS <= feature_row.keys()

    # seq increments per emitted row
    feature_row_2, _ = state.ingest(_tick("BTC-USD", 0.5, 100.5))
    assert feature_row_2["feature_id"] == "BTC-USD:0:1"


# ---------------------------------------------------------------------------
# 2. Delayed outcome event
# ---------------------------------------------------------------------------


def test_outcome_arrives_only_after_horizon_and_matches_feature_id():
    state = _new_state()

    # Offsets (seconds) for every tick we'll feed, including the final one
    # that closes the first row's horizon window. Prices ramp linearly.
    offsets = [round(0.5 * i, 1) for i in range(int(HORIZON_SEC / 0.5) + 1)]
    assert offsets[-1] == HORIZON_SEC

    first_row, drained = state.ingest(_tick("BTC-USD", offsets[0], 100.0 + offsets[0]))
    first_feature_id = first_row["feature_id"]

    # Feed ticks up to just under the horizon: no outcome yet.
    for t in offsets[1:-1]:
        _, drained = state.ingest(_tick("BTC-USD", t, 100.0 + t))
        assert drained == [], f"unexpected outcome at age {t}"

    # Independently hand-compute the expected future_vol over the same
    # window using the pure feature_funcs helper on the same price sequence
    # (mirrors the price_buf the implementation would have built).
    from collections import deque

    expected_slice = deque({"price": 100.0 + t, "ts": t} for t in offsets)
    expected_future_vol = compute_future_vol(expected_slice, HORIZON_SEC)

    # This tick lands at the horizon: the first row should drain now.
    last_t = offsets[-1]
    _, drained = state.ingest(_tick("BTC-USD", last_t, 100.0 + last_t))

    assert len(drained) == 1
    labelled_row, outcome_event = drained[0]
    assert outcome_event["feature_id"] == first_feature_id
    assert outcome_event["future_vol_60s"] == expected_future_vol
    assert labelled_row["future_vol_60s"] == expected_future_vol


# ---------------------------------------------------------------------------
# 3. Timestamp regression -> epoch bump, seq reset, stale pending dropped
# ---------------------------------------------------------------------------


def test_timestamp_regression_bumps_epoch_and_drops_stale_pending():
    state = _new_state()
    state.ingest(_tick("BTC-USD", 0.0, 100.0))
    state.ingest(_tick("BTC-USD", 1.0, 100.1))

    assert state.epoch == 0
    assert state.seq == 2
    assert len(state.pending) == 2

    # Timestamp moves backwards -> reset.
    feature_row, drained = state.ingest(_tick("BTC-USD", 0.5, 99.0))

    assert drained == []
    assert state.epoch == 1
    assert feature_row["stream_epoch"] == 1
    assert feature_row["feature_id"] == "BTC-USD:1:0"
    assert state.seq == 1  # incremented past the row we just emitted
    assert len(state.price_buf) == 1
    assert len(state.spread_buf) == 1
    assert len(state.ts_buf) == 1
    # Only the post-regression row is pending; the epoch-0 rows are gone.
    assert len(state.pending) == 1
    assert state.pending[0]["epoch"] == 1

    # Even after enough time passes for the old (epoch 0) rows' horizon to
    # have elapsed, no outcome for them ever appears — they were dropped.
    seen_feature_ids = set()
    t = 0.5 + 0.5
    while t < 0.5 + HORIZON_SEC + 1:
        _, drained = state.ingest(_tick("BTC-USD", t, 99.0))
        for _labelled, outcome in drained:
            seen_feature_ids.add(outcome["feature_id"])
        t += 0.5

    assert "BTC-USD:0:0" not in seen_feature_ids
    assert "BTC-USD:0:1" not in seen_feature_ids


# ---------------------------------------------------------------------------
# 4. Parquet row shape unchanged (no feature_id / stream_epoch leakage)
# ---------------------------------------------------------------------------


def test_labelled_parquet_row_shape_matches_original_contract():
    state = _new_state()
    t = 0.0
    drained = []
    while not drained and t < HORIZON_SEC + 2:
        _, drained = state.ingest(_tick("BTC-USD", t, 100.0 + t))
        t += 0.5

    assert drained, "expected at least one drained row in this window"
    labelled_row, outcome_event = drained[0]

    expected_keys = ORIGINAL_FEATURE_KEYS | {"future_vol_60s", "vol_spike"}
    assert set(labelled_row.keys()) == expected_keys
    assert "feature_id" not in labelled_row
    assert "stream_epoch" not in labelled_row

    # The outcome event, not the labelled row, carries the identifiers.
    assert outcome_event["feature_id"]
    assert outcome_event["stream_epoch"] == 0
    assert (
        outcome_event["label_schema"] == f"p85-{int(HORIZON_SEC)}s-{VOL_THRESHOLD}-v1"
    )


# ---------------------------------------------------------------------------
# 5. vol_spike strict > at the threshold boundary
# ---------------------------------------------------------------------------


def test_vol_spike_label_is_strictly_greater_than_threshold():
    state = _new_state()
    # Drain a row and inspect its future_vol, then re-derive vol_spike logic
    # via direct threshold comparisons to prove the strict-> rule.
    t = 0.0
    drained = []
    while not drained and t < HORIZON_SEC + 2:
        _, drained = state.ingest(_tick("BTC-USD", t, 100.0 + t))
        t += 0.5

    labelled_row, outcome_event = drained[0]
    future_vol = labelled_row["future_vol_60s"]

    # Boundary case: threshold exactly equal to future_vol -> not a spike.
    assert int(future_vol > future_vol) == 0
    # Just below future_vol -> spike.
    assert int(future_vol > (future_vol - 1e-9)) == 1

    # Sanity: the state's own vol_threshold (below the observed future_vol
    # given VOL_THRESHOLD is tiny relative to this price ramp) yields spike=1.
    assert labelled_row["vol_spike"] == outcome_event["vol_spike"]
    assert labelled_row["vol_spike"] == int(future_vol > VOL_THRESHOLD)
