from datetime import timedelta
from decimal import Decimal

import pandas as pd
import pytest

from btcspiker_data.contracts import BookState, MODEL_TICK_COLUMNS
from btcspiker_data.materialize import (
    join_trades_to_books,
    materialize_segmented_features,
)


def _book(second, *, segment_id=0):
    return BookState(
        product_id="BTC-USD",
        observed_through=second,
        sequence_start=100 + segment_id,
        sequence_end=100 + segment_id,
        best_bid=Decimal("90000.00"),
        bid_size=Decimal("2.0"),
        best_ask=Decimal("90000.10"),
        ask_size=Decimal("1.0"),
        segment_id=segment_id,
    )


def _trade(second, trade_id, *, segment_id=0):
    return {
        "product_id": "BTC-USD",
        "trade_id": str(trade_id),
        "event_time": second + timedelta(microseconds=500_000),
        "price": Decimal("90000.05"),
        "size": Decimal("0.01"),
        "reported_side": "BUY",
        "segment_id": segment_id,
    }


def test_trade_uses_last_fully_observed_book_second():
    from tests.data.conftest import DAY_START

    joined = join_trades_to_books(
        [_trade(DAY_START + timedelta(seconds=2), "100")],
        [_book(DAY_START + timedelta(seconds=1)), _book(DAY_START + timedelta(seconds=2))],
    )

    row = joined.loc[joined["trade_id"] == "100"].iloc[0]
    assert row["book_observed_through"] < row["timestamp"].floor("s")
    assert row["book_observed_through"] == DAY_START + timedelta(seconds=1)


def test_join_excludes_unsafe_trades_and_rejects_duplicate_ids():
    from tests.data.conftest import DAY_START

    assert join_trades_to_books(
        [_trade(DAY_START, "early")], [_book(DAY_START)]
    ).empty
    with pytest.raises(ValueError, match="duplicate trade_id"):
        join_trades_to_books(
            [_trade(DAY_START + timedelta(seconds=2), "100"), _trade(DAY_START + timedelta(seconds=3), "100")],
            [_book(DAY_START + timedelta(seconds=1)), _book(DAY_START + timedelta(seconds=2))],
        )


def test_join_never_crosses_segments_and_returns_tick_contract():
    from tests.data.conftest import DAY_START

    joined = join_trades_to_books(
        [_trade(DAY_START + timedelta(seconds=2), "100", segment_id=1)],
        [_book(DAY_START + timedelta(seconds=1), segment_id=0)],
    )

    assert joined.empty
    assert tuple(joined.columns) == MODEL_TICK_COLUMNS


def test_materializes_each_feature_set_without_crossing_segment_boundaries():
    from tests.data.conftest import DAY_START

    books = []
    trades = []
    for segment_id, offset in ((0, 0), (1, 200)):
        for second in range(62):
            observed = DAY_START + timedelta(seconds=offset + second)
            books.append(_book(observed, segment_id=segment_id))
            if second:
                trades.append(_trade(observed, f"{segment_id}-{second}", segment_id=segment_id))

    outputs = materialize_segmented_features(trades, books)

    assert set(outputs) == {"core_v1", "multi_window_v1", "microstructure_v1"}
    for frame in outputs.values():
        assert not frame.empty
        assert set(frame["segment_id"]) == {0, 1}
        assert frame.groupby("segment_id").size().to_dict() == {0: 1, 1: 1}
        assert frame["timestamp"].is_monotonic_increasing
