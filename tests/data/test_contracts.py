from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from btcspiker_data.contracts import (
    MODEL_TICK_COLUMNS,
    RAW_BOOK_COLUMNS,
    RAW_TRADE_COLUMNS,
    BookDelta,
    BookState,
    QualityIncident,
    TradeEvent,
    validate_utc_window,
)


UTC = timezone.utc
NOW = datetime(2026, 4, 24, tzinfo=UTC)


def test_book_state_rejects_crossed_market():
    with pytest.raises(ValueError, match="crossed book"):
        BookState(
            product_id="BTC-USD",
            observed_through=NOW,
            sequence_start=10,
            sequence_end=11,
            best_bid=Decimal("101"),
            bid_size=Decimal("1"),
            best_ask=Decimal("100"),
            ask_size=Decimal("1"),
        )


def test_book_delta_rejects_regressed_sequence():
    with pytest.raises(ValueError, match="sequence range regressed"):
        BookDelta(
            product_id="BTC-USD",
            changed_second=NOW,
            sequence_start=11,
            sequence_end=10,
            best_bid=Decimal("100"),
            best_ask=Decimal("101"),
            changes=(),
        )


def test_book_delta_rejects_unknown_side_and_negative_quantity():
    with pytest.raises(ValueError, match="invalid L2 change"):
        BookDelta(
            product_id="BTC-USD",
            changed_second=NOW,
            sequence_start=10,
            sequence_end=11,
            best_bid=Decimal("100"),
            best_ask=Decimal("101"),
            changes=(("ask", Decimal("101"), Decimal("-1")),),
        )


def test_trade_event_preserves_reported_side():
    event = TradeEvent(
        product_id="BTC-USD",
        trade_id="42",
        event_time=NOW,
        price=Decimal("90000"),
        size=Decimal("0.01"),
        reported_side="SELL",
        source="coinbase_public_trades",
    )
    assert event.reported_side == "SELL"
    assert event.side_semantics == "coinbase_reported_unspecified"


def test_trade_event_rejects_nonpositive_values():
    with pytest.raises(ValueError, match="positive"):
        TradeEvent("BTC-USD", "42", NOW, Decimal("0"), Decimal("1"), "SELL", "source")


def test_window_requires_utc_and_positive_duration():
    with pytest.raises(ValueError, match="UTC"):
        validate_utc_window(datetime(2026, 4, 24), NOW + timedelta(days=1))
    with pytest.raises(ValueError, match="before"):
        validate_utc_window(NOW, NOW)


def test_quality_incident_requires_utc_ordered_window():
    with pytest.raises(ValueError, match="before"):
        QualityIncident("gap", NOW, NOW - timedelta(seconds=1), "error", "regression")


def test_ordered_schema_contracts_are_unique_and_feature_complete():
    for columns in (RAW_BOOK_COLUMNS, RAW_TRADE_COLUMNS, MODEL_TICK_COLUMNS):
        assert len(columns) == len(set(columns))
    assert "segment_id" in RAW_BOOK_COLUMNS
    assert {"bid_size", "ask_size", "book_observed_through", "segment_id"}.issubset(
        MODEL_TICK_COLUMNS
    )
