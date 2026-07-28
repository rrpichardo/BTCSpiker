from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest


UTC = timezone.utc
DAY_START = datetime(2026, 4, 24, tzinfo=UTC)


@pytest.fixture
def replay_anchor() -> dict[str, object]:
    return {
        "product_id": "BTC-USD",
        "anchor_second": DAY_START,
        "source_sequence_num": 100,
        "best_bid": Decimal("90000.00"),
        "best_ask": Decimal("90000.10"),
        "bid_book": {Decimal("90000.00"): Decimal("1.25")},
        "ask_book": {Decimal("90000.10"): Decimal("0.75")},
    }


@pytest.fixture
def replay_delta_rows() -> list[dict[str, object]]:
    return [
        {
            "product_id": "BTC-USD",
            "changed_second": DAY_START + timedelta(seconds=offset),
            "source_sequence_num_start": 100 + offset,
            "source_sequence_num_end": 100 + offset,
            "best_bid": Decimal("90000.00"),
            "best_ask": Decimal("90000.10"),
            "changes": (
                ("bid", Decimal("90000.00"), Decimal(str(1.25 + offset / 10))),
            ),
        }
        for offset in (1, 2, 3)
    ]


@pytest.fixture
def trade_rows() -> list[dict[str, object]]:
    return [
        {
            "product_id": "BTC-USD",
            "trade_id": str(100 - offset),
            "event_time": DAY_START + timedelta(seconds=offset, microseconds=500_000),
            "price": Decimal("90000.05"),
            "size": Decimal("0.01"),
            "reported_side": "SELL" if offset % 2 else "BUY",
            "source": "coinbase_public_trades",
        }
        for offset in (1, 2, 3, 4)
    ]


@pytest.fixture
def overlapping_trade_page(trade_rows) -> list[dict[str, object]]:
    return [trade_rows[2], trade_rows[3]]


@pytest.fixture
def sequence_regression(replay_delta_rows) -> dict[str, object]:
    return {
        **replay_delta_rows[-1],
        "source_sequence_num_start": 99,
        "source_sequence_num_end": 98,
    }


@pytest.fixture
def explicit_source_gap() -> tuple[datetime, datetime]:
    return DAY_START + timedelta(seconds=10), DAY_START + timedelta(seconds=20)
