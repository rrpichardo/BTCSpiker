"""Frozen cross-module contracts for historical Coinbase market data."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal


RAW_BOOK_COLUMNS = (
    "source",
    "product_id",
    "observed_through",
    "sequence_start",
    "sequence_end",
    "best_bid",
    "bid_size",
    "best_ask",
    "ask_size",
    "changes_json",
    "source_revision",
    "source_date",
)
RAW_TRADE_COLUMNS = (
    "source",
    "product_id",
    "trade_id",
    "event_time",
    "price",
    "size",
    "reported_side",
    "side_semantics",
    "source_date",
)
MODEL_TICK_COLUMNS = (
    "product_id",
    "timestamp",
    "price",
    "best_bid",
    "best_ask",
    "bid_size",
    "ask_size",
    "trade_id",
    "trade_size",
    "reported_side",
    "book_observed_through",
    "segment_id",
)


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError("timestamp must be UTC")


def validate_utc_window(start: datetime, end: datetime) -> None:
    """Validate a non-empty, half-open UTC interval."""
    _require_utc(start)
    _require_utc(end)
    if start >= end:
        raise ValueError("start must be before end")


@dataclass(frozen=True)
class BookState:
    product_id: str
    observed_through: datetime
    sequence_start: int
    sequence_end: int
    best_bid: Decimal
    bid_size: Decimal
    best_ask: Decimal
    ask_size: Decimal
    segment_id: int = 0

    def __post_init__(self) -> None:
        _require_utc(self.observed_through)
        if self.sequence_start > self.sequence_end:
            raise ValueError("sequence range regressed")
        if self.best_bid > self.best_ask:
            raise ValueError("crossed book")
        if self.bid_size < 0 or self.ask_size < 0:
            raise ValueError("book quantities must be non-negative")
        if self.segment_id < 0:
            raise ValueError("segment_id must be non-negative")


@dataclass(frozen=True)
class BookDelta:
    product_id: str
    changed_second: datetime
    sequence_start: int
    sequence_end: int
    best_bid: Decimal
    best_ask: Decimal
    changes: tuple[tuple[str, Decimal, Decimal], ...]

    def __post_init__(self) -> None:
        _require_utc(self.changed_second)
        if self.sequence_start > self.sequence_end:
            raise ValueError("sequence range regressed")
        if self.best_bid > self.best_ask:
            raise ValueError("crossed book")
        if any(
            side not in {"bid", "offer"} or quantity < 0
            for side, _, quantity in self.changes
        ):
            raise ValueError("invalid L2 change")


@dataclass(frozen=True)
class TradeEvent:
    product_id: str
    trade_id: str
    event_time: datetime
    price: Decimal
    size: Decimal
    reported_side: str
    source: str
    side_semantics: str = "coinbase_reported_unspecified"

    def __post_init__(self) -> None:
        _require_utc(self.event_time)
        if self.price <= 0 or self.size <= 0:
            raise ValueError("trade price and size must be positive")


@dataclass(frozen=True)
class QualityIncident:
    code: str
    start: datetime
    end: datetime
    severity: str
    detail: str

    def __post_init__(self) -> None:
        validate_utc_window(self.start, self.end)
