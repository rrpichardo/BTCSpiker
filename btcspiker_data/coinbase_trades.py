"""Bounded public Coinbase historical-trades client."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
import time as clock
from typing import Callable, Iterator

from .contracts import TradeEvent


_ENDPOINT = "https://api.coinbase.com/api/v3/brokerage/market/products/BTC-USD/ticker"
_RETRYABLE = {429, 500, 502, 503, 504}


class TradePageStalledError(RuntimeError):
    """Raised when inclusive pagination no longer makes backward progress."""


class CoinbaseTradeClient:
    def __init__(
        self,
        session,
        *,
        product_id: str = "BTC-USD",
        sleep: Callable[[float], None] = clock.sleep,
        monotonic: Callable[[], float] = clock.monotonic,
    ) -> None:
        self.session = session
        self.product_id = product_id
        self._sleep = sleep
        self._monotonic = monotonic
        self._tokens = 8.0
        self._last_refill = monotonic()

    def iter_day_trades(self, source_date: date) -> Iterator[TradeEvent]:
        day_start = datetime.combine(source_date, time.min, tzinfo=timezone.utc)
        day_end = day_start + timedelta(days=1)
        start_epoch = int(day_start.timestamp())
        end_epoch = int(day_end.timestamp())
        page_end = end_epoch
        seen: set[str] = set()
        events: list[TradeEvent] = []
        nonadvancing_pages = 0
        previous_oldest: datetime | None = None

        while True:
            payload = self._get_page(source_date, page_end, start_epoch)
            page_events = [self._parse_trade(item) for item in payload.get("trades", ())]
            in_window = [event for event in page_events if day_start <= event.event_time < day_end]
            fresh = [event for event in in_window if event.trade_id not in seen]
            seen.update(event.trade_id for event in fresh)
            events.extend(fresh)

            if not in_window:
                raise TradePageStalledError(f"no in-window trades for {source_date} at end={page_end}")
            oldest = min(event.event_time for event in in_window)
            if previous_oldest is not None and oldest >= previous_oldest:
                nonadvancing_pages += 1
            elif not fresh:
                nonadvancing_pages += 1
            else:
                nonadvancing_pages = 0
            if nonadvancing_pages >= 2:
                raise TradePageStalledError(f"pagination stalled for {source_date} at end={page_end}")
            if oldest <= day_start:
                break
            previous_oldest = oldest
            page_end = int(oldest.timestamp())

        yield from sorted(events, key=lambda event: (event.event_time, event.trade_id))

    def _get_page(self, source_date: date, page_end: int, start_epoch: int) -> dict:
        for attempt in range(5):
            self._acquire_token()
            response = self.session.get(
                _ENDPOINT,
                params={"limit": 1000, "start": start_epoch, "end": page_end},
                headers={"User-Agent": "BTCSpiker-research/1"},
                timeout=(5, 30),
            )
            if response.status_code not in _RETRYABLE:
                if 200 <= response.status_code < 300:
                    return response.json()
                raise RuntimeError(
                    f"Coinbase trades request failed date={source_date} end={page_end} status={response.status_code}"
                )
            if attempt == 4:
                raise RuntimeError(
                    f"Coinbase trades request failed date={source_date} end={page_end} status={response.status_code}"
                )
            retry_after = response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after is not None else min(16.0, 2.0**attempt)
            self._sleep(delay)
        raise AssertionError("unreachable")

    def _acquire_token(self) -> None:
        now = self._monotonic()
        self._tokens = min(8.0, self._tokens + (now - self._last_refill) * 8.0)
        self._last_refill = now
        if self._tokens < 1:
            delay = (1 - self._tokens) / 8.0
            self._sleep(delay)
            self._last_refill = self._monotonic()
            self._tokens = 0.0
        else:
            self._tokens -= 1

    def _parse_trade(self, item: dict) -> TradeEvent:
        event_time = datetime.fromisoformat(item["time"].replace("Z", "+00:00")).astimezone(timezone.utc)
        return TradeEvent(
            product_id=item.get("product_id", self.product_id),
            trade_id=str(item["trade_id"]),
            event_time=event_time,
            price=Decimal(item["price"]),
            size=Decimal(item["size"]),
            reported_side=item["side"],
            source="coinbase_public_trades",
        )


def iter_day_trades(
    source_date: date,
    *,
    session,
    product_id: str = "BTC-USD",
    sleep: Callable[[float], None] = clock.sleep,
) -> Iterator[TradeEvent]:
    """Fetch one UTC date of public Coinbase trades without authentication."""
    return CoinbaseTradeClient(
        session=session, product_id=product_id, sleep=sleep
    ).iter_day_trades(source_date)
