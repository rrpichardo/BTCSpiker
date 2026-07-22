from datetime import date, datetime, timezone

import pytest

from btcspiker_data.coinbase_trades import (
    CoinbaseTradeClient,
    TradePageStalledError,
    iter_day_trades,
)


UTC = timezone.utc
DAY = date(2026, 4, 24)
DAY_START = int(datetime(2026, 4, 24, tzinfo=UTC).timestamp())
DAY_END = int(datetime(2026, 4, 25, tzinfo=UTC).timestamp())


class Response:
    def __init__(self, status_code, payload, headers=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        return self._payload


class TradeSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def get(self, url, *, params, headers, timeout):
        self.calls.append((url, params, headers, timeout))
        return next(self.responses)


def trade(trade_id, second, side="SELL"):
    return {
        "trade_id": str(trade_id),
        "product_id": "BTC-USD",
        "price": "90000.25",
        "size": "0.01",
        "time": datetime.fromtimestamp(second, UTC).isoformat().replace("+00:00", "Z"),
        "side": side,
    }


@pytest.fixture
def fake_trade_session():
    return TradeSession(
        [
            Response(200, {"trades": [trade("100", DAY_END - 1), trade("99", DAY_START + 20)], "best_bid": "1"}),
            Response(200, {"trades": [trade("99", DAY_START + 20), trade("98", DAY_START + 10), trade("97", DAY_START)]}),
        ]
    )


@pytest.fixture
def stalled_trade_session():
    return TradeSession(
        [
            Response(200, {"trades": [trade("100", DAY_END - 1)]}),
            Response(200, {"trades": [trade("100", DAY_END - 1)]}),
            Response(200, {"trades": [trade("100", DAY_END - 1)]}),
        ]
    )


def client(session):
    return CoinbaseTradeClient(session=session, sleep=lambda _: None)


def test_backfill_deduplicates_inclusive_end_overlap(fake_trade_session):
    trades = list(client(fake_trade_session).iter_day_trades(DAY))
    assert [trade.trade_id for trade in trades] == ["97", "98", "99", "100"]
    assert fake_trade_session.calls[0][1] == {"limit": 1000, "start": DAY_START, "end": DAY_END}
    assert fake_trade_session.calls[1][1]["end"] == DAY_START + 20


def test_backfill_rejects_page_that_cannot_advance(stalled_trade_session):
    with pytest.raises(TradePageStalledError):
        list(client(stalled_trade_session).iter_day_trades(DAY))


def test_response_bbo_is_not_copied_into_trade_events(fake_trade_session):
    event = next(client(fake_trade_session).iter_day_trades(DAY))
    assert not hasattr(event, "best_bid")


def test_backfill_preserves_reported_side_and_uses_public_request_shape(fake_trade_session):
    event = next(client(fake_trade_session).iter_day_trades(DAY))
    _, _, headers, timeout = fake_trade_session.calls[0]
    assert event.reported_side == "SELL"
    assert event.side_semantics == "coinbase_reported_unspecified"
    assert headers == {"User-Agent": "BTCSpiker-research/1"}
    assert timeout == (5, 30)


def test_retries_retryable_status_with_retry_after_before_success():
    session = TradeSession(
        [
            Response(429, {}, {"Retry-After": "2"}),
            Response(200, {"trades": [trade("1", DAY_START)]}),
        ]
    )
    sleeps = []
    trades = list(CoinbaseTradeClient(session=session, sleep=sleeps.append).iter_day_trades(DAY))
    assert [item.trade_id for item in trades] == ["1"]
    assert sleeps == [2.0]


def test_module_iterator_delegates_to_the_public_client(fake_trade_session):
    trades = list(iter_day_trades(DAY, session=fake_trade_session, sleep=lambda _: None))
    assert [trade.trade_id for trade in trades] == ["97", "98", "99", "100"]
