from datetime import date, datetime, timezone

import pytest

from btcspiker_data.coinbase_trades import (
    CoinbaseTradeClient,
    TradeDayCompletion,
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


class ManualClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


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
    assert len(stalled_trade_session.calls) == 2


def test_empty_pages_require_two_consecutive_pages_before_stall():
    session = TradeSession(
        [
            Response(200, {"trades": []}),
            Response(200, {"trades": []}),
        ]
    )

    with pytest.raises(TradePageStalledError, match="no new trade IDs"):
        list(client(session).iter_day_trades(DAY))

    assert len(session.calls) == 2


def test_completion_evidence_exists_only_after_success(fake_trade_session, stalled_trade_session):
    successful = client(fake_trade_session)
    list(successful.iter_day_trades(DAY))
    assert successful.last_completion == TradeDayCompletion(
        product_id="BTC-USD",
        source_date=DAY,
        day_start_epoch=DAY_START,
        day_end_epoch=DAY_END,
    )

    stalled = client(stalled_trade_session)
    with pytest.raises(TradePageStalledError):
        list(stalled.iter_day_trades(DAY))
    assert stalled.last_completion is None


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


def test_fifth_retryable_failure_includes_request_context():
    session = TradeSession([Response(503, {}) for _ in range(5)])
    sleeps = []
    trade_client = CoinbaseTradeClient(session=session, sleep=sleeps.append)

    with pytest.raises(
        RuntimeError,
        match=rf"date={DAY} end={DAY_END} status=503",
    ):
        list(trade_client.iter_day_trades(DAY))

    assert len(session.calls) == 5
    assert sleeps == [1.0, 2.0, 4.0, 8.0]


def test_nonretryable_4xx_fails_immediately():
    session = TradeSession([Response(400, {})])
    sleeps = []
    trade_client = CoinbaseTradeClient(session=session, sleep=sleeps.append)

    with pytest.raises(RuntimeError, match="status=400"):
        list(trade_client.iter_day_trades(DAY))

    assert len(session.calls) == 1
    assert sleeps == []


def test_module_iterator_delegates_to_the_public_client(fake_trade_session):
    trades = list(iter_day_trades(DAY, session=fake_trade_session, sleep=lambda _: None))
    assert [trade.trade_id for trade in trades] == ["97", "98", "99", "100"]


def test_ninth_immediate_request_waits_for_token_refill():
    page_seconds = [DAY_END - index * 100 for index in range(1, 9)] + [DAY_START]
    session = TradeSession(
        [Response(200, {"trades": [trade(str(index), second)]})
         for index, second in enumerate(page_seconds)]
    )
    clock = ManualClock()
    trade_client = CoinbaseTradeClient(
        session=session,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    list(trade_client.iter_day_trades(DAY))

    assert len(session.calls) == 9
    assert clock.sleeps == [0.125]


def test_configured_rate_limit_controls_token_bucket():
    page_seconds = [DAY_END - index * 100 for index in range(1, 5)] + [DAY_START]
    session = TradeSession(
        [Response(200, {"trades": [trade(str(index), second)]})
         for index, second in enumerate(page_seconds)]
    )
    clock = ManualClock()

    list(CoinbaseTradeClient(
        session=session,
        max_rps=4,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    ).iter_day_trades(DAY))

    assert clock.sleeps == [0.25]


def test_configured_product_controls_request_endpoint():
    session = TradeSession(
        [Response(200, {"trades": [trade("eth-1", DAY_START)]})]
    )
    list(CoinbaseTradeClient(session=session, product_id="ETH-USD").iter_day_trades(DAY))
    assert session.calls[0][0].endswith("/products/ETH-USD/ticker")


def test_short_page_inside_day_start_epoch_second_completes_day():
    first_second = DAY_START + 0.5
    session = TradeSession(
        [Response(200, {"trades": [trade("first", first_second)]})]
    )
    trade_client = client(session)

    trades = list(trade_client.iter_day_trades(DAY))

    assert trades[0].event_time.microsecond == 500_000
    assert trade_client.last_completion is not None


def test_full_page_inside_day_start_epoch_second_cannot_certify_completion():
    full_page = [
        trade(str(index), DAY_START + (1000 - index) / 2000)
        for index in range(1000)
    ]
    session = TradeSession(
        [
            Response(200, {"trades": full_page}),
            Response(200, {"trades": full_page}),
        ]
    )
    trade_client = client(session)

    with pytest.raises(TradePageStalledError, match="did not move backward"):
        list(trade_client.iter_day_trades(DAY))

    assert len(session.calls) == 2
    assert trade_client.last_completion is None
