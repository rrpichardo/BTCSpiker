from btcspiker_ml.tick_identity import tick_dedupe_key

TICK = {
    "product_id": "BTC-USD",
    "timestamp": "2026-04-06T15:02:34.590029066Z",
    "price": "69700.12",
    "best_bid": "69700.10",
    "best_ask": "69700.14",
    "volume_24_h": "12345.6",
}


def test_key_covers_only_feature_affecting_fields():
    assert tick_dedupe_key(TICK) == (
        "BTC-USD",
        "2026-04-06T15:02:34.590029066Z",
        "69700.12",
        "69700.10",
        "69700.14",
    )


def test_volume_24_h_is_excluded_from_the_key():
    other_volume = {**TICK, "volume_24_h": "999999.9"}
    assert tick_dedupe_key(TICK) == tick_dedupe_key(other_volume)


def test_a_real_price_move_changes_the_key():
    moved = {**TICK, "price": "69701.00"}
    assert tick_dedupe_key(TICK) != tick_dedupe_key(moved)


def test_missing_fields_do_not_raise():
    assert tick_dedupe_key({}) == (None, None, None, None, None)
