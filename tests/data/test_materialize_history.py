from datetime import timedelta
from decimal import Decimal
import hashlib

import pandas as pd
import pytest

from btcspiker_data.contracts import BookState, MODEL_TICK_COLUMNS
from btcspiker_data.materialize import (
    join_trades_to_books,
    materialize_segmented_features,
    write_feature_outputs_atomic,
)


def _book(second, *, segment_id=0, product_id="BTC-USD"):
    return BookState(
        product_id=product_id,
        observed_through=second,
        sequence_start=100 + segment_id,
        sequence_end=100 + segment_id,
        best_bid=Decimal("90000.00"),
        bid_size=Decimal("2.0"),
        best_ask=Decimal("90000.10"),
        ask_size=Decimal("1.0"),
        segment_id=segment_id,
    )


def _trade(second, trade_id, *, segment_id=0, product_id="BTC-USD"):
    return {
        "product_id": product_id,
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


def test_join_infers_segment_only_from_exact_prior_book_second():
    from tests.data.conftest import DAY_START
    gap_trade = _trade(DAY_START + timedelta(seconds=50), "gap")
    after_gap_trade = _trade(DAY_START + timedelta(seconds=101), "after-gap")
    gap_trade.pop("segment_id")
    after_gap_trade.pop("segment_id")

    joined = join_trades_to_books(
        [gap_trade, after_gap_trade],
        [
            _book(DAY_START, segment_id=0),
            _book(DAY_START + timedelta(seconds=100), segment_id=1),
        ],
    )

    assert joined["trade_id"].tolist() == ["after-gap"]
    assert joined["segment_id"].tolist() == [1]


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


def test_materializes_products_independently_when_segment_ids_match():
    from tests.data.conftest import DAY_START

    books = []
    trades = []
    for product_id in ("BTC-USD", "ETH-USD"):
        for second in range(62):
            observed = DAY_START + timedelta(seconds=second)
            books.append(_book(observed, product_id=product_id))
            if second:
                trades.append(
                    _trade(observed, f"{product_id}-{second}", product_id=product_id)
                )

    core = materialize_segmented_features(trades, books)["core_v1"]

    assert core.groupby("product_id").size().to_dict() == {
        "BTC-USD": 1,
        "ETH-USD": 1,
    }
    # The one emitted row per product is the first tick, whose feature snapshot
    # must see only itself. Shared engine state would make one product see two.
    assert set(core["n_ticks_60s"]) == {1}


def test_writes_all_feature_sets_by_content_hash_and_reuses_identical_files(tmp_path):
    outputs = {
        feature_set_id: pd.DataFrame(
            [{"product_id": "BTC-USD", "timestamp": "2026-04-24T00:00:00.500000+00:00", "value": index}]
        )
        for index, feature_set_id in enumerate(
            ("core_v1", "multi_window_v1", "microstructure_v1")
        )
    }

    paths = write_feature_outputs_atomic(outputs, tmp_path)
    mtimes = {key: path.stat().st_mtime_ns for key, path in paths.items()}
    reused = write_feature_outputs_atomic(outputs, tmp_path)

    assert set(paths) == set(outputs)
    assert reused == paths
    assert {key: path.stat().st_mtime_ns for key, path in reused.items()} == mtimes
    for feature_set_id, path in paths.items():
        digest = path.stem.removeprefix("part-")
        assert path.parent == tmp_path / "features" / f"feature_set={feature_set_id}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        pd.testing.assert_frame_equal(pd.read_parquet(path), outputs[feature_set_id])


def test_content_addressed_writer_rejects_corrupted_existing_file(tmp_path):
    outputs = {
        feature_set_id: pd.DataFrame([{"value": index}])
        for index, feature_set_id in enumerate(
            ("core_v1", "multi_window_v1", "microstructure_v1")
        )
    }
    paths = write_feature_outputs_atomic(outputs, tmp_path)
    paths["core_v1"].write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="content digest mismatch"):
        write_feature_outputs_atomic(outputs, tmp_path)
