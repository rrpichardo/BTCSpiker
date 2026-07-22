from datetime import datetime, timedelta, timezone

import pyarrow as pa
import pytest

from btcspiker_data.storage import write_empty_partition_atomic, write_partition_atomic


def _trade_table() -> pa.Table:
    return pa.table(
        {
            "source": ["coinbase"],
            "product_id": ["BTC-USD"],
            "trade_id": ["42"],
            "event_time": [datetime(2026, 4, 24, 3, tzinfo=timezone.utc)],
            "price": ["90000.05"],
            "size": ["0.01"],
            "reported_side": ["SELL"],
            "side_semantics": ["coinbase_reported_unspecified"],
            "source_date": ["2026-04-24"],
        }
    )


def _book_table(observed_through, sequence_starts, sequence_ends) -> pa.Table:
    rows = len(observed_through)
    return pa.table(
        {
            "source": ["coinbase"] * rows,
            "product_id": ["BTC-USD"] * rows,
            "observed_through": observed_through,
            "sequence_start": sequence_starts,
            "sequence_end": sequence_ends,
            "best_bid": ["90000.00"] * rows,
            "bid_size": ["1.0"] * rows,
            "best_ask": ["90000.10"] * rows,
            "ask_size": ["1.0"] * rows,
            "segment_id": [0] * rows,
            "changes_json": ["[]"] * rows,
            "source_revision": ["r1"] * rows,
            "source_date": ["2026-04-24"] * rows,
        }
    )


def test_partition_path_is_content_addressed(tmp_path):
    table = _trade_table()
    left = write_partition_atomic(table, tmp_path, "trades", "BTC-USD")
    right = write_partition_atomic(table, tmp_path, "trades", "BTC-USD")
    assert left.path == right.path
    assert left.sha256 == right.sha256
    assert left.path.name == f"part-{left.sha256}.parquet"


def test_partition_rejects_unordered_columns(tmp_path):
    table = _trade_table().select(list(reversed(_trade_table().column_names)))
    with pytest.raises(ValueError, match="ordered"):
        write_partition_atomic(table, tmp_path, "trades", "BTC-USD")


def test_book_states_allow_repeated_sequence_range_at_distinct_timestamps(tmp_path):
    start = datetime(2026, 4, 24, 3, tzinfo=timezone.utc)
    table = _book_table([start, start + timedelta(seconds=1)], [42, 42], [42, 42])
    record = write_partition_atomic(table, tmp_path, "book_states", "BTC-USD")
    assert record.row_count == 2


def test_book_deltas_reject_duplicate_observed_through_even_with_distinct_sequences(tmp_path):
    timestamp = datetime(2026, 4, 24, 3, tzinfo=timezone.utc)
    table = _book_table([timestamp, timestamp], [42, 43], [42, 43])
    with pytest.raises(ValueError, match="duplicate stable keys"):
        write_partition_atomic(table, tmp_path, "book_deltas", "BTC-USD")


@pytest.mark.parametrize("kind", ["trades", "book_deltas", "book_states"])
def test_empty_hour_partition_has_canonical_schema_and_location(tmp_path, kind):
    source = "coinbase_public_trades" if kind == "trades" else "cbb26"
    hour = datetime(2026, 4, 24, 7, tzinfo=timezone.utc)

    record = write_empty_partition_atomic(
        tmp_path,
        kind,
        "BTC-USD",
        source=source,
        hour=hour,
    )

    assert record.row_count == 0
    assert f"kind={kind}" in record.path.as_posix()
    assert "hour=07" in record.path.as_posix()
