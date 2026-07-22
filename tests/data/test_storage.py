from datetime import datetime, timezone

import pyarrow as pa
import pytest

from btcspiker_data.storage import write_partition_atomic


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
