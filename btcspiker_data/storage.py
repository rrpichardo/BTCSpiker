"""Immutable, content-addressed local raw-data partitions."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import pyarrow as pa
import pyarrow.parquet as pq

from .contracts import RAW_BOOK_COLUMNS, RAW_TRADE_COLUMNS


@dataclass(frozen=True)
class PartitionRecord:
    path: Path
    sha256: str
    row_count: int
    size_bytes: int


_COLUMNS = {"trades": RAW_TRADE_COLUMNS, "book_deltas": RAW_BOOK_COLUMNS, "book_states": RAW_BOOK_COLUMNS}
_TIME_COLUMN = {"trades": "event_time", "book_deltas": "observed_through", "book_states": "observed_through"}
_KEY_COLUMNS = {
    "trades": ("trade_id",),
    "book_deltas": ("observed_through",),
    "book_states": ("observed_through",),
}
_EMPTY_TYPES = {
    "trades": {
        "event_time": pa.timestamp("us", tz="UTC"),
    },
    "book_deltas": {
        "observed_through": pa.timestamp("us", tz="UTC"),
        "sequence_start": pa.int64(),
        "sequence_end": pa.int64(),
        "segment_id": pa.int64(),
    },
    "book_states": {
        "observed_through": pa.timestamp("us", tz="UTC"),
        "sequence_start": pa.int64(),
        "sequence_end": pa.int64(),
        "segment_id": pa.int64(),
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate(table: pa.Table, kind: str, product_id: str) -> tuple[str, datetime]:
    if kind not in _COLUMNS:
        raise ValueError(f"unknown partition kind: {kind}")
    if tuple(table.column_names) != _COLUMNS[kind]:
        raise ValueError("table columns must match the required ordered schema")
    if table.num_rows == 0:
        raise ValueError("partition must not be empty")
    if set(table["product_id"].to_pylist()) != {product_id}:
        raise ValueError("partition product_id does not match destination")
    sources = set(table["source"].to_pylist())
    if len(sources) != 1 or not next(iter(sources)):
        raise ValueError("partition must contain exactly one source")
    timestamps = table[_TIME_COLUMN[kind]].to_pylist()
    if any(not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value) for value in timestamps):
        raise ValueError("partition timestamps must be timezone-aware UTC")
    hour = timestamps[0].replace(minute=0, second=0, microsecond=0)
    if any(value.replace(minute=0, second=0, microsecond=0) != hour for value in timestamps):
        raise ValueError("partition must contain one UTC hour")
    keys = list(zip(*(table[name].to_pylist() for name in _KEY_COLUMNS[kind])))
    if len(keys) != len(set(keys)):
        raise ValueError("partition contains duplicate stable keys")
    return next(iter(sources)), hour


def _persist_partition(
    table: pa.Table,
    root: str | Path,
    kind: str,
    product_id: str,
    source: str,
    hour: datetime,
) -> PartitionRecord:
    directory = Path(root) / "raw" / f"kind={kind}" / f"source={source}" / f"product={product_id}" / f"date={hour:%Y-%m-%d}" / f"hour={hour:%H}"
    directory.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=directory, prefix=".part-", suffix=".parquet", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        pq.write_table(table, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        digest = _sha256(temporary)
        destination = directory / f"part-{digest}.parquet"
        if destination.exists():
            if _sha256(destination) != digest:
                raise ValueError("existing content-addressed path has an invalid digest")
            temporary.unlink()
        else:
            os.replace(temporary, destination)
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return PartitionRecord(destination, digest, table.num_rows, destination.stat().st_size)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_partition_atomic(table: pa.Table, root: str | Path, kind: str, product_id: str) -> PartitionRecord:
    """Write an immutable non-empty hourly Parquet file."""
    source, hour = _validate(table, kind, product_id)
    return _persist_partition(table, root, kind, product_id, source, hour)


def write_empty_partition_atomic(
    root: str | Path,
    kind: str,
    product_id: str,
    *,
    source: str,
    hour: datetime,
) -> PartitionRecord:
    """Write an explicit zero-row hour without manufacturing market events."""
    if kind not in _COLUMNS:
        raise ValueError(f"unknown partition kind: {kind}")
    if not source or not product_id:
        raise ValueError("empty partition source and product_id are required")
    if (
        hour.tzinfo is None
        or hour.utcoffset() != timezone.utc.utcoffset(hour)
        or hour != hour.replace(minute=0, second=0, microsecond=0)
    ):
        raise ValueError("empty partition hour must be aligned UTC")
    arrays = [
        pa.array([], type=_EMPTY_TYPES[kind].get(name, pa.string()))
        for name in _COLUMNS[kind]
    ]
    table = pa.Table.from_arrays(arrays, names=_COLUMNS[kind])
    return _persist_partition(table, root, kind, product_id, source, hour)
