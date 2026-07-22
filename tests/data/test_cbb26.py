from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from btcspiker_data.cbb26 import (
    CBB26IntegrityError,
    CBB26Shard,
    InsufficientWorkingSpace,
    UnverifiedRemoteArtifact,
    download_shard_to_cache,
    list_btc_shards,
    read_sidecar,
    remove_verified_shard_cache,
    restore_shard,
    sha256_file,
)


UTC = timezone.utc


def _tree(start: date, end: date) -> list[dict[str, object]]:
    nodes = []
    current = start
    while current <= end:
        root = f"data/{current.isoformat()}"
        nodes.extend(
            [
                {"path": f"{root}/BTC-USD.dump", "size": 4},
                {"path": f"{root}/BTC-USD.json", "size": 1},
            ]
        )
        current = date.fromordinal(current.toordinal() + 1)
    return nodes


def test_inventory_requires_every_requested_day():
    start, end = date(2026, 4, 24), date(2026, 5, 28)
    shards = list_btc_shards(_tree(start, end), start=start, end=end, product="BTC-USD")
    assert len(shards) == 35
    with pytest.raises(CBB26IntegrityError, match="missing dump or sidecar"):
        list_btc_shards(_tree(start, end)[:-1], start=start, end=end, product="BTC-USD")


def test_download_reuses_matching_local_file(tmp_path: Path):
    shard = CBB26Shard(date(2026, 4, 24), "BTC-USD", "data/2026-04-24/BTC-USD.dump", "data/2026-04-24/BTC-USD.json", 4)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        destination = Path(kwargs["local_dir"]) / kwargs["filename"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"dump")
        return str(destination)

    first = download_shard_to_cache(shard, tmp_path, download, disk_usage=lambda _: (0, 0, 2**33))
    again = download_shard_to_cache(shard, tmp_path, download, disk_usage=lambda _: (0, 0, 2**33))
    assert again == first
    assert sha256_file(again) == hashlib.sha256(b"dump").hexdigest()
    assert len(calls) == 1


def test_download_rejects_low_space_before_calling_hub(tmp_path: Path):
    shard = CBB26Shard(date(2026, 4, 24), "BTC-USD", "data/2026-04-24/BTC-USD.dump", "data/2026-04-24/BTC-USD.json", 4)
    with pytest.raises(InsufficientWorkingSpace, match="observed=1 required=4294967296"):
        download_shard_to_cache(shard, tmp_path, lambda **_: pytest.fail("downloaded"), disk_usage=lambda _: (0, 0, 1))


def test_read_sidecar_requires_size_and_schema(tmp_path: Path):
    sidecar = tmp_path / "BTC-USD.json"
    sidecar.write_text(json.dumps({"dump_size_bytes": 4, "export_schema": "cbb26_hf_export_staging"}))
    assert read_sidecar(sidecar)["dump_size_bytes"] == 4
    sidecar.write_text("{}")
    with pytest.raises(CBB26IntegrityError, match="dump_size_bytes"):
        read_sidecar(sidecar)


def test_restore_fails_for_count_mismatch_or_bad_temporal_integrity(tmp_path: Path):
    shard = CBB26Shard(date(2026, 4, 24), "BTC-USD", "x.dump", "x.json", 4)
    dump = tmp_path / "x.dump"
    dump.write_bytes(b"dump")
    sidecar = {"row_counts": {"orderbook_replay_anchors": 1, "orderbook_second_deltas": 1, "orderbook_checkpoints": 1, "orderbook_replay_metadata": 1}, "day_start_utc": "2026-04-24T00:00:00+00:00", "day_end_utc": "2026-04-24T23:59:59+00:00"}

    class Cursor:
        def execute(self, *_): pass
        def fetchone(self): return (0,)
    class Connection:
        def cursor(self): return Cursor()
        def commit(self): pass

    with pytest.raises(CBB26IntegrityError, match="row count"):
        restore_shard(shard, dump, sidecar, Connection(), run=lambda *_args, **_kwargs: type("R", (), {"returncode": 0, "stderr": ""})(), disk_usage=lambda _: (0, 0, 2**33))


@pytest.mark.parametrize(
    ("result_values", "message"),
    [([1, 1, 1, 1, 0], "missing anchor"), ([1, 1, 1, 1, 1, 1], "outside sidecar UTC bounds")],
)
def test_restore_rejects_missing_anchor_and_out_of_window_rows(tmp_path: Path, result_values, message: str):
    shard = CBB26Shard(date(2026, 4, 24), "BTC-USD", "x.dump", "x.json", 4)
    dump = tmp_path / "x.dump"
    dump.write_bytes(b"dump")
    sidecar = {"row_counts": {table: 1 for table in ("orderbook_replay_anchors", "orderbook_second_deltas", "orderbook_checkpoints", "orderbook_replay_metadata")}, "day_start_utc": "2026-04-24T00:00:00+00:00", "day_end_utc": "2026-04-24T23:59:59+00:00"}

    class Cursor:
        def execute(self, *_): pass
        def fetchone(self): return (result_values.pop(0),)
    class Connection:
        def cursor(self): return Cursor()
        def commit(self): pytest.fail("must not commit an invalid restore")

    with pytest.raises(CBB26IntegrityError, match=message):
        restore_shard(shard, dump, sidecar, Connection(), run=lambda *_args, **_kwargs: type("R", (), {"returncode": 0, "stderr": ""})(), disk_usage=lambda _: (0, 0, 2**33))


def test_remove_cache_requires_verified_partition_receipts(tmp_path: Path):
    shard = CBB26Shard(date(2026, 4, 24), "BTC-USD", "x.dump", "x.json", 4)
    shard_dir = tmp_path / "2026-04-24"
    shard_dir.mkdir()
    (shard_dir / "BTC-USD.dump").write_bytes(b"dump")
    (shard_dir / "BTC-USD.json").write_text("{}")
    with pytest.raises(UnverifiedRemoteArtifact):
        remove_verified_shard_cache(shard, tmp_path, receipts=[], connection=None)
    assert (shard_dir / "BTC-USD.dump").exists()
