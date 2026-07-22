from __future__ import annotations

import hashlib
import json
from datetime import date, timezone
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
TABLES = (
    "orderbook_replay_anchors",
    "orderbook_second_deltas",
    "orderbook_checkpoints",
    "orderbook_replay_metadata",
)


def _sidecar(**overrides):
    value = {
        "schema": "cbb26_timeseries_shard_manifest_v1",
        "trade_date": "2026-04-24",
        "product_id": "BTC-USD",
        "day_start_utc": "2026-04-24T00:00:00+00:00",
        "day_end_utc": "2026-04-24T23:59:59+00:00",
        "row_counts": {table: 1 for table in TABLES},
        "dump_relpath": "data/2026-04-24/BTC-USD.dump",
        "dump_size_bytes": 4,
        "export_schema": "cbb26_hf_export_staging",
        "format": "pg_dump_custom_fc",
    }
    value.update(overrides)
    return value


def _shard():
    return CBB26Shard(
        date(2026, 4, 24),
        "BTC-USD",
        "data/2026-04-24/BTC-USD.dump",
        "data/2026-04-24/BTC-USD.json",
        4,
    )


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
    shard = _shard()
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        destination = Path(kwargs["local_dir"]) / kwargs["filename"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if kwargs["filename"].endswith(".json"):
            destination.write_text(json.dumps(_sidecar()))
        else:
            destination.write_bytes(b"dump")
        return str(destination)

    first = download_shard_to_cache(shard, tmp_path, download, disk_usage=lambda _: (0, 0, 2**33))
    again = download_shard_to_cache(shard, tmp_path, download, disk_usage=lambda _: (0, 0, 2**33))
    assert again == first
    assert sha256_file(again) == hashlib.sha256(b"dump").hexdigest()
    assert [call["filename"] for call in calls] == [shard.sidecar_path, shard.dump_path]
    assert all(call["revision"] == "c1e89eded9915e1c75a18911298edfbbbe4050ce" for call in calls)
    assert not list(first.parent.glob(".download*"))


def test_download_does_not_accept_dump_without_valid_sidecar(tmp_path: Path):
    shard = _shard()
    target = tmp_path / "c1e89eded9915e1c75a18911298edfbbbe4050ce" / "2026-04-24" / "BTC-USD.dump"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"dump")

    def download(**kwargs):
        downloaded = Path(kwargs["local_dir"]) / kwargs["filename"]
        downloaded.parent.mkdir(parents=True, exist_ok=True)
        downloaded.write_text(json.dumps(_sidecar(product_id="ETH-USD")))
        return str(downloaded)

    with pytest.raises(CBB26IntegrityError, match="product_id"):
        download_shard_to_cache(shard, tmp_path, download, disk_usage=lambda _: (0, 0, 2**33))


def test_download_rejects_low_space_before_calling_hub(tmp_path: Path):
    shard = _shard()
    with pytest.raises(InsufficientWorkingSpace, match="observed=1 required=4294967296"):
        download_shard_to_cache(shard, tmp_path, lambda **_: pytest.fail("downloaded"), disk_usage=lambda _: (0, 0, 1))


def test_space_check_uses_nearest_existing_parent(tmp_path: Path):
    missing_root = tmp_path / "not-yet-created" / "cache"
    checked = []

    def disk_usage(path):
        checked.append(Path(path))
        assert Path(path).exists()
        return (0, 0, 1)

    with pytest.raises(InsufficientWorkingSpace):
        download_shard_to_cache(_shard(), missing_root, lambda **_: pytest.fail("downloaded"), disk_usage=disk_usage)
    assert checked == [tmp_path]


def test_read_sidecar_requires_size_and_schema(tmp_path: Path):
    sidecar = tmp_path / "BTC-USD.json"
    sidecar.write_text(json.dumps(_sidecar()))
    assert read_sidecar(sidecar, _shard())["dump_size_bytes"] == 4
    sidecar.write_text("{}")
    with pytest.raises(CBB26IntegrityError, match="dump_size_bytes"):
        read_sidecar(sidecar)


def test_restore_fails_for_count_mismatch_or_bad_temporal_integrity(tmp_path: Path):
    shard = _shard()
    dump = tmp_path / "x.dump"
    dump.write_bytes(b"dump")
    sidecar = _sidecar()

    class Cursor:
        def execute(self, *_): pass
        def fetchone(self): return (0,)
    class Connection:
        def cursor(self): return Cursor()
        def commit(self): pass

    with pytest.raises(CBB26IntegrityError, match="row count"):
        restore_shard(shard, dump, sidecar, Connection(), cache_root=tmp_path, run=lambda *_args, **_kwargs: type("R", (), {"returncode": 0, "stderr": ""})(), disk_usage=lambda _: (0, 0, 2**33))


@pytest.mark.parametrize(
    ("result_values", "message"),
    [([1, 1, 1, 1, 0], "missing anchor"), ([1, 1, 1, 1, 1, 1], "outside sidecar UTC bounds")],
)
def test_restore_rejects_missing_anchor_and_out_of_window_rows(tmp_path: Path, result_values, message: str):
    shard = _shard()
    dump = tmp_path / "x.dump"
    dump.write_bytes(b"dump")
    sidecar = _sidecar()

    class Cursor:
        def execute(self, *_): pass
        def fetchone(self): return (result_values.pop(0),)
    class Connection:
        def cursor(self): return Cursor()
        def commit(self): pytest.fail("must not commit an invalid restore")

    with pytest.raises(CBB26IntegrityError, match=message):
        restore_shard(shard, dump, sidecar, Connection(), cache_root=tmp_path, run=lambda *_args, **_kwargs: type("R", (), {"returncode": 0, "stderr": ""})(), disk_usage=lambda _: (0, 0, 2**33))


@pytest.mark.parametrize(
    "overrides",
    [
        {"day_start_utc": None},
        {"day_end_utc": "not-a-time"},
        {"day_start_utc": "2026-04-24T00:00:00-04:00"},
        {"day_start_utc": "2026-04-25T00:00:00+00:00"},
    ],
)
def test_restore_requires_valid_sidecar_utc_bounds(tmp_path: Path, overrides):
    dump = tmp_path / "x.dump"
    dump.write_bytes(b"dump")
    with pytest.raises(CBB26IntegrityError, match="UTC bounds"):
        restore_shard(
            _shard(), dump, _sidecar(**overrides), object(), cache_root=tmp_path,
            run=lambda *_args, **_kwargs: pytest.fail("restored"), disk_usage=lambda _: (0, 0, 2**33),
        )


def test_restore_runs_pg_restore_inside_compose_with_import_path(tmp_path: Path):
    dump = tmp_path / "c1e89eded9915e1c75a18911298edfbbbe4050ce" / "2026-04-24" / "BTC-USD.dump"
    dump.parent.mkdir(parents=True)
    dump.write_bytes(b"dump")
    commands = []

    class Cursor:
        values = iter([1, 1, 1, 1, 1, 0, 0, 0, 0])
        def execute(self, *_): pass
        def fetchone(self): return (next(self.values),)
    class Connection:
        def cursor(self): return Cursor()
        def commit(self): pass

    restore_shard(
        _shard(), dump, _sidecar(), Connection(), cache_root=tmp_path,
        run=lambda command, **_: commands.append(command) or type("R", (), {"returncode": 0, "stderr": ""})(),
        disk_usage=lambda _: (0, 0, 2**33),
    )
    compose_prefix = [
        "docker", "compose", "-f", str(Path(__file__).resolve().parents[2] / "docker-compose.data.yaml"),
        "exec", "-T", "cbb26-staging",
    ]
    assert commands == [
        compose_prefix + [
            "psql", "--dbname=postgresql://btcspiker:btcspiker@127.0.0.1:5432/btcspiker",
            "--set=ON_ERROR_STOP=1",
            "--command=TRUNCATE cbb26_hf_export_staging.orderbook_replay_anchors, cbb26_hf_export_staging.orderbook_second_deltas, cbb26_hf_export_staging.orderbook_checkpoints, cbb26_hf_export_staging.orderbook_replay_metadata",
        ],
        compose_prefix + [
            "pg_restore", "--data-only", "--no-owner", "--no-privileges",
            "--schema=cbb26_hf_export_staging",
            "--dbname=postgresql://btcspiker:btcspiker@127.0.0.1:5432/btcspiker",
            "/imports/c1e89eded9915e1c75a18911298edfbbbe4050ce/2026-04-24/BTC-USD.dump",
        ],
    ]


def test_restore_nonzero_pg_restore_raises_without_commit(tmp_path: Path):
    dump = tmp_path / "x.dump"
    dump.write_bytes(b"dump")

    class Connection:
        def cursor(self): pytest.fail("must not inspect or commit a failed restore")
        def commit(self): pytest.fail("must not commit")

    results = iter([0, 1])

    with pytest.raises(CBB26IntegrityError, match="pg_restore failed"):
        restore_shard(
            _shard(), dump, _sidecar(), Connection(), cache_root=tmp_path,
            run=lambda *_args, **_kwargs: type("R", (), {"returncode": next(results), "stderr": "bad dump"})(),
            disk_usage=lambda _: (0, 0, 2**33),
        )


def test_remove_cache_requires_verified_partition_receipts(tmp_path: Path):
    shard = _shard()
    shard_dir = tmp_path / "c1e89eded9915e1c75a18911298edfbbbe4050ce" / "2026-04-24"
    shard_dir.mkdir(parents=True)
    (shard_dir / "BTC-USD.dump").write_bytes(b"dump")
    (shard_dir / "BTC-USD.json").write_text("{}")
    artifact = tmp_path / "normalized" / "part.parquet"
    artifact.parent.mkdir()
    artifact.write_bytes(b"normalized")
    with pytest.raises(UnverifiedRemoteArtifact):
        remove_verified_shard_cache(shard, tmp_path, expected_artifacts=[artifact], receipts=[], connection=None)
    assert (shard_dir / "BTC-USD.dump").exists()


def test_remove_cache_verifies_inventory_and_removes_only_target_day(tmp_path: Path):
    shard = _shard()
    target = tmp_path / "c1e89eded9915e1c75a18911298edfbbbe4050ce" / "2026-04-24"
    other = tmp_path / "c1e89eded9915e1c75a18911298edfbbbe4050ce" / "2026-04-25"
    target.mkdir(parents=True)
    other.mkdir(parents=True)
    for directory in (target, other):
        (directory / "BTC-USD.dump").write_bytes(b"dump")
        (directory / "BTC-USD.json").write_text("{}")
    artifact = tmp_path / "normalized" / "part.parquet"
    artifact.parent.mkdir()
    artifact.write_bytes(b"normalized")
    digest = sha256_file(artifact)

    class Cursor:
        def __init__(self): self.executions = []
        def execute(self, sql, params): self.executions.append((sql, params))
    class Connection:
        def __init__(self): self._cursor = Cursor(); self.committed = False
        def cursor(self): return self._cursor
        def commit(self): self.committed = True

    connection = Connection()
    remove_verified_shard_cache(
        shard, tmp_path, expected_artifacts=[artifact],
        receipts=[{"artifact_path": str(artifact), "remote_sha256": digest, "commit_sha": "a" * 40, "success": True}],
        connection=connection,
    )
    assert not target.exists()
    assert other.exists()
    assert artifact.exists()
    assert connection.committed
    assert len(connection._cursor.executions) == 4


def test_remove_cache_rejects_floating_upload_receipt(tmp_path: Path):
    artifact = tmp_path / "part.parquet"
    artifact.write_bytes(b"normalized")
    with pytest.raises(UnverifiedRemoteArtifact, match="commit-pinned"):
        remove_verified_shard_cache(
            _shard(), tmp_path, expected_artifacts=[artifact],
            receipts=[{"artifact_path": str(artifact), "remote_sha256": sha256_file(artifact), "commit_sha": "main", "success": True}],
            connection=None,
        )
