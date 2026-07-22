from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, time, timezone
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
    load_replay_rows,
    process_replay_day,
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


def _daily_artifacts(root: Path):
    artifacts = []
    receipts = []
    for kind in ("book_deltas", "book_states", "trades"):
        source = "coinbase_public_trades" if kind == "trades" else "cbb26"
        for hour in range(24):
            content = f"{kind}-{hour:02d}".encode()
            digest = hashlib.sha256(content).hexdigest()
            relative = Path(
                "raw",
                f"kind={kind}",
                f"source={source}",
                "product=BTC-USD",
                "date=2026-04-24",
                f"hour={hour:02d}",
                f"part-{digest}.parquet",
            )
            artifact = root / relative
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(content)
            artifacts.append(artifact)
            receipts.append(
                {
                    "revision": "a" * 40,
                    "remote_path": relative.as_posix(),
                    "sha256": digest,
                    "size_bytes": len(content),
                    "success": True,
                }
            )
    return artifacts, receipts


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


def test_compose_mount_matches_required_cbb26_cache_root():
    compose = (Path(__file__).resolve().parents[2] / "docker-compose.data.yaml").read_text()
    assert "${BTCSPIKER_CBB26_CACHE_ROOT:-./data/coinbase_history/cache/cbb26}:/imports:ro" in compose


def test_restore_passes_explicit_compose_environment(tmp_path: Path):
    dump = tmp_path / "c1e89eded9915e1c75a18911298edfbbbe4050ce" / "2026-04-24" / "BTC-USD.dump"
    dump.parent.mkdir(parents=True)
    dump.write_bytes(b"dump")
    observed_environments = []

    class Cursor:
        values = iter([1, 1, 1, 1, 1, 0, 0, 0, 0])
        def execute(self, *_): pass
        def fetchone(self): return (next(self.values),)
    class Connection:
        def cursor(self): return Cursor()
        def commit(self): pass

    restore_shard(
        _shard(), dump, _sidecar(), Connection(), cache_root=tmp_path,
        run=lambda _command, **kwargs: observed_environments.append(kwargs.get("env")) or type("R", (), {"returncode": 0, "stderr": ""})(),
        disk_usage=lambda _: (0, 0, 2**33),
        compose_environment={"BTCSPIKER_CBB26_CACHE_ROOT": str(tmp_path)},
    )

    assert observed_environments == [
        {"BTCSPIKER_CBB26_CACHE_ROOT": str(tmp_path)},
        {"BTCSPIKER_CBB26_CACHE_ROOT": str(tmp_path)},
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
        remove_verified_shard_cache(
            shard,
            tmp_path,
            expected_artifacts=[artifact],
            receipts=[{"artifact_path": str(artifact), "remote_sha256": sha256_file(artifact), "commit_sha": "a" * 40, "success": True}],
            connection=None,
        )
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
    artifacts, receipts = _daily_artifacts(tmp_path / "normalized")

    class Cursor:
        def __init__(self): self.executions = []
        def execute(self, sql, params): self.executions.append((sql, params))
    class Connection:
        def __init__(self): self._cursor = Cursor(); self.committed = False
        def cursor(self): return self._cursor
        def commit(self): self.committed = True

    connection = Connection()
    remove_verified_shard_cache(
        shard, tmp_path, expected_artifacts=artifacts, receipts=receipts,
        connection=connection,
    )
    assert not target.exists()
    assert other.exists()
    assert all(artifact.exists() for artifact in artifacts)
    assert connection.committed
    assert len(connection._cursor.executions) == 4
    start = datetime.combine(shard.trade_date, time.min, tzinfo=timezone.utc)
    end = datetime.combine(shard.trade_date, time(23, 59, 59), tzinfo=timezone.utc)
    assert connection._cursor.executions[0][1] == ("BTC-USD", start, end, "BTC-USD", start)
    assert connection._cursor.executions[1][1] == ("BTC-USD", start, end)
    assert connection._cursor.executions[2][1] == ("BTC-USD", start, end)
    assert connection._cursor.executions[3][1] == ("BTC-USD", end, start)
    sql = [statement for statement, _ in connection._cursor.executions]
    assert "product_id=%s AND (anchor_second BETWEEN %s AND %s" in sql[0]
    assert "product_id=%s AND changed_second BETWEEN %s AND %s" in sql[1]
    assert "product_id=%s AND checkpoint_hour BETWEEN %s AND %s" in sql[2]
    assert "product_id=%s AND window_start <= %s AND window_end >= %s" in sql[3]


def test_remove_cache_rejects_missing_hour_and_keeps_dump(tmp_path: Path):
    shard_dir = tmp_path / "c1e89eded9915e1c75a18911298edfbbbe4050ce" / "2026-04-24"
    shard_dir.mkdir(parents=True)
    dump = shard_dir / "BTC-USD.dump"
    dump.write_bytes(b"dump")
    (shard_dir / "BTC-USD.json").write_text("{}")
    artifacts, receipts = _daily_artifacts(tmp_path / "normalized")

    with pytest.raises(UnverifiedRemoteArtifact, match="daily normalized inventory"):
        remove_verified_shard_cache(
            _shard(), tmp_path, expected_artifacts=artifacts[:-1], receipts=receipts[:-1], connection=None,
        )
    assert dump.exists()


def test_remove_cache_rejects_partition_for_another_date(tmp_path: Path):
    artifacts, receipts = _daily_artifacts(tmp_path / "normalized")
    wrong_date = Path(str(artifacts[0]).replace("date=2026-04-24", "date=2026-04-25"))
    wrong_date.parent.mkdir(parents=True)
    artifacts[0].replace(wrong_date)
    artifacts[0] = wrong_date

    with pytest.raises(UnverifiedRemoteArtifact, match="path does not match"):
        remove_verified_shard_cache(
            _shard(), tmp_path, expected_artifacts=artifacts, receipts=receipts, connection=None,
        )


def test_remove_cache_rejects_floating_upload_receipt(tmp_path: Path):
    artifacts, receipts = _daily_artifacts(tmp_path / "normalized")
    receipts[0]["revision"] = "main"
    with pytest.raises(UnverifiedRemoteArtifact, match="commit-pinned"):
        remove_verified_shard_cache(
            _shard(), tmp_path, expected_artifacts=artifacts, receipts=receipts, connection=None,
        )


def test_remove_cache_rejects_extra_receipt(tmp_path: Path):
    artifacts, receipts = _daily_artifacts(tmp_path / "normalized")
    receipts.append({**receipts[0], "remote_path": "raw/unexpected.parquet"})
    with pytest.raises(UnverifiedRemoteArtifact, match="receipt inventory"):
        remove_verified_shard_cache(
            _shard(), tmp_path, expected_artifacts=artifacts, receipts=receipts, connection=None,
        )


def test_remove_cache_requires_explicit_success(tmp_path: Path):
    artifacts, receipts = _daily_artifacts(tmp_path / "normalized")
    receipts[0].pop("success")
    with pytest.raises(UnverifiedRemoteArtifact, match="unsuccessful"):
        remove_verified_shard_cache(
            _shard(), tmp_path, expected_artifacts=artifacts, receipts=receipts, connection=None,
        )


@pytest.mark.parametrize("mutation", ["duplicate", "missing_path"])
def test_remove_cache_rejects_duplicate_or_missing_receipt_path(tmp_path: Path, mutation: str):
    artifacts, receipts = _daily_artifacts(tmp_path / "normalized")
    if mutation == "duplicate":
        receipts[-1]["remote_path"] = receipts[0]["remote_path"]
    else:
        receipts[-1].pop("remote_path")
    with pytest.raises(UnverifiedRemoteArtifact, match="receipt inventory"):
        remove_verified_shard_cache(
            _shard(), tmp_path, expected_artifacts=artifacts, receipts=receipts, connection=None,
        )


def test_remove_cache_rejects_noncanonical_source(tmp_path: Path):
    artifacts, receipts = _daily_artifacts(tmp_path / "normalized")
    wrong_source = Path(str(artifacts[0]).replace("source=cbb26", "source=other"))
    wrong_source.parent.mkdir(parents=True)
    artifacts[0].replace(wrong_source)
    artifacts[0] = wrong_source
    receipts[0]["remote_path"] = receipts[0]["remote_path"].replace("source=cbb26", "source=other")

    with pytest.raises(UnverifiedRemoteArtifact, match="source"):
        remove_verified_shard_cache(
            _shard(), tmp_path, expected_artifacts=artifacts, receipts=receipts, connection=None,
        )


def test_load_replay_rows_uses_bounded_deterministic_queries():
    shard = _shard()
    start = datetime.combine(shard.trade_date, time.min, tzinfo=UTC)
    end = datetime.combine(shard.trade_date, time(23, 59, 59), tzinfo=UTC)
    anchor = ("BTC-USD", start, 100, "90000", "90001", {"90000": "1"}, {"90001": "1"})
    newer_anchor = ("BTC-USD", start, 200, "90000", "90001", {"90000": "2"}, {"90001": "1"})
    newest_checkpoint = ("BTC-USD", start, 300, "90000", "90001", {"90000": "3"}, {"90001": "1"})
    delta = ("BTC-USD", start, 101, 101, "90000", "90001", [["bid", "90000", "2"]])
    metadata = ("BTC-USD", start, end, "complete", 0, {})

    class Cursor:
        def __init__(self):
            self.executions = []
            self.results = iter([[anchor], [newer_anchor], [newest_checkpoint], [delta], [metadata]])
        def execute(self, sql, params): self.executions.append((sql, params))
        def fetchall(self): return next(self.results)
    class Connection:
        def __init__(self): self._cursor = Cursor()
        def cursor(self): return self._cursor

    connection = Connection()
    rows = load_replay_rows(connection, shard)

    assert len(rows.anchors) == 1
    assert rows.anchors[0]["anchor_second"] == start
    assert rows.anchors[0]["source_sequence_num"] == 300
    assert rows.deltas[0]["changes"] == [["bid", "90000", "2"]]
    assert rows.metadata[0]["window_end"] == end
    assert len(connection._cursor.executions) == 5
    assert connection._cursor.executions[0][1] == ("BTC-USD", start)
    assert "ORDER BY anchor_second DESC, source_sequence_num DESC LIMIT 1" in connection._cursor.executions[0][0]
    assert connection._cursor.executions[1][1] == ("BTC-USD", start, end)
    assert connection._cursor.executions[2][1] == ("BTC-USD", start, end)
    assert connection._cursor.executions[3][1] == ("BTC-USD", start, end)
    assert connection._cursor.executions[4][1] == ("BTC-USD", end, start)
    assert all("ORDER BY" in sql for sql, _ in connection._cursor.executions)


def test_process_replay_day_feeds_bounded_rows_to_replay_and_publish(tmp_path: Path):
    shard = _shard()
    start = datetime.combine(shard.trade_date, time.min, tzinfo=UTC)
    end = datetime.combine(shard.trade_date, time(23, 59, 59), tzinfo=UTC)
    replay_calls, publish_calls = [], []
    rows = type("Rows", (), {"anchors": ["anchor"], "deltas": ["delta"], "metadata": ["gap"]})()

    def load(_connection, actual_shard):
        assert actual_shard == shard
        return rows
    def replay(**kwargs):
        replay_calls.append(kwargs)
        return iter(["state"])
    def publish(**kwargs):
        publish_calls.append(kwargs)
        return ["receipt"]

    result = process_replay_day(
        object(), shard, tmp_path, load=load, replay=replay, publish=publish
    )

    assert result == ["receipt"]
    assert replay_calls == [{
        "anchors": ["anchor"], "deltas": ["delta"], "metadata": ["gap"],
        "day_start": start, "day_end": end, "product_id": "BTC-USD",
    }]
    assert publish_calls == [{
        "deltas": ["delta"], "states": ["state"], "root": tmp_path,
        "source_revision": "c1e89eded9915e1c75a18911298edfbbbe4050ce",
        "source_date": "2026-04-24",
    }]
