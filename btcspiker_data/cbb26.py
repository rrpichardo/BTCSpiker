"""Pinned, one-day-at-a-time acquisition of CBB26 replay shards."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable, Iterable, Mapping, Sequence


CBB26_REPO = "deusmos/cbb26-timeseries-db"
CBB26_REVISION = "c1e89eded9915e1c75a18911298edfbbbe4050ce"
DEFAULT_START = date(2026, 4, 24)
DEFAULT_END = date(2026, 5, 28)
WORKING_SPACE_FLOOR = 4_294_967_296
STAGING_SCHEMA = "cbb26_hf_export_staging"
STAGING_TABLES = (
    "orderbook_replay_anchors",
    "orderbook_second_deltas",
    "orderbook_checkpoints",
    "orderbook_replay_metadata",
)


class CBB26IntegrityError(RuntimeError):
    """A Hub shard or its restored contents do not meet the pinned contract."""


class InsufficientWorkingSpace(CBB26IntegrityError):
    """The one-day operation cannot safely start on the available disk."""


class UnverifiedRemoteArtifact(CBB26IntegrityError):
    """A local shard cannot be deleted until all normalized outputs are verified."""


@dataclass(frozen=True)
class CBB26Shard:
    trade_date: date
    product_id: str
    dump_path: str
    sidecar_path: str
    expected_size: int


def _node_value(node: object, name: str) -> Any:
    return node.get(name) if isinstance(node, Mapping) else getattr(node, name, None)


def _required_space(cache_root: Path, disk_usage: Callable[[str | Path], Any]) -> None:
    usage = disk_usage(cache_root)
    free = usage.free if hasattr(usage, "free") else usage[2]
    if free < WORKING_SPACE_FLOOR:
        raise InsufficientWorkingSpace(
            f"insufficient working space: observed={free} required={WORKING_SPACE_FLOOR}"
        )


def list_btc_shards(
    tree: Iterable[object], *, start: date = DEFAULT_START, end: date = DEFAULT_END, product: str = "BTC-USD"
) -> list[CBB26Shard]:
    """Build the exact daily inventory from a pinned Hub tree response."""
    if end < start:
        raise ValueError("end must not precede start")
    entries = {str(_node_value(node, "path")): _node_value(node, "size") for node in tree}
    shards: list[CBB26Shard] = []
    day = start
    while day <= end:
        root = f"data/{day.isoformat()}/{product}"
        dump_path, sidecar_path = f"{root}.dump", f"{root}.json"
        size = entries.get(dump_path)
        if not isinstance(size, int) or sidecar_path not in entries:
            raise CBB26IntegrityError(f"missing dump or sidecar for {product} on {day.isoformat()}")
        if size <= 0:
            raise CBB26IntegrityError(f"invalid dump size for {dump_path}: {size}")
        shards.append(CBB26Shard(day, product, dump_path, sidecar_path, size))
        day += timedelta(days=1)
    return shards


def discover_btc_shards(
    *, start: date = DEFAULT_START, end: date = DEFAULT_END, product: str = "BTC-USD", api: Any = None
) -> list[CBB26Shard]:
    """Query Hugging Face using the immutable dataset revision."""
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    tree = api.list_repo_tree(CBB26_REPO, repo_type="dataset", recursive=True, revision=CBB26_REVISION)
    return list_btc_shards(tree, start=start, end=end, product=product)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_paths(shard: CBB26Shard, cache_root: Path) -> tuple[Path, Path]:
    directory = cache_root / CBB26_REVISION / shard.trade_date.isoformat()
    return directory / f"{shard.product_id}.dump", directory / f"{shard.product_id}.json"


def read_sidecar(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CBB26IntegrityError(f"invalid sidecar {path}: {error}") from error
    if not isinstance(value, dict) or not isinstance(value.get("dump_size_bytes"), int):
        raise CBB26IntegrityError("sidecar missing dump_size_bytes")
    if value.get("export_schema") != STAGING_SCHEMA:
        raise CBB26IntegrityError(f"sidecar must declare export_schema={STAGING_SCHEMA}")
    return value


def download_shard_to_cache(
    shard: CBB26Shard,
    cache_root: str | Path,
    download: Callable[..., str] | None = None,
    *,
    disk_usage: Callable[[str | Path], Any] = shutil.disk_usage,
) -> Path:
    """Download only a missing daily dump and preserve a matching partial cache file."""
    root = Path(cache_root)
    target, sidecar_target = _local_paths(shard, root)
    if target.is_file() and target.stat().st_size == shard.expected_size:
        if sidecar_target.is_file():
            sidecar = read_sidecar(sidecar_target)
            if sidecar["dump_size_bytes"] != shard.expected_size:
                raise CBB26IntegrityError("sidecar and Hub dump sizes disagree")
        return target
    _required_space(root, disk_usage)
    target.parent.mkdir(parents=True, exist_ok=True)
    if download is None:
        from huggingface_hub import hf_hub_download

        download = hf_hub_download
    temporary_root = target.parent / ".download"
    downloaded = Path(
        download(
            repo_id=CBB26_REPO,
            repo_type="dataset",
            revision=CBB26_REVISION,
            filename=shard.dump_path,
            local_dir=str(temporary_root),
        )
    )
    if downloaded.stat().st_size != shard.expected_size:
        raise CBB26IntegrityError(
            f"download size mismatch for {shard.dump_path}: observed={downloaded.stat().st_size} expected={shard.expected_size}"
        )
    # os.replace is atomic; a valid existing cache file was returned above and is never overwritten.
    os.replace(downloaded, target)
    return target


def _count(cursor: Any, table: str) -> int:
    cursor.execute(f"SELECT count(*) FROM {STAGING_SCHEMA}.{table}")
    return int(cursor.fetchone()[0])


def restore_shard(
    shard: CBB26Shard,
    dump_path: str | Path,
    sidecar: Mapping[str, Any],
    connection: Any,
    *,
    run: Callable[..., Any] = subprocess.run,
    disk_usage: Callable[[str | Path], Any] = shutil.disk_usage,
    cache_root: str | Path | None = None,
    database_url: str = "postgresql://btcspiker:btcspiker@127.0.0.1:55432/btcspiker",
) -> dict[str, int]:
    """Restore a custom dump into plain PostgreSQL and validate every source count."""
    _required_space(Path(cache_root or Path(dump_path).parent), disk_usage)
    required_counts = sidecar.get("row_counts")
    if not isinstance(required_counts, Mapping):
        raise CBB26IntegrityError("sidecar missing row_counts")
    cursor = connection.cursor()
    cursor.execute(f"TRUNCATE {', '.join(f'{STAGING_SCHEMA}.{table}' for table in STAGING_TABLES)}")
    result = run(
        [
            "pg_restore", "--data-only", "--no-owner", "--no-privileges",
            f"--schema={STAGING_SCHEMA}", f"--dbname={database_url}", str(dump_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise CBB26IntegrityError(f"pg_restore failed: {getattr(result, 'stderr', '')}")
    counts = {table: _count(cursor, table) for table in STAGING_TABLES}
    for table, actual in counts.items():
        expected = required_counts.get(table)
        if actual != expected:
            raise CBB26IntegrityError(f"row count mismatch for {table}: observed={actual} expected={expected}")
    day_start = sidecar.get("day_start_utc")
    day_end = sidecar.get("day_end_utc")
    if day_start and day_end:
        cursor.execute(
            f"SELECT count(*) FROM {STAGING_SCHEMA}.orderbook_replay_anchors WHERE product_id=%s AND anchor_second <= %s",
            (shard.product_id, day_start),
        )
        if int(cursor.fetchone()[0]) < 1:
            raise CBB26IntegrityError("missing anchor at or before day start")
        cursor.execute(
            f"SELECT count(*) FROM {STAGING_SCHEMA}.orderbook_second_deltas WHERE product_id=%s AND (changed_second < %s OR changed_second > %s)",
            (shard.product_id, day_start, day_end),
        )
        if int(cursor.fetchone()[0]):
            raise CBB26IntegrityError("row outside sidecar UTC bounds")
    connection.commit()
    return counts


def remove_verified_shard_cache(
    shard: CBB26Shard, cache_root: str | Path, receipts: Sequence[Mapping[str, Any]], connection: Any
) -> None:
    """Delete one day's transient dump only after all 24 hourly artifacts verify remotely."""
    day = shard.trade_date.isoformat()
    eligible = [r for r in receipts if str(r.get("date", r.get("source_date", ""))) == day]
    by_hour = {int(r["hour"]): r for r in eligible if "hour" in r}
    if set(by_hour) != set(range(24)) or any(r.get("sha256") != r.get("remote_sha256") for r in by_hour.values()):
        raise UnverifiedRemoteArtifact(f"unverified remote artifacts for {day}")
    dump, sidecar = _local_paths(shard, Path(cache_root))
    for path in (dump, sidecar):
        if path.exists():
            path.unlink()
    if connection is not None:
        cursor = connection.cursor()
        cursor.execute(
            f"DELETE FROM {STAGING_SCHEMA}.orderbook_replay_anchors WHERE product_id=%s",
            (shard.product_id,),
        )
        cursor.execute(
            f"DELETE FROM {STAGING_SCHEMA}.orderbook_second_deltas WHERE product_id=%s",
            (shard.product_id,),
        )
        cursor.execute(
            f"DELETE FROM {STAGING_SCHEMA}.orderbook_checkpoints WHERE product_id=%s",
            (shard.product_id,),
        )
        cursor.execute(
            f"DELETE FROM {STAGING_SCHEMA}.orderbook_replay_metadata WHERE product_id=%s",
            (shard.product_id,),
        )
        connection.commit()
