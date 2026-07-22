"""Pinned, one-day-at-a-time acquisition of CBB26 replay shards."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
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
SIDECAR_SCHEMA = "cbb26_timeseries_shard_manifest_v1"
COMPOSE_FILE = Path(__file__).resolve().parents[1] / "docker-compose.data.yaml"
CONTAINER_DATABASE_URL = "postgresql://btcspiker:btcspiker@127.0.0.1:5432/btcspiker"


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
    existing = cache_root
    while not existing.exists() and existing.parent != existing:
        existing = existing.parent
    usage = disk_usage(existing)
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


def _sidecar_bounds(value: Mapping[str, Any], shard: CBB26Shard | None) -> tuple[datetime, datetime]:
    try:
        trade_date = date.fromisoformat(str(value["trade_date"]))
        start = datetime.fromisoformat(str(value["day_start_utc"]))
        end = datetime.fromisoformat(str(value["day_end_utc"]))
    except (KeyError, TypeError, ValueError) as error:
        raise CBB26IntegrityError("sidecar has invalid UTC bounds") from error
    expected_start = datetime.combine(trade_date, time.min, tzinfo=timezone.utc)
    expected_end = datetime.combine(trade_date, time(23, 59, 59), tzinfo=timezone.utc)
    if (
        start.utcoffset() != timedelta(0)
        or end.utcoffset() != timedelta(0)
        or start != expected_start
        or end != expected_end
        or (shard is not None and trade_date != shard.trade_date)
    ):
        raise CBB26IntegrityError("sidecar has invalid UTC bounds for shard date")
    return start, end


def _validate_sidecar(value: Mapping[str, Any], shard: CBB26Shard | None = None) -> dict[str, Any]:
    if not isinstance(value.get("dump_size_bytes"), int) or isinstance(value.get("dump_size_bytes"), bool):
        raise CBB26IntegrityError("sidecar missing dump_size_bytes")
    if value.get("schema") != SIDECAR_SCHEMA:
        raise CBB26IntegrityError(f"sidecar must declare schema={SIDECAR_SCHEMA}")
    if value.get("export_schema") != STAGING_SCHEMA:
        raise CBB26IntegrityError(f"sidecar must declare export_schema={STAGING_SCHEMA}")
    if not isinstance(value.get("product_id"), str) or not value["product_id"]:
        raise CBB26IntegrityError("sidecar missing product_id")
    _sidecar_bounds(value, shard)
    counts = value.get("row_counts")
    if not isinstance(counts, Mapping) or set(counts) != set(STAGING_TABLES):
        raise CBB26IntegrityError("sidecar row_counts must cover all staging tables")
    if any(not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in counts.values()):
        raise CBB26IntegrityError("sidecar row_counts must be non-negative integers")
    if shard is not None:
        if value["product_id"] != shard.product_id:
            raise CBB26IntegrityError("sidecar product_id does not match shard")
        if value.get("dump_relpath") != shard.dump_path:
            raise CBB26IntegrityError("sidecar dump_relpath does not match shard")
        if value["dump_size_bytes"] != shard.expected_size:
            raise CBB26IntegrityError("sidecar and Hub dump sizes disagree")
    return dict(value)


def read_sidecar(path: str | Path, shard: CBB26Shard | None = None) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CBB26IntegrityError(f"invalid sidecar {path}: {error}") from error
    if not isinstance(value, dict):
        raise CBB26IntegrityError("sidecar must contain a JSON object")
    return _validate_sidecar(value, shard)


def download_shard_to_cache(
    shard: CBB26Shard,
    cache_root: str | Path,
    download: Callable[..., str] | None = None,
    *,
    disk_usage: Callable[[str | Path], Any] = shutil.disk_usage,
) -> Path:
    """Acquire and validate one pinned daily dump plus its JSON sidecar."""
    root = Path(cache_root)
    target, sidecar_target = _local_paths(shard, root)
    dump_matches = target.is_file() and target.stat().st_size == shard.expected_size
    if dump_matches and sidecar_target.is_file():
        sidecar = read_sidecar(sidecar_target, shard)
        if sidecar["dump_size_bytes"] == target.stat().st_size:
            return target
    _required_space(root, disk_usage)
    target.parent.mkdir(parents=True, exist_ok=True)
    if download is None:
        from huggingface_hub import hf_hub_download

        download = hf_hub_download
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".download-") as temporary:
        temporary_root = Path(temporary)
        downloaded_sidecar = Path(
            download(
                repo_id=CBB26_REPO,
                repo_type="dataset",
                revision=CBB26_REVISION,
                filename=shard.sidecar_path,
                local_dir=str(temporary_root),
            )
        )
        sidecar = read_sidecar(downloaded_sidecar, shard)
        downloaded_dump: Path | None = None
        local_size = target.stat().st_size if dump_matches else None
        if not dump_matches:
            downloaded_dump = Path(
                download(
                    repo_id=CBB26_REPO,
                    repo_type="dataset",
                    revision=CBB26_REVISION,
                    filename=shard.dump_path,
                    local_dir=str(temporary_root),
                )
            )
            local_size = downloaded_dump.stat().st_size
        if local_size != shard.expected_size or sidecar["dump_size_bytes"] != local_size:
            raise CBB26IntegrityError(
                f"download size mismatch for {shard.dump_path}: observed={local_size} expected={shard.expected_size}"
            )
        if downloaded_dump is not None:
            os.replace(downloaded_dump, target)
        os.replace(downloaded_sidecar, sidecar_target)
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
    database_url: str = CONTAINER_DATABASE_URL,
) -> dict[str, int]:
    """Restore a custom dump into plain PostgreSQL and validate every source count."""
    validated_sidecar = _validate_sidecar(sidecar, shard)
    start, end = _sidecar_bounds(validated_sidecar, shard)
    root = Path(cache_root or Path(dump_path).parent)
    _required_space(root, disk_usage)
    path = Path(dump_path)
    if path.stat().st_size != shard.expected_size:
        raise CBB26IntegrityError("local dump size does not match Hub tree and sidecar")
    try:
        import_path = Path("/imports") / path.relative_to(root)
    except ValueError as error:
        raise CBB26IntegrityError("dump path must be within cache_root") from error
    required_counts = validated_sidecar["row_counts"]
    compose_prefix = [
        "docker", "compose", "-f", str(COMPOSE_FILE), "exec", "-T", "cbb26-staging",
    ]
    truncate_sql = f"TRUNCATE {', '.join(f'{STAGING_SCHEMA}.{table}' for table in STAGING_TABLES)}"
    truncate_result = run(
        compose_prefix
        + [
            "psql", f"--dbname={database_url}", "--set=ON_ERROR_STOP=1",
            f"--command={truncate_sql}",
        ],
        capture_output=True,
        text=True,
    )
    if truncate_result.returncode:
        raise CBB26IntegrityError(f"staging truncate failed: {getattr(truncate_result, 'stderr', '')}")
    result = run(
        compose_prefix
        + [
            "pg_restore", "--data-only", "--no-owner", "--no-privileges",
            f"--schema={STAGING_SCHEMA}", f"--dbname={database_url}", str(import_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise CBB26IntegrityError(f"pg_restore failed: {getattr(result, 'stderr', '')}")
    cursor = connection.cursor()
    counts = {table: _count(cursor, table) for table in STAGING_TABLES}
    for table, actual in counts.items():
        expected = required_counts.get(table)
        if actual != expected:
            raise CBB26IntegrityError(f"row count mismatch for {table}: observed={actual} expected={expected}")
    cursor.execute(
        f"SELECT count(*) FROM {STAGING_SCHEMA}.orderbook_replay_anchors WHERE product_id=%s AND anchor_second <= %s",
        (shard.product_id, start),
    )
    if int(cursor.fetchone()[0]) < 1:
        raise CBB26IntegrityError("missing anchor at or before day start")
    bound_checks = (
        ("orderbook_replay_anchors", "anchor_second > %s", (end,)),
        ("orderbook_second_deltas", "changed_second < %s OR changed_second > %s", (start, end)),
        ("orderbook_checkpoints", "checkpoint_hour < %s OR checkpoint_hour > %s", (start, end)),
        (
            "orderbook_replay_metadata",
            "window_start < %s OR window_start > %s OR window_end < %s OR window_end > %s OR window_end < window_start",
            (start, end, start, end),
        ),
    )
    for table, predicate, bounds in bound_checks:
        cursor.execute(
            f"SELECT count(*) FROM {STAGING_SCHEMA}.{table} WHERE product_id=%s AND ({predicate})",
            (shard.product_id, *bounds),
        )
        if int(cursor.fetchone()[0]):
            raise CBB26IntegrityError(f"row outside sidecar UTC bounds in {table}")
    connection.commit()
    return counts


def remove_verified_shard_cache(
    shard: CBB26Shard,
    cache_root: str | Path,
    expected_artifacts: Sequence[str | Path],
    receipts: Sequence[Mapping[str, Any]],
    connection: Any,
) -> None:
    """Delete one transient shard after its explicit normalized inventory is verified."""
    expected = [Path(path).resolve() for path in expected_artifacts]
    if not expected or len(set(expected)) != len(expected):
        raise UnverifiedRemoteArtifact("expected artifact inventory must be non-empty and unique")
    receipts_by_path: dict[Path, list[Mapping[str, Any]]] = {}
    for receipt in receipts:
        if "artifact_path" in receipt:
            receipts_by_path.setdefault(Path(str(receipt["artifact_path"])).resolve(), []).append(receipt)
    for artifact in expected:
        matches = receipts_by_path.get(artifact, [])
        if len(matches) != 1 or not artifact.is_file():
            raise UnverifiedRemoteArtifact(f"missing unique upload receipt for {artifact}")
        receipt = matches[0]
        if receipt.get("success") is not True or not isinstance(receipt.get("commit_sha"), str) or re.fullmatch(r"[0-9a-f]{40}", receipt["commit_sha"]) is None:
            raise UnverifiedRemoteArtifact(f"upload receipt is not commit-pinned for {artifact}")
        if receipt.get("remote_sha256") != sha256_file(artifact):
            raise UnverifiedRemoteArtifact(f"remote digest mismatch for {artifact}")
    start = datetime.combine(shard.trade_date, time.min, tzinfo=timezone.utc)
    end = datetime.combine(shard.trade_date, time(23, 59, 59), tzinfo=timezone.utc)
    if connection is not None:
        cursor = connection.cursor()
        cursor.execute(
            f"DELETE FROM {STAGING_SCHEMA}.orderbook_replay_anchors WHERE product_id=%s AND (anchor_second BETWEEN %s AND %s OR anchor_second=(SELECT max(anchor_second) FROM {STAGING_SCHEMA}.orderbook_replay_anchors WHERE product_id=%s AND anchor_second <= %s))",
            (shard.product_id, start, end, shard.product_id, start),
        )
        cursor.execute(
            f"DELETE FROM {STAGING_SCHEMA}.orderbook_second_deltas WHERE product_id=%s AND changed_second BETWEEN %s AND %s",
            (shard.product_id, start, end),
        )
        cursor.execute(
            f"DELETE FROM {STAGING_SCHEMA}.orderbook_checkpoints WHERE product_id=%s AND checkpoint_hour BETWEEN %s AND %s",
            (shard.product_id, start, end),
        )
        cursor.execute(
            f"DELETE FROM {STAGING_SCHEMA}.orderbook_replay_metadata WHERE product_id=%s AND window_start <= %s AND window_end >= %s",
            (shard.product_id, end, start),
        )
        connection.commit()
    dump, sidecar = _local_paths(shard, Path(cache_root))
    for path in (dump, sidecar):
        if path.exists():
            path.unlink()
    try:
        dump.parent.rmdir()
    except OSError:
        pass
