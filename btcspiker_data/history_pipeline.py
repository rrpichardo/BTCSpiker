"""One-day-at-a-time acquisition into a private, immutable Hub dataset."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
import json
import hashlib
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable

import pandas as pd
import pyarrow as pa
import requests

from .book_replay import publish_replay_partitions, replay_day
from .cbb26 import (
    CBB26_REPO,
    CBB26_REVISION,
    CBB26Shard,
    COMPOSE_FILE,
    DEFAULT_END,
    DEFAULT_START,
    discover_btc_shards,
    download_shard_to_cache,
    load_replay_rows,
    read_sidecar,
    remove_verified_shard_cache,
    restore_shard,
)
from .coinbase_trades import CoinbaseTradeClient, TradeEvent
from .contracts import RAW_BOOK_COLUMNS, RAW_TRADE_COLUMNS
from .hub_storage import PrivateHubStore, UploadReceipt
from .materialize import join_trades_to_books, materialize_segmented_features
from .quality import QUALIFIED_SECONDS_MIN, QualityReport, audit_dataset, write_quality_report
from .raw_manifest import (
    RawDatasetManifest,
    raw_manifest_id,
    publish_raw_manifest,
    serialize_trade_day_completion,
)
from .storage import PartitionRecord, write_empty_partition_atomic, write_partition_atomic


UTC = timezone.utc


@dataclass(frozen=True)
class HistoryDownloadConfig:
    cache_root: Path
    start: date = DEFAULT_START
    end: date = DEFAULT_END
    product: str = "BTC-USD"
    revision: str = CBB26_REVISION
    max_rps: int = 8

    def __post_init__(self) -> None:
        object.__setattr__(self, "cache_root", self.cache_root.expanduser().resolve())
        if self.revision != CBB26_REVISION:
            raise ValueError("CBB26 revision must remain pinned")
        if self.product != "BTC-USD":
            raise ValueError("only BTC-USD is supported by the frozen pipeline")
        if self.start > self.end:
            raise ValueError("start date must not follow end date")
        if self.start < DEFAULT_START or self.end > DEFAULT_END:
            raise ValueError("dates must remain inside the reviewed 2026-04-24 through 2026-05-28 window")
        if self.max_rps != 8:
            raise ValueError("the reviewed public rate limit is exactly 8 requests/second")


@dataclass(frozen=True)
class DownloadSummary:
    dataset_id: str
    repo_id: str
    revision: str
    manifest_path: Path
    quality_report_path: Path
    quality_status: str
    qualified_seconds: int
    downloaded_files: int
    uploaded_files: int
    reused_files: int
    bytes_downloaded: int


@dataclass
class _DayResult:
    source_date: date
    partitions: list[PartitionRecord]
    receipts: list[UploadReceipt]
    manifest_partitions: list[dict[str, Any]]
    completion: dict[str, Any]
    report: QualityReport
    gaps: list[dict[str, datetime]]


def _hourly_trade_partitions(
    trades: Iterable[TradeEvent], root: Path, source_date: date, product: str
) -> list[PartitionRecord]:
    grouped: dict[int, list[TradeEvent]] = {}
    for trade in trades:
        grouped.setdefault(trade.event_time.hour, []).append(trade)
    records: list[PartitionRecord] = []
    day_start = datetime.combine(source_date, time.min, tzinfo=UTC)
    for hour in range(24):
        rows = grouped.get(hour, [])
        if not rows:
            records.append(
                write_empty_partition_atomic(
                    root,
                    "trades",
                    product,
                    source="coinbase_public_trades",
                    hour=day_start + timedelta(hours=hour),
                )
            )
            continue
        values = {
            "source": ["coinbase_public_trades"] * len(rows),
            "product_id": [product] * len(rows),
            "trade_id": [row.trade_id for row in rows],
            "event_time": [row.event_time for row in rows],
            "price": [str(row.price) for row in rows],
            "size": [str(row.size) for row in rows],
            "reported_side": [row.reported_side for row in rows],
            "side_semantics": [row.side_semantics for row in rows],
            "source_date": [source_date.isoformat()] * len(rows),
        }
        table = pa.table({name: values[name] for name in RAW_TRADE_COLUMNS})
        records.append(write_partition_atomic(table, root, "trades", product))
    return records


def _state_intervals(states: Iterable[Any]) -> list[dict[str, datetime]]:
    ordered = sorted(states, key=lambda row: (row.segment_id, row.observed_through))
    intervals: list[dict[str, datetime]] = []
    start: datetime | None = None
    previous: datetime | None = None
    segment: int | None = None
    for state in ordered:
        if start is None or state.segment_id != segment or state.observed_through != previous + timedelta(seconds=1):
            if start is not None and previous is not None:
                intervals.append({"start": start, "end": previous + timedelta(seconds=1)})
            start = state.observed_through
            segment = state.segment_id
        previous = state.observed_through
    if start is not None and previous is not None:
        intervals.append({"start": start, "end": previous + timedelta(seconds=1)})
    return intervals


def _gap_intervals(metadata: Iterable[dict[str, Any]]) -> list[dict[str, datetime]]:
    result = []
    for row in metadata:
        status = str(row.get("status", "")).lower()
        if status in {"gap", "excluded", "incomplete", "missing", "error"} or int(row.get("gap_count", 0)) > 0:
            result.append({"start": row["window_start"], "end": row["window_end"]})
    return result


def _receipt_payload(receipt: UploadReceipt) -> dict[str, Any]:
    return {**asdict(receipt), "success": True}


def _remote_path(record: PartitionRecord, normalized_root: Path) -> str:
    try:
        return record.path.relative_to(normalized_root).as_posix()
    except ValueError as error:
        raise ValueError("normalized partition escaped its data root") from error


def _combine_day_reports(reports: list[QualityReport]) -> QualityReport:
    qualified = sum(report.qualified_seconds for report in reports)
    failures = {failure for report in reports for failure in report.failures}
    if qualified < QUALIFIED_SECONDS_MIN:
        failures.add("qualified_seconds_below_minimum")
    per_day = {key: value for report in reports for key, value in report.per_day.items()}
    first_values = [report.first_event for report in reports if report.first_event is not None]
    last_values = [report.last_event for report in reports if report.last_event is not None]
    sequences = [report.sequence_range for report in reports if report.sequence_range is not None]
    return QualityReport(
        status="PASS" if not failures else "FAIL",
        qualified_seconds=qualified,
        calendar_span_seconds=sum(report.calendar_span_seconds for report in reports),
        per_day=per_day,
        failures=tuple(sorted(failures)),
        duplicate_count=sum(report.duplicate_count for report in reports),
        first_event=min(first_values, default=None),
        last_event=max(last_values, default=None),
        sequence_range=(min(value[0] for value in sequences), max(value[1] for value in sequences)) if sequences else None,
        gap_incidents=tuple(item for report in reports for item in report.gap_incidents),
        exclusions=tuple(item for report in reports for item in report.exclusions),
    )


def _manifest_partition(record: PartitionRecord, receipt: UploadReceipt, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "remote_path": receipt.remote_path,
        "rows": record.row_count,
        "sha256": record.sha256,
        "verified_receipt": _receipt_payload(receipt),
    }


def raw_manifest_payload_id(payload: dict[str, Any]) -> str:
    """Recompute the immutable raw identity without creation-time noise."""
    identity = dict(payload)
    identity.pop("created_at", None)
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _feature_engine_git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError("feature engine revision is not an exact git SHA")
    return value


def run_history_download(config: HistoryDownloadConfig) -> DownloadSummary:
    """Acquire, audit, upload, and manifest the configured pinned window."""
    store = PrivateHubStore.connect()
    cbb_root = config.cache_root / "cbb26"
    compose_environment = {
        **os.environ,
        "BTCSPIKER_CBB26_CACHE_ROOT": str(cbb_root),
    }
    subprocess.run(
        [
            "docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d",
            "--wait", "--wait-timeout", "120", "cbb26-staging",
        ],
        check=True,
        env=compose_environment,
    )
    try:
        return _run_connected_history_download(config, store, compose_environment)
    finally:
        subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), "stop", "cbb26-staging"],
            check=False,
            env=compose_environment,
        )


def _run_connected_history_download(
    config: HistoryDownloadConfig,
    store: PrivateHubStore,
    compose_environment: dict[str, str],
) -> DownloadSummary:
    shards = discover_btc_shards(start=config.start, end=config.end, product=config.product)
    config.cache_root.mkdir(parents=True, exist_ok=True)
    cbb_root = config.cache_root / "cbb26"
    normalized_root = config.cache_root / "normalized"
    reports_root = config.cache_root / "reports"
    import psycopg

    connection = psycopg.connect("postgresql://btcspiker:btcspiker@127.0.0.1:55432/btcspiker")
    results: list[_DayResult] = []
    downloaded_bytes = 0
    downloaded_files = 0
    reused_files = 0
    with connection, requests.Session() as session:
        for shard in shards:
            source_directory = cbb_root / CBB26_REVISION / shard.trade_date.isoformat()
            source_was_cached = all(
                path.is_file()
                for path in (
                    source_directory / f"{shard.product_id}.dump",
                    source_directory / f"{shard.product_id}.json",
                )
            )
            dump = download_shard_to_cache(shard, cbb_root)
            sidecar_path = dump.with_suffix(".json")
            sidecar = read_sidecar(sidecar_path, shard)
            restore_shard(
                shard,
                dump,
                sidecar,
                connection,
                cache_root=cbb_root,
                compose_environment=compose_environment,
            )
            replay_rows = load_replay_rows(connection, shard)
            day_start = datetime.combine(shard.trade_date, time.min, tzinfo=UTC)
            day_end = day_start + timedelta(days=1) - timedelta(seconds=1)
            states = list(replay_day(
                anchors=replay_rows.anchors,
                deltas=replay_rows.deltas,
                metadata=replay_rows.metadata,
                day_start=day_start,
                day_end=day_end,
                product_id=shard.product_id,
            ))
            records = publish_replay_partitions(
                deltas=replay_rows.deltas,
                states=states,
                root=normalized_root,
                source_revision=CBB26_REVISION,
                source_date=shard.trade_date.isoformat(),
            )
            trade_client = CoinbaseTradeClient(
                session,
                product_id=config.product,
                max_rps=config.max_rps,
            )
            trades = list(trade_client.iter_day_trades(shard.trade_date))
            if trade_client.last_completion is None:
                raise RuntimeError(f"missing trade completion evidence for {shard.trade_date}")
            records.extend(_hourly_trade_partitions(trades, normalized_root, shard.trade_date, config.product))
            receipts = [store.upload_partition(record, _remote_path(record, normalized_root)) for record in records]
            kinds = [record.path.relative_to(normalized_root).parts[1].split("=", 1)[1] for record in records]
            manifest_partitions = [
                _manifest_partition(record, receipt, kind)
                for record, receipt, kind in zip(records, receipts, kinds)
            ]
            completion = serialize_trade_day_completion(trade_client.last_completion)
            gaps = _gap_intervals(replay_rows.metadata)
            day_manifest = RawDatasetManifest(
                source_revision=CBB26_REVISION,
                source_url=f"https://huggingface.co/datasets/{CBB26_REPO}",
                repo_id=store.repo_id,
                revision=receipts[-1].revision,
                usage_scope="research_unverified",
                schemas={"book": list(RAW_BOOK_COLUMNS), "trades": list(RAW_TRADE_COLUMNS)},
                partitions=[{**item, "local_path": str(record.path)} for item, record in zip(manifest_partitions, records)],
                coverage_seconds=0,
                missing_seconds=0,
                duplicate_counts={},
                sequence_incidents=gaps,
                excluded_intervals=gaps,
                created_at=datetime.now(UTC),
                trade_day_completions=[completion],
            )
            joined = join_trades_to_books(trades, states)
            report = audit_dataset(
                day_manifest,
                book_intervals=_state_intervals(states),
                trades=trades,
                book_states=states,
                replay_incidents=gaps,
                joined_ticks=joined.to_dict("records"),
                output_dir=reports_root / shard.trade_date.isoformat(),
                minimum_qualified_seconds=0,
            )
            results.append(_DayResult(shard.trade_date, records, receipts, manifest_partitions, completion, report, gaps))
            if source_was_cached:
                reused_files += 2
            else:
                downloaded_files += 2
                downloaded_bytes += shard.expected_size + sidecar_path.stat().st_size
            reused_files += sum(receipt.reused for receipt in receipts)
            if report.status == "PASS":
                remove_verified_shard_cache(
                    shard,
                    cbb_root,
                    [record.path for record in records],
                    [_receipt_payload(receipt) for receipt in receipts],
                    connection=connection,
                )
                for record in records:
                    record.path.unlink(missing_ok=True)
            else:
                break

    combined = _combine_day_reports([result.report for result in results])
    write_quality_report(combined, reports_root / "final")
    all_partitions = [item for result in results for item in result.manifest_partitions]
    all_completions = [result.completion for result in results]
    all_gaps = [gap for result in results for gap in result.gaps]
    manifest = RawDatasetManifest(
        source_revision=CBB26_REVISION,
        source_url=f"https://huggingface.co/datasets/{CBB26_REPO}",
        repo_id=store.repo_id,
        revision=results[-1].receipts[-1].revision,
        usage_scope="research_unverified",
        schemas={"book": list(RAW_BOOK_COLUMNS), "trades": list(RAW_TRADE_COLUMNS)},
        partitions=all_partitions,
        coverage_seconds=combined.qualified_seconds,
        missing_seconds=max(0, ((config.end - config.start).days + 1) * 86_400 - combined.qualified_seconds),
        duplicate_counts={"trades": combined.duplicate_count},
        sequence_incidents=all_gaps,
        excluded_intervals=all_gaps,
        created_at=datetime.now(UTC),
        trade_day_completions=all_completions,
    )
    dataset_id = raw_manifest_id(manifest)
    manifest_dir = config.cache_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{dataset_id}.json"
    manifest_path.write_text(json.dumps(asdict(manifest), default=lambda value: value.isoformat(), sort_keys=True, indent=2) + "\n")
    final_revision = results[-1].receipts[-1].revision
    uploaded_files = sum(not receipt.reused for result in results for receipt in result.receipts)
    if combined.status == "PASS":
        manifest_receipt = publish_raw_manifest(manifest, store)
        final_revision = manifest_receipt.revision
        reused_files += int(manifest_receipt.reused)
        uploaded_files += int(not manifest_receipt.reused)
    return DownloadSummary(
        dataset_id=dataset_id,
        repo_id=store.repo_id,
        revision=final_revision,
        manifest_path=manifest_path.resolve(),
        quality_report_path=(reports_root / "final" / "quality.json").resolve(),
        quality_status=combined.status,
        qualified_seconds=combined.qualified_seconds,
        downloaded_files=downloaded_files,
        uploaded_files=uploaded_files,
        reused_files=reused_files,
        bytes_downloaded=downloaded_bytes,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _remote_digest(info: Any) -> str | None:
    lfs = getattr(info, "lfs", None)
    if isinstance(lfs, dict):
        return lfs.get("sha256")
    return getattr(lfs, "sha256", None)


def load_verified_manifest(path: Path, *, api: Any = None) -> dict[str, Any]:
    """Validate a raw manifest and every local or remote referenced artifact."""
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid raw manifest: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("raw manifest must be an object")
    required = {
        "source_revision", "repo_id", "revision", "usage_scope", "partitions",
        "excluded_intervals", "created_at",
    }
    if not required.issubset(payload):
        raise ValueError("raw manifest is missing required fields")
    if payload["source_revision"] != CBB26_REVISION:
        raise ValueError("raw manifest source revision is not pinned")
    repo_id = payload["repo_id"]
    if not isinstance(repo_id, str) or re.fullmatch(r"[^/]+/btcspiker-coinbase-history", repo_id) is None:
        raise ValueError("raw manifest repository is not canonical")
    revision = payload["revision"]
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision) is None:
        raise ValueError("raw manifest revision is not an exact commit SHA")
    if payload["usage_scope"] != "research_unverified":
        raise ValueError("raw manifest usage scope is not research_unverified")
    partitions = payload["partitions"]
    if not isinstance(partitions, list) or not partitions:
        raise ValueError("raw manifest has no partitions")

    remote_partitions: list[dict[str, Any]] = []
    pattern = re.compile(
        r"raw/kind=(book_deltas|book_states|trades)/source=([^/]+)/product=BTC-USD/"
        r"date=(\d{4}-\d{2}-\d{2})/hour=(\d{2})/part-([0-9a-f]{64})\.parquet"
    )
    expected_sources = {
        "book_deltas": "cbb26",
        "book_states": "cbb26",
        "trades": "coinbase_public_trades",
    }
    for partition in partitions:
        if not isinstance(partition, dict):
            raise ValueError("raw manifest partition is invalid")
        kind = partition.get("kind")
        expected = partition.get("sha256")
        remote_path = partition.get("remote_path")
        match = pattern.fullmatch(remote_path or "")
        if (
            kind not in expected_sources
            or not isinstance(expected, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected) is None
            or match is None
            or match.group(1) != kind
            or match.group(2) != expected_sources[kind]
            or int(match.group(4)) > 23
            or match.group(5) != expected
        ):
            raise ValueError("raw manifest partition path or checksum is invalid")
        receipt = partition.get("verified_receipt")
        if not isinstance(receipt, dict) or not (
            receipt.get("success") is True
            and receipt.get("repo_id") == repo_id
            and receipt.get("remote_path") == remote_path
            and receipt.get("sha256") == expected
            and isinstance(receipt.get("revision"), str)
            and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", receipt["revision"]) is not None
        ):
            raise ValueError("raw manifest partition receipt is invalid")
        local = partition.get("local_path") or partition.get("path")
        if local:
            file_path = Path(local)
            if not file_path.is_file():
                raise ValueError(f"manifest partition is missing: {file_path}")
            if _file_sha256(file_path) != expected:
                raise ValueError(f"manifest partition checksum mismatch: {file_path}")
        else:
            remote_partitions.append(partition)

    if remote_partitions:
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi()
        for offset in range(0, len(remote_partitions), 100):
            batch = remote_partitions[offset : offset + 100]
            remote_paths = [partition["remote_path"] for partition in batch]
            infos = api.get_paths_info(
                repo_id=repo_id,
                repo_type="dataset",
                paths=remote_paths,
                revision=revision,
            )
            by_path = {getattr(info, "path", None): info for info in infos}
            for partition in batch:
                remote_path = partition["remote_path"]
                info = by_path.get(remote_path)
                digest = _remote_digest(info) if info is not None else None
                if digest is None:
                    if hasattr(api, "hf_hub_download"):
                        downloaded = Path(api.hf_hub_download(
                            repo_id=repo_id,
                            repo_type="dataset",
                            filename=remote_path,
                            revision=revision,
                        ))
                    else:
                        from huggingface_hub import hf_hub_download

                        downloaded = Path(hf_hub_download(
                            repo_id=repo_id,
                            repo_type="dataset",
                            filename=remote_path,
                            revision=revision,
                        ))
                    digest = _file_sha256(downloaded)
                if digest != partition["sha256"]:
                    raise ValueError(f"remote manifest partition checksum mismatch: {remote_path}")
    return payload


def _resolve_partition(partition: dict[str, Any], manifest: dict[str, Any]) -> Path:
    local = partition.get("local_path") or partition.get("path")
    if local and Path(local).is_file():
        return Path(local)
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(
        repo_id=manifest["repo_id"],
        repo_type="dataset",
        filename=partition["remote_path"],
        revision=manifest["revision"],
    ))


def materialize_history(raw_manifest: Path, feature_set: str, output_root: Path) -> Path:
    manifest = load_verified_manifest(raw_manifest)
    if feature_set not in {"core_v1", "multi_window_v1", "microstructure_v1"}:
        raise ValueError("unknown feature set")
    by_day: dict[str, dict[str, list[pd.DataFrame]]] = {}
    for partition in manifest.get("partitions", []):
        kind = partition.get("kind")
        if kind not in {"trades", "book_states"}:
            continue
        local = _resolve_partition(partition, manifest)
        import hashlib
        if hashlib.sha256(local.read_bytes()).hexdigest() != partition["sha256"]:
            raise ValueError(f"manifest partition checksum mismatch: {local}")
        frame = pd.read_parquet(local)
        if frame.empty:
            continue
        source_date = str(frame["source_date"].iloc[0])
        by_day.setdefault(source_date, {}).setdefault(kind, []).append(frame)
    pieces = []
    for source_date, kinds in sorted(by_day.items()):
        if set(kinds) != {"trades", "book_states"}:
            raise ValueError(f"raw manifest is incomplete for {source_date}")
        trades = pd.concat(kinds["trades"], ignore_index=True)
        books = pd.concat(kinds["book_states"], ignore_index=True)
        trades["price"] = trades["price"].astype(float)
        trades["size"] = trades["size"].astype(float)
        for column in ("best_bid", "best_ask", "bid_size", "ask_size"):
            books[column] = books[column].astype(float)
        outputs = materialize_segmented_features(trades, books)
        if not outputs[feature_set].empty:
            pieces.append(outputs[feature_set])
    if not pieces:
        raise ValueError("materialization produced no feature rows")
    result = pd.concat(pieces, ignore_index=True).sort_values(["timestamp", "product_id"], kind="mergesort")
    destination = output_root.expanduser().resolve() / "features" / feature_set / "features.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".parquet.tmp")
    result.to_parquet(temporary, index=False)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    lineage = {
        "parent_dataset_id": raw_manifest_payload_id(manifest),
        "source_manifest_path": str(raw_manifest.expanduser().resolve()),
        "feature_set_id": feature_set,
        "feature_engine_git_sha": _feature_engine_git_sha(),
        "excluded_intervals": manifest.get("excluded_intervals", []),
    }
    lineage_path = destination.with_suffix(destination.suffix + ".lineage.json")
    lineage_path.write_text(json.dumps(lineage, sort_keys=True, indent=2) + "\n")
    return destination
