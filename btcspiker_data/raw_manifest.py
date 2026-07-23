"""Deterministic identities and publication for raw-data manifests."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any


@dataclass(frozen=True)
class RawDatasetManifest:
    source_revision: str
    source_url: str
    repo_id: str
    revision: str
    usage_scope: str
    schemas: dict[str, list[str]]
    partitions: list[dict[str, Any]]
    coverage_seconds: int
    missing_seconds: int
    duplicate_counts: dict[str, int]
    sequence_incidents: list[dict[str, Any]]
    excluded_intervals: list[dict[str, Any]]
    created_at: datetime
    # Completion evidence is deliberately part of the identity: a successful
    # L2 download alone must never qualify a UTC day.
    trade_day_completions: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.usage_scope != "research_unverified":
            raise ValueError("usage_scope must be research_unverified")

    def identity_payload(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("created_at")
        return value


def serialize_trade_day_completion(completion: Any) -> dict[str, Any]:
    """Serialize proven Coinbase pagination exhaustion for manifest storage.

    ``TradeDayCompletion`` is created by ``CoinbaseTradeClient`` only after the
    paginator reaches the UTC day boundary. This adapter is the sole place that
    turns that production evidence into ``trade_pages_complete=True``.
    """
    from .coinbase_trades import TradeDayCompletion

    if not isinstance(completion, TradeDayCompletion):
        raise TypeError("completion must be TradeDayCompletion")
    if not isinstance(completion.product_id, str) or not completion.product_id:
        raise ValueError("completion product_id must be non-empty")
    if type(completion.source_date) is not date:
        raise ValueError("completion source_date must be a date")
    day_start = datetime.combine(completion.source_date, datetime.min.time(), tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)
    if (
        type(completion.day_start_epoch) is not int
        or type(completion.day_end_epoch) is not int
        or completion.day_start_epoch != int(day_start.timestamp())
        or completion.day_end_epoch != int(day_end.timestamp())
    ):
        raise ValueError("completion epochs must match the UTC source date")
    return {
        "product_id": completion.product_id,
        "source_date": completion.source_date.isoformat(),
        "day_start_epoch": completion.day_start_epoch,
        "day_end_epoch": completion.day_end_epoch,
        "trade_pages_complete": True,
    }


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=lambda item: item.isoformat()).encode("utf-8")


def raw_manifest_id(manifest: RawDatasetManifest) -> str:
    return hashlib.sha256(_canonical(manifest.identity_payload())).hexdigest()


def publish_raw_manifest(manifest: RawDatasetManifest, store: Any) -> Any:
    """Publish a named immutable manifest through a compatible private store."""
    dataset_id = raw_manifest_id(manifest)
    content = _canonical(asdict(manifest))
    content_sha = hashlib.sha256(content).hexdigest()
    remote_path = f"manifests/{dataset_id}/manifest-{content_sha}.json"
    return store.upload_bytes(remote_path, content)
