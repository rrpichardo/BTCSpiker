"""Deterministic identities and publication for raw-data manifests."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
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

    def __post_init__(self) -> None:
        if self.usage_scope != "research_unverified":
            raise ValueError("usage_scope must be research_unverified")

    def identity_payload(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("created_at")
        return value


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=lambda item: item.isoformat()).encode("utf-8")


def raw_manifest_id(manifest: RawDatasetManifest) -> str:
    return hashlib.sha256(_canonical(manifest.identity_payload())).hexdigest()


def publish_raw_manifest(manifest: RawDatasetManifest, store: Any) -> Any:
    """Publish a named immutable manifest through a compatible private store."""
    dataset_id = raw_manifest_id(manifest)
    return store.upload_bytes(f"manifests/{dataset_id}.json", _canonical(asdict(manifest)))
