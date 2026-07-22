from dataclasses import asdict, dataclass, field
import hashlib
import json


@dataclass(frozen=True)
class DatasetManifest:
    source: str
    product: str
    rows: int
    start_time: str
    end_time: str
    quality: dict[str, int | float | str]
    partitions: list[dict[str, str | int]]
    parent_dataset_id: str | None = None
    source_manifest_path: str | None = None
    feature_set_id: str | None = None
    feature_engine_git_sha: str | None = None
    excluded_intervals: list[dict[str, str]] = field(default_factory=list)


def manifest_id(manifest: DatasetManifest) -> str:
    value = asdict(manifest)
    for optional_name in (
        "parent_dataset_id",
        "source_manifest_path",
        "feature_set_id",
        "feature_engine_git_sha",
    ):
        if value[optional_name] is None:
            value.pop(optional_name)
    if not value["excluded_intervals"]:
        value.pop("excluded_intervals")
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
