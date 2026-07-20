from dataclasses import asdict, dataclass
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


def manifest_id(manifest: DatasetManifest) -> str:
    payload = json.dumps(asdict(manifest), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
