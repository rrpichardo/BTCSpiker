from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil


@dataclass(frozen=True)
class PublishedFile:
    path: Path
    sha256: str
    size_bytes: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_capacity(path: Path, required_bytes: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free < required_bytes:
        raise OSError(f"insufficient free space: required={required_bytes} free={free}")


def atomic_publish(source: Path, destination: Path) -> PublishedFile:
    destination.parent.mkdir(parents=True, exist_ok=True)
    ensure_capacity(destination.parent, source.stat().st_size * 2)
    # shutil.copy (not copy2) so the temp file — and therefore the destination
    # after replace — carries the publish-time mtime, not the source's mtime.
    # prune_cache sorts by mtime, so preserving source mtime would invert LRU.
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copy(source, temporary)
        source_hash = sha256_file(source)
        if sha256_file(temporary) != source_hash:
            raise OSError("checksum mismatch during publication")
        temporary.replace(destination)
    finally:
        # No-op on the success path (temporary was renamed by .replace); on any
        # failure, this prevents .tmp litter from eroding the cache budget.
        temporary.unlink(missing_ok=True)
    return PublishedFile(destination, source_hash, destination.stat().st_size)


def prune_cache(root: Path, max_bytes: int) -> list[Path]:
    files = sorted((path for path in root.rglob("*") if path.is_file()), key=lambda path: path.stat().st_mtime)
    total = sum(path.stat().st_size for path in files)
    removed: list[Path] = []
    for path in files:
        if total <= max_bytes:
            break
        size = path.stat().st_size
        path.unlink()
        total -= size
        removed.append(path)
    return removed
