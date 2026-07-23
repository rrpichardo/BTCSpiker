import os
import shutil
import time
from collections import namedtuple
from pathlib import Path

import pytest

from btcspiker_ml.storage import (
    atomic_publish,
    ensure_capacity,
    prune_cache,
    sha256_file,
)


def test_atomic_publish_verifies_bytes(tmp_path: Path):
    source = tmp_path / "source.parquet"
    source.write_bytes(b"verified")
    result = atomic_publish(source, tmp_path / "artifacts" / "part.parquet")
    assert result.sha256 == sha256_file(result.path)
    assert result.path.read_bytes() == b"verified"


def test_ensure_capacity_raises_when_disk_full(tmp_path: Path, monkeypatch):
    fake_usage = namedtuple("usage", ["total", "used", "free"])(
        total=100, used=99, free=1
    )
    monkeypatch.setattr(shutil, "disk_usage", lambda path: fake_usage)

    with pytest.raises(OSError, match="insufficient free space"):
        ensure_capacity(tmp_path / "cache", required_bytes=1_000_000)


def test_prune_cache_removes_oldest_files_first(tmp_path: Path):
    # three files, 100 bytes each, mtimes ordered oldest -> newest
    paths = []
    for index, name in enumerate(["oldest.bin", "middle.bin", "newest.bin"]):
        path = tmp_path / name
        path.write_bytes(b"x" * 100)
        # stagger mtimes so ordering is unambiguous regardless of filesystem resolution
        mtime = 1_000_000 + index * 10
        os.utime(path, (mtime, mtime))
        paths.append(path)

    # total is 300 bytes; a 150 byte ceiling forces removal of the two oldest files
    removed = prune_cache(tmp_path, max_bytes=150)

    assert removed == [paths[0], paths[1]]
    remaining = {p.name for p in tmp_path.iterdir()}
    assert remaining == {"newest.bin"}


def test_atomic_publish_cleans_up_temp_file_on_copy_failure(
    tmp_path: Path, monkeypatch
):
    source = tmp_path / "source.parquet"
    source.write_bytes(b"data")
    destination = tmp_path / "artifacts" / "part.parquet"

    def failing_copy(src, dst):
        # simulate a copy that got partway through, then failed (disk full,
        # signal, permission error mid-write, etc.)
        Path(dst).write_bytes(b"partial")
        raise OSError("simulated copy failure")

    monkeypatch.setattr(shutil, "copy", failing_copy)

    with pytest.raises(OSError, match="simulated copy failure"):
        atomic_publish(source, destination)

    leftover = [p for p in destination.parent.iterdir() if p.name.endswith(".tmp")]
    assert leftover == [], f"expected no .tmp litter, found: {leftover}"


def test_atomic_publish_records_publish_time_not_source_time(tmp_path: Path):
    source = tmp_path / "source.parquet"
    source.write_bytes(b"data")
    # force source mtime to look 24 hours old, so if publish preserved the
    # source mtime the destination would too, and LRU pruning would evict it early
    old_mtime = time.time() - 86_400
    os.utime(source, (old_mtime, old_mtime))

    result = atomic_publish(source, tmp_path / "artifacts" / "part.parquet")

    published_mtime = result.path.stat().st_mtime
    now = time.time()
    assert abs(published_mtime - now) < 5, (
        f"expected mtime within 5s of publish time {now}, got {published_mtime} "
        f"(delta={now - published_mtime}s) — publish is leaking source mtime"
    )
