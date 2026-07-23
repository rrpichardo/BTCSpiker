import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from btcspiker_ml.datasets import (
    REQUIRED_FEATURE_COLUMNS,
    inspect_existing_dataset,
    publish_existing_manifest,
    resolve_existing_dataset,
)

HANDOFF_SAMPLE = Path("handoff/data_sample/features_slice.parquet")
REPO_ROOT = Path(__file__).resolve().parents[2]


def _copy_handoff(tmp_path: Path) -> Path:
    # copy the checked-in handoff sample into an isolated location so we can
    # exercise the resolver without depending on the caller's working directory
    destination_dir = tmp_path / "handoff" / "data_sample"
    destination_dir.mkdir(parents=True)
    destination = destination_dir / "features_slice.parquet"
    shutil.copy(HANDOFF_SAMPLE.resolve(), destination)
    return destination


def test_explicit_existing_dataset_wins(tmp_path: Path, monkeypatch):
    supplied = tmp_path / "collected.parquet"
    supplied.write_bytes(b"collected")
    monkeypatch.setenv("BTCSPIKER_EXISTING_DATA", str(supplied))
    assert resolve_existing_dataset(None) == supplied.resolve()


def test_resolver_fails_instead_of_generating_data(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("BTCSPIKER_EXISTING_DATA", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="existing collected dataset"):
        resolve_existing_dataset(None)


def test_inspection_rejects_unlabelled_or_empty_data(tmp_path: Path):
    path = tmp_path / "empty.parquet"
    path.write_bytes(b"")
    with pytest.raises(ValueError):
        inspect_existing_dataset(path)


def test_resolver_falls_back_to_handoff_sample(tmp_path: Path, monkeypatch):
    # mirror only the handoff sample into a clean cwd (no data/processed/) so
    # the resolver has to reach for the handoff fallback
    _copy_handoff(tmp_path)
    monkeypatch.delenv("BTCSPIKER_EXISTING_DATA", raising=False)
    monkeypatch.chdir(tmp_path)

    resolved = resolve_existing_dataset(None)
    assert (
        resolved
        == (tmp_path / "handoff" / "data_sample" / "features_slice.parquet").resolve()
    )


def test_inspection_reads_handoff_sample(tmp_path: Path):
    dataset = inspect_existing_dataset(HANDOFF_SAMPLE.resolve())
    assert dataset.rows > 0
    assert dataset.start_time.endswith("+00:00")
    assert dataset.end_time.endswith("+00:00")
    assert len(dataset.sha256) == 64
    assert REQUIRED_FEATURE_COLUMNS.issubset(set(dataset.columns))


def test_inspection_rejects_missing_feature_column(tmp_path: Path):
    df = pd.read_parquet(HANDOFF_SAMPLE.resolve())
    df = df.drop(columns=["vol_60s"])
    path = tmp_path / "missing_vol.parquet"
    df.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="vol_60s"):
        inspect_existing_dataset(path)


def test_inspection_rejects_non_monotonic_timestamps(tmp_path: Path):
    df = pd.read_parquet(HANDOFF_SAMPLE.resolve()).copy()
    # swap two timestamps to create a strict backwards jump
    df.loc[df.index[10], "timestamp"] = df["timestamp"].iloc[0]
    path = tmp_path / "unsorted.parquet"
    df.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="monotonic"):
        inspect_existing_dataset(path)


def test_inspection_rejects_non_binary_target(tmp_path: Path):
    df = pd.read_parquet(HANDOFF_SAMPLE.resolve()).copy()
    df.loc[df.index[0], "vol_spike"] = 2
    path = tmp_path / "non_binary.parquet"
    df.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="binary"):
        inspect_existing_dataset(path)


def test_publish_manifest_writes_json_and_returns_id(tmp_path: Path):
    dataset = inspect_existing_dataset(HANDOFF_SAMPLE.resolve())
    artifact_root = tmp_path / "artifacts"

    dataset_id, manifest_path = publish_existing_manifest(dataset, artifact_root)

    assert manifest_path == artifact_root / "manifests" / f"existing-{dataset_id}.json"
    assert manifest_path.is_file()
    payload = json.loads(manifest_path.read_text())
    assert set(payload["quality"].keys()) == {
        "sha256",
        "prevalence",
        "null_counts",
        "duplicate_timestamps",
        "input_mode",
        "absolute_source_path",
    }
    assert payload["quality"]["input_mode"] == "existing_collected"
    assert payload["quality"]["sha256"] == dataset.sha256
    assert payload["rows"] == dataset.rows
    assert payload["partitions"][0]["path"] == str(dataset.path.resolve())


def test_publish_manifest_is_deterministic(tmp_path: Path):
    dataset = inspect_existing_dataset(HANDOFF_SAMPLE.resolve())
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"

    id_a, _ = publish_existing_manifest(dataset, root_a)
    id_b, _ = publish_existing_manifest(dataset, root_b)
    assert id_a == id_b


def test_publish_manifest_records_optional_raw_lineage(tmp_path: Path):
    dataset = inspect_existing_dataset(HANDOFF_SAMPLE.resolve())
    dataset_id, manifest_path = publish_existing_manifest(
        dataset,
        tmp_path / "artifacts",
        parent_dataset_id="raw-1",
        source_manifest_path="/data/raw-1.json",
        feature_set_id="core_v1",
        feature_engine_git_sha="a" * 40,
        excluded_intervals=[{"start": "2026-04-24T01:00:00+00:00", "end": "2026-04-24T01:01:00+00:00"}],
    )

    payload = json.loads(manifest_path.read_text())
    assert payload["parent_dataset_id"] == "raw-1"
    assert payload["source_manifest_path"] == "/data/raw-1.json"
    assert payload["feature_set_id"] == "core_v1"
    assert payload["feature_engine_git_sha"] == "a" * 40
    assert payload["excluded_intervals"][0]["start"].startswith("2026-04-24")
    assert dataset_id


def test_inspection_carries_adjacent_lineage_into_default_manifest(tmp_path: Path):
    feature_path = _copy_handoff(tmp_path)
    lineage = {
        "parent_dataset_id": "b" * 64,
        "source_manifest_path": "/data/raw.json",
        "feature_set_id": "core_v1",
        "feature_engine_git_sha": "a" * 40,
        "excluded_intervals": [{"start": "2026-04-24T01:00:00+00:00", "end": "2026-04-24T01:01:00+00:00"}],
    }
    feature_path.with_suffix(".parquet.lineage.json").write_text(json.dumps(lineage))

    dataset = inspect_existing_dataset(feature_path)
    _, manifest_path = publish_existing_manifest(dataset, tmp_path / "artifacts")
    payload = json.loads(manifest_path.read_text())

    assert payload["parent_dataset_id"] == "b" * 64
    assert payload["source_manifest_path"] == "/data/raw.json"
    assert payload["feature_set_id"] == "core_v1"
    assert payload["feature_engine_git_sha"] == "a" * 40


def test_bind_cli_runs_without_pythonpath(tmp_path: Path):
    # Explicitly strip PYTHONPATH so the sys.path shim in the script is what
    # makes btcspiker_ml importable. This locks in the pattern for the whole
    # scripts/ family — later tasks add more entry points that must behave
    # the same way when invoked as bare `python scripts/foo.py`.
    result = subprocess.run(
        [
            sys.executable,
            "scripts/bind_existing_dataset.py",
            "--config",
            "experiment.yaml",
        ],
        cwd=REPO_ROOT,
        env={
            "BTCSPIKER_EXISTING_DATA": str(REPO_ROOT / HANDOFF_SAMPLE),
            "PATH": "/usr/bin:/bin",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "dataset_id:" in result.stdout
