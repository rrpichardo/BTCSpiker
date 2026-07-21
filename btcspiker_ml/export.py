"""Immutable, checksum-verified local exports of MLflow run artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

from mlflow.tracking import MlflowClient


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ExportManifest:
    run_id: str
    destination: Path
    files: Mapping[str, str]

    def verify(self) -> bool:
        return _verify_export(self.destination, self.files)


def _verify_export(destination: Path, files: Mapping[str, str]) -> bool:
    for relative_path, expected_digest in files.items():
        path = destination / relative_path
        if not path.is_file() or _sha256(path) != expected_digest:
            return False
    return True


def _artifact_paths(client: MlflowClient, run_id: str, prefix: str = "") -> list[str]:
    paths: list[str] = []
    for item in client.list_artifacts(run_id, prefix):
        if item.is_dir:
            paths.extend(_artifact_paths(client, run_id, item.path))
        else:
            paths.append(item.path)
    return paths


def export_run(run_id: str, artifact_root: Path) -> ExportManifest:
    """Download a run into a new local immutable export and verify every byte.

    MLflow is read-only in this operation: artifacts are downloaded, never
    moved or deleted.  An existing run export is deliberately never replaced.
    """
    client = MlflowClient()
    # Ensure an invalid id fails before any local directory is made.
    client.get_run(run_id)
    destination = Path(artifact_root) / "mlflow-exports" / f"run_id={run_id}"
    if destination.exists():
        raise FileExistsError(f"immutable export already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary = Path(tempfile.mkdtemp(prefix=f".run_id={run_id}.", dir=destination.parent))
    published = False
    try:
        files: dict[str, str] = {}
        for artifact_path in _artifact_paths(client, run_id):
            downloaded = Path(client.download_artifacts(run_id, artifact_path, str(temporary)))
            relative_path = downloaded.relative_to(temporary).as_posix()
            files[relative_path] = _sha256(downloaded)

        manifest_path = temporary / "export-manifest.json"
        manifest_path.write_text(
            json.dumps({"run_id": run_id, "files": files}, indent=2, sort_keys=True) + "\n"
        )
        temporary.replace(destination)
        published = True
        if not _verify_export(destination, files):
            # This directory was created by this call and failed its required
            # verification, so do not leave a poisoned immutable destination.
            shutil.rmtree(destination)
            published = False
            raise OSError(f"checksum verification failed for export {destination}")
        return ExportManifest(run_id=run_id, destination=destination, files=files)
    finally:
        if not published:
            shutil.rmtree(temporary, ignore_errors=True)
