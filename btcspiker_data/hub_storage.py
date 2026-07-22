"""Private Hugging Face Hub persistence with commit-pinned verification."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .storage import PartitionRecord


@dataclass(frozen=True)
class UploadReceipt:
    repo_id: str
    revision: str
    remote_path: str
    sha256: str
    size_bytes: int


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PrivateHubStore:
    def __init__(self, api: Any, repo_id: str) -> None:
        self._api = api
        self.repo_id = repo_id

    @classmethod
    def connect(
        cls,
        api_factory: Callable[[], Any] | None = None,
        repository_not_found_error: type[BaseException] | None = None,
    ) -> "PrivateHubStore":
        repository_not_found_exceptions: tuple[type[BaseException], ...] = ()
        if api_factory is None:
            from huggingface_hub import HfApi
            from huggingface_hub.errors import RepositoryNotFoundError

            api_factory = HfApi
            repository_not_found_error = RepositoryNotFoundError
        if repository_not_found_error is not None:
            repository_not_found_exceptions = (repository_not_found_error,)
        api = api_factory()
        identity = api.whoami()
        name = identity.get("name") if isinstance(identity, dict) else getattr(identity, "name", None)
        if not name:
            raise RuntimeError("Hugging Face authentication is required")
        repo_id = f"{name}/btcspiker-coinbase-history"
        try:
            api.repo_info(repo_id=repo_id, repo_type="dataset")
        except repository_not_found_exceptions:
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
        store = cls(api, repo_id)
        store._assert_private()
        return store

    def _assert_private(self) -> Any:
        info = self._api.repo_info(repo_id=self.repo_id, repo_type="dataset")
        if not getattr(info, "private", False):
            raise ValueError("Hub destination must be private")
        return info

    def _verify(self, remote_path: str, revision: str, expected: str) -> None:
        try:
            self._api.repo_info(repo_id=self.repo_id, repo_type="dataset", revision=revision)
        except Exception as error:
            raise RuntimeError("cannot read committed revision") from error
        metadata = self._api.get_paths_info(repo_id=self.repo_id, paths=[remote_path], repo_type="dataset", revision=revision)
        candidate = metadata[0] if metadata else None
        lfs = getattr(candidate, "lfs", None)
        remote_digest = getattr(lfs, "sha256", None)
        if remote_digest is None:
            downloaded = Path(self._api.hf_hub_download(repo_id=self.repo_id, filename=remote_path, repo_type="dataset", revision=revision))
            remote_digest = _file_sha256(downloaded)
        if remote_digest != expected:
            raise ValueError("remote checksum does not match local partition")

    def upload_partition(self, partition: PartitionRecord, remote_path: str) -> UploadReceipt:
        info = self._assert_private()
        if not remote_path.startswith("raw/"):
            raise ValueError("remote partition path must be under raw/")
        if _file_sha256(partition.path) != partition.sha256:
            raise ValueError("local partition checksum does not match its record")
        if self._api.file_exists(repo_id=self.repo_id, filename=remote_path, repo_type="dataset"):
            revision = getattr(info, "sha", None) or "main"
            self._verify(remote_path, revision, partition.sha256)
            return UploadReceipt(self.repo_id, revision, remote_path, partition.sha256, partition.size_bytes)
        commit = self._api.upload_file(path_or_fileobj=str(partition.path), path_in_repo=remote_path, repo_id=self.repo_id, repo_type="dataset")
        revision = getattr(commit, "oid", None) or getattr(commit, "commit_url", "").rstrip("/").split("/").pop()
        if not revision:
            raise RuntimeError("upload did not return a committed revision")
        self._verify(remote_path, revision, partition.sha256)
        return UploadReceipt(self.repo_id, revision, remote_path, partition.sha256, partition.size_bytes)

    def upload_bytes(self, remote_path: str, content: bytes) -> UploadReceipt:
        """Upload manifest bytes; manifests are verified by their SHA-256."""
        import io
        self._assert_private()
        digest = hashlib.sha256(content).hexdigest()
        commit = self._api.upload_file(path_or_fileobj=io.BytesIO(content), path_in_repo=remote_path, repo_id=self.repo_id, repo_type="dataset")
        revision = getattr(commit, "oid", None)
        if not revision:
            raise RuntimeError("upload did not return a committed revision")
        self._verify(remote_path, revision, digest)
        return UploadReceipt(self.repo_id, revision, remote_path, digest, len(content))
