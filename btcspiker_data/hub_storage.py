"""Private Hugging Face Hub persistence with commit-pinned verification."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date
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
    reused: bool = False


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
        store._assert_destination()
        return store

    def _assert_destination(self) -> Any:
        identity = self._api.whoami()
        name = identity.get("name") if isinstance(identity, dict) else getattr(identity, "name", None)
        if not name:
            raise RuntimeError("Hugging Face authentication is required")
        expected_repo_id = f"{name}/btcspiker-coinbase-history"
        if self.repo_id != expected_repo_id:
            raise ValueError("Hub destination must use the authenticated namespace")
        info = self._api.repo_info(repo_id=self.repo_id, repo_type="dataset")
        if not getattr(info, "private", False):
            raise ValueError("Hub destination must be private")
        return info

    @staticmethod
    def _exact_revision(revision: Any, error_message: str) -> str:
        if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision) is None:
            raise RuntimeError(error_message)
        return revision

    @classmethod
    def _existing_revision(cls, info: Any) -> str:
        return cls._exact_revision(
            getattr(info, "sha", None),
            "existing remote file has no exact commit SHA",
        )

    @staticmethod
    def _validate_partition_path(remote_path: str, expected_sha256: str) -> None:
        match = re.fullmatch(
            r"raw/kind=(book_deltas|book_states|trades)/source=([^/]+)/"
            r"product=BTC-USD/date=(\d{4}-\d{2}-\d{2})/hour=(\d{2})/"
            r"part-([0-9a-f]{64})\.parquet",
            remote_path,
        )
        if match is None:
            raise ValueError("remote partition path does not match the required layout")
        try:
            date.fromisoformat(match.group(3))
        except ValueError as error:
            raise ValueError("remote partition path does not match the required layout") from error
        if int(match.group(4)) > 23 or match.group(5) != expected_sha256:
            raise ValueError("remote partition path does not match the required layout")

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
            raise ValueError("uploaded content checksum does not match expected content")

    def upload_partition(self, partition: PartitionRecord, remote_path: str) -> UploadReceipt:
        info = self._assert_destination()
        self._validate_partition_path(remote_path, partition.sha256)
        if _file_sha256(partition.path) != partition.sha256:
            raise ValueError("local partition checksum does not match its record")
        if self._api.file_exists(repo_id=self.repo_id, filename=remote_path, repo_type="dataset"):
            revision = self._existing_revision(info)
            self._verify(remote_path, revision, partition.sha256)
            return UploadReceipt(self.repo_id, revision, remote_path, partition.sha256, partition.size_bytes, reused=True)
        commit = self._api.upload_file(path_or_fileobj=str(partition.path), path_in_repo=remote_path, repo_id=self.repo_id, repo_type="dataset")
        revision = self._exact_revision(
            getattr(commit, "oid", None),
            "upload did not return an exact commit SHA",
        )
        self._verify(remote_path, revision, partition.sha256)
        return UploadReceipt(self.repo_id, revision, remote_path, partition.sha256, partition.size_bytes)

    def upload_bytes(self, remote_path: str, content: bytes) -> UploadReceipt:
        """Upload manifest bytes; manifests are verified by their SHA-256."""
        import io
        info = self._assert_destination()
        digest = hashlib.sha256(content).hexdigest()
        if self._api.file_exists(repo_id=self.repo_id, filename=remote_path, repo_type="dataset"):
            revision = self._existing_revision(info)
            self._verify(remote_path, revision, digest)
            return UploadReceipt(self.repo_id, revision, remote_path, digest, len(content), reused=True)
        commit = self._api.upload_file(path_or_fileobj=io.BytesIO(content), path_in_repo=remote_path, repo_id=self.repo_id, repo_type="dataset")
        revision = self._exact_revision(
            getattr(commit, "oid", None),
            "upload did not return an exact commit SHA",
        )
        self._verify(remote_path, revision, digest)
        return UploadReceipt(self.repo_id, revision, remote_path, digest, len(content))
