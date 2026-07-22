from pathlib import Path

import pytest

from btcspiker_data.hub_storage import PrivateHubStore
from btcspiker_data.storage import PartitionRecord


class Info:
    private = True


class FakeApi:
    def __init__(self, *, private=True, digest=None, readable=True):
        self.info = Info()
        self.info.private = private
        self.digest = digest
        self.readable = readable
        self.uploads = 0
        self.calls = []

    def whoami(self):
        return {"name": "alice"}

    def repo_info(self, **kwargs):
        self.calls.append(("repo_info", kwargs))
        if kwargs.get("revision") and not self.readable:
            raise RuntimeError("not found")
        return self.info

    def create_repo(self, **kwargs):
        self.calls.append(("create_repo", kwargs))

    def file_exists(self, **kwargs):
        return False

    def upload_file(self, **kwargs):
        self.uploads += 1
        self.calls.append(("upload_file", kwargs))
        return type("Commit", (), {"oid": "commit-1"})()

    def get_paths_info(self, **kwargs):
        return [type("PathInfo", (), {"lfs": type("Lfs", (), {"sha256": self.digest})()})()]


def _partition(tmp_path):
    path = tmp_path / "part-abc.parquet"
    path.write_bytes(b"partition")
    from hashlib import sha256
    return PartitionRecord(path, sha256(b"partition").hexdigest(), 1, path.stat().st_size)


def test_connect_derives_private_authenticated_destination():
    api = FakeApi()
    store = PrivateHubStore.connect(api_factory=lambda: api)
    assert store.repo_id == "alice/btcspiker-coinbase-history"
    assert any(name == "repo_info" for name, _ in api.calls)


def test_connect_rejects_public_destination():
    with pytest.raises(ValueError, match="private"):
        PrivateHubStore.connect(api_factory=lambda: FakeApi(private=False))


def test_connect_fails_without_authenticated_identity():
    class UnauthenticatedApi(FakeApi):
        def whoami(self):
            raise RuntimeError("not authenticated")
    with pytest.raises(RuntimeError, match="not authenticated"):
        PrivateHubStore.connect(api_factory=UnauthenticatedApi)


def test_upload_verifies_exact_commit_digest(tmp_path):
    part = _partition(tmp_path)
    api = FakeApi(digest=part.sha256)
    receipt = PrivateHubStore.connect(api_factory=lambda: api).upload_partition(part, "raw/part.parquet")
    assert receipt.revision == "commit-1"
    assert receipt.sha256 == part.sha256
    assert api.uploads == 1


def test_upload_fails_when_committed_digest_mismatches_without_deleting_local_file(tmp_path):
    part = _partition(tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        PrivateHubStore.connect(api_factory=lambda: FakeApi(digest="0" * 64)).upload_partition(part, "raw/part.parquet")
    assert part.path.exists()


def test_upload_fails_if_committed_revision_cannot_be_read(tmp_path):
    part = _partition(tmp_path)
    with pytest.raises(RuntimeError, match="cannot read"):
        PrivateHubStore.connect(api_factory=lambda: FakeApi(digest=part.sha256, readable=False)).upload_partition(part, "raw/part.parquet")


def test_upload_reuses_already_uploaded_matching_partition(tmp_path):
    part = _partition(tmp_path)
    class ExistingApi(FakeApi):
        def __init__(self):
            super().__init__(digest=part.sha256)
            self.info.sha = "existing-commit"
        def file_exists(self, **kwargs):
            return True
    api = ExistingApi()
    receipt = PrivateHubStore.connect(api_factory=lambda: api).upload_partition(part, "raw/part.parquet")
    assert receipt.revision == "existing-commit"
    assert api.uploads == 0
