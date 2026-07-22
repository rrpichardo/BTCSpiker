from pathlib import Path
from hashlib import sha256

import pytest

from btcspiker_data.hub_storage import PrivateHubStore
from btcspiker_data.storage import PartitionRecord


class Info:
    private = True


class FakeRepositoryNotFoundError(Exception):
    pass


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
    return PartitionRecord(path, sha256(b"partition").hexdigest(), 1, path.stat().st_size)


def _remote_path(partition):
    return (
        "raw/kind=trades/source=coinbase/product=BTC-USD/"
        f"date=2026-04-24/hour=03/part-{partition.sha256}.parquet"
    )


def test_connect_derives_private_authenticated_destination():
    api = FakeApi()
    store = PrivateHubStore.connect(api_factory=lambda: api)
    assert store.repo_id == "alice/btcspiker-coinbase-history"
    assert any(name == "repo_info" for name, _ in api.calls)


def test_connect_rejects_public_destination():
    with pytest.raises(ValueError, match="private"):
        PrivateHubStore.connect(api_factory=lambda: FakeApi(private=False))


def test_connect_creates_only_an_absent_repository_then_verifies_privacy():
    class AbsentApi(FakeApi):
        def __init__(self):
            super().__init__()
            self.absent = True
        def repo_info(self, **kwargs):
            self.calls.append(("repo_info", kwargs))
            if self.absent:
                raise FakeRepositoryNotFoundError("missing")
            return self.info
        def create_repo(self, **kwargs):
            self.calls.append(("create_repo", kwargs))
            self.absent = False
    api = AbsentApi()
    PrivateHubStore.connect(
        api_factory=lambda: api,
        repository_not_found_error=FakeRepositoryNotFoundError,
    )
    assert ("create_repo", {"repo_id": "alice/btcspiker-coinbase-history", "repo_type": "dataset", "private": True, "exist_ok": True}) in api.calls
    assert [name for name, _ in api.calls].count("repo_info") == 2


def test_connect_propagates_service_error_without_attempting_creation():
    class BrokenApi(FakeApi):
        def repo_info(self, **kwargs):
            raise ConnectionError("service unavailable")
    api = BrokenApi()
    with pytest.raises(ConnectionError, match="service unavailable"):
        PrivateHubStore.connect(
            api_factory=lambda: api,
            repository_not_found_error=FakeRepositoryNotFoundError,
        )
    assert not any(name == "create_repo" for name, _ in api.calls)


def test_connect_fails_without_authenticated_identity():
    class UnauthenticatedApi(FakeApi):
        def whoami(self):
            raise RuntimeError("not authenticated")
    with pytest.raises(RuntimeError, match="not authenticated"):
        PrivateHubStore.connect(api_factory=UnauthenticatedApi)


def test_upload_verifies_exact_commit_digest(tmp_path):
    part = _partition(tmp_path)
    api = FakeApi(digest=part.sha256)
    receipt = PrivateHubStore.connect(api_factory=lambda: api).upload_partition(part, _remote_path(part))
    assert receipt.revision == "commit-1"
    assert receipt.sha256 == part.sha256
    assert api.uploads == 1


def test_upload_fails_when_committed_digest_mismatches_without_deleting_local_file(tmp_path):
    part = _partition(tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        PrivateHubStore.connect(api_factory=lambda: FakeApi(digest="0" * 64)).upload_partition(part, _remote_path(part))
    assert part.path.exists()


def test_upload_fails_if_committed_revision_cannot_be_read(tmp_path):
    part = _partition(tmp_path)
    with pytest.raises(RuntimeError, match="cannot read"):
        PrivateHubStore.connect(api_factory=lambda: FakeApi(digest=part.sha256, readable=False)).upload_partition(part, _remote_path(part))


def test_upload_fallback_downloads_and_hashes_exact_committed_revision(tmp_path):
    part = _partition(tmp_path)
    api = FakeApi(digest=None)
    api.downloaded = []
    def download(**kwargs):
        api.downloaded.append(kwargs)
        return str(part.path)
    api.hf_hub_download = download
    remote_path = _remote_path(part)
    receipt = PrivateHubStore.connect(api_factory=lambda: api).upload_partition(part, remote_path)
    assert receipt.sha256 == part.sha256
    assert api.downloaded == [{
        "repo_id": "alice/btcspiker-coinbase-history",
        "filename": remote_path,
        "repo_type": "dataset",
        "revision": "commit-1",
    }]


def test_upload_partition_rechecks_privacy_after_connect(tmp_path):
    part = _partition(tmp_path)
    api = FakeApi(digest=part.sha256)
    store = PrivateHubStore.connect(api_factory=lambda: api)
    api.info.private = False
    with pytest.raises(ValueError, match="private"):
        store.upload_partition(part, _remote_path(part))
    assert api.uploads == 0


def test_upload_bytes_rechecks_privacy_after_connect():
    api = FakeApi(digest=None)
    store = PrivateHubStore.connect(api_factory=lambda: api)
    api.info.private = False
    with pytest.raises(ValueError, match="private"):
        store.upload_bytes("manifests/id.json", b"{}")
    assert api.uploads == 0


def test_upload_reuses_already_uploaded_matching_partition(tmp_path):
    part = _partition(tmp_path)
    class ExistingApi(FakeApi):
        def __init__(self):
            super().__init__(digest=part.sha256)
            self.info.sha = "a" * 40
        def file_exists(self, **kwargs):
            return True
    api = ExistingApi()
    receipt = PrivateHubStore.connect(api_factory=lambda: api).upload_partition(part, _remote_path(part))
    assert receipt.revision == "a" * 40
    assert api.uploads == 0


def test_direct_constructor_rejects_noncanonical_private_repo_before_upload(tmp_path):
    part = _partition(tmp_path)
    api = FakeApi(digest=part.sha256)
    store = PrivateHubStore(api, "mallory/private-data")
    with pytest.raises(ValueError, match="authenticated namespace"):
        store.upload_partition(part, _remote_path(part))
    assert api.uploads == 0


def test_existing_partition_without_exact_commit_sha_fails_closed(tmp_path):
    part = _partition(tmp_path)
    class ExistingWithoutShaApi(FakeApi):
        def __init__(self):
            super().__init__(digest=part.sha256)
            self.info.sha = None
        def file_exists(self, **kwargs):
            return True
    api = ExistingWithoutShaApi()
    with pytest.raises(RuntimeError, match="commit SHA"):
        PrivateHubStore.connect(api_factory=lambda: api).upload_partition(part, _remote_path(part))
    assert api.uploads == 0


def test_existing_partition_rejects_symbolic_revision(tmp_path):
    part = _partition(tmp_path)
    class ExistingAtMainApi(FakeApi):
        def __init__(self):
            super().__init__(digest=part.sha256)
            self.info.sha = "main"
        def file_exists(self, **kwargs):
            return True
    api = ExistingAtMainApi()
    with pytest.raises(RuntimeError, match="commit SHA"):
        PrivateHubStore.connect(api_factory=lambda: api).upload_partition(part, _remote_path(part))
    assert api.uploads == 0


def test_partition_rejects_noncanonical_remote_path(tmp_path):
    part = _partition(tmp_path)
    api = FakeApi(digest=part.sha256)
    with pytest.raises(ValueError, match="layout"):
        PrivateHubStore.connect(api_factory=lambda: api).upload_partition(part, "raw/part.parquet")
    assert api.uploads == 0
