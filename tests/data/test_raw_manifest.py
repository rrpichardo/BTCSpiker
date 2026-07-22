import json
from datetime import date, datetime, timedelta, timezone

import pytest

from btcspiker_data.hub_storage import PrivateHubStore
from btcspiker_data.coinbase_trades import TradeDayCompletion
from btcspiker_data.quality import audit_dataset
from btcspiker_data.raw_manifest import (
    RawDatasetManifest,
    publish_raw_manifest,
    raw_manifest_id,
    serialize_trade_day_completion,
)


def _manifest_values():
    return dict(
        source_revision="r1",
        source_url="https://example.test/data",
        repo_id="user/btcspiker-coinbase-history",
        revision="abc123",
        usage_scope="research_unverified",
        schemas={"trades": ["source", "trade_id"]},
        partitions=[{"remote_path": "raw/a", "row_count": 1, "sha256": "a" * 64}],
        coverage_seconds=1,
        missing_seconds=0,
        duplicate_counts={"trades": 0, "book": 0},
        sequence_incidents=[],
        excluded_intervals=[],
    )


def test_manifest_id_ignores_dictionary_order():
    common = _manifest_values()
    left = RawDatasetManifest(created_at=datetime(2026, 1, 1, tzinfo=timezone.utc), **common)
    right = RawDatasetManifest(created_at=datetime(2030, 1, 1, tzinfo=timezone.utc), **dict(reversed(list(common.items()))))
    assert raw_manifest_id(left) == raw_manifest_id(right)


def test_manifest_rejects_non_research_usage_scope():
    values = _manifest_values()
    values["usage_scope"] = "production"
    with pytest.raises(ValueError, match="research_unverified"):
        RawDatasetManifest(created_at=datetime(2026, 1, 1, tzinfo=timezone.utc), **values)


def test_trade_completion_evidence_is_part_of_deterministic_identity():
    values = _manifest_values()
    left = RawDatasetManifest(created_at=datetime(2026, 1, 1, tzinfo=timezone.utc), **values)
    right = RawDatasetManifest(
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc), **values,
        trade_day_completions=[{
            "product_id": "BTC-USD", "source_date": "2026-01-01",
            "day_start_epoch": 1767225600, "day_end_epoch": 1767312000,
            "trade_pages_complete": True,
        }],
    )
    assert raw_manifest_id(left) != raw_manifest_id(right)


def test_real_trade_completion_adapter_produces_canonical_credited_evidence():
    source_date = date(2026, 1, 1)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    completion = TradeDayCompletion(
        product_id="BTC-USD",
        source_date=source_date,
        day_start_epoch=int(start.timestamp()),
        day_end_epoch=int((start + timedelta(days=1)).timestamp()),
    )

    serialized = serialize_trade_day_completion(completion)

    assert serialized == {
        "product_id": "BTC-USD",
        "source_date": "2026-01-01",
        "day_start_epoch": 1767225600,
        "day_end_epoch": 1767312000,
        "trade_pages_complete": True,
    }
    assert json.dumps(serialized, separators=(",", ":")) == (
        '{"product_id":"BTC-USD","source_date":"2026-01-01",'
        '"day_start_epoch":1767225600,"day_end_epoch":1767312000,'
        '"trade_pages_complete":true}'
    )
    values = _manifest_values()
    values["partitions"] = []
    manifest = RawDatasetManifest(
        created_at=start,
        trade_day_completions=[serialized],
        **values,
    )
    report = audit_dataset(manifest, book_intervals=[(start, start + timedelta(days=1))])
    assert report.per_day["2026-01-01"]["trade_pages_complete"] is True
    assert report.per_day["2026-01-01"]["valid_seconds"] == 86_400

    for unadapted in (completion, {key: value for key, value in serialized.items()
                                   if key != "trade_pages_complete"}):
        invalid_manifest = RawDatasetManifest(
            created_at=start,
            trade_day_completions=[unadapted],
            **values,
        )
        invalid = audit_dataset(
            invalid_manifest,
            book_intervals=[(start, start + timedelta(days=1))],
        )
        assert invalid.per_day["2026-01-01"]["valid_seconds"] == 0
        assert "invalid_trade_completion_evidence" in invalid.failures


def test_same_identity_different_creation_times_publish_distinct_immutable_bytes():
    from hashlib import sha256

    class Info:
        private = True
        sha = None

    class MemoryApi:
        def __init__(self):
            self.info = Info()
            self.files = {}
            self.uploads = 0
        def whoami(self):
            return {"name": "alice"}
        def repo_info(self, **kwargs):
            return self.info
        def file_exists(self, *, filename, **kwargs):
            return filename in self.files
        def upload_file(self, *, path_or_fileobj, path_in_repo, **kwargs):
            content = path_or_fileobj.getvalue()
            self.files[path_in_repo] = content
            self.uploads += 1
            self.info.sha = f"{self.uploads:040x}"
            return type("Commit", (), {"oid": self.info.sha})()
        def get_paths_info(self, *, paths, **kwargs):
            digest = sha256(self.files[paths[0]]).hexdigest()
            lfs = type("Lfs", (), {"sha256": digest})()
            return [type("PathInfo", (), {"lfs": lfs})()]

    values = _manifest_values()
    first = RawDatasetManifest(created_at=datetime(2026, 1, 1, tzinfo=timezone.utc), **values)
    second = RawDatasetManifest(created_at=datetime(2026, 1, 2, tzinfo=timezone.utc), **values)
    api = MemoryApi()
    store = PrivateHubStore.connect(api_factory=lambda: api)
    first_receipt = publish_raw_manifest(first, store)
    second_receipt = publish_raw_manifest(second, store)

    assert raw_manifest_id(first) == raw_manifest_id(second)
    assert first_receipt.remote_path != second_receipt.remote_path
    assert first_receipt.sha256 != second_receipt.sha256
    assert api.files[first_receipt.remote_path] != api.files[second_receipt.remote_path]
    assert api.uploads == 2


def test_upload_bytes_reuses_matching_existing_content_at_pinned_commit():
    from hashlib import sha256

    content = b'{"created_at":"2026-01-01T00:00:00+00:00"}'
    remote_path = f"manifests/id/manifest-{sha256(content).hexdigest()}.json"

    class Info:
        private = True
        sha = "b" * 40

    class ExistingApi:
        def __init__(self):
            self.info = Info()
            self.uploads = 0
        def whoami(self):
            return {"name": "alice"}
        def repo_info(self, **kwargs):
            return self.info
        def file_exists(self, **kwargs):
            return True
        def upload_file(self, **kwargs):
            self.uploads += 1
        def get_paths_info(self, **kwargs):
            lfs = type("Lfs", (), {"sha256": sha256(content).hexdigest()})()
            return [type("PathInfo", (), {"lfs": lfs})()]

    api = ExistingApi()
    receipt = PrivateHubStore.connect(api_factory=lambda: api).upload_bytes(remote_path, content)
    assert receipt.revision == "b" * 40
    assert receipt.sha256 == sha256(content).hexdigest()
    assert api.uploads == 0


def test_upload_bytes_never_overwrites_existing_different_content():
    from hashlib import sha256

    existing_content = b"old"
    new_content = b"new"

    class Info:
        private = True
        sha = "c" * 40

    class ExistingApi:
        def __init__(self):
            self.info = Info()
            self.uploads = 0
        def whoami(self):
            return {"name": "alice"}
        def repo_info(self, **kwargs):
            return self.info
        def file_exists(self, **kwargs):
            return True
        def upload_file(self, **kwargs):
            self.uploads += 1
        def get_paths_info(self, **kwargs):
            lfs = type("Lfs", (), {"sha256": sha256(existing_content).hexdigest()})()
            return [type("PathInfo", (), {"lfs": lfs})()]

    api = ExistingApi()
    store = PrivateHubStore.connect(api_factory=lambda: api)
    with pytest.raises(ValueError, match="checksum"):
        store.upload_bytes("manifests/id/manifest-content.json", new_content)
    assert api.uploads == 0
