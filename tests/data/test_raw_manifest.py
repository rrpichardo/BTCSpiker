from datetime import datetime, timezone

from btcspiker_data.raw_manifest import RawDatasetManifest, raw_manifest_id


def test_manifest_id_ignores_dictionary_order():
    common = dict(
        source_revision="r1",
        source_url="https://example.test/data",
        repo_id="user/btcspiker-coinbase-history",
        revision="abc123",
        usage_scope="research",
        schemas={"trades": ["source", "trade_id"]},
        partitions=[{"remote_path": "raw/a", "row_count": 1, "sha256": "a" * 64}],
        coverage_seconds=1,
        missing_seconds=0,
        duplicate_counts={"trades": 0, "book": 0},
        sequence_incidents=[],
        excluded_intervals=[],
    )
    left = RawDatasetManifest(created_at=datetime(2026, 1, 1, tzinfo=timezone.utc), **common)
    right = RawDatasetManifest(created_at=datetime(2030, 1, 1, tzinfo=timezone.utc), **dict(reversed(list(common.items()))))
    assert raw_manifest_id(left) == raw_manifest_id(right)
