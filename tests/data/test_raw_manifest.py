from datetime import datetime, timezone

import pytest

from btcspiker_data.raw_manifest import RawDatasetManifest, raw_manifest_id


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
