import hashlib
import json

from btcspiker_ml.manifest import DatasetManifest, manifest_id


def test_manifest_id_ignores_dictionary_order():
    left = DatasetManifest("coinbase", "BTC-USD", 10, "a", "b", {"x": 1, "y": 2}, [])
    right = DatasetManifest("coinbase", "BTC-USD", 10, "a", "b", {"y": 2, "x": 1}, [])
    assert manifest_id(left) == manifest_id(right)


def test_optional_lineage_fields_are_deterministic_and_backward_compatible():
    base = DatasetManifest("coinbase", "BTC-USD", 10, "a", "b", {}, [])
    lineage = DatasetManifest(
        "coinbase",
        "BTC-USD",
        10,
        "a",
        "b",
        {},
        [],
        parent_dataset_id="raw-1",
        source_manifest_path="manifests/raw-1.json",
        feature_set_id="core_v1",
        feature_engine_git_sha="a" * 40,
        excluded_intervals=[{"start": "a", "end": "b"}],
    )
    assert base.parent_dataset_id is None
    assert manifest_id(lineage) == manifest_id(lineage)
    assert manifest_id(base) != manifest_id(lineage)
    legacy_payload = {
        "source": "coinbase",
        "product": "BTC-USD",
        "rows": 10,
        "start_time": "a",
        "end_time": "b",
        "quality": {},
        "partitions": [],
    }
    expected = hashlib.sha256(
        json.dumps(legacy_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert manifest_id(base) == expected
