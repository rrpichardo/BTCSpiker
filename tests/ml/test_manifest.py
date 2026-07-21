from btcspiker_ml.manifest import DatasetManifest, manifest_id


def test_manifest_id_ignores_dictionary_order():
    left = DatasetManifest("coinbase", "BTC-USD", 10, "a", "b", {"x": 1, "y": 2}, [])
    right = DatasetManifest("coinbase", "BTC-USD", 10, "a", "b", {"y": 2, "x": 1}, [])
    assert manifest_id(left) == manifest_id(right)
