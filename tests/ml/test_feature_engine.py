import pandas as pd
import pytest

from btcspiker_ml.features import FEATURE_SETS, FeatureEngine, materialize_features


def test_microstructure_v1_materializes_the_frozen_level2_schema(raw_ticks):
    level2_ticks = [
        {**tick, "bid_size": "2.0", "ask_size": "1.0"} for tick in raw_ticks
    ]

    materialized = materialize_features(pd.DataFrame(level2_ticks), "microstructure_v1")

    assert set(FEATURE_SETS["microstructure_v1"].columns).issubset(materialized.columns)


def test_microstructure_v1_reports_missing_level2_inputs(raw_ticks):
    with pytest.raises(
        ValueError, match=r"missing raw columns: \['ask_size', 'bid_size'\]"
    ):
        materialize_features(pd.DataFrame(raw_ticks), "microstructure_v1")


def test_batch_and_stream_features_match(raw_ticks):
    batch = materialize_features(pd.DataFrame(raw_ticks), "core_v1")
    engine = FeatureEngine("core_v1", horizon_seconds=60, threshold=0.000048)
    streamed = [row for tick in raw_ticks for row in engine.ingest(tick)]

    # A non-empty assertion prevents an empty-vs-empty parity pass when the
    # fixture does not span the delayed 60-second label horizon.
    assert streamed
    pd.testing.assert_frame_equal(
        batch.reset_index(drop=True),
        pd.DataFrame(streamed).reset_index(drop=True),
        check_exact=False,
        rtol=1e-10,
        atol=1e-12,
    )


def test_features_do_not_change_when_future_ticks_are_modified(raw_ticks):
    cutoff = len(raw_ticks) // 2
    original = materialize_features(pd.DataFrame(raw_ticks), "multi_window_v1")
    mutated_ticks = [dict(row) for row in raw_ticks]
    for row in mutated_ticks[cutoff + 1 :]:
        row["price"] = str(float(row["price"]) * 10)
    mutated = materialize_features(pd.DataFrame(mutated_ticks), "multi_window_v1")

    causal_columns = list(FEATURE_SETS["multi_window_v1"].columns)
    pd.testing.assert_frame_equal(
        original.loc[:, causal_columns].iloc[:cutoff].reset_index(drop=True),
        mutated.loc[:, causal_columns].iloc[:cutoff].reset_index(drop=True),
        check_exact=False,
        rtol=1e-10,
        atol=1e-12,
    )
