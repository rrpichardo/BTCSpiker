import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from btcspiker_ml.eda import (
    DataProfile,
    profile_dataset,
    write_profile_artifacts,
)


def test_profile_exposes_time_span_duplicates_and_daily_prevalence():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z", "2026-01-01T00:00:01Z"]
            ),
            "vol_spike": [0, 1, 1],
            "price": [1.0, 1.1, 1.1],
        }
    )
    profile = profile_dataset(frame, "vol_spike", "timestamp")
    assert profile.rows == 3
    assert profile.duplicate_timestamps == 1
    assert profile.positive_rate == 2 / 3
    assert profile.start_time == "2026-01-01T00:00:00+00:00"


def test_profile_counts_non_finite_per_numeric_column():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:01Z",
                    "2026-01-01T00:00:02Z",
                    "2026-01-01T00:00:03Z",
                ]
            ),
            "vol_spike": [0, 1, 0, 1],
            "price": [1.0, np.nan, 3.0, np.inf],
            "spread_bps": [0.1, 0.2, -np.inf, 0.4],
        }
    )
    profile = profile_dataset(frame, "vol_spike", "timestamp")
    # price has one NaN + one +inf = 2 non-finite
    assert profile.non_finite_by_column["price"] == 2
    # spread_bps has one -inf = 1 non-finite
    assert profile.non_finite_by_column["spread_bps"] == 1
    # vol_spike (int) has no non-finite values
    assert profile.non_finite_by_column["vol_spike"] == 0


def test_daily_prevalence_groups_by_utc_day_not_local():
    # timestamps that straddle a UTC midnight but not local midnight in most
    # timezones. Two ticks pre-midnight UTC (2026-01-01), two post-midnight UTC
    # (2026-01-02). All would look like the same local day in most timezones,
    # but UTC-partitioned prevalence must see them as two distinct days.
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T23:30:00Z",
                    "2026-01-01T23:45:00Z",
                    "2026-01-02T00:15:00Z",
                    "2026-01-02T00:30:00Z",
                ]
            ),
            "vol_spike": [0, 0, 1, 1],
        }
    )
    profile = profile_dataset(frame, "vol_spike", "timestamp")
    assert set(profile.daily_prevalence.keys()) == {"2026-01-01", "2026-01-02"}
    assert profile.daily_prevalence["2026-01-01"] == 0.0
    assert profile.daily_prevalence["2026-01-02"] == 1.0


def test_write_profile_artifacts_roundtrips_profile_json(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:01Z",
                    "2026-01-01T00:00:02Z",
                    "2026-01-02T00:00:00Z",
                ]
            ),
            "vol_spike": [0, 1, 0, 1],
            "price": [1.0, 1.1, 1.05, 1.2],
        }
    )
    profile = profile_dataset(frame, "vol_spike", "timestamp")
    output_dir = tmp_path / "eda"
    paths = write_profile_artifacts(
        profile, frame, "vol_spike", "timestamp", output_dir
    )

    profile_json = output_dir / "profile.json"
    assert profile_json in paths
    payload = json.loads(profile_json.read_text())
    # Ensure JSON serialization matches original dataclass field-for-field.
    assert payload == asdict(profile)


def test_write_profile_artifacts_refuses_to_overwrite(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:01Z",
                ]
            ),
            "vol_spike": [0, 1],
            "price": [1.0, 1.1],
        }
    )
    profile = profile_dataset(frame, "vol_spike", "timestamp")
    output_dir = tmp_path / "eda"
    write_profile_artifacts(profile, frame, "vol_spike", "timestamp", output_dir)
    with pytest.raises(FileExistsError):
        write_profile_artifacts(profile, frame, "vol_spike", "timestamp", output_dir)


def test_write_profile_artifacts_produces_correlations_and_daily_csv(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:01Z",
                    "2026-01-01T00:00:02Z",
                    "2026-01-02T00:00:00Z",
                ]
            ),
            "vol_spike": [0, 1, 0, 1],
            "price": [1.0, 1.1, 1.05, 1.2],
            "log_return": [0.0, 0.09, -0.05, 0.14],
        }
    )
    profile = profile_dataset(frame, "vol_spike", "timestamp")
    output_dir = tmp_path / "eda"
    paths = write_profile_artifacts(
        profile, frame, "vol_spike", "timestamp", output_dir
    )

    daily_csv = output_dir / "daily_prevalence.csv"
    corr_csv = output_dir / "correlations.csv"
    assert daily_csv in paths
    assert corr_csv in paths
    daily = pd.read_csv(daily_csv)
    assert set(daily.columns) == {"day", "prevalence"}
    corr = pd.read_csv(corr_csv, index_col=0)
    # correlation matrix is square over the numeric columns
    assert corr.shape[0] == corr.shape[1]
    assert "price" in corr.columns


def test_profile_dataset_is_deterministic():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:01Z",
                    "2026-01-01T00:00:02Z",
                ]
            ),
            "vol_spike": [0, 1, 1],
            "price": [1.0, 1.1, 1.2],
        }
    )
    p1 = profile_dataset(frame.copy(), "vol_spike", "timestamp")
    p2 = profile_dataset(frame.copy(), "vol_spike", "timestamp")
    assert p1 == p2
    assert isinstance(p1, DataProfile)


def test_profile_reports_provisional_qualification_for_under_thirty_day_corpus():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-11T23:59:59Z",
                ]
            ),
            "vol_spike": [0, 1],
            "price": [1.0, 1.1],
        }
    )

    profile = profile_dataset(frame, "vol_spike", "timestamp")

    assert profile.coverage_days == pytest.approx(11.0)
    assert profile.qualification_data is False
    assert profile.neural_data_eligible is False
