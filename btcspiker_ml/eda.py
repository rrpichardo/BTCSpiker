"""Deterministic exploratory data analysis for the collected corpus.

Pure library: takes a pandas DataFrame plus target / timestamp column names and
produces a ``DataProfile`` snapshot plus a bundle of on-disk artifacts.  No
MLflow, no config parsing, no dataset resolution — those belong to the CLI
orchestrator in ``scripts/profile_dataset.py``.

Determinism matters: given the same input frame the profile JSON must be
byte-for-byte identical across runs, so downstream comparisons (e.g. "did the
corpus change between two publications?") reduce to a diff on the emitted
files.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import matplotlib

# Force a headless backend before importing pyplot; the CLI runs on
# machines without a display and any GUI backend would abort.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


AUTOCORR_LAG_SECONDS = (1, 5, 60, 300)
GAP_THRESHOLD_SECONDS = 60.0


@dataclass(frozen=True)
class DataProfile:
    """Snapshot of quality-relevant statistics for a labelled feature frame."""

    rows: int
    start_time: str
    end_time: str
    duplicate_timestamps: int
    positive_rate: float
    non_finite_by_column: dict[str, int]
    daily_prevalence: dict[str, float]
    coverage_days: float = 0.0
    qualification_data: bool = False
    neural_data_eligible: bool = False
    inter_arrival_ms: dict[str, float] = field(default_factory=dict)
    gap_summary: dict[str, float | str] = field(default_factory=dict)
    autocorrelation: dict[str, dict[str, float | None]] = field(default_factory=dict)
    daily_feature_means: dict[str, dict[str, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _utc_timestamps(frame: pd.DataFrame, timestamp_column: str) -> pd.DatetimeIndex:
    parsed = pd.to_datetime(frame[timestamp_column], utc=True)
    return pd.DatetimeIndex(parsed)


def _inter_arrival_ms(timestamps: pd.DatetimeIndex) -> dict[str, float]:
    """Median / p95 / p99 inter-arrival gap in milliseconds."""
    if len(timestamps) < 2:
        return {"median_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    deltas = timestamps.to_series().diff().dropna().dt.total_seconds() * 1000.0
    return {
        "median_ms": float(deltas.quantile(0.50)),
        "p95_ms": float(deltas.quantile(0.95)),
        "p99_ms": float(deltas.quantile(0.99)),
    }


def _gap_summary(timestamps: pd.DatetimeIndex) -> dict[str, float | str]:
    """Count gaps > threshold, longest gap length (s), and when it started."""
    if len(timestamps) < 2:
        return {
            "gap_threshold_seconds": GAP_THRESHOLD_SECONDS,
            "count_over_threshold": 0,
            "longest_gap_seconds": 0.0,
            "longest_gap_start": "",
        }
    # Work in positional space (numpy) to avoid ambiguity when the timestamp
    # index contains duplicates — .loc / idxmax return Series on repeated labels.
    seconds = np.asarray(
        timestamps.to_series().diff().dt.total_seconds().to_numpy(), dtype=float
    )
    valid_mask = ~np.isnan(seconds)
    over_threshold = int(np.sum(seconds[valid_mask] > GAP_THRESHOLD_SECONDS))
    if not valid_mask.any():
        longest_seconds = 0.0
        longest_start_iso = ""
    else:
        # The gap value at position i is timestamps[i] - timestamps[i-1], so
        # argmax gives the tick immediately *after* the biggest gap.
        position_after = int(np.nanargmax(seconds))
        longest_seconds = float(seconds[position_after])
        start_position = max(0, position_after - 1)
        longest_start_iso = timestamps[start_position].isoformat()
    return {
        "gap_threshold_seconds": GAP_THRESHOLD_SECONDS,
        "count_over_threshold": over_threshold,
        "longest_gap_seconds": longest_seconds,
        "longest_gap_start": longest_start_iso,
    }


def _autocorrelation(
    frame: pd.DataFrame, timestamps: pd.DatetimeIndex, columns: list[str]
) -> dict[str, dict[str, float]]:
    """Autocorrelation at AUTOCORR_LAG_SECONDS for the requested columns.

    Approach: resample the (irregular) tick series to 1-second bins by taking the
    mean within each bin, then compute Pearson autocorrelation at the requested
    integer-second lags.  This documents-and-simplifies the alignment: tick
    inter-arrival is sub-second, so aggregating to whole seconds is coarse
    enough to give a stable lag definition and cheap enough on ~800k rows.
    """
    result: dict[str, dict[str, float | None]] = {}
    if not columns or len(timestamps) < 2:
        return result
    for column in columns:
        if column not in frame.columns:
            continue
        series = pd.Series(frame[column].to_numpy(), index=timestamps, copy=False)
        # Coerce to float so booleans / ints all resample cleanly.
        series = series.astype(float, copy=False)
        resampled = series.resample("1s").mean()
        column_result: dict[str, float | None] = {}
        for lag in AUTOCORR_LAG_SECONDS:
            if len(resampled) <= lag:
                # Record as None (JSON null) rather than NaN so profile.json is
                # strict JSON and DataProfile equality stays reflexive.
                column_result[f"lag_{lag}s"] = None
                continue
            paired = pd.concat(
                [resampled, resampled.shift(lag)], axis=1, copy=False
            ).dropna()
            if (
                len(paired) < 2
                or paired.iloc[:, 0].nunique() < 2
                or paired.iloc[:, 1].nunique() < 2
            ):
                column_result[f"lag_{lag}s"] = None
                continue
            value = paired.iloc[:, 0].corr(paired.iloc[:, 1])
            column_result[f"lag_{lag}s"] = float(value) if pd.notna(value) else None
        result[column] = column_result
    return result


def _daily_feature_means(
    frame: pd.DataFrame, timestamps: pd.DatetimeIndex, columns: list[str]
) -> dict[str, dict[str, float]]:
    """Per-UTC-day mean for each requested column."""
    if not columns:
        return {}
    days = timestamps.strftime("%Y-%m-%d")
    result: dict[str, dict[str, float]] = {}
    for column in columns:
        if column not in frame.columns:
            continue
        try:
            grouped = frame[column].groupby(days).mean()
        except TypeError:
            continue
        result[column] = {str(day): float(value) for day, value in grouped.items()}
    return result


# ---------------------------------------------------------------------------
# Core profile
# ---------------------------------------------------------------------------

def profile_dataset(
    frame: pd.DataFrame, target_column: str, timestamp_column: str
) -> DataProfile:
    """Summarise the labelled feature frame in a deterministic, JSON-friendly form.

    Sort the incoming frame by timestamp before computing quality stats so gap
    and autocorrelation calculations are order-invariant.  Duplicate timestamps
    (i.e. two ticks with the same event time) still count as duplicates after
    the sort.
    """
    ordered = frame.sort_values(timestamp_column, kind="mergesort").reset_index(drop=True)
    timestamps = _utc_timestamps(ordered, timestamp_column)
    numeric = ordered.select_dtypes(include=[np.number])
    daily = (
        ordered.assign(_day=timestamps.strftime("%Y-%m-%d"))
        .groupby("_day", sort=True)[target_column]
        .mean()
    )
    coverage_days = float((timestamps[-1].date() - timestamps[0].date()).days + 1)
    positive_rows = int(ordered[target_column].sum())
    autocorr_targets = [
        column for column in (target_column, "log_return") if column in ordered.columns
    ]
    feature_mean_targets = [
        column
        for column in ("log_return", "vol_60s", "spread_bps", "trade_intensity_60s")
        if column in ordered.columns
    ]
    return DataProfile(
        rows=int(len(ordered)),
        start_time=timestamps[0].isoformat(),
        end_time=timestamps[-1].isoformat(),
        duplicate_timestamps=int(timestamps.duplicated().sum()),
        positive_rate=float(ordered[target_column].mean()),
        non_finite_by_column={
            column: int((~np.isfinite(numeric[column])).sum()) for column in numeric.columns
        },
        daily_prevalence={str(day): float(value) for day, value in daily.items()},
        # These are data-only gates. Candidate qualification and the final neural
        # stage also require later-task validation and search-state evidence.
        coverage_days=coverage_days,
        qualification_data=coverage_days >= 30.0,
        neural_data_eligible=len(ordered) >= 100_000 and positive_rows >= 500,
        inter_arrival_ms=_inter_arrival_ms(timestamps),
        gap_summary=_gap_summary(timestamps),
        autocorrelation=_autocorrelation(ordered, timestamps, autocorr_targets),
        daily_feature_means=_daily_feature_means(
            ordered, timestamps, feature_mean_targets
        ),
    )


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------

def _write_prevalence_plot(
    daily_prevalence: dict[str, float], output_path: Path
) -> None:
    """Bar plot: target prevalence per UTC day."""
    fig, axis = plt.subplots(figsize=(8, 4))
    if daily_prevalence:
        days = list(daily_prevalence.keys())
        values = [daily_prevalence[day] for day in days]
        axis.bar(range(len(days)), values, color="#2b7bba")
        axis.set_xticks(range(len(days)))
        axis.set_xticklabels(days, rotation=45, ha="right")
    axis.set_title("Target prevalence per UTC day")
    axis.set_ylabel("positive rate")
    axis.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def _write_daily_feature_means_plot(
    daily_feature_means: dict[str, dict[str, float]], output_path: Path
) -> None:
    """Line plot: per-day mean of key features (drift visual)."""
    fig, axis = plt.subplots(figsize=(8, 4))
    for feature_name, values in daily_feature_means.items():
        if not values:
            continue
        days = list(values.keys())
        y = [values[day] for day in days]
        axis.plot(range(len(days)), y, marker="o", label=feature_name)
        axis.set_xticks(range(len(days)))
        axis.set_xticklabels(days, rotation=45, ha="right")
    axis.set_title("Daily mean of key features (drift signal)")
    axis.set_ylabel("daily mean")
    if daily_feature_means:
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def _write_inter_arrival_plot(
    timestamps: pd.DatetimeIndex, output_path: Path
) -> None:
    """Histogram of inter-arrival gaps in seconds (log-scaled x)."""
    fig, axis = plt.subplots(figsize=(8, 4))
    if len(timestamps) >= 2:
        deltas = timestamps.to_series().diff().dropna().dt.total_seconds()
        # Clip to a positive minimum so log-scale doesn't choke on zero-second
        # gaps (duplicate timestamps).
        positive = deltas.clip(lower=1e-6)
        axis.hist(positive, bins=50, color="#2b7bba", log=True)
        axis.set_xscale("log")
    axis.set_title("Inter-arrival gap distribution")
    axis.set_xlabel("gap (s, log)")
    axis.set_ylabel("count (log)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def write_profile_artifacts(
    profile: DataProfile,
    frame: pd.DataFrame,
    target_column: str,
    timestamp_column: str,
    output_dir: Path,
) -> list[Path]:
    """Serialise a DataProfile plus supporting CSV/PNG diagnostics.

    Refuses to overwrite an existing directory: silent overwrites would erase
    prior EDA snapshots that MLflow may already reference.  Callers should
    pass a fresh (nonexistent) path — the CLI mints one under
    ``artifact_root/eda/<dataset_id>/<utc-stamp>/`` to keep prior runs.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing EDA directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)

    written: list[Path] = []

    # 1. profile.json — canonical, sorted keys, trailing newline.
    profile_path = output_dir / "profile.json"
    profile_path.write_text(
        json.dumps(asdict(profile), indent=2, sort_keys=True) + "\n"
    )
    written.append(profile_path)

    # 2. daily_prevalence.csv — ordered by day.
    prevalence_path = output_dir / "daily_prevalence.csv"
    if profile.daily_prevalence:
        prevalence_df = pd.DataFrame(
            sorted(profile.daily_prevalence.items()),
            columns=["day", "prevalence"],
        )
    else:
        prevalence_df = pd.DataFrame(columns=["day", "prevalence"])
    prevalence_df.to_csv(prevalence_path, index=False)
    written.append(prevalence_path)

    # 3. correlations.csv — only when numeric features exist.
    numeric = frame.select_dtypes(include=[np.number])
    if not numeric.empty:
        correlations_path = output_dir / "correlations.csv"
        numeric.corr(method="pearson").to_csv(correlations_path)
        written.append(correlations_path)

    # 4. daily_feature_means.csv — one row per day, one column per feature.
    if profile.daily_feature_means:
        feature_means_path = output_dir / "daily_feature_means.csv"
        pd.DataFrame(profile.daily_feature_means).sort_index().to_csv(feature_means_path)
        written.append(feature_means_path)

    # 5. PNG diagnostics.
    prevalence_png = output_dir / "target_prevalence.png"
    _write_prevalence_plot(profile.daily_prevalence, prevalence_png)
    written.append(prevalence_png)

    if profile.daily_feature_means:
        feature_drift_png = output_dir / "daily_feature_means.png"
        _write_daily_feature_means_plot(profile.daily_feature_means, feature_drift_png)
        written.append(feature_drift_png)

    inter_arrival_png = output_dir / "inter_arrival_gaps.png"
    timestamps = _utc_timestamps(frame.sort_values(timestamp_column), timestamp_column)
    _write_inter_arrival_plot(timestamps, inter_arrival_png)
    written.append(inter_arrival_png)

    return written
