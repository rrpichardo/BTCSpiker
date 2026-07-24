"""
Pure-Python assembly for the /predictions/timeline endpoint.

No Kafka, no SQLite, no import side effects: takes the predictions LEFT JOIN
outcomes row list (same shape `materializer.py`'s PERFORMANCE_JOIN_SQL
produces) plus the requested window bounds, and returns chart-ready points
keyed on feature_ts. `materializer.py` owns the SQL query; this module does
the per-row classification and bucketing.

Every row is classified into exactly one of six states BEFORE any bucketing,
so a bucket can never pair one row's prediction with a different row's
outcome to manufacture a call that never happened. See `classify_row`.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta

from evaluation import LABEL_HORIZON_SECONDS, _parse_dt

MAX_TIMELINE_POINTS = 1000

# Candidate bucket widths (seconds) for auto-resolution — kept to human-
# legible steps so bucket boundaries land on clean times, not e.g. every 83s.
_RESOLUTION_STEPS = [
    1,
    2,
    5,
    10,
    15,
    30,
    60,
    120,
    300,
    600,
    900,
    1800,
    3600,
    7200,
    14400,
    21600,
    43200,
    86400,
]


def _auto_resolution(range_seconds: float) -> int:
    """Smallest step in `_RESOLUTION_STEPS` that keeps the bucket count within
    MAX_TIMELINE_POINTS; falls back to the largest step for very wide ranges."""
    for step in _RESOLUTION_STEPS:
        if range_seconds / step <= MAX_TIMELINE_POINTS:
            return step
    return _RESOLUTION_STEPS[-1]


def classify_row(row: dict, anchor_dt: datetime | None) -> str:
    """Classify one joined predictions+outcomes row into exactly one of:

    correct_call / false_alarm / missed_spike / correct_quiet -- outcome is
        present, scored on time (api_ts < written_at, same rule as
        evaluation._split_graded), and score/tau/vol_spike are all usable.
    pending -- no outcome yet, and feature_ts is strictly less than
        LABEL_HORIZON_SECONDS old relative to `anchor_dt` (the newest
        feature_ts in the whole table) -- ground truth genuinely hasn't
        matured yet. At exactly the horizon (age == LABEL_HORIZON_SECONDS)
        the featurizer has already had enough hindsight to emit a label
        (btcspiker_ml/features.py gates its skip on `age < horizon_seconds`,
        i.e. it labels once `age >= horizon_seconds`), so a still-missing
        outcome there is a pipeline gap, not "pending".
    unavailable -- everything else that can't be honestly graded: an
        outcome-less row older than the maturity horizon (outcome pipeline
        gap), a scored-late row (api_ts >= written_at -- the model's own
        score was recorded after we already knew the answer, so crediting it
        would let replay fake foresight), or malformed/missing score/tau/
        vol_spike/timestamps.

    anchor_dt is the caller-supplied "now" for maturity purposes -- always
    the newest feature_ts in the data (live or replayed), never wall-clock
    time, so replay and seeded fixtures behave identically to live traffic.
    """
    written_at = row.get("written_at")

    if written_at is None:
        feature_dt = _parse_dt(row.get("feature_ts"))
        if feature_dt is None or anchor_dt is None:
            return "unavailable"
        age_seconds = (anchor_dt - feature_dt).total_seconds()
        if 0 <= age_seconds < LABEL_HORIZON_SECONDS:
            return "pending"
        return "unavailable"

    api_dt = _parse_dt(row.get("api_ts"))
    written_dt = _parse_dt(written_at)
    if api_dt is None or written_dt is None or not (api_dt < written_dt):
        return "unavailable"  # scored late (or unparsable) -- never crediting foresight

    score = row.get("score")
    tau = row.get("tau")
    vol_spike = row.get("vol_spike")
    if (
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not isinstance(tau, (int, float))
        or isinstance(tau, bool)
        or vol_spike is None
    ):
        return "unavailable"

    predicted = score >= tau
    actual = bool(vol_spike)
    if predicted and actual:
        return "correct_call"
    if predicted and not actual:
        return "false_alarm"
    if not predicted and actual:
        return "missed_spike"
    return "correct_quiet"


def _raw_point(row: dict, cls: str) -> dict:
    return {
        "feature_ts": row.get("feature_ts"),
        "api_ts": row.get("api_ts"),
        "feature_id": row.get("feature_id"),
        "market_price": row.get("market_price"),
        "score": row.get("score"),
        "tau": row.get("tau"),
        "class": cls,
    }


def _bucket_points(
    classified_rows: list[tuple[dict, str]], from_dt: datetime, resolution: int
) -> list[dict]:
    buckets: dict[int, list[tuple[dict, str]]] = {}
    for row, cls in classified_rows:
        feature_dt = _parse_dt(row.get("feature_ts"))
        if feature_dt is None:
            continue
        key = int((feature_dt - from_dt).total_seconds() // resolution)
        buckets.setdefault(key, []).append((row, cls))

    points = []
    for key in sorted(buckets):
        bucket_rows = buckets[key]
        bucket_start = from_dt + timedelta(seconds=key * resolution)

        best_score_row = None
        best_score = None
        last_price_ts = None
        last_price = None
        classes: Counter = Counter()

        for row, cls in bucket_rows:
            classes[cls] += 1

            score = row.get("score")
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                if best_score is None or score > best_score:
                    best_score = score
                    best_score_row = row

            price = row.get("market_price")
            feature_ts = row.get("feature_ts")
            if isinstance(price, (int, float)) and not isinstance(price, bool):
                if last_price_ts is None or (feature_ts or "") >= last_price_ts:
                    last_price_ts = feature_ts
                    last_price = price

        points.append(
            {
                "feature_ts": bucket_start.isoformat(),
                "market_price": last_price,
                "score": best_score,
                "tau": best_score_row.get("tau") if best_score_row else None,
                "classes": dict(classes),
            }
        )
    return points


def build_timeline(
    joined_rows: list[dict],
    from_dt: datetime,
    to_dt: datetime,
    anchor_dt: datetime | None,
    resolution: int | None = None,
) -> tuple[list[dict], bool, int | None]:
    """Classify every row, then return (points, aggregated, resolution_used).

    joined_rows must already be filtered/ordered by feature_ts ascending
    within [from_dt, to_dt) -- this function does not re-filter by window.
    """
    classified = [(row, classify_row(row, anchor_dt)) for row in joined_rows]

    if resolution is None and len(classified) <= MAX_TIMELINE_POINTS:
        points = [_raw_point(row, cls) for row, cls in classified]
        return points, False, None

    range_seconds = max((to_dt - from_dt).total_seconds(), 1.0)
    resolution_used = resolution or _auto_resolution(range_seconds)
    points = _bucket_points(classified, from_dt, resolution_used)
    return points, True, resolution_used
