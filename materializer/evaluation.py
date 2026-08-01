"""
Pure-Python evaluation layer for the /predictions/performance snapshot.

No Kafka, no SQLite, no import side effects: every function here takes plain
dicts/lists and returns plain dicts. `materializer.py` owns the SQL LEFT JOIN
(predictions LEFT JOIN outcomes on feature_id) and calls `compute_performance`
with the resulting row list; this module just does the grading math.

Stdlib only — no numpy/sklearn, matching the materializer's existing
dependency footprint (confluent-kafka/fastapi/uvicorn).

`compute_performance` can only see what's in `joined_rows`, so it computes
`window.n_joined/n_graded/n_scored_late/n_predictions_unmatched/
median_lead_seconds` from that list. It CANNOT know about outcomes that never
matched a prediction (a LEFT JOIN rooted at predictions never surfaces those
rows) or the requested window's actual timestamp bounds — those are the
caller's responsibility to merge into `window` afterward (see
`materializer.py`'s `/predictions/performance` handler).
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime
from statistics import median

# ---------------------------------------------------------------------------
# Reference constants — held-out evaluation numbers, not computed here.
# Source: handoff/docs/model_card_v1.md
# ---------------------------------------------------------------------------
REFERENCE = {
    "source": "handoff/docs/model_card_v1.md (held-out evaluation)",
    "ml": {
        "pr_auc_test": 0.1459,
        "f1_test": 0.1359,
        "pr_auc_val": 0.3580,
        "f1_val": 0.4463,
    },
    "baseline_reference": {
        "pr_auc_test": 0.1340,
        "f1_test": 0.1487,
        "pr_auc_val": 0.3270,
        "f1_val": 0.3868,
        "scorer_note": (
            "sigmoid-calibrated z-score (training-time) — a different scorer "
            "than the live binary baseline"
        ),
    },
}

TRAINING_BENCHMARK_PR_AUC = REFERENCE["ml"]["pr_auc_test"]
FALLBACK_TAU = 0.7015
LABEL_HORIZON_SECONDS = 60
BASELINE_DECISION_THRESHOLD = 0.5
MAX_CHART_POINTS = 1500


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def tie_aware_average_precision(scores: list[float], labels: list[int]) -> float | None:
    """Average precision with sklearn-equivalent tie handling.

    Scores are processed in descending order, but groups of EQUAL scores are
    treated as one unit: the whole tied group's TP/FP counts are accumulated
    before taking the precision/recall step for that group. This matters
    because a naive per-row AP implementation over-credits ties (it can rank
    a false positive "before" a tied true positive purely by iteration
    order); grouping removes that artifact. Sanity check: if every score is
    tied, there's exactly one group and AP collapses to the prevalence.

    Returns None if there are zero positives in `labels` (undefined).
    """
    total_positives = sum(1 for label in labels if label)
    if total_positives == 0:
        return None

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    groups: list[list[int]] = []
    for i in order:
        if groups and scores[groups[-1][0]] == scores[i]:
            groups[-1].append(i)
        else:
            groups.append([i])

    cum_tp = 0
    cum_fp = 0
    prev_recall = 0.0
    ap = 0.0
    for group in groups:
        group_tp = sum(1 for i in group if labels[i])
        group_fp = len(group) - group_tp
        cum_tp += group_tp
        cum_fp += group_fp
        recall = cum_tp / total_positives
        precision = cum_tp / (cum_tp + cum_fp)
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return ap


def confusion(scores: list[float], labels: list[int], threshold: float) -> dict:
    """Confusion matrix + derived metrics for the `score >= threshold` rule.

    Pinned conventions (see module callers): precision is 0.0 (not None)
    when the model alerted but never correctly; precision is None only when
    the model never alerted at all (tp + fp == 0, genuinely undefined).
    Symmetrically, recall is None only when there are zero real positives.
    f1 is None iff either input metric is None; 0.0 is otherwise a valid f1.
    """
    tp = fp = fn = tn = 0
    for score, label in zip(scores, labels):
        alerted = score >= threshold
        actual = bool(label)
        if alerted and actual:
            tp += 1
        elif alerted and not actual:
            fp += 1
        elif not alerted and actual:
            fn += 1
        else:
            tn += 1

    if tp + fp == 0:
        precision = None
    elif tp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    recall = None if tp + fn == 0 else tp / (tp + fn)

    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def adaptive_threshold(vols: list[float], percentile: float = 85) -> float | None:
    """Nearest-rank percentile of `vols`. None below 50 samples (unreliable)."""
    n = len(vols)
    if n < 50:
        return None
    ordered = sorted(vols)
    rank = math.ceil(percentile / 100 * n) - 1
    rank = max(0, min(n - 1, rank))
    return ordered[rank]


# ---------------------------------------------------------------------------
# Timestamp / grading helpers
# ---------------------------------------------------------------------------


def _parse_dt(value) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _split_graded(joined_rows: list[dict]) -> tuple[list[dict], list[float], int, int]:
    """Split joined rows into (graded_rows, lead_seconds, n_scored_late,
    n_predictions_unmatched) per the online-grading rule:

    A row is GRADED only if it has a non-null outcome (written_at is set)
    AND api_ts < written_at — the prediction's own scoring timestamp must
    strictly precede the moment its outcome fact was written to our DB, i.e.
    wall-clock ordering of OUR OWN writes (works identically in replay and
    live modes; unlike feature_ts comparisons it can't be gamed by replay
    speed). Rows with an outcome but api_ts >= written_at count as
    "scored late" — excluded from grading, never silently folded in. Rows
    with no outcome yet aren't graded or late; they're just pending.
    """
    graded = []
    lead_seconds = []
    n_scored_late = 0
    n_predictions_unmatched = 0

    for row in joined_rows:
        if row.get("written_at") is None:
            n_predictions_unmatched += 1
            continue
        if not isinstance(row.get("score"), (int, float)):
            # Malformed/old-format prediction (e.g. missing score) that still
            # has a matched outcome — exclude rather than crash grading math
            # downstream (sorting/thresholding assumes a numeric score).
            continue
        api_dt = _parse_dt(row.get("api_ts"))
        written_dt = _parse_dt(row.get("written_at"))
        if api_dt is None or written_dt is None:
            # Can't establish ordering — exclude from both graded and late
            # counts rather than guess.
            continue
        if api_dt < written_dt:
            lead = (written_dt - api_dt).total_seconds()
            graded.append(row)
            lead_seconds.append(lead)
        else:
            n_scored_late += 1

    return graded, lead_seconds, n_scored_late, n_predictions_unmatched


def _group_key(row: dict) -> tuple:
    return (row.get("model_variant"), row.get("model_version"), row.get("run_id"))


def _group_tau(rows: list[dict]) -> tuple[float, str]:
    taus = [r["tau"] for r in rows if r.get("tau") is not None]
    if taus:
        return Counter(taus).most_common(1)[0][0], "model"
    return FALLBACK_TAU, "fallback"


def _series_name(model_variant: str, model_version) -> str:
    if model_variant == "baseline":
        return "Baseline"
    return f"ML {model_version}" if model_version else "ML"


def _build_ml_series(rows: list[dict], labels: list[int]) -> dict:
    """Build an ml series entry, minus pr_auc gating (done by the caller,
    which knows `min_positives`) — `scores`/`labels` are carried along so
    `_finalize_series_pr_auc` can compute AP without recomputing the group.
    """
    tau, tau_source = _group_tau(rows)
    scores = [r.get("score") for r in rows]
    n = len(rows)
    n_positives = sum(1 for label in labels if label)
    stats = confusion(scores, labels, tau)

    key = _group_key(rows[0])
    return {
        "name": _series_name("ml", key[1]),
        "model_variant": "ml",
        "model_version": key[1],
        "run_id": key[2],
        "tau": tau,
        "tau_source": tau_source,
        "n": n,
        "n_positives": n_positives,
        "scores": scores,  # stripped before returning to caller
        "labels": labels,  # stripped before returning to caller
        **stats,
    }


def _finalize_series_pr_auc(series: dict, min_positives: int) -> dict:
    scores = series.pop("scores")
    labels = series.pop("labels")
    if series["n_positives"] >= min_positives:
        series["pr_auc"] = tie_aware_average_precision(scores, labels)
        series["pr_auc_reason"] = None
    else:
        series["pr_auc"] = None
        series["pr_auc_reason"] = (
            f"fewer than {min_positives} real spikes in this window — "
            "PR-AUC would be unreliable"
        )
    return series


def _build_baseline_series(
    rows: list[dict], labels: list[int], baseline_vol_threshold: float
) -> dict:
    baseline_scores = [
        1.0 if (r.get("vol_60s") or 0.0) > baseline_vol_threshold else 0.0 for r in rows
    ]
    stats = confusion(baseline_scores, labels, BASELINE_DECISION_THRESHOLD)
    return {
        "name": "Baseline",
        "model_variant": "baseline",
        "model_version": None,
        "run_id": None,
        "tau": None,
        "tau_source": None,
        "n": len(rows),
        "n_positives": sum(1 for label in labels if label),
        "pr_auc": None,
        "pr_auc_reason": "binary scorer — ranking metrics don't apply",
        **stats,
    }


def _benchmark_note(
    mode_key: str,
    ml_series: list[dict],
    n_graded: int,
    min_note_sample_n: int,
    drift_pr_auc_ratio: float,
) -> dict:
    if mode_key != "official":
        return {
            "triggered": False,
            "detail": "suppressed — activity view is self-normalizing",
        }

    primary = max(ml_series, key=lambda s: s["n"]) if ml_series else None
    if primary is None or primary["pr_auc"] is None or n_graded < min_note_sample_n:
        return {"triggered": False, "detail": "not enough graded data yet"}

    pr_auc = primary["pr_auc"]
    if pr_auc < TRAINING_BENCHMARK_PR_AUC * drift_pr_auc_ratio:
        detail = (
            f"Live PR-AUC ({pr_auc:.3f}) is well below the training benchmark (0.146) "
            "in this window. This can mean model drift or a different market regime — "
            "the project's retraining trigger uses a 7-day window; investigate before acting."
        )
        return {"triggered": True, "detail": detail}
    return {
        "triggered": False,
        "detail": "no drift detected — live PR-AUC is within expected range",
    }


def _build_mode(
    mode_key: str,
    graded_rows: list[dict],
    labels: list[int],
    *,
    min_positives: int,
    min_note_sample_n: int,
    drift_pr_auc_ratio: float,
    baseline_vol_threshold: float,
    threshold_info: str,
    percentile: float | None = None,
) -> dict:
    n_graded = len(graded_rows)
    n_positives = sum(1 for label in labels if label)
    base_rate = (n_positives / n_graded) if n_graded else None

    groups: dict[tuple, list[int]] = {}
    for idx, row in enumerate(graded_rows):
        groups.setdefault(_group_key(row), []).append(idx)

    ml_series = []
    for key, idx_list in groups.items():
        if key[0] != "ml":
            continue
        group_rows = [graded_rows[i] for i in idx_list]
        group_labels = [labels[i] for i in idx_list]
        series = _build_ml_series(group_rows, group_labels)
        series = _finalize_series_pr_auc(series, min_positives)
        ml_series.append(series)

    series_list = list(ml_series)
    if n_graded:
        series_list.append(
            _build_baseline_series(graded_rows, labels, baseline_vol_threshold)
        )

    note = _benchmark_note(
        mode_key, ml_series, n_graded, min_note_sample_n, drift_pr_auc_ratio
    )

    return {
        "threshold_info": threshold_info,
        "base_rate": base_rate,
        "ml_available": len(ml_series) > 0,
        "series": series_list,
        "note": note,
        "percentile": percentile,
    }


# ---------------------------------------------------------------------------
# Chart rows
# ---------------------------------------------------------------------------


def _chart_rows(
    joined_rows: list[dict],
    adaptive_thresh: float | None,
    baseline_vol_threshold: float,
) -> tuple[int | None, list[dict]]:
    epochs = {
        r["stream_epoch"] for r in joined_rows if r.get("stream_epoch") is not None
    }
    max_epoch = max(epochs) if epochs else None

    if max_epoch is None:
        selected = list(joined_rows)
    else:
        selected = [r for r in joined_rows if r.get("stream_epoch") == max_epoch]

    selected = sorted(selected, key=lambda r: r.get("feature_ts") or "")

    rows = []
    for row in selected:
        has_outcome = row.get("written_at") is not None
        vol_60s = row.get("vol_60s")
        baseline_score = (
            1.0 if (vol_60s is not None and vol_60s > baseline_vol_threshold) else 0.0
        )
        outcome_official = None
        outcome_adaptive = None
        if has_outcome:
            vol_spike = row.get("vol_spike")
            outcome_official = bool(vol_spike) if vol_spike is not None else None
            future_vol = row.get("future_vol_60s")
            if adaptive_thresh is not None and future_vol is not None:
                outcome_adaptive = future_vol > adaptive_thresh
        rows.append(
            {
                "feature_ts": row.get("feature_ts"),
                "score": row.get("score"),
                "model_variant": row.get("model_variant"),
                "baseline_score": baseline_score,
                "outcome_official": outcome_official,
                "outcome_adaptive": outcome_adaptive,
            }
        )

    return max_epoch, _downsample_chart_rows(rows)


def _downsample_chart_rows(
    rows: list[dict], max_points: int = MAX_CHART_POINTS
) -> list[dict]:
    """Thin `rows` to <= max_points, outcome-aware.

    Never drops a row with outcome_official or outcome_adaptive True, and
    never drops the first or last row. Thinning only removes rows from runs
    of outcome-false/None rows, evenly across those runs.
    """
    n = len(rows)
    if n <= max_points:
        return rows

    must_keep = {0, n - 1}
    for i, row in enumerate(rows):
        if row["outcome_official"] is True or row["outcome_adaptive"] is True:
            must_keep.add(i)

    if len(must_keep) >= max_points:
        return [rows[i] for i in sorted(must_keep)]

    fillable = [i for i in range(n) if i not in must_keep]
    budget = max_points - len(must_keep)
    step = len(fillable) / budget
    chosen = set()
    pos = 0.0
    while len(chosen) < budget and int(pos) < len(fillable):
        chosen.add(fillable[int(pos)])
        pos += step

    combined = sorted(must_keep | chosen)
    return [rows[i] for i in combined]


# ---------------------------------------------------------------------------
# Top-level assembly
# ---------------------------------------------------------------------------


def compute_performance(
    joined_rows: list[dict],
    *,
    min_positives: int = 10,
    min_note_sample_n: int = 1000,
    drift_pr_auc_ratio: float = 0.7,
    baseline_vol_threshold: float = 4.8e-5,
    adaptive_percentile: float = 85,
    stream_id: str | None = None,
) -> dict:
    """Assemble the evaluation half of the /predictions/performance snapshot.

    `joined_rows` is the result of predictions LEFT JOIN outcomes (see module
    docstring) — every prediction column plus the outcome-only columns
    (future_vol_60s, vol_spike, label_schema, written_at).

    `stream_id` identifies the single ingest segment `joined_rows` was
    already scoped to by the caller's SQL (materializer.py's
    performance_window) — it is not used to filter here. Echoing it in
    `window` lets a caller (or a test) confirm the metrics and the chart
    (`chart.stream_epoch`) describe the same population, rather than the
    chart silently re-deriving its own notion of "current" via max().

    The returned `window` sub-dict only carries what's derivable from
    `joined_rows` (n_joined, n_graded, n_scored_late, n_predictions_unmatched,
    median_lead_seconds, stream_id). The caller merges in from_feature_ts,
    to_feature_ts, n_outcomes_unmatched, and n_unknown_lineage, which require
    separate SQL queries. Likewise `as_of`, `window_minutes`, and `complete`
    are added by the caller.
    """
    n_joined = len(joined_rows)
    graded_rows, lead_seconds, n_scored_late, n_predictions_unmatched = _split_graded(
        joined_rows
    )
    n_graded = len(graded_rows)
    median_lead_seconds = median(lead_seconds) if lead_seconds else None

    official_labels = [
        int(row.get("vol_spike") or 0) if row.get("vol_spike") is not None else 0
        for row in graded_rows
    ]
    official_threshold_info = (
        f"score >= tau (each model's own shipped decision threshold); "
        f"baseline alerts when vol_60s > {baseline_vol_threshold:.3g}"
    )
    official_mode = _build_mode(
        "official",
        graded_rows,
        official_labels,
        min_positives=min_positives,
        min_note_sample_n=min_note_sample_n,
        drift_pr_auc_ratio=drift_pr_auc_ratio,
        baseline_vol_threshold=baseline_vol_threshold,
        threshold_info=official_threshold_info,
    )

    future_vols = [
        row.get("future_vol_60s")
        for row in graded_rows
        if row.get("future_vol_60s") is not None
    ]
    adaptive_thresh = adaptive_threshold(future_vols, adaptive_percentile)
    if adaptive_thresh is None:
        adaptive_mode = {
            "threshold_info": (
                "insufficient graded samples (<50) to compute an adaptive threshold"
            ),
            "base_rate": None,
            "ml_available": False,
            "series": [],
            "note": {
                "triggered": False,
                "detail": "suppressed — activity view is self-normalizing",
            },
            "percentile": adaptive_percentile,
        }
    else:
        # `is not None` (not `or float("-inf")`): a genuine future_vol_60s of
        # 0.0 is falsy in Python, so the old `or` idiom silently substituted
        # -inf for it here even though adaptive_threshold() above correctly
        # included that same 0.0 via `is not None`. Doesn't change any label
        # in practice (realized volatility is never negative, so 0.0 and
        # -inf both fail a `> adaptive_thresh` comparison the same way) --
        # fixed for correctness/clarity, not because it was misgrading rows.
        adaptive_labels = [
            1 if row.get("future_vol_60s") is not None and row["future_vol_60s"] > adaptive_thresh else 0
            for row in graded_rows
        ]
        adaptive_threshold_info = (
            f"score >= tau; spike = future_vol_60s above this window's "
            f"P{adaptive_percentile:g} ({adaptive_thresh:.3g})"
        )
        adaptive_mode = _build_mode(
            "adaptive",
            graded_rows,
            adaptive_labels,
            min_positives=min_positives,
            min_note_sample_n=min_note_sample_n,
            drift_pr_auc_ratio=drift_pr_auc_ratio,
            baseline_vol_threshold=baseline_vol_threshold,
            threshold_info=adaptive_threshold_info,
            percentile=adaptive_percentile,
        )

    max_epoch, chart_rows = _chart_rows(
        joined_rows, adaptive_thresh, baseline_vol_threshold
    )

    return {
        "window": {
            "n_joined": n_joined,
            "n_graded": n_graded,
            "n_scored_late": n_scored_late,
            "n_predictions_unmatched": n_predictions_unmatched,
            "median_lead_seconds": median_lead_seconds,
            "stream_id": stream_id,
        },
        "params": {
            "label_horizon_seconds": LABEL_HORIZON_SECONDS,
            "min_positives": min_positives,
            "adaptive_percentile": adaptive_percentile,
        },
        "modes": {
            "official": official_mode,
            "adaptive": adaptive_mode,
        },
        "reference": REFERENCE,
        "chart": {
            "stream_epoch": max_epoch,
            # Post-downsample count -- compare against window.n_joined (the
            # pre-downsample source population) to see how much _chart_rows'
            # MAX_CHART_POINTS cap actually thinned.
            "rendered_row_count": len(chart_rows),
            "rows": chart_rows,
        },
    }
