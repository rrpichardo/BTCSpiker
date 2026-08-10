"""Unit tests for the evaluation layer (materializer/evaluation.py) plus a
couple of endpoint-level checks for /predictions/performance that live here
because that's where the grading-behavior tests naturally belong.

Run from the repo root with:

    python3 -m pytest tests/test_evaluation.py -q
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATERIALIZER_DIR = PROJECT_ROOT / "materializer"
sys.path.insert(0, str(MATERIALIZER_DIR))

import evaluation  # noqa: E402
import materializer  # noqa: E402

# ---------------------------------------------------------------------------
# tie_aware_average_precision
# ---------------------------------------------------------------------------


def test_ap_unique_scores_hand_computed():
    scores = [0.9, 0.8, 0.7, 0.6]
    labels = [1, 0, 1, 0]

    ap = evaluation.tie_aware_average_precision(scores, labels)

    assert ap == pytest.approx((1.0 + 2 / 3) / 2, abs=1e-9)


def test_ap_all_tied_scores_equals_prevalence():
    # This is the bug a naive (non-tie-grouped) AP implementation gets wrong:
    # with every score identical, ranking carries no information, so AP must
    # collapse to exactly the prevalence (2/4 positives here).
    scores = [0.5, 0.5, 0.5, 0.5]
    labels = [1, 0, 1, 0]

    ap = evaluation.tie_aware_average_precision(scores, labels)

    assert ap == pytest.approx(0.5, abs=1e-12)


def test_ap_mixed_tie_groups_hand_computed():
    # Groups (descending): [0.9]->tp1, [0.7,0.7]->tp1/fp1, [0.5,0.5,0.5]->tp1/fp2
    # AP = 1/3*(1) + 1/3*(2/3) + 1/3*(1/2) = 13/18
    scores = [0.9, 0.7, 0.7, 0.5, 0.5, 0.5]
    labels = [1, 1, 0, 1, 0, 0]

    ap = evaluation.tie_aware_average_precision(scores, labels)

    assert ap == pytest.approx(13 / 18, abs=1e-9)


def test_ap_returns_none_with_zero_positives():
    assert evaluation.tie_aware_average_precision([0.9, 0.1], [0, 0]) is None


# ---------------------------------------------------------------------------
# confusion()
# ---------------------------------------------------------------------------


def test_confusion_precision_zero_when_tp_zero_fp_positive():
    result = evaluation.confusion([0.9, 0.8], [0, 0], threshold=0.5)
    assert result["tp"] == 0
    assert result["fp"] == 2
    assert result["precision"] == 0.0


def test_confusion_precision_none_when_never_alerted():
    result = evaluation.confusion([0.1, 0.2], [1, 0], threshold=0.5)
    assert result["tp"] == 0
    assert result["fp"] == 0
    assert result["precision"] is None


def test_confusion_recall_none_only_with_zero_positives():
    # Zero real positives -> recall undefined regardless of alerts.
    result = evaluation.confusion([0.9, 0.1], [0, 0], threshold=0.5)
    assert result["recall"] is None

    # Positives exist but all missed -> recall is a valid 0.0, not None.
    result = evaluation.confusion([0.1, 0.1], [1, 1], threshold=0.5)
    assert result["recall"] == 0.0


def test_confusion_f1_propagation():
    # precision or recall None -> f1 None.
    result = evaluation.confusion([0.1, 0.2], [1, 0], threshold=0.5)
    assert result["precision"] is None
    assert result["f1"] is None

    # Both defined and zero -> f1 is a valid 0.0.
    result = evaluation.confusion([0.9, 0.9], [0, 0], threshold=0.5)
    assert result["precision"] == 0.0
    assert result["recall"] is None  # zero positives here too
    assert result["f1"] is None

    # Both positive -> standard harmonic mean.
    result = evaluation.confusion([0.9, 0.9, 0.1], [1, 0, 1], threshold=0.5)
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# adaptive_threshold()
# ---------------------------------------------------------------------------


def test_adaptive_threshold_nearest_rank_hand_fixture():
    vols = list(range(1, 101))  # 1..100
    # nearest-rank P85: ceil(0.85*100) - 1 = 84 (0-indexed) -> value 85
    assert evaluation.adaptive_threshold(vols, percentile=85) == 85


def test_adaptive_threshold_below_50_samples_is_none():
    assert evaluation.adaptive_threshold(list(range(49)), percentile=85) is None
    assert evaluation.adaptive_threshold(list(range(50)), percentile=85) is not None


# ---------------------------------------------------------------------------
# compute_performance() — row fixtures
# ---------------------------------------------------------------------------

BASE_TS = datetime(2026, 7, 16, 19, 0, 0, tzinfo=timezone.utc)


def _iso(dt):
    return dt.isoformat()


def _row(
    idx,
    *,
    api_ts,
    written_at=None,
    model_variant="ml",
    model_version="v1.0",
    run_id="run-1",
    tau=0.7,
    score=0.5,
    vol_60s=0.00003,
    vol_spike=0,
    future_vol_60s=0.00003,
    stream_epoch=1,
):
    """Build one predictions-LEFT-JOIN-outcomes row dict, matching the exact
    shape materializer.performance_window() produces."""
    feature_ts = BASE_TS + timedelta(seconds=idx)
    return {
        "event_id": f"e{idx}",
        "source_partition": 0,
        "source_offset": idx,
        "feature_ts": _iso(feature_ts),
        "api_ts": _iso(api_ts),
        "model_variant": model_variant,
        "model_version": model_version,
        "score": score,
        "vol_60s": vol_60s,
        "spread_bps": 1.0,
        "log_return": 0.0001,
        "trade_intensity_60s": 5.0,
        "feature_id": f"BTC-USD:{stream_epoch}:{idx}",
        "stream_epoch": stream_epoch,
        "tau": tau,
        "run_id": run_id,
        "product_id": "BTC-USD" if written_at else None,
        "future_vol_60s": future_vol_60s if written_at else None,
        "vol_spike": vol_spike if written_at else None,
        "label_schema": "p85-60s-4.8e-05-v1" if written_at else None,
        "written_at": _iso(written_at) if written_at else None,
    }


# ---------------------------------------------------------------------------
# Online / late split
# ---------------------------------------------------------------------------


def test_online_late_split_and_median_lead_seconds():
    api_ts = BASE_TS
    graded_a = _row(
        0,
        api_ts=api_ts,
        written_at=api_ts + timedelta(seconds=60),
        score=0.9,
        vol_spike=1,
    )
    graded_b = _row(
        1,
        api_ts=api_ts,
        written_at=api_ts + timedelta(seconds=70),
        score=0.1,
        vol_spike=0,
    )
    late = _row(
        2,
        api_ts=api_ts + timedelta(seconds=5),
        written_at=api_ts,
        score=0.5,
        vol_spike=0,
    )
    pending = _row(3, api_ts=api_ts, written_at=None)

    result = evaluation.compute_performance(
        [graded_a, graded_b, late, pending], min_positives=1
    )

    window = result["window"]
    assert window["n_joined"] == 4
    assert window["n_graded"] == 2
    assert window["n_scored_late"] == 1
    assert window["n_predictions_unmatched"] == 1
    assert window["median_lead_seconds"] == pytest.approx(65.0)


def test_null_score_row_excluded_from_grading_not_crashed():
    # A malformed/old-format prediction event (missing score) that still has
    # a matched outcome must be excluded from grading, not raise TypeError
    # when the AP/confusion math tries to sort/threshold a None score.
    api_ts = BASE_TS
    good = _row(0, api_ts=api_ts, written_at=api_ts + timedelta(seconds=60), score=0.8)
    null_score = _row(
        1, api_ts=api_ts, written_at=api_ts + timedelta(seconds=60), score=None
    )

    result = evaluation.compute_performance([good, null_score], min_positives=1)

    assert result["window"]["n_graded"] == 1


def test_mode_percentile_field_present_for_ui_badge():
    # ui/src/pages/PerformancePage.jsx reads modes.adaptive.percentile to
    # render the "top N% by score" badge text; official mode has no
    # percentile concept and must report None so the UI hides the badge.
    api_ts = BASE_TS
    rows = [
        _row(i, api_ts=api_ts, written_at=api_ts + timedelta(seconds=60))
        for i in range(60)
    ]

    result = evaluation.compute_performance(
        rows, min_positives=1, adaptive_percentile=85
    )

    assert result["modes"]["official"]["percentile"] is None
    assert result["modes"]["adaptive"]["percentile"] == 85


def test_mode_percentile_present_even_with_insufficient_adaptive_samples():
    api_ts = BASE_TS
    rows = [
        _row(i, api_ts=api_ts, written_at=api_ts + timedelta(seconds=60))
        for i in range(10)  # below the 50-sample adaptive_threshold floor
    ]

    result = evaluation.compute_performance(
        rows, min_positives=1, adaptive_percentile=85
    )

    assert result["modes"]["adaptive"]["percentile"] == 85


# ---------------------------------------------------------------------------
# Per-group tau + fallback marker
# ---------------------------------------------------------------------------


def test_per_group_own_tau_used_for_confusion():
    api_ts = BASE_TS
    rows = [
        _row(
            0,
            api_ts=api_ts,
            written_at=api_ts + timedelta(seconds=60),
            tau=0.75,
            score=0.8,
            vol_spike=1,
        ),
        _row(
            1,
            api_ts=api_ts,
            written_at=api_ts + timedelta(seconds=60),
            tau=0.75,
            score=0.6,
            vol_spike=0,
        ),
    ]
    result = evaluation.compute_performance(rows, min_positives=1)
    ml_series = next(
        s for s in result["modes"]["official"]["series"] if s["model_variant"] == "ml"
    )
    assert ml_series["tau"] == 0.75
    assert ml_series["tau_source"] == "model"
    # score 0.8 >= tau 0.75 -> alert & correct; score 0.6 < 0.75 -> no alert & correct
    assert ml_series["tp"] == 1
    assert ml_series["tn"] == 1


def test_tau_fallback_marker_when_group_has_no_tau():
    api_ts = BASE_TS
    rows = [
        _row(0, api_ts=api_ts, written_at=api_ts + timedelta(seconds=60), tau=None),
        _row(1, api_ts=api_ts, written_at=api_ts + timedelta(seconds=60), tau=None),
    ]
    result = evaluation.compute_performance(rows, min_positives=1)
    ml_series = next(
        s for s in result["modes"]["official"]["series"] if s["model_variant"] == "ml"
    )
    assert ml_series["tau"] == evaluation.FALLBACK_TAU
    assert ml_series["tau_source"] == "fallback"


# ---------------------------------------------------------------------------
# ml-absent scenario
# ---------------------------------------------------------------------------


def test_ml_absent_baseline_never_appears_under_ml_label():
    api_ts = BASE_TS
    rows = [
        _row(
            i,
            api_ts=api_ts,
            written_at=api_ts + timedelta(seconds=60),
            model_variant="baseline",
            vol_60s=0.0001,
            vol_spike=1,
        )
        for i in range(5)
    ]
    result = evaluation.compute_performance(rows, min_positives=1)
    official = result["modes"]["official"]
    assert official["ml_available"] is False
    assert all(s["model_variant"] != "ml" for s in official["series"])
    # The baseline entry must still be present and derived from vol_60s, not
    # smuggled in under an "ml" label.
    baseline = next(s for s in official["series"] if s["model_variant"] == "baseline")
    assert baseline["n"] == 5


# ---------------------------------------------------------------------------
# Note triggering
# ---------------------------------------------------------------------------


def _low_pr_auc_rows(n, api_ts=BASE_TS):
    """n graded ml rows, deliberately worst-ranked: every positive scores
    below every negative, so pr_auc comes out low."""
    rows = []
    n_positive = max(11, n // 10)
    for i in range(n):
        is_positive = i < n_positive
        # Positives get low scores, negatives get high scores (inverted rank).
        score = 0.1 if is_positive else 0.9
        rows.append(
            _row(
                i,
                api_ts=api_ts,
                written_at=api_ts + timedelta(seconds=60),
                score=score,
                vol_spike=1 if is_positive else 0,
                tau=0.5,
            )
        )
    return rows


def test_note_triggered_when_below_ratio_and_minimums_met():
    rows = _low_pr_auc_rows(1000)
    result = evaluation.compute_performance(
        rows, min_positives=10, min_note_sample_n=1000, drift_pr_auc_ratio=0.7
    )
    official = result["modes"]["official"]
    ml_entry = next(s for s in official["series"] if s["model_variant"] == "ml")
    assert ml_entry["pr_auc"] is not None
    assert official["note"]["triggered"] is True
    assert f"{ml_entry['pr_auc']:.3f}" in official["note"]["detail"]
    assert "0.146" in official["note"]["detail"]
    assert "7-day window" in official["note"]["detail"]


def test_note_not_triggered_below_min_sample_n():
    rows = _low_pr_auc_rows(50)
    result = evaluation.compute_performance(
        rows, min_positives=10, min_note_sample_n=1000, drift_pr_auc_ratio=0.7
    )
    official = result["modes"]["official"]
    assert official["note"]["triggered"] is False
    assert official["note"]["detail"] == "not enough graded data yet"


def test_adaptive_note_always_suppressed():
    rows = _low_pr_auc_rows(1000)
    result = evaluation.compute_performance(
        rows, min_positives=10, min_note_sample_n=1000, drift_pr_auc_ratio=0.7
    )
    adaptive = result["modes"]["adaptive"]
    assert adaptive["note"] == {
        "triggered": False,
        "detail": "suppressed — activity view is self-normalizing",
    }


# ---------------------------------------------------------------------------
# Chart: epoch filtering + outcome-aware downsampling
# ---------------------------------------------------------------------------


def test_chart_filters_to_max_stream_epoch():
    api_ts = BASE_TS
    old_epoch_rows = [
        _row(
            i, api_ts=api_ts, written_at=api_ts + timedelta(seconds=60), stream_epoch=1
        )
        for i in range(3)
    ]
    new_epoch_rows = [
        _row(
            100 + i,
            api_ts=api_ts,
            written_at=api_ts + timedelta(seconds=60),
            stream_epoch=2,
        )
        for i in range(4)
    ]
    result = evaluation.compute_performance(
        old_epoch_rows + new_epoch_rows, min_positives=1
    )
    assert result["chart"]["stream_epoch"] == 2
    assert len(result["chart"]["rows"]) == 4


def test_chart_downsampling_never_drops_positive_outcome_rows():
    api_ts = BASE_TS
    rows = []
    positive_indices = set(range(0, 3000, 137))  # sparse positives scattered throughout
    for i in range(3000):
        rows.append(
            _row(
                i,
                api_ts=api_ts,
                written_at=api_ts + timedelta(seconds=60),
                vol_spike=1 if i in positive_indices else 0,
            )
        )

    result = evaluation.compute_performance(rows, min_positives=1)
    chart_rows = result["chart"]["rows"]

    assert len(chart_rows) <= evaluation.MAX_CHART_POINTS
    n_positive_in_chart = sum(1 for r in chart_rows if r["outcome_official"] is True)
    assert n_positive_in_chart == len(positive_indices)


# ---------------------------------------------------------------------------
# Endpoint-level: window validation + feature_ts normalization
# ---------------------------------------------------------------------------


def test_performance_endpoint_rejects_out_of_range_window(tmp_path, monkeypatch):
    db_path = tmp_path / "predictions.db"
    materializer.init_db(db_path).close()
    monkeypatch.setattr(materializer, "PREDICTIONS_DB_PATH", str(db_path))
    materializer._state.ready = True
    materializer._reset_perf_cache()
    client = TestClient(materializer.app, raise_server_exceptions=False)

    assert client.get("/predictions/performance?window_minutes=0").status_code == 422
    assert client.get("/predictions/performance?window_minutes=121").status_code == 422
    assert client.get("/predictions/performance?window_minutes=30").status_code == 200


def test_performance_endpoint_normalizes_feature_ts_for_window_filter(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "predictions.db"
    conn = materializer.init_db(db_path)

    # Two predictions at (nearly) the same instant, one with "Z" suffix, one
    # with an explicit "+00:00" offset -- these must sort/compare identically
    # after normalization.
    now = datetime.now(timezone.utc).replace(microsecond=0)
    z_ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    offset_ts = (now + timedelta(seconds=1)).isoformat()

    materializer.insert_events(
        conn,
        [
            {
                "event_id": "z-event",
                "source_partition": 0,
                "source_offset": 1,
                "feature_ts": z_ts,
                "api_ts": z_ts,
                "model_variant": "ml",
                "model_version": "v1.0",
                "score": 0.5,
                "vol_60s": 0.00003,
                "spread_bps": 1.0,
                "log_return": 0.0001,
                "trade_intensity_60s": 5.0,
                "feature_id": "BTC-USD:1:1",
                "stream_epoch": 1,
                "tau": 0.7,
                "run_id": "run-1",
            },
            {
                "event_id": "offset-event",
                "source_partition": 0,
                "source_offset": 2,
                "feature_ts": offset_ts,
                "api_ts": offset_ts,
                "model_variant": "ml",
                "model_version": "v1.0",
                "score": 0.5,
                "vol_60s": 0.00003,
                "spread_bps": 1.0,
                "log_return": 0.0001,
                "trade_intensity_60s": 5.0,
                "feature_id": "BTC-USD:1:2",
                "stream_epoch": 1,
                "tau": 0.7,
                "run_id": "run-1",
            },
        ],
    )
    conn.close()

    monkeypatch.setattr(materializer, "PREDICTIONS_DB_PATH", str(db_path))
    materializer._state.ready = True
    materializer._reset_perf_cache()
    client = TestClient(materializer.app, raise_server_exceptions=False)

    response = client.get("/predictions/performance?window_minutes=30")
    assert response.status_code == 200
    body = response.json()
    # Both rows are within the last 30 minutes of "now" -> both counted.
    assert body["window"]["n_joined"] == 2


# ---------------------------------------------------------------------------
# Endpoint-level: stream_id segment scoping (a looping replay must not
# double-count a prior pass into the same window)
# ---------------------------------------------------------------------------


def _pred_event(
    event_id: str,
    offset: int,
    *,
    feature_ts: str,
    api_ts: str,
    stream_id,
    stream_epoch=0,
) -> dict:
    """Minimal raw PredictionEvent dict, insertable via materializer.insert_events."""
    return {
        "event_id": event_id,
        "source_partition": 0,
        "source_offset": offset,
        "feature_ts": feature_ts,
        "api_ts": api_ts,
        "model_variant": "ml",
        "model_version": "v1.0",
        "score": 0.5,
        "vol_60s": 0.00003,
        "spread_bps": 1.0,
        "log_return": 0.0001,
        "trade_intensity_60s": 5.0,
        "feature_id": f"BTC-USD:{stream_id}:{offset}",
        "stream_epoch": stream_epoch,
        "tau": 0.7,
        "run_id": "run-1",
        "stream_id": stream_id,
        "ingest_mode": "replay",
        "product_id": "BTC-USD",
    }


def _performance_body(db_path, monkeypatch, window_minutes=30):
    monkeypatch.setattr(materializer, "PREDICTIONS_DB_PATH", str(db_path))
    materializer._state.ready = True
    materializer._reset_perf_cache()
    client = TestClient(materializer.app, raise_server_exceptions=False)
    response = client.get(f"/predictions/performance?window_minutes={window_minutes}")
    assert response.status_code == 200
    return response.json()


def test_performance_window_excludes_prior_segment_at_same_feature_ts(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "predictions.db"
    conn = materializer.init_db(db_path)

    now = datetime.now(timezone.utc).replace(microsecond=0)
    # A looping replay: pass 1 (older boot) and pass 2 (current boot) cover
    # the IDENTICAL feature_ts range -- exactly what a `--loop` replay does.
    # Only pass 2's stream_id is the active segment (its api_ts is newest).
    pass_1 = [
        _pred_event(
            f"p1-{i}",
            i,
            feature_ts=(now - timedelta(seconds=i)).isoformat(),
            api_ts=(now - timedelta(minutes=20)).isoformat(),
            stream_id="boot1:0",
        )
        for i in range(5)
    ]
    pass_2 = [
        _pred_event(
            f"p2-{i}",
            100 + i,
            feature_ts=(now - timedelta(seconds=i)).isoformat(),
            api_ts=now.isoformat(),
            stream_id="boot2:0",
        )
        for i in range(5)
    ]
    materializer.insert_events(conn, pass_1 + pass_2)
    conn.close()

    body = _performance_body(db_path, monkeypatch)

    assert body["window"]["n_joined"] == 5  # pass_2 only, not 10
    assert body["window"]["stream_id"] == "boot2:0"
    assert body["chart"]["rows"]
    assert body["chart"]["rendered_row_count"] == len(body["chart"]["rows"])


def test_active_segment_follows_newest_boot_not_max_epoch(tmp_path, monkeypatch):
    db_path = tmp_path / "predictions.db"
    conn = materializer.init_db(db_path)

    now = datetime.now(timezone.utc).replace(microsecond=0)
    # boot_a has a HIGHER epoch (3) but an OLDER api_ts -- a stale prior run.
    # boot_b has epoch 0 (a fresh restart) but the NEWEST api_ts -- the live
    # stream. MAX(stream_epoch) would wrongly pick boot_a; api_ts must win.
    old_boot_high_epoch = _pred_event(
        "old",
        1,
        feature_ts=(now - timedelta(minutes=5)).isoformat(),
        api_ts=(now - timedelta(minutes=20)).isoformat(),
        stream_id="boot_a:3",
        stream_epoch=3,
    )
    new_boot_low_epoch = _pred_event(
        "new",
        2,
        feature_ts=now.isoformat(),
        api_ts=now.isoformat(),
        stream_id="boot_b:0",
        stream_epoch=0,
    )
    materializer.insert_events(conn, [old_boot_high_epoch, new_boot_low_epoch])
    conn.close()

    body = _performance_body(db_path, monkeypatch)

    assert body["window"]["stream_id"] == "boot_b:0"
    assert body["window"]["n_joined"] == 1


def test_null_stream_id_rows_are_reported_as_unknown_lineage(tmp_path, monkeypatch):
    db_path = tmp_path / "predictions.db"
    conn = materializer.init_db(db_path)

    now = datetime.now(timezone.utc).replace(microsecond=0)
    current = _pred_event(
        "current",
        1,
        feature_ts=now.isoformat(),
        api_ts=now.isoformat(),
        stream_id="boot1:0",
    )
    legacy = _pred_event(
        "legacy",
        2,
        feature_ts=(now - timedelta(minutes=1)).isoformat(),
        api_ts=(now - timedelta(minutes=1)).isoformat(),
        stream_id=None,
    )
    materializer.insert_events(conn, [current, legacy])
    conn.close()

    body = _performance_body(db_path, monkeypatch)

    assert body["window"]["stream_id"] == "boot1:0"
    assert body["window"]["n_joined"] == 1  # legacy row excluded, not silently merged
    assert body["window"]["n_unknown_lineage"] == 1
