from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import mlflow
import pytest
from sklearn.dummy import DummyClassifier

from btcspiker_ml.qualification import CandidateEvidence, qualify


def _passing_evidence() -> CandidateEvidence:
    return CandidateEvidence(
        coverage_days=30.0,
        quote_trade_coverage=True,
        folds_won=4,
        bootstrap_lower=0.001,
        brier_ratio=1.05,
        event_f1_delta=0.0,
        final_pr_auc_delta=0.001,
        p95_latency_ms=800,
        deployable=True,
        parity_passed=True,
    )


def test_candidate_passes_all_staging_gates_at_their_boundaries():
    result = qualify(_passing_evidence())

    assert result.passed
    assert result.reasons == ()


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"coverage_days": 29.99}, "coverage_under_thirty_days"),
        ({"quote_trade_coverage": False}, "quote_trade_coverage_missing"),
        ({"folds_won": 3}, "fewer_than_four_folds_won"),
        ({"bootstrap_lower": 0.0}, "bootstrap_lower_not_positive"),
        ({"brier_ratio": 1.051}, "brier_regression_over_five_percent"),
        ({"event_f1_delta": -0.001}, "event_f1_regressed"),
        ({"final_pr_auc_delta": 0.0}, "final_pr_auc_not_improved"),
        ({"p95_latency_ms": 801}, "p95_latency_over_800ms"),
        ({"deployable": False}, "feature_set_not_deployable"),
        ({"parity_passed": False}, "feature_parity_failed"),
    ],
)
def test_candidate_fails_each_staging_gate_with_a_stable_reason_code(change, reason):
    result = qualify(replace(_passing_evidence(), **change))

    assert not result.passed
    assert result.reasons == (reason,)


def test_candidate_reports_all_failed_gates_in_gate_order():
    evidence = CandidateEvidence(
        coverage_days=0.0,
        quote_trade_coverage=False,
        folds_won=0,
        bootstrap_lower=0.0,
        brier_ratio=1.06,
        event_f1_delta=-0.01,
        final_pr_auc_delta=0.0,
        p95_latency_ms=801,
        deployable=False,
        parity_passed=False,
    )

    assert qualify(evidence).reasons == (
        "coverage_under_thirty_days",
        "quote_trade_coverage_missing",
        "fewer_than_four_folds_won",
        "bootstrap_lower_not_positive",
        "brier_regression_over_five_percent",
        "event_f1_regressed",
        "final_pr_auc_not_improved",
        "p95_latency_over_800ms",
        "feature_set_not_deployable",
        "feature_parity_failed",
    )


def test_qualification_cli_promotes_only_a_passing_candidate_to_staging_once(tmp_path: Path):
    tracking_uri = (tmp_path / "mlruns").as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("qualification-cli")
    with mlflow.start_run() as run:
        classifier = DummyClassifier(strategy="prior").fit([[0], [1]], [0, 1])
        mlflow.sklearn.log_model(classifier, "model")
        run_id = run.info.run_id

    from btcspiker_ml.search import SearchState

    state_path = tmp_path / "state.json"
    state = SearchState.new("search-1", "dataset-1", wall_clock_seconds=60)
    state.completed_stages = ["baseline", "linear", "trees", "ablation", "ensemble"]
    state.save(state_path)
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(_passing_evidence().__dict__))
    command = [
        sys.executable, "scripts/qualify_candidate.py", run_id, str(evidence_path),
        "--search-state", str(state_path), "--tracking-uri", tracking_uri,
        "--artifact-root", str(tmp_path / "artifacts"),
    ]

    first = subprocess.run(command, text=True, capture_output=True, check=True)

    client = mlflow.tracking.MlflowClient(tracking_uri)
    staging = client.get_latest_versions("btc-volatility-candidate", stages=["Staging"])
    assert len(staging) == 1
    assert staging[0].run_id == run_id
    assert "Production unchanged" in first.stdout
    assert not client.get_latest_versions("btc-volatility-candidate", stages=["Production"])
    saved_state = SearchState.load(state_path)
    assert saved_state.final_holdout_accessed_at is not None

    second = subprocess.run(command, text=True, capture_output=True)
    assert second.returncode == 1
    assert "final holdout is sealed" in second.stderr
