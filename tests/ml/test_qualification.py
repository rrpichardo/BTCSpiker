from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from btcspiker_ml.datasets import inspect_existing_dataset, publish_existing_manifest
from btcspiker_ml.qualification import CandidateEvidence, qualify
from btcspiker_ml.search import SearchState


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


def _qualification_fixture(tmp_path: Path) -> tuple[str, str, Path, Path, str]:
    timestamps = pd.date_range("2026-01-01", periods=300, freq="6h", tz="UTC")
    signal = np.tile([0.0, 0.0, 1.0, 1.0], 75)
    target = signal.astype(int)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "log_return": signal,
            "spread_bps": np.full(len(signal), 1.0),
            "vol_60s": signal,
            "mean_return_60s": signal,
            "trade_intensity_60s": np.full(len(signal), 2.0),
            "n_ticks_60s": np.full(len(signal), 3.0),
            "spread_mean_60s": np.full(len(signal), 1.0),
            "vol_spike": target,
        }
    )
    dataset_path = tmp_path / "features.parquet"
    frame.to_parquet(dataset_path, index=False)
    artifact_root = tmp_path / "artifacts"
    dataset_id, _ = publish_existing_manifest(
        inspect_existing_dataset(dataset_path), artifact_root
    )
    tracking_uri = (tmp_path / "mlruns").as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("qualification-cli")
    lineage = {
        "dataset_id": dataset_id,
        "search_id": "search-1",
        "feature_set_id": "core_v1",
        "target_version": "vol_spike_v1",
        "validation_version": "walkforward_v1",
        "git_sha": "abc123",
    }
    with mlflow.start_run(run_name="baseline") as baseline:
        mlflow.set_tags(
            {
                **lineage,
                "candidate_stage": "baseline",
                "model_family": "development_prevalence",
                "deployable": "false",
            }
        )
        mlflow.log_metrics({f"fold_{index}_pr_auc": 0.5 for index in range(5)})
        mlflow.log_metric("aggregate_pr_auc", 0.5)
        baseline_run_id = baseline.info.run_id
    with mlflow.start_run(run_name="candidate") as candidate:
        mlflow.set_tags(
            {
                **lineage,
                "candidate_stage": "linear",
                "model_family": "logistic",
                "deployable": "true",
                "feature_parity_passed": "true",
            }
        )
        mlflow.log_params(
            {
                "feature_columns": json.dumps(["log_return"]),
                "final_holdout": "sealed",
                "tau": "0.5",
            }
        )
        mlflow.log_text(
            """validation:
  folds: 5
  final_holdout_fraction: 0.20
  bootstrap_block_minutes: 30
  bootstrap_resamples: 2000
  random_seed: 42
""",
            "experiment-config.yaml",
        )
        mlflow.log_metrics({f"fold_{index}_pr_auc": 1.0 for index in range(5)})
        mlflow.log_metric("aggregate_pr_auc", 1.0)
        classifier = LogisticRegression().fit(signal[:240, None], target[:240])
        mlflow.sklearn.log_model(classifier, "model")
        candidate_run_id = candidate.info.run_id

    state_path = tmp_path / "state.json"
    state = SearchState.new("search-1", dataset_id, wall_clock_seconds=60)
    state.completed_stages = ["baseline", "linear", "trees", "ablation", "ensemble"]
    state.best_run_ids = {
        "baseline": baseline_run_id,
        "linear": candidate_run_id,
    }
    state.experiment_contract = {
        "dataset_id": dataset_id,
        "feature_set_id": "core_v1",
        "target_version": "vol_spike_v1",
        "validation_version": "walkforward_v1",
        "git_sha": "abc123",
    }
    state.save(state_path)
    return candidate_run_id, baseline_run_id, state_path, artifact_root, tracking_uri


def _qualification_command(
    run_id: str, state_path: Path, artifact_root: Path, tracking_uri: str
) -> list[str]:
    return [
        sys.executable,
        "scripts/qualify_candidate.py",
        run_id,
        "--search-state",
        str(state_path),
        "--tracking-uri",
        tracking_uri,
        "--artifact-root",
        str(artifact_root),
    ]


def test_qualification_cli_derives_evidence_and_promotes_only_recorded_winner_once(
    tmp_path: Path,
):
    run_id, _, state_path, artifact_root, tracking_uri = _qualification_fixture(
        tmp_path
    )
    command = [
        *_qualification_command(run_id, state_path, artifact_root, tracking_uri),
    ]

    first = subprocess.run(command, text=True, capture_output=True, check=True)

    client = mlflow.tracking.MlflowClient(tracking_uri)
    staging = client.get_latest_versions("btc-volatility-candidate", stages=["Staging"])
    assert len(staging) == 1
    assert staging[0].run_id == run_id
    assert "Production unchanged" in first.stdout
    assert not client.get_latest_versions(
        "btc-volatility-candidate", stages=["Production"]
    )
    saved_state = SearchState.load(state_path)
    assert saved_state.final_holdout_accessed_at is not None

    second = subprocess.run(command, text=True, capture_output=True)
    assert second.returncode == 1
    assert "final holdout is sealed" in second.stderr


def test_qualification_cli_refuses_caller_authored_evidence_file(tmp_path: Path):
    run_id, _, state_path, artifact_root, tracking_uri = _qualification_fixture(
        tmp_path
    )
    fabricated = tmp_path / "fabricated-evidence.json"
    fabricated.write_text(json.dumps(_passing_evidence().__dict__))
    command = _qualification_command(run_id, state_path, artifact_root, tracking_uri)
    command.insert(3, str(fabricated))

    result = subprocess.run(command, text=True, capture_output=True)

    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr
    assert SearchState.load(state_path).final_holdout_opened is False


def test_qualification_cli_rejects_wrong_lineage_before_opening_holdout(tmp_path: Path):
    _, _, state_path, artifact_root, tracking_uri = _qualification_fixture(tmp_path)
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(run_name="unrelated") as unrelated:
        mlflow.set_tags(
            {
                "dataset_id": "other-dataset",
                "search_id": "other-search",
                "candidate_stage": "linear",
                "deployable": "true",
            }
        )
        mlflow.log_param("feature_columns", json.dumps(["log_return"]))
        model = LogisticRegression().fit([[0.0], [1.0]], [0, 1])
        mlflow.sklearn.log_model(model, "model")
        unrelated_run_id = unrelated.info.run_id

    result = subprocess.run(
        _qualification_command(
            unrelated_run_id, state_path, artifact_root, tracking_uri
        ),
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "recorded development winner" in result.stderr
    assert SearchState.load(state_path).final_holdout_opened is False


def test_qualification_cli_rejects_recorded_run_when_lineage_mismatches_state(
    tmp_path: Path,
):
    run_id, _, state_path, artifact_root, tracking_uri = _qualification_fixture(
        tmp_path
    )
    client = mlflow.tracking.MlflowClient(tracking_uri)
    client.set_tag(run_id, "dataset_id", "other-dataset")

    result = subprocess.run(
        _qualification_command(run_id, state_path, artifact_root, tracking_uri),
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "lineage dataset_id" in result.stderr
    assert SearchState.load(state_path).final_holdout_opened is False


def test_qualification_cli_requires_immutable_search_config_artifact(tmp_path: Path):
    run_id, _, state_path, artifact_root, tracking_uri = _qualification_fixture(
        tmp_path
    )
    client = mlflow.tracking.MlflowClient(tracking_uri)
    Path(client.download_artifacts(run_id, "experiment-config.yaml")).unlink()

    result = subprocess.run(
        _qualification_command(run_id, state_path, artifact_root, tracking_uri),
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "experiment-config.yaml" in result.stderr
    assert SearchState.load(state_path).final_holdout_opened is False
