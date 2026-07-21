"""Artifact-derived, one-time Staging qualification for completed candidates."""

from __future__ import annotations

import ast
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
from pathlib import Path
import time
from typing import Iterator, Sequence

import mlflow
from mlflow.entities import Run
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import yaml

from btcspiker_ml.manifest import DatasetManifest, manifest_id
from btcspiker_ml.metrics import evaluate_predictions, paired_block_bootstrap
from btcspiker_ml.search import SearchState
from btcspiker_ml.storage import sha256_file


QUOTE_COLUMNS = frozenset({"spread_bps", "spread_mean_60s"})
TRADE_COLUMNS = frozenset(
    {"log_return", "vol_60s", "mean_return_60s", "trade_intensity_60s", "n_ticks_60s"}
)


@dataclass(frozen=True)
class CandidateEvidence:
    coverage_days: float
    quote_trade_coverage: bool
    folds_won: int
    bootstrap_lower: float
    brier_ratio: float
    event_f1_delta: float
    final_pr_auc_delta: float
    p95_latency_ms: float
    deployable: bool
    parity_passed: bool


@dataclass(frozen=True)
class QualificationResult:
    passed: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ArtifactQualification:
    evidence: CandidateEvidence
    result: QualificationResult
    baseline_run_id: str
    dataset_manifest_path: Path
    final_holdout_accessed_at: str


@dataclass(frozen=True)
class EvaluationContract:
    folds: int
    final_holdout_fraction: float
    bootstrap_block_minutes: int
    bootstrap_resamples: int
    random_seed: int


def qualify(evidence: CandidateEvidence) -> QualificationResult:
    """Apply every Staging gate, retaining stable failure codes in gate order."""
    checks = {
        "coverage_under_thirty_days": evidence.coverage_days >= 30.0,
        "quote_trade_coverage_missing": evidence.quote_trade_coverage,
        "fewer_than_four_folds_won": evidence.folds_won >= 4,
        "bootstrap_lower_not_positive": evidence.bootstrap_lower > 0,
        "brier_regression_over_five_percent": evidence.brier_ratio <= 1.05,
        "event_f1_regressed": evidence.event_f1_delta >= 0,
        "final_pr_auc_not_improved": evidence.final_pr_auc_delta > 0,
        "p95_latency_over_800ms": evidence.p95_latency_ms <= 800,
        "feature_set_not_deployable": evidence.deployable,
        "feature_parity_failed": evidence.parity_passed,
    }
    reasons = tuple(reason for reason, passed in checks.items() if not passed)
    return QualificationResult(passed=not reasons, reasons=reasons)


def qualify_recorded_candidate(
    *,
    client: MlflowClient,
    run_id: str,
    state_path: Path,
    artifact_root: Path,
) -> ArtifactQualification:
    """Evaluate the recorded search winner against its sealed holdout exactly once.

    Caller-authored metric files are deliberately not accepted.  The search state,
    content-addressed dataset manifest, completed MLflow runs, and fitted model are
    the only authorities.  A process lock plus a persisted write-ahead seal ensures
    two qualifiers cannot evaluate the same holdout.
    """
    with _locked_state(state_path):
        state = SearchState.load(state_path)
        candidate, baseline, manifest_path, manifest, model, feature_columns, tau, contract = (
            _validate_authorities(client, run_id, state, artifact_root)
        )

        # Persist the one-time access marker before any partition bytes are read.
        # If evaluation crashes, retry remains forbidden because the holdout was in
        # fact exposed.  The lock remains held through the complete evaluation.
        state.open_final_holdout(requesting_stage="qualification")
        state.save(state_path)
        evidence = _derive_evidence(
            candidate=candidate,
            baseline=baseline,
            manifest=manifest,
            model=model,
            feature_columns=feature_columns,
            tau=tau,
            contract=contract,
        )
        accessed_at = state.final_holdout_accessed_at
        if accessed_at is None:  # pragma: no cover - SearchState invariant
            raise RuntimeError("final holdout access timestamp was not persisted")

    return ArtifactQualification(
        evidence=evidence,
        result=qualify(evidence),
        baseline_run_id=baseline.info.run_id,
        dataset_manifest_path=manifest_path,
        final_holdout_accessed_at=accessed_at,
    )


@contextmanager
def _locked_state(state_path: Path) -> Iterator[None]:
    lock_path = state_path.with_suffix(state_path.suffix + ".qualification.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _validate_authorities(
    client: MlflowClient,
    run_id: str,
    state: SearchState,
    artifact_root: Path,
) -> tuple[
    Run, Run, Path, DatasetManifest, object, tuple[str, ...], float, EvaluationContract
]:
    recorded = set(state.best_run_ids.values())
    if run_id not in recorded:
        raise ValueError(f"run {run_id} is not a recorded development winner")

    candidate = client.get_run(run_id)
    if candidate.info.status != "FINISHED":
        raise ValueError(f"run {run_id} is not completed")
    candidate_stage = candidate.data.tags.get("candidate_stage")
    if not candidate_stage or state.best_run_ids.get(candidate_stage) != run_id:
        raise ValueError(f"run {run_id} is not the recorded winner for its stage")
    _verify_lineage(candidate, state)

    candidate_score = candidate.data.metrics.get("aggregate_pr_auc")
    if candidate_score is None:
        raise ValueError("candidate run lacks aggregate_pr_auc")
    for stage, recorded_run_id in state.best_run_ids.items():
        recorded_run = client.get_run(recorded_run_id)
        _verify_lineage(recorded_run, state)
        score = recorded_run.data.metrics.get("aggregate_pr_auc")
        if score is None:
            raise ValueError(f"recorded {stage} winner lacks aggregate_pr_auc")
        if score > candidate_score:
            raise ValueError(
                f"run {run_id} is not the strongest recorded development winner"
            )

    baseline_id = state.best_run_ids.get("baseline")
    if not baseline_id:
        raise ValueError("search state has no recorded baseline winner")
    baseline = client.get_run(baseline_id)
    if baseline.info.status != "FINISHED":
        raise ValueError("recorded baseline run is not completed")
    _verify_lineage(baseline, state)

    manifest_path = (
        Path(artifact_root)
        / "manifests"
        / f"existing-{state.dataset_id}.json"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"dataset manifest not found: {manifest_path}")
    manifest = DatasetManifest(**json.loads(manifest_path.read_text()))
    if manifest_id(manifest) != state.dataset_id:
        raise ValueError("dataset manifest does not match state dataset_id")
    if len(manifest.partitions) != 1:
        raise ValueError("qualification requires exactly one immutable dataset partition")

    if candidate.data.params.get("final_holdout") != "sealed":
        raise ValueError("candidate run does not record a sealed final holdout")
    feature_columns = _feature_columns(candidate.data.params.get("feature_columns"))
    try:
        tau = float(candidate.data.params["tau"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("candidate run lacks a valid tau parameter") from exc
    if not 0 <= tau <= 1:
        raise ValueError("candidate tau must be in [0, 1]")

    config_path = Path(client.download_artifacts(run_id, "experiment-config.yaml"))
    raw_config = yaml.safe_load(config_path.read_text())
    try:
        validation = raw_config["validation"]
        contract = EvaluationContract(
            folds=int(validation["folds"]),
            final_holdout_fraction=float(validation["final_holdout_fraction"]),
            bootstrap_block_minutes=int(validation["bootstrap_block_minutes"]),
            bootstrap_resamples=int(validation["bootstrap_resamples"]),
            random_seed=int(validation["random_seed"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("candidate experiment-config.yaml lacks validation contract") from exc
    if (
        contract.folds < 1
        or not 0 < contract.final_holdout_fraction < 1
        or contract.bootstrap_block_minutes <= 0
        or contract.bootstrap_resamples <= 0
    ):
        raise ValueError("candidate experiment-config.yaml has invalid validation contract")

    if not any(item.path == "model" and item.is_dir for item in client.list_artifacts(run_id)):
        raise ValueError("candidate run lacks a fitted model artifact")
    model_path = client.download_artifacts(run_id, "model")
    model = mlflow.sklearn.load_model(model_path)
    return candidate, baseline, manifest_path, manifest, model, feature_columns, tau, contract


def _verify_lineage(run: Run, state: SearchState) -> None:
    expected = {
        "dataset_id": state.dataset_id,
        "search_id": state.search_id,
        **state.experiment_contract,
    }
    for key, value in expected.items():
        actual = run.data.tags.get(key)
        if actual != str(value):
            raise ValueError(
                f"run {run.info.run_id} lineage {key}={actual!r} does not match {value!r}"
            )


def _feature_columns(raw: str | None) -> tuple[str, ...]:
    if not raw:
        raise ValueError("candidate run lacks feature_columns")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(raw)
    if not isinstance(parsed, (list, tuple)) or not parsed or not all(
        isinstance(column, str) and column for column in parsed
    ):
        raise ValueError("candidate feature_columns must be a non-empty string list")
    if len(set(parsed)) != len(parsed):
        raise ValueError("candidate feature_columns contain duplicates")
    return tuple(parsed)


def _derive_evidence(
    *,
    candidate: Run,
    baseline: Run,
    manifest: DatasetManifest,
    model: object,
    feature_columns: Sequence[str],
    tau: float,
    contract: EvaluationContract,
) -> CandidateEvidence:
    partition = manifest.partitions[0]
    path = Path(str(partition["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"dataset partition not found: {path}")
    expected_sha = str(partition["sha256"])
    if sha256_file(path) != expected_sha:
        raise ValueError("dataset partition checksum does not match immutable manifest")
    frame = pd.read_parquet(path)
    if len(frame) != manifest.rows:
        raise ValueError("dataset row count does not match immutable manifest")
    if "timestamp" not in frame or "vol_spike" not in frame:
        raise ValueError("dataset lacks timestamp or vol_spike")
    missing = set(feature_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"dataset lacks candidate features: {sorted(missing)}")

    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("dataset timestamps are not monotonic")
    target = frame["vol_spike"].to_numpy(dtype=int)
    if not np.isin(target, [0, 1]).all():
        raise ValueError("dataset target is not binary")
    development_end = len(frame) - int(
        np.ceil(len(frame) * contract.final_holdout_fraction)
    )
    if development_end <= 0:
        raise ValueError("dataset is too small for the final holdout contract")
    development = np.arange(development_end)
    holdout = np.arange(development_end, len(frame))
    if len(np.unique(target[holdout])) != 2:
        raise ValueError("final holdout must contain both target classes")

    source = frame.loc[:, list(feature_columns)].replace([np.inf, -np.inf], np.nan)
    medians = source.iloc[development].median().fillna(0.0)
    holdout_x = source.iloc[holdout].fillna(medians).to_numpy(dtype=float)
    candidate_scores = _positive_probabilities(model, holdout_x)
    baseline_probability = float(target[development].mean())
    baseline_scores = np.full(len(holdout), baseline_probability)
    holdout_timestamps = timestamps.iloc[holdout]
    candidate_metrics = evaluate_predictions(
        target[holdout], candidate_scores, holdout_timestamps, threshold=tau
    )
    baseline_metrics = evaluate_predictions(
        target[holdout], baseline_scores, holdout_timestamps, threshold=tau
    )
    interval = paired_block_bootstrap(
        target[holdout],
        candidate_scores,
        baseline_scores,
        holdout_timestamps,
        contract.bootstrap_block_minutes,
        contract.bootstrap_resamples,
        contract.random_seed,
    )

    folds_won = sum(
        candidate.data.metrics.get(f"fold_{index}_pr_auc", float("-inf"))
        > baseline.data.metrics.get(f"fold_{index}_pr_auc", float("inf"))
        for index in range(contract.folds)
    )
    coverage_days = (
        pd.Timestamp(manifest.end_time) - pd.Timestamp(manifest.start_time)
    ).total_seconds() / 86_400
    coverage_columns = QUOTE_COLUMNS | TRADE_COLUMNS
    quote_trade_coverage = coverage_columns.issubset(frame.columns) and bool(
        np.isfinite(frame.loc[:, sorted(coverage_columns)].to_numpy(dtype=float)).all()
    )
    p95_latency_ms = _p95_latency_ms(model, holdout_x)
    deployable = candidate.data.tags.get("deployable", "").lower() == "true"
    parity_passed = (
        candidate.data.tags.get("feature_parity_passed", "").lower() == "true"
    )
    brier_ratio = (
        candidate_metrics.brier / baseline_metrics.brier
        if baseline_metrics.brier > 0
        else float("inf")
    )
    return CandidateEvidence(
        coverage_days=float(coverage_days),
        quote_trade_coverage=quote_trade_coverage,
        folds_won=folds_won,
        bootstrap_lower=interval.lower,
        brier_ratio=float(brier_ratio),
        event_f1_delta=candidate_metrics.event_f1 - baseline_metrics.event_f1,
        final_pr_auc_delta=candidate_metrics.pr_auc - baseline_metrics.pr_auc,
        p95_latency_ms=p95_latency_ms,
        deployable=deployable,
        parity_passed=parity_passed,
    )


def _positive_probabilities(model: object, values: np.ndarray) -> np.ndarray:
    predict_proba = getattr(model, "predict_proba", None)
    if not callable(predict_proba):
        raise ValueError("candidate model does not expose predict_proba")
    probabilities = np.asarray(predict_proba(values), dtype=float)
    classes = np.asarray(getattr(model, "classes_", []))
    matches = np.flatnonzero(classes == 1)
    if probabilities.ndim != 2 or len(matches) != 1:
        raise ValueError("candidate model lacks one binary positive-class probability")
    scores = probabilities[:, int(matches[0])]
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("candidate model returned invalid probabilities")
    return scores


def _p95_latency_ms(model: object, values: np.ndarray) -> float:
    sample_count = min(len(values), 50)
    sample_indices = np.linspace(0, len(values) - 1, sample_count, dtype=int)
    latencies: list[float] = []
    for index in sample_indices:
        started = time.perf_counter_ns()
        _positive_probabilities(model, values[index : index + 1])
        latencies.append((time.perf_counter_ns() - started) / 1_000_000)
    return float(np.percentile(latencies, 95))
