"""Resumable, staged MLflow tournament orchestration.

This module deliberately records research outcomes only; it never registers or
promotes a model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from concurrent.futures import Future, ThreadPoolExecutor
import json
from pathlib import Path
import time
from typing import Any, Mapping

from btcspiker_ml.tracking import ExperimentTracker


VALID_STAGES = {"baseline", "linear", "trees", "ablation", "ensemble", "neural"}


@dataclass
class SearchState:
    search_id: str
    dataset_id: str
    completed_stages: list[str] = field(default_factory=list)
    best_run_ids: dict[str, str] = field(default_factory=dict)
    remaining_wall_clock_seconds: float = 0.0
    final_holdout_opened: bool = False
    final_holdout_accessed_at: str | None = None
    failure_counts: dict[str, int] = field(default_factory=dict)
    completed_trial_ids: dict[str, list[str]] = field(default_factory=dict)
    experiment_contract: dict[str, str] = field(default_factory=dict)
    best_scores: dict[str, float] = field(default_factory=dict)
    stage_parent_run_ids: dict[str, str] = field(default_factory=dict)
    stage_score_history: dict[str, list[float]] = field(default_factory=dict)

    @classmethod
    def new(cls, search_id: str, dataset_id: str, *, wall_clock_seconds: float) -> "SearchState":
        """Create a state with its final evaluation split sealed."""
        return cls(
            search_id=search_id,
            dataset_id=dataset_id,
            remaining_wall_clock_seconds=wall_clock_seconds,
        )

    def open_final_holdout(self, *, requesting_stage: str) -> None:
        """Permit holdout access only to the explicit post-search qualifier."""
        required = {"baseline", "linear", "trees", "ablation", "ensemble"}
        if requesting_stage != "qualification" or self.final_holdout_opened or not required.issubset(self.completed_stages):
            raise PermissionError("final holdout is sealed")
        self.final_holdout_opened = True
        self.final_holdout_accessed_at = datetime.now(timezone.utc).isoformat()

    @classmethod
    def load(cls, path: Path) -> "SearchState":
        return cls(**json.loads(path.read_text()))

    @classmethod
    def load_or_create(cls, path: Path, *, search_id: str, dataset_id: str, remaining_wall_clock_seconds: float) -> "SearchState":
        if path.exists():
            state = cls.load(path)
            if state.search_id != search_id or state.dataset_id != dataset_id:
                raise ValueError("experiment state does not match search_id and dataset_id")
            return state
        return cls(search_id=search_id, dataset_id=dataset_id, remaining_wall_clock_seconds=remaining_wall_clock_seconds)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(self), sort_keys=True, indent=2) + "\n")
        temporary.replace(path)


@dataclass(frozen=True)
class StageResult:
    parent_run_id: str
    status: str
    completed_trial_ids: tuple[str, ...]
    skipped_reason: str | None = None


def run_stage(config: Mapping[str, Any], dataset_id: str, feature_set_id: str, stage: str) -> StageResult:
    if stage not in VALID_STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {', '.join(sorted(VALID_STAGES))}")
    values = dict(config)
    search_id = str(values["search_id"])
    state_path = Path(values.get("state_dir", ".experiment-state")) / f"{search_id}.json"
    budget = float(values.get("remaining_wall_clock_seconds", values.get("max_hours", 24) * 3600))
    contract = _contract(values, dataset_id, feature_set_id)
    if state_path.exists() and not values.get("resume", False):
        raise ValueError("existing experiment state requires --resume")
    state = SearchState.load_or_create(state_path, search_id=search_id, dataset_id=dataset_id, remaining_wall_clock_seconds=budget)
    if state.experiment_contract and state.experiment_contract != contract:
        raise ValueError("immutable experiment contract does not match existing state")
    if not state.experiment_contract:
        state.experiment_contract = contract
        state.save(state_path)
    reason = _neural_ineligibility_reason(values, state) if stage == "neural" else None
    done = set(state.completed_trial_ids.get(stage, []))
    trials = list(values.get("trials", []))
    pending = [
        (number, trial) for number, trial in enumerate(trials)
        if str(trial.get("id", number)) not in done
    ]
    tracker = ExperimentTracker(str(values.get("experiment_name", "btc-volatility-tournament")), values.get("tracking_uri"))
    recovered_state = False
    if stage in state.best_run_ids and stage not in state.best_scores:
        recovered_score = tracker.run_metric(state.best_run_ids[stage], "aggregate_pr_auc")
        if recovered_score is not None:
            state.best_scores[stage] = float(recovered_score)
            recovered_state = True
    existing_parent_id = state.stage_parent_run_ids.get(stage) or tracker.find_stage_parent(
        search_id=search_id, stage=stage,
    )
    if existing_parent_id and stage not in state.stage_parent_run_ids:
        state.stage_parent_run_ids[stage] = existing_parent_id
        recovered_state = True
    if recovered_state:
        state.save(state_path)
    if not pending and stage in state.completed_stages and existing_parent_id:
        status = "skipped" if reason else "completed"
        return StageResult(existing_parent_id, status, (), reason)

    parent_lineage = _lineage(values, dataset_id, feature_set_id, stage, model_family=stage, deployable=False)
    if existing_parent_id:
        parent_id = tracker.start_run(parent_lineage, run_id=existing_parent_id)
    else:
        parent_id = tracker.start_run(parent_lineage, run_name=f"{stage}-parent")
        state.stage_parent_run_ids[stage] = parent_id
        state.save(state_path)
    started = time.monotonic()
    if reason:
        state.completed_stages.append(stage) if stage not in state.completed_stages else None
        state.remaining_wall_clock_seconds = max(0.0, state.remaining_wall_clock_seconds - (time.monotonic() - started))
        state.save(state_path)
        tracker.end_run("FINISHED", extra_tags={"stage_status": "skipped", "skip_reason": reason})
        return StageResult(parent_id, "skipped", (), reason)

    completed_now: list[str] = []
    failures = 0
    try:
        evaluated_trials = _evaluate_pending_trials(
            pending, max_parallel_jobs=int(values.get("max_parallel_jobs", 1)),
            remaining_wall_clock_seconds=state.remaining_wall_clock_seconds, started=started,
        )
        for number, trial, evaluated in evaluated_trials:
            trial_id = str(trial.get("id", number))
            child_lineage = _lineage(
                values, dataset_id, feature_set_id, stage,
                model_family=str(trial.get("model_family", stage)),
                deployable=bool(trial.get("deployable", values.get("deployable", False))),
            )
            child_id = tracker.start_run(child_lineage, nested=True, run_name=f"{stage}-trial-{trial_id}")
            failed_this_trial = False
            try:
                if state.remaining_wall_clock_seconds <= time.monotonic() - started:
                    trial = {**trial, "outcome": "skipped", "skip_reason": "search wall-clock budget exhausted"}
                if isinstance(evaluated, BaseException):
                    raise evaluated
                record = {**trial, **evaluated}
                outcome = str(record.get("outcome", "finished"))
                if outcome == "failed":
                    raise RuntimeError(str(record.get("exception", "trial failed")))
                tracker.log_params({"trial_id": trial_id, **dict(record.get("params", {}))})
                tracker.log_metrics(dict(record.get("metrics", {})))
                tracker.log_tags(dict(record.get("tags", {})))
                artifacts = record.get("lineage_artifacts")
                if artifacts:
                    tracker.log_lineage_artifacts(**dict(artifacts))
                timing = record.get("resource_timing")
                if timing:
                    tracker.log_resource_timing(dict(timing))
                model = record.get("model")
                if model is not None:
                    tracker.log_model(model, artifact_path="model")
                if outcome == "pruned":
                    tracker.end_run("KILLED", run_status="pruned")
                elif outcome == "skipped":
                    tracker.end_run(
                        "FINISHED", run_status="skipped",
                        extra_tags={"skip_reason": record.get("skip_reason", "trial skipped")},
                    )
                elif outcome == "finished":
                    metrics = dict(record.get("metrics", {}))
                    eligible = _selection_eligible(metrics, values)
                    tracker.log_tags({"selection_eligible": eligible})
                    tracker.end_run("FINISHED")
                    score = metrics.get("aggregate_pr_auc")
                    if score is not None and eligible and float(score) > state.best_scores.get(stage, float("-inf")):
                        state.best_run_ids[stage] = child_id
                        state.best_scores[stage] = float(score)
                    if score is not None:
                        state.stage_score_history.setdefault(stage, []).append(float(score))
                else:
                    raise ValueError(f"unsupported trial outcome {outcome!r}")
            except Exception as exc:
                failures += 1
                failed_this_trial = True
                tracker.log_failure(exc)
                tracker.end_run("FAILED", run_status="failed")
            done.add(trial_id)
            completed_now.append(trial_id)
            state.completed_trial_ids[stage] = sorted(done)
            state.failure_counts[stage] = state.failure_counts.get(stage, 0) + int(failed_this_trial)
            state.save(state_path)
            # The threshold is over the fixed stage trial budget, not just the
            # partial prefix executed in this invocation.  The numerator is
            # persisted, so a resume cannot reset accumulated failures.
            if trials and state.failure_counts[stage] / len(trials) > 0.20:
                raise RuntimeError(f"stage {stage} exceeded 20% trial failure threshold")
        if stage not in state.completed_stages:
            state.completed_stages.append(stage)
        tracker.end_run("FINISHED", extra_tags={"stage_status": "completed"})
        return StageResult(parent_id, "completed", tuple(completed_now))
    except Exception as exc:
        tracker.log_failure(exc)
        tracker.end_run("FAILED", run_status="failed", extra_tags={"stage_status": "failed"})
        raise
    finally:
        state.remaining_wall_clock_seconds = max(0.0, state.remaining_wall_clock_seconds - (time.monotonic() - started))
        state.save(state_path)


def _selection_eligible(metrics: Mapping[str, Any], values: Mapping[str, Any]) -> bool:
    """Refuse a stage win to a trial that already fails qualification's bars.

    Selection used to rank on mean fold score alone, so a candidate could take
    a stage on one lucky fold, or while badly calibrated, and only lose
    qualification afterwards — once the sealed holdout had been spent.  Both
    bars are the same ones qualification applies.
    """
    max_brier_ratio = values.get("max_development_brier_ratio")
    if max_brier_ratio is not None:
        ratio = metrics.get("development_brier_ratio")
        if ratio is not None and float(ratio) > float(max_brier_ratio):
            return False
    minimum_folds = values.get("min_development_folds_won")
    if minimum_folds is not None:
        won = metrics.get("development_folds_won")
        if won is not None and float(won) < float(minimum_folds):
            return False
    return True


def _lineage(
    values: Mapping[str, Any], dataset_id: str, feature_set_id: str, stage: str,
    *, model_family: str, deployable: bool | None = None,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id, "feature_set_id": feature_set_id,
        "target_version": values.get("target_version", "vol_spike_v1"),
        "validation_version": values.get("validation_version", "walkforward_v1"),
        "git_sha": values.get("git_sha", "unknown"), "search_id": values["search_id"],
        "model_family": model_family,
        "deployable": values.get("deployable", False) if deployable is None else deployable,
        "candidate_stage": stage,
    }


def _contract(values: Mapping[str, Any], dataset_id: str, feature_set_id: str) -> dict[str, str]:
    """Fields that cannot change once a resumable search has begun."""
    return {
        "dataset_id": dataset_id,
        "feature_set_id": feature_set_id,
        "target_version": str(values.get("target_version", "vol_spike_v1")),
        "validation_version": str(values.get("validation_version", "walkforward_v1")),
        "git_sha": str(values.get("git_sha", "unknown")),
    }


def _neural_ineligibility_reason(values: Mapping[str, Any], state: SearchState) -> str | None:
    if values.get("neural_skip_reason"):
        return str(values["neural_skip_reason"])
    if int(values.get("labelled_rows", 0)) < 100_000:
        return "neural stage requires at least 100,000 labelled rows"
    positives = values.get("development_fold_positive_events", [])
    if not positives or any(int(count) < 100 for count in positives):
        return "neural stage requires at least 100 positive events in every development fold"
    missing = [stage for stage in ("trees", "ablation") if stage not in state.completed_stages]
    if missing:
        return f"neural stage requires completed stages: {', '.join(missing)}"
    scores = state.stage_score_history.get("trees", [])
    minimum_trials = int(values.get("boosted_tree_plateau_min_trials", 10))
    if len(scores) < minimum_trials:
        return f"neural stage requires measured boosted-tree plateau from at least {minimum_trials} tree trials"
    midpoint = len(scores) // 2
    earlier_best = max(scores[:midpoint])
    later_best = max(scores[midpoint:])
    max_improvement = float(values.get("boosted_tree_plateau_max_improvement", 0.001))
    if later_best - earlier_best > max_improvement:
        return (
            "neural stage requires boosted-tree plateau "
            f"(improvement {later_best - earlier_best:.6f} exceeds {max_improvement:.6f})"
        )
    return None


def _evaluate_pending_trials(
    pending: list[tuple[int, Mapping[str, Any]]], *, max_parallel_jobs: int,
    remaining_wall_clock_seconds: float, started: float,
) -> list[tuple[int, Mapping[str, Any], dict[str, Any] | BaseException]]:
    """Evaluate independent development trials concurrently, then log them serially.

    MLflow's fluent run context is process-global, so evaluations run in worker
    threads while parent/child MLflow logging remains ordered on the caller.
    """
    results: list[tuple[int, Mapping[str, Any], dict[str, Any] | BaseException]] = []
    workers = max(1, max_parallel_jobs)
    for start_index in range(0, len(pending), workers):
        batch = pending[start_index:start_index + workers]
        if time.monotonic() - started >= remaining_wall_clock_seconds:
            results.extend((number, {**trial, "outcome": "skipped", "skip_reason": "search wall-clock budget exhausted"}, {}) for number, trial in batch)
            continue
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures: list[Future[dict[str, Any]]] = []
            runnable: list[bool] = []
            for _number, trial in batch:
                evaluator = trial.get("evaluate")
                should_evaluate = callable(evaluator) and trial.get("outcome", "finished") == "finished"
                runnable.append(should_evaluate)
                if should_evaluate:
                    futures.append(executor.submit(evaluator))
            future_index = 0
            for (number, trial), should_evaluate in zip(batch, runnable):
                if not should_evaluate:
                    results.append((number, trial, {}))
                    continue
                future = futures[future_index]
                future_index += 1
                try:
                    results.append((number, trial, dict(future.result())))
                except BaseException as exc:
                    results.append((number, trial, exc))
    return results
