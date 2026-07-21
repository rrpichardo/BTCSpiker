"""Resumable, staged MLflow tournament orchestration.

This module deliberately records research outcomes only; it never registers or
promotes a model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    failure_counts: dict[str, int] = field(default_factory=dict)
    completed_trial_ids: dict[str, list[str]] = field(default_factory=dict)

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
    state = SearchState.load_or_create(state_path, search_id=search_id, dataset_id=dataset_id, remaining_wall_clock_seconds=budget)
    tracker = ExperimentTracker(str(values.get("experiment_name", "btc-volatility-tournament")), values.get("tracking_uri"))
    parent_lineage = _lineage(values, dataset_id, feature_set_id, stage, model_family=stage)
    parent_id = tracker.start_run(parent_lineage, run_name=f"{stage}-parent")
    started = time.monotonic()
    reason = _neural_ineligibility_reason(values, state) if stage == "neural" else None
    if reason:
        state.completed_stages.append(stage) if stage not in state.completed_stages else None
        state.remaining_wall_clock_seconds = max(0.0, state.remaining_wall_clock_seconds - (time.monotonic() - started))
        state.save(state_path)
        tracker.end_run("FINISHED", extra_tags={"stage_status": "skipped", "skip_reason": reason})
        return StageResult(parent_id, "skipped", (), reason)

    done = set(state.completed_trial_ids.get(stage, []))
    completed_now: list[str] = []
    failures = 0
    trials = list(values.get("trials", []))
    try:
        for number, trial in enumerate(trials):
            trial_id = str(trial.get("id", number))
            if trial_id in done:
                continue
            child_lineage = _lineage(values, dataset_id, feature_set_id, stage, model_family=str(trial.get("model_family", stage)))
            child_id = tracker.start_run(child_lineage, nested=True, run_name=f"{stage}-trial-{trial_id}")
            failed_this_trial = False
            try:
                outcome = str(trial.get("outcome", "finished"))
                if outcome == "failed":
                    raise RuntimeError(str(trial.get("exception", "trial failed")))
                tracker.log_params({"trial_id": trial_id, **dict(trial.get("params", {}))})
                tracker.log_metrics(dict(trial.get("metrics", {})))
                if outcome == "pruned":
                    tracker.end_run("KILLED", run_status="pruned")
                elif outcome == "finished":
                    tracker.end_run("FINISHED")
                    score = trial.get("metrics", {}).get("aggregate_pr_auc")
                    if score is not None and (stage not in state.best_run_ids or float(score) > float(values.get("_best_score", "-inf"))):
                        state.best_run_ids[stage] = child_id
                        values["_best_score"] = float(score)
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
            if trials and failures / len(trials) > 0.20:
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


def _lineage(values: Mapping[str, Any], dataset_id: str, feature_set_id: str, stage: str, *, model_family: str) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id, "feature_set_id": feature_set_id,
        "target_version": values.get("target_version", "vol_spike_v1"),
        "validation_version": values.get("validation_version", "walkforward_v1"),
        "git_sha": values.get("git_sha", "unknown"), "search_id": values["search_id"],
        "model_family": model_family, "deployable": values.get("deployable", False),
        "candidate_stage": stage,
    }


def _neural_ineligibility_reason(values: Mapping[str, Any], state: SearchState) -> str | None:
    if int(values.get("labelled_rows", 0)) < 100_000:
        return "neural stage requires at least 100,000 labelled rows"
    positives = values.get("development_fold_positive_events", [])
    if not positives or any(int(count) < 100 for count in positives):
        return "neural stage requires at least 100 positive events in every development fold"
    missing = [stage for stage in ("trees", "ablation") if stage not in state.completed_stages]
    if missing:
        return f"neural stage requires completed stages: {', '.join(missing)}"
    return None
