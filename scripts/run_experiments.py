"""Run one resumable development-tournament stage; never opens the holdout or promotes models."""

from __future__ import annotations

import os

# Set BEFORE numpy/scipy/sklearn/torch import, not applied later via
# threadpoolctl.threadpool_limits() at call time: threadpoolctl only patches
# native libraries already loaded into the process at the moment it's
# entered. A library first dlopen'd lazily -- e.g. scipy's lbfgs solver,
# loaded on a trial's first LogisticRegression.fit() call, not at import time
# -- would slip through uncapped even inside a `with threadpool_limits():`
# block, letting that trial oversubscribe while sibling trials
# (max_parallel_jobs > 1) are still running. Env vars are read by each
# library at ITS OWN init time, so they hold regardless of load order.
# Verified empirically: a context-manager-only version of this fix still let
# a concurrently-loaded OpenMP library report num_threads=11 mid-run.
#
# THREAD_LIMIT_ENV_VARS is intentionally a literal copy of
# btcspiker_ml.search's constant of the same name, not an import of it:
# `import btcspiker_ml.search` pulls in btcspiker_ml.tracking ->
# mlflow.sklearn -> sklearn/numpy transitively, which would defeat the
# entire "before numpy is ever imported" point of this block by importing
# numpy right here. Drift between the two copies is caught by
# tests/ml/test_run_experiments.py::test_thread_limit_env_vars_match_search_module.
THREAD_LIMIT_ENV_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)
for _thread_env in THREAD_LIMIT_ENV_VARS:
    os.environ.setdefault(_thread_env, "1")

import argparse
import importlib.util
import pickle
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import psutil
import threadpoolctl
import yaml
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import _num_samples

try:
    import optuna
except ImportError:  # The standard API image omits requirements-ml.txt.
    optuna = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from btcspiker_ml.config import ExperimentConfig, load_experiment_config
from btcspiker_ml.datasets import resolve_existing_dataset
from btcspiker_ml.features import FEATURE_SETS
from btcspiker_ml.metrics import evaluate_predictions
from btcspiker_ml.models import build_model, model_families, suggest_params
from btcspiker_ml.search import VALID_STAGES, run_stage
from btcspiker_ml.splits import make_temporal_splits


_OPTIONAL_MODULES = {"lightgbm": "lightgbm", "xgboost": "xgboost", "catboost": "catboost"}

# Platt scaling over a chronological tail of each training window.  Two
# parameters cannot overfit a small calibration slice the way isotonic can.
CALIBRATION_METHOD = "sigmoid"
CALIBRATION_FRACTION = 0.25
# Qualification fails a candidate whose Brier score regresses more than 5%
# against the prevalence baseline; development selection applies the same bar
# so the tournament cannot crown a candidate that is already disqualified.
MAX_DEVELOPMENT_BRIER_RATIO = 1.05
# Qualification also requires at least four folds beaten.  Selecting on the mean
# fold score alone let one lucky fold carry an otherwise unbeaten-by-prevalence
# candidate to a stage win.
MIN_DEVELOPMENT_FOLDS_WON = 4


def build_stage_trials(config: ExperimentConfig, raw: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    """Build a deterministic, bounded trial plan from the configured stage budget.

    Trial IDs, family rotation, Optuna suggestions, and ablations are stable so
    a resumed process can regenerate the plan and skip persisted trial IDs.
    """
    if stage not in VALID_STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    budget = 1 if stage == "baseline" else int(config.search.stage_trials.get(stage, 0))
    if budget < 0:
        raise ValueError(f"trial budget for {stage} cannot be negative")

    columns = list(raw.get("feature_columns", []))
    if stage == "baseline":
        families = ("development_prevalence",)
    elif stage == "linear":
        families = ("logistic", "sgd_logistic")
    elif stage == "trees":
        families = tuple(family for family in model_families("all") if family not in {"logistic", "sgd_logistic"})
    elif stage == "ablation":
        families = ("logistic", "hist_gradient_boosting", "extra_trees")
    elif stage == "ensemble":
        families = ("logistic_hist_gradient_soft_voting",)
    else:
        families = ("neural",)

    seed = config.validation.random_seed + sorted(VALID_STAGES).index(stage)
    if optuna is not None:
        sampler = optuna.samplers.RandomSampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        suggestion_trials = [study.ask() for _ in range(budget)]
    else:
        study = None
        suggestion_trials = [_DeterministicTrial(seed + number) for number in range(budget)]
    frame: pd.DataFrame | None = None
    trials: list[dict[str, Any]] = []
    for number in range(budget):
        family = families[number % len(families)]
        optuna_trial = suggestion_trials[number]
        if family == "development_prevalence":
            params: dict[str, Any] = {}
        elif family == "logistic_hist_gradient_soft_voting":
            params = {"tree_weight": optuna_trial.suggest_float("tree_weight", 0.2, 0.8)}
        elif family == "neural":
            # window is the model's entire reason to exist -- it is what lets
            # a sequence model see short-term dynamics a row-independent model
            # cannot. Searching only hidden_width (as this used to) gives 3
            # distinct configs across the stage's 10 trials; window and epochs
            # are searched too so the budget actually explores the space that
            # matters.
            params = {
                "hidden_width": optuna_trial.suggest_categorical("hidden_width", [32, 64, 128]),
                "window": optuna_trial.suggest_categorical("window", [10, 20, 40]),
                "epochs": optuna_trial.suggest_int("epochs", 5, 30),
            }
        else:
            configured = _configured_params(raw, family)
            params = {**configured, **suggest_params(optuna_trial, family)}
        if study is not None:
            study.tell(optuna_trial, 0.0)

        trial_id = f"{stage}-{number:04d}-{family}"
        spec: dict[str, Any] = {
            "id": trial_id,
            "model_family": family,
            "params": params,
            "deployable": stage != "baseline" and family != "neural",
        }
        if stage == "ablation":
            if not columns:
                raise ValueError("ablation stage requires configured feature_columns")
            spec["removed_feature"] = columns[number % len(columns)]
        optional_module = _OPTIONAL_MODULES.get(family)
        if optional_module and importlib.util.find_spec(optional_module) is None:
            spec.update(outcome="skipped", skip_reason=f"optional model backend {optional_module} is not installed")
        elif family == "neural" and importlib.util.find_spec("torch") is None:
            spec.update(outcome="skipped", skip_reason="neural trials require the isolated neural environment")
        else:
            if frame is None:
                frame = pd.read_parquet(resolve_existing_dataset(config.storage.existing_data))
            spec["evaluate"] = _bind_evaluator(config, raw, stage, spec, frame)
        trials.append(spec)
    return trials


class _DeterministicTrial:
    """Small bounded fallback used only when requirements-ml is unavailable."""

    def __init__(self, seed: int):
        self._rng = np.random.default_rng(seed)

    def suggest_float(self, _name: str, low: float, high: float, *, log: bool = False) -> float:
        if log:
            return float(np.exp(self._rng.uniform(np.log(low), np.log(high))))
        return float(self._rng.uniform(low, high))

    def suggest_int(self, _name: str, low: int, high: int) -> int:
        return int(self._rng.integers(low, high + 1))

    def suggest_categorical(self, _name: str, choices: list[Any]) -> Any:
        return choices[int(self._rng.integers(0, len(choices)))]


def _configured_params(raw: dict[str, Any], family: str) -> dict[str, Any]:
    if family == "logistic":
        return dict(raw.get("logistic_params", {}))
    if family == "hist_gradient_boosting":
        return dict(raw.get("tree_params", {}))
    return dict(raw.get(f"{family}_params", {}))


def _runtime_feature_contract(raw: dict[str, Any]) -> tuple[str, str]:
    """Resolve the runtime feature identity from the registered feature set."""
    feature_set_id = str(raw.get("feature_set_id", ""))
    feature_set = FEATURE_SETS.get(feature_set_id)
    if feature_set is None:
        raise ValueError(f"unknown runtime feature_set_id: {feature_set_id!r}")
    return feature_set.feature_set_id, feature_set.schema_version


def _bind_evaluator(
    config: ExperimentConfig, raw: dict[str, Any], stage: str,
    spec: dict[str, Any], frame: pd.DataFrame,
):
    def evaluate() -> dict[str, Any]:
        return _evaluate_development_trial(config, raw, stage, spec, frame)

    return evaluate


class _TemporalCalibrationSplit:
    """One chronological fit/calibration split for probability calibration.

    Rows reach the estimator in ascending event-time order, so a plain
    ``StratifiedKFold`` would calibrate on rows that precede its own training
    window.  This yields a single split that always trains on the earlier rows
    and calibrates on the later ones.
    """

    def __init__(self, calibration_fraction: float):
        if not 0.0 < calibration_fraction < 1.0:
            raise ValueError("calibration_fraction must be between 0 and 1")
        self._fraction = calibration_fraction

    def split(self, X, y=None, groups=None):
        rows = _num_samples(X)
        cut = rows - int(np.ceil(rows * self._fraction))
        if cut < 1:
            raise ValueError("not enough rows to hold out a calibration window")
        yield np.arange(cut), np.arange(cut, rows)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return 1


def _candidate_model(spec: dict[str, Any], config: ExperimentConfig):
    family = str(spec["model_family"])
    params = dict(spec.get("params", {}))
    # Trial-level scheduling owns the configured parallelism.  Keep each
    # estimator single-threaded so four concurrent trials do not create a
    # second, unbounded pool of worker processes.
    n_jobs = 1
    if family == "development_prevalence":
        # The prevalence reference must stay raw; it is what calibration is scored against.
        estimator = DummyClassifier(strategy="prior")
    elif family == "logistic_hist_gradient_soft_voting":
        weight = float(params["tree_weight"])
        estimator = _calibrated(VotingClassifier(
            estimators=[
                ("linear", build_model("logistic", {}, config.validation.random_seed, 1)),
                ("tree", build_model("hist_gradient_boosting", {}, config.validation.random_seed, 1)),
            ],
            voting="soft", weights=[1.0 - weight, weight], n_jobs=n_jobs,
        ))
    else:
        estimator = _calibrated(build_model(family, params, config.validation.random_seed, n_jobs))
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", estimator)])


def _calibrated(estimator):
    """Fit a chronological probability calibrator around a candidate estimator.

    Ranking quality alone does not survive qualification: an uncalibrated
    candidate is scored on Brier against the prevalence baseline, and a
    confident-but-drifting model loses that comparison badly.
    """
    return CalibratedClassifierCV(
        estimator, method=CALIBRATION_METHOD,
        cv=_TemporalCalibrationSplit(CALIBRATION_FRACTION), ensemble=True,
        # Trial-level scheduling owns the parallelism; the calibrator must not
        # open a second pool on top of the concurrent trials.
        n_jobs=1,
    )


def _evaluate_development_trial(
    config: ExperimentConfig, raw: dict[str, Any], stage: str,
    spec: dict[str, Any], frame: pd.DataFrame,
) -> dict[str, Any]:
    """Evaluate and refit a candidate on development rows only."""
    started = time.perf_counter()
    rss_before = psutil.Process().memory_info().rss
    columns = list(raw.get("feature_columns", []))
    if not columns:
        columns = [
            column for column in frame.select_dtypes(include=["number"]).columns
            if column not in {"vol_spike", "future_vol_60s"}
        ]
    removed = spec.get("removed_feature")
    if removed:
        columns = [column for column in columns if column != removed]
    if not columns:
        raise ValueError("no usable development feature columns")

    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    target = frame["vol_spike"].to_numpy(dtype=int)
    plan = make_temporal_splits(
        timestamps, config.validation, targets=target, event_keys=np.arange(len(frame)),
    )
    source = frame.loc[:, columns].replace([np.inf, -np.inf], np.nan)
    fold_metrics: dict[str, float] = {}
    scores: list[float] = []
    folds_won = 0
    candidate_brier_total = 0.0
    baseline_brier_total = 0.0
    fit_seconds = 0.0
    inference_seconds = 0.0
    per_row_latency_ms: list[float] = []
    oof_parts: list[pd.DataFrame] = []
    development_rows: set[int] = set()
    fold_boundaries: list[dict[str, Any]] = []

    # Trial-level scheduling (max_parallel_jobs) owns concurrency; every estimator
    # must stay single-threaded or concurrent trials oversubscribe the machine.
    # n_jobs=1 alone doesn't guarantee this: HistGradientBoosting has no n_jobs
    # parameter at all and threads internally via OpenMP by default, and the BLAS
    # backend behind plain matrix ops threads independently of any sklearn param.
    # The actual cap is applied once, at process scope, by main() -- see the
    # threadpool_limits comment there for why a per-trial context manager here
    # doesn't work under concurrency.
    for fold_number, fold in enumerate(plan.folds):
        train = np.asarray(fold.train)
        validation = np.asarray(fold.validation)
        development_rows.update(fold.train)
        development_rows.update(fold.validation)
        model = _candidate_model(spec, config)
        before_fit = time.perf_counter()
        model.fit(source.iloc[train], target[train])
        fit_seconds += time.perf_counter() - before_fit
        before_inference = time.perf_counter()
        prediction = model.predict_proba(source.iloc[validation])[:, 1]
        duration = time.perf_counter() - before_inference
        inference_seconds += duration
        per_row_latency_ms.extend([duration * 1_000 / len(validation)] * len(validation))
        metric = evaluate_predictions(target[validation], prediction, timestamps.iloc[validation], threshold=0.5)
        fold_metrics[f"fold_{fold_number}_pr_auc"] = metric.pr_auc
        scores.append(metric.pr_auc)
        # A prevalence-constant predictor scores PR-AUC == prevalence, so this
        # counts the folds that actually beat the baseline qualification uses.
        folds_won += int(metric.pr_auc > metric.prevalence)
        # The baseline refits per fold and predicts that fold's training
        # prevalence, so the comparison has to move with it.  Scoring against a
        # single pooled constant instead measures prevalence drift, which
        # penalises every model equally — including the baseline itself.
        fold_baseline = np.full(len(validation), float(target[train].mean()))
        candidate_brier_total += brier_score_loss(target[validation], prediction) * len(validation)
        baseline_brier_total += brier_score_loss(target[validation], fold_baseline) * len(validation)
        oof_parts.append(pd.DataFrame({
            "row_index": validation, "timestamp": timestamps.iloc[validation].astype(str).to_numpy(),
            "target": target[validation], "prediction": prediction, "fold": fold_number,
        }))
        fold_boundaries.append({
            "fold": fold_number, "train_rows": len(train), "validation_rows": len(validation),
            "train_start": str(timestamps.iloc[train[0]]), "train_end": str(timestamps.iloc[train[-1]]),
            "validation_start": str(timestamps.iloc[validation[0]]), "validation_end": str(timestamps.iloc[validation[-1]]),
        })

    oof = pd.concat(oof_parts, ignore_index=True).sort_values("row_index")
    tau = _best_threshold(oof["target"].to_numpy(), oof["prediction"].to_numpy())
    if baseline_brier_total <= 0.0:
        raise ValueError("prevalence baseline has no Brier score to compare against")
    development_brier_ratio = float(candidate_brier_total / baseline_brier_total)
    fitted = _candidate_model(spec, config)
    development = np.asarray(sorted(development_rows), dtype=int)
    before_refit = time.perf_counter()
    fitted.fit(source.iloc[development], target[development])
    fit_seconds += time.perf_counter() - before_refit
    parity_passed = _roundtrip_feature_parity(fitted, source.iloc[development[: min(256, len(development))]])
    model_size_bytes = len(pickle.dumps(fitted, protocol=pickle.HIGHEST_PROTOCOL))
    peak_rss_mb = max(rss_before, psutil.Process().memory_info().rss) / (1024 * 1024)
    total_seconds = time.perf_counter() - started
    precision, recall, _ = precision_recall_curve(oof["target"], oof["prediction"])
    fraction_positive, mean_predicted = calibration_curve(oof["target"], oof["prediction"], n_bins=10, strategy="quantile")
    p95_latency_ms = float(np.percentile(per_row_latency_ms, 95))
    family = str(spec["model_family"])
    feature_set_id, feature_schema_version = _runtime_feature_contract(raw)

    return {
        "outcome": "finished",
        "params": {
            "model_family": family, "model_params": dict(spec.get("params", {})),
            "feature_cols": ",".join(columns), "feature_columns": columns,
            "feature_set_id": feature_set_id,
            "feature_schema_version": feature_schema_version,
            "tau": tau, "final_holdout": "sealed",
            "max_parallel_jobs": config.search.max_parallel_jobs,
        },
        "tags": {"feature_parity_passed": parity_passed},
        "metrics": {
            "aggregate_pr_auc": float(np.mean(scores)), "p95_latency_ms": p95_latency_ms,
            "development_brier_ratio": development_brier_ratio,
            "development_folds_won": float(folds_won),
            **fold_metrics,
        },
        "model": fitted,
        "resource_timing": {
            "fit_seconds": fit_seconds, "inference_seconds": inference_seconds,
            "total_seconds": total_seconds, "peak_rss_mb": peak_rss_mb,
            "model_size_bytes": float(model_size_bytes), "p95_latency_ms": p95_latency_ms,
        },
        "lineage_artifacts": {
            "config": raw,
            "dataset_manifest": {
                "path": str(resolve_existing_dataset(config.storage.existing_data).resolve()),
                "rows": len(frame), "start_time": str(timestamps.iloc[0]), "end_time": str(timestamps.iloc[-1]),
                "target_prevalence": float(target.mean()),
            },
            "feature_manifest": {
                "feature_set_id": feature_set_id,
                "feature_schema_version": feature_schema_version,
                "columns": columns,
                "removed_feature": removed,
            },
            "fold_boundaries": fold_boundaries,
            "oof_predictions": oof.to_csv(index=False),
            "pr_plot": pd.DataFrame({"precision": precision, "recall": recall}).to_csv(index=False),
            "calibration_plot": pd.DataFrame({"fraction_positive": fraction_positive, "mean_predicted": mean_predicted}).to_csv(index=False),
            "regime_table": {"all_development": {"rows": len(oof), "positive_events": int(oof["target"].sum())}},
        },
    }


def _best_threshold(target: np.ndarray, prediction: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(target, prediction)
    if not len(thresholds):
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], np.finfo(float).eps)
    return float(thresholds[int(np.nanargmax(f1))])


def _roundtrip_feature_parity(model: Any, values: pd.DataFrame) -> bool:
    """Verify the serialized runtime model preserves development predictions."""
    expected = model.predict_proba(values)
    runtime_model = pickle.loads(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    actual = runtime_model.predict_proba(values.to_numpy())
    return bool(np.allclose(expected, actual, rtol=0.0, atol=1e-12))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--stage", required=True, choices=sorted(VALID_STAGES))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text())
    config = load_experiment_config(args.config)
    frame = pd.read_parquet(resolve_existing_dataset(config.storage.existing_data), columns=["vol_spike"])
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    values = {
        "tracking_uri": config.mlflow.tracking_uri, "experiment_name": config.mlflow.experiment_name,
        "search_id": raw.get("search_id", args.dataset_id), "git_sha": git_sha,
        "target_version": raw.get("target_version", config.target.name),
        # v2 opens the expanding window on an equal block of the development
        # span instead of a bare embargo interval; folds are not comparable
        # across the two, so the contract has to name which one produced a run.
        "validation_version": raw.get("validation_version", "purged_walkforward_v2"),
        "deployable": False, "max_hours": config.search.max_hours,
        "max_parallel_jobs": config.search.max_parallel_jobs,
        "trials": build_stage_trials(config, raw, args.stage), "labelled_rows": int(frame.shape[0]),
        "max_development_brier_ratio": MAX_DEVELOPMENT_BRIER_RATIO,
        "min_development_folds_won": MIN_DEVELOPMENT_FOLDS_WON,
        "development_fold_positive_events": raw.get("development_fold_positive_events", []),
        "resume": args.resume,
    }
    # threadpoolctl's limits (BLAS/OpenMP) and torch's own thread pool are both
    # process-global state, not thread-local: a context manager or
    # set_num_threads() call entered/exited PER TRIAL (the old approach) races
    # when trials run concurrently (max_parallel_jobs > 1, see search.py's
    # _evaluate_pending_trials) -- trial A finishing and restoring/resetting
    # threads while B-D are still mid-fit. Applying both caps once, here, for
    # the whole stage's run is the only placement that actually holds them for
    # the process's entire trial-fitting lifetime. threadpoolctl doesn't track
    # torch's own intra-op thread pool, so both calls are needed.
    if importlib.util.find_spec("torch") is not None:
        import torch

        torch.set_num_threads(1)
    with threadpoolctl.threadpool_limits(limits=1):
        result = run_stage(values, args.dataset_id, raw.get("feature_set_id", "unknown"), args.stage)
    print(f"{args.stage}: {result.status} ({result.parent_run_id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
