"""Deterministic model factories used by the tabular model tournament."""

from __future__ import annotations

import importlib.util
from typing import Any

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


_BASELINE_FAMILIES = ("logistic", "sgd_logistic", "random_forest", "extra_trees", "hist_gradient_boosting")
_OPTIONAL_FAMILIES = ("lightgbm", "xgboost", "catboost")
_ALL_FAMILIES = _BASELINE_FAMILIES + _OPTIONAL_FAMILIES


def model_families(stage: str) -> tuple[str, ...]:
    """Return the model families available for a tournament stage.

    ``baseline`` provides only scikit-learn models.  ``all`` (and the later
    ``boosted``/``tournament`` stages) adds optional boosted-tree libraries.
    """
    if stage == "baseline":
        return _BASELINE_FAMILIES
    if stage in {"all", "boosted", "tournament"}:
        return _ALL_FAMILIES
    raise ValueError(f"Unknown model stage {stage!r}; expected baseline, boosted, tournament, or all")


def _optional_library(module: str, family: str) -> None:
    if importlib.util.find_spec(module) is None:
        raise ImportError(
            f"Model family '{family}' requires the optional '{module}' package. "
            f"Install it to enable {family}."
        )


def _with_defaults(params: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    return {**defaults, **params}


def build_model(family: str, params: dict[str, Any], seed: int, n_jobs: int):
    """Build one seeded classifier without nested parallelism."""
    if family == "logistic":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(**_with_defaults(params, {"C": 1.0, "max_iter": 1_000, "random_state": seed}))),
            ]
        )
    if family == "sgd_logistic":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", SGDClassifier(**_with_defaults(params, {"loss": "log_loss", "alpha": 0.0001, "max_iter": 1_000, "tol": 1e-3, "random_state": seed}))),
            ]
        )
    if family == "random_forest":
        return RandomForestClassifier(**_with_defaults(params, {"n_estimators": 300, "min_samples_leaf": 5, "random_state": seed, "n_jobs": n_jobs}))
    if family == "extra_trees":
        return ExtraTreesClassifier(**_with_defaults(params, {"n_estimators": 300, "min_samples_leaf": 5, "random_state": seed, "n_jobs": n_jobs}))
    if family == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(**_with_defaults(params, {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 5, "early_stopping": True, "validation_fraction": 0.5, "random_state": seed}))
    if family == "neural":
        width = int(params.pop("hidden_width", 64))
        return MLPClassifier(**_with_defaults(params, {"hidden_layer_sizes": (width,), "max_iter": 300, "early_stopping": True, "random_state": seed}))
    if family == "lightgbm":
        _optional_library("lightgbm", family)
        from lightgbm import LGBMClassifier

        return LGBMClassifier(**_with_defaults(params, {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 5, "subsample": 1.0, "colsample_bytree": 1.0, "random_state": seed, "n_jobs": n_jobs, "verbosity": -1}))
    if family == "xgboost":
        _optional_library("xgboost", family)
        from xgboost import XGBClassifier

        return XGBClassifier(**_with_defaults(params, {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 5, "subsample": 1.0, "colsample_bytree": 1.0, "random_state": seed, "n_jobs": n_jobs, "tree_method": "hist", "eval_metric": "logloss"}))
    if family == "catboost":
        _optional_library("catboost", family)
        from catboost import CatBoostClassifier

        return CatBoostClassifier(**_with_defaults(params, {"iterations": 300, "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 1.0, "random_seed": seed, "thread_count": n_jobs, "verbose": False, "allow_writing_files": False}))
    raise ValueError(f"Unknown model family {family!r}; expected one of {', '.join(_ALL_FAMILIES)}")


def suggest_params(trial: Any, family: str) -> dict[str, Any]:
    """Suggest a deliberately bounded hyperparameter set for ``family``."""
    if family == "logistic":
        return {"C": trial.suggest_float("C", 1e-4, 1e2, log=True)}
    if family == "sgd_logistic":
        return {"alpha": trial.suggest_float("alpha", 1e-4, 1e2, log=True)}
    if family in {"random_forest", "extra_trees"}:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 500),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0),
        }
    if family == "hist_gradient_boosting":
        return {
            "max_iter": trial.suggest_int("max_iter", 100, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 256),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 500),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 1e2, log=True),
        }
    if family == "lightgbm":
        return _boosted_tree_params(trial, "num_leaves", "min_child_samples", "n_estimators")
    if family == "xgboost":
        return _boosted_tree_params(trial, "max_depth", "min_child_weight", "n_estimators")
    if family == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 100, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "depth": trial.suggest_int("depth", 2, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-4, 1e2, log=True),
            "rsm": trial.suggest_float("rsm", 0.5, 1.0),
        }
    raise ValueError(f"Unknown model family {family!r}")


def _boosted_tree_params(trial: Any, complexity_name: str, child_name: str, estimators_name: str) -> dict[str, Any]:
    complexity_bounds = (8, 256) if complexity_name == "num_leaves" else (2, 10)
    return {
        estimators_name: trial.suggest_int(estimators_name, 100, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        complexity_name: trial.suggest_int(complexity_name, *complexity_bounds),
        child_name: trial.suggest_int(child_name, 5, 500),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1e2, log=True),
    }
