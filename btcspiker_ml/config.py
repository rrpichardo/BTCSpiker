from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class StorageConfig:
    existing_data: Path
    artifact_root: Path
    local_cache: Path
    local_cache_max_gib: int


@dataclass(frozen=True)
class TargetConfig:
    name: str
    horizon_seconds: int
    volatility_threshold: float
    price_field: str


@dataclass(frozen=True)
class ValidationConfig:
    folds: int
    final_holdout_fraction: float
    max_feature_lookback_seconds: int
    bootstrap_block_minutes: int
    bootstrap_resamples: int
    random_seed: int


@dataclass(frozen=True)
class SearchConfig:
    max_hours: int
    max_parallel_jobs: int
    stage_trials: dict[str, int]


@dataclass(frozen=True)
class MlflowConfig:
    tracking_uri: str
    experiment_name: str
    registered_model_name: str


@dataclass(frozen=True)
class ExperimentConfig:
    storage: StorageConfig
    target: TargetConfig
    validation: ValidationConfig
    search: SearchConfig
    mlflow: MlflowConfig


def load_experiment_config(path: Path) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text())
    storage = dict(raw["storage"])
    storage["existing_data"] = Path(storage["existing_data"]).expanduser()
    storage["artifact_root"] = Path(storage["artifact_root"]).expanduser()
    storage["local_cache"] = Path(storage["local_cache"]).expanduser()
    return ExperimentConfig(
        storage=StorageConfig(**storage),
        target=TargetConfig(**raw["target"]),
        validation=ValidationConfig(**raw["validation"]),
        search=SearchConfig(**raw["search"]),
        mlflow=MlflowConfig(**raw["mlflow"]),
    )
