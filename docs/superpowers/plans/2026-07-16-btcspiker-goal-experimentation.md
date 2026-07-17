# BTCSpiker Goal-Driven Experimentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible, MLflow-tracked experimentation system that searches broadly for a stronger out-of-sample predictor of BTCSpiker's existing 60-second Coinbase BTC-USD volatility-spike target and registers only fully qualified, deployable candidates as Staging.

**Architecture:** Store immutable raw and curated Parquet partitions plus completed-run exports in iCloud, while keeping the active cache and MLflow SQLite volume local. A single causal feature engine feeds batch, replay, and streaming paths; five-fold purged walk-forward validation drives a progressive model tournament, with an untouched final holdout and explicit Staging gates.

**Tech Stack:** Python 3.11+, pandas, NumPy, PyArrow, scikit-learn, Optuna, LightGBM, XGBoost, CatBoost, MLflow 2.12.x, pytest, Docker Compose, Coinbase Exchange/Advanced Trade public market data, Binance public-data archives, iCloud Drive.

## Global Constraints

- Keep the primary target at a 60-second horizon and threshold `0.000048`.
- Version the operational target as trade-price realized volatility, matching `features/featurizer.py`; do not silently switch it to midprice.
- Use only information available at or before each prediction timestamp.
- Use free public data and local M3 Pro compute only; no paid APIs or paid cloud compute.
- Limit each major model-search cycle to 24 hours and bound parallel work to avoid exhausting 18 GB RAM.
- Keep the active MLflow SQLite database and Docker volume local; export completed artifacts to iCloud.
- Default local cache ceiling is 4 GiB because only about 10 GiB was free during planning.
- Use five expanding walk-forward folds, a 20% untouched final holdout, and an embargo of `max_feature_lookback + 60 seconds`.
- Primary metric is PR-AUC; always log prevalence lift, calibration, event metrics, regime metrics, and latency.
- Never use the final holdout for feature selection, tuning, calibration, threshold selection, or ensemble construction.
- Never auto-promote a candidate to Production. Passing candidates may be registered as Staging only.
- Preserve the current logistic artifact and deterministic rule as rollback targets.
- Preserve the `/predict` p95 SLO of 800 ms.

## File Structure

### New files

- `requirements-ml.txt` — research-only dependency set, separate from API/worker images.
- `experiment.yaml` — immutable defaults for storage, target, splits, metrics, search budgets, and model stages.
- `btcspiker_ml/__init__.py` — package marker and version.
- `btcspiker_ml/config.py` — typed experiment configuration loader.
- `btcspiker_ml/storage.py` — local cache, atomic iCloud publication, checksums, and capacity checks.
- `btcspiker_ml/manifest.py` — dataset and feature manifests with stable IDs.
- `btcspiker_ml/sources/coinbase.py` — Coinbase paginated trade backfill normalization.
- `btcspiker_ml/sources/binance.py` — official archive URL construction and checksum verification.
- `btcspiker_ml/eda.py` — target audit, data quality, temporal EDA, and report artifacts.
- `btcspiker_ml/features.py` — shared causal feature engine and versioned feature-set registry.
- `btcspiker_ml/splits.py` — untouched holdout and purged walk-forward folds.
- `btcspiker_ml/metrics.py` — row, calibration, event, regime, latency, and paired-bootstrap metrics.
- `btcspiker_ml/models.py` — bounded model factories and Optuna search spaces.
- `btcspiker_ml/tracking.py` — required MLflow tags, parameters, metrics, and artifacts.
- `btcspiker_ml/search.py` — staged tournament orchestration and resume state.
- `btcspiker_ml/qualification.py` — exact Staging gates and reason codes.
- `btcspiker_ml/export.py` — completed-run export and checksum index for iCloud.
- `scripts/collect_dataset.py` — backfill/live-compaction CLI.
- `scripts/profile_dataset.py` — EDA and target-audit CLI.
- `scripts/run_experiments.py` — model-tournament CLI.
- `scripts/qualify_candidate.py` — final-holdout, replay, latency, Staging, and export CLI.
- `docs/goals/prediction-quality-goal.md` — durable `/goal` charter and operating checkpoints.
- `tests/ml/` — focused unit and integration tests for the research pipeline.

### Existing files to modify

- `.gitignore` — ignore local cache, run state, and generated datasets without ignoring manifests or reports.
- `scripts/ws_ingest.py` — capture bid/ask quantities and stage larger local segments for hourly Parquet compaction.
- `features/featurizer.py` — consume the shared feature engine and emit feature-schema metadata.
- `scripts/replay.py` — consume the same shared feature engine.
- `scripts/feature_to_predict_bridge.py` — forward the versioned deployable payload instead of seven hardcoded fields.
- `api/main.py` — accept backward-compatible extra features and validate the registered feature contract.
- `scripts/log_model_to_mlflow.py` — keep legacy bootstrap but remove candidate auto-promotion responsibilities.
- `docker-compose.yaml` — expose candidate model name/stage configuration without changing the default Production model.
- `handoff/docs/feature_spec.md` — correct the target's trade-price wording and document schema versions.
- `README.md`, `docs/runbook.md`, `docs/results.md` — add experiment, MLflow, iCloud, Staging, resume, and rollback instructions.

---

### Task 1: Freeze the experiment contract and dependency boundary

**Files:**
- Create: `requirements-ml.txt`
- Create: `experiment.yaml`
- Create: `btcspiker_ml/__init__.py`
- Create: `btcspiker_ml/config.py`
- Create: `tests/ml/test_config.py`
- Modify: `.gitignore:15-20`

**Interfaces:**
- Produces: `ExperimentConfig`, `load_experiment_config(path: Path) -> ExperimentConfig`.
- Consumes: no earlier task interfaces.

- [ ] **Step 1: Write the failing configuration test**

```python
from pathlib import Path

from btcspiker_ml.config import load_experiment_config


def test_loads_frozen_target_and_validation_contract():
    cfg = load_experiment_config(Path("experiment.yaml"))
    assert cfg.target.horizon_seconds == 60
    assert cfg.target.volatility_threshold == 0.000048
    assert cfg.target.price_field == "price"
    assert cfg.validation.folds == 5
    assert cfg.validation.final_holdout_fraction == 0.20
    assert cfg.validation.max_feature_lookback_seconds == 300
    assert cfg.search.max_hours == 24
    assert cfg.storage.local_cache_max_gib == 4
```

- [ ] **Step 2: Run the test and confirm the missing package failure**

Run: `pytest tests/ml/test_config.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'btcspiker_ml'`.

- [ ] **Step 3: Add exact research dependencies and YAML defaults**

```text
# requirements-ml.txt
-r handoff/requirements.txt
httpx>=0.27,<1
optuna>=3.6,<5
scipy>=1.11,<2
psutil>=5.9,<8
lightgbm>=4.3,<5
xgboost>=2.0,<4
catboost>=1.2,<2
```

```yaml
# experiment.yaml
storage:
  cloud_root: "~/Library/Mobile Documents/com~apple~CloudDocs/BTCSpiker"
  local_cache: ".cache/btcspiker"
  local_cache_max_gib: 4
target:
  name: "trade_price_future_vol_spike_60s_v1"
  horizon_seconds: 60
  volatility_threshold: 0.000048
  price_field: "price"
validation:
  folds: 5
  final_holdout_fraction: 0.20
  max_feature_lookback_seconds: 300
  bootstrap_block_minutes: 30
  bootstrap_resamples: 2000
  random_seed: 42
search:
  max_hours: 24
  max_parallel_jobs: 4
  stage_trials: {linear: 30, trees: 80, ablation: 60, ensemble: 30, neural: 10}
mlflow:
  tracking_uri: "http://localhost:5001"
  experiment_name: "btc-volatility-tournament"
  registered_model_name: "btc-volatility-candidate"
```

- [ ] **Step 4: Implement typed configuration loading**

```python
# btcspiker_ml/config.py
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class StorageConfig:
    cloud_root: Path
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
    storage["cloud_root"] = Path(storage["cloud_root"]).expanduser()
    storage["local_cache"] = Path(storage["local_cache"]).expanduser()
    return ExperimentConfig(
        storage=StorageConfig(**storage),
        target=TargetConfig(**raw["target"]),
        validation=ValidationConfig(**raw["validation"]),
        search=SearchConfig(**raw["search"]),
        mlflow=MlflowConfig(**raw["mlflow"]),
    )
```

Add `.cache/btcspiker/` and `.experiment-state/` to `.gitignore`; keep `reports/` trackable.

- [ ] **Step 5: Run tests and commit**

Run: `pytest tests/ml/test_config.py -q`

Expected: `1 passed`.

```bash
git add .gitignore requirements-ml.txt experiment.yaml btcspiker_ml tests/ml/test_config.py
git commit -m "feat: freeze ML experiment contract"
```

---

### Task 2: Add immutable manifests and safe iCloud publication

**Files:**
- Create: `btcspiker_ml/manifest.py`
- Create: `btcspiker_ml/storage.py`
- Create: `tests/ml/test_manifest.py`
- Create: `tests/ml/test_storage.py`

**Interfaces:**
- Produces: `DatasetManifest`, `manifest_id(manifest) -> str`, `sha256_file(path) -> str`, `atomic_publish(source, destination) -> PublishedFile`, `ensure_capacity(path, required_bytes) -> None`, `prune_cache(root, max_bytes) -> list[Path]`.
- Consumes: `StorageConfig` from Task 1.

- [ ] **Step 1: Write failing determinism and atomic-publication tests**

```python
from pathlib import Path

from btcspiker_ml.manifest import DatasetManifest, manifest_id
from btcspiker_ml.storage import atomic_publish, sha256_file


def test_manifest_id_ignores_dictionary_order():
    left = DatasetManifest("coinbase", "BTC-USD", 10, "a", "b", {"x": 1, "y": 2}, [])
    right = DatasetManifest("coinbase", "BTC-USD", 10, "a", "b", {"y": 2, "x": 1}, [])
    assert manifest_id(left) == manifest_id(right)


def test_atomic_publish_verifies_bytes(tmp_path: Path):
    source = tmp_path / "source.parquet"
    source.write_bytes(b"verified")
    result = atomic_publish(source, tmp_path / "cloud" / "part.parquet")
    assert result.sha256 == sha256_file(result.path)
    assert result.path.read_bytes() == b"verified"
```

- [ ] **Step 2: Verify failure before implementation**

Run: `pytest tests/ml/test_manifest.py tests/ml/test_storage.py -q`

Expected: import errors for `btcspiker_ml.manifest` and `btcspiker_ml.storage`.

- [ ] **Step 3: Implement stable manifests and checksum publication**

```python
# btcspiker_ml/manifest.py
from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class DatasetManifest:
    source: str
    product: str
    rows: int
    start_time: str
    end_time: str
    quality: dict[str, int | float | str]
    partitions: list[dict[str, str | int]]


def manifest_id(manifest: DatasetManifest) -> str:
    payload = json.dumps(asdict(manifest), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
```

```python
# btcspiker_ml/storage.py
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil


@dataclass(frozen=True)
class PublishedFile:
    path: Path
    sha256: str
    size_bytes: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_capacity(path: Path, required_bytes: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free < required_bytes:
        raise OSError(f"insufficient free space: required={required_bytes} free={free}")


def atomic_publish(source: Path, destination: Path) -> PublishedFile:
    destination.parent.mkdir(parents=True, exist_ok=True)
    ensure_capacity(destination.parent, source.stat().st_size * 2)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copy2(source, temporary)
    source_hash = sha256_file(source)
    if sha256_file(temporary) != source_hash:
        temporary.unlink(missing_ok=True)
        raise OSError("checksum mismatch during publication")
    temporary.replace(destination)
    return PublishedFile(destination, source_hash, destination.stat().st_size)


def prune_cache(root: Path, max_bytes: int) -> list[Path]:
    files = sorted((path for path in root.rglob("*") if path.is_file()), key=lambda path: path.stat().st_mtime)
    total = sum(path.stat().st_size for path in files)
    removed: list[Path] = []
    for path in files:
        if total <= max_bytes:
            break
        size = path.stat().st_size
        path.unlink()
        total -= size
        removed.append(path)
    return removed
```

- [ ] **Step 4: Test capacity failure and commit**

Run: `pytest tests/ml/test_manifest.py tests/ml/test_storage.py -q`

Expected: all tests pass, including a monkeypatched `shutil.disk_usage` case that raises `insufficient free space` and an LRU case that removes the oldest file until the cache is within its byte ceiling.

```bash
git add btcspiker_ml/manifest.py btcspiker_ml/storage.py tests/ml/test_manifest.py tests/ml/test_storage.py
git commit -m "feat: add immutable dataset publication"
```

---

### Task 3: Acquire target-aligned Coinbase data and external Binance context

**Files:**
- Create: `btcspiker_ml/sources/__init__.py`
- Create: `btcspiker_ml/sources/coinbase.py`
- Create: `btcspiker_ml/sources/binance.py`
- Create: `scripts/collect_dataset.py`
- Create: `tests/ml/test_coinbase_source.py`
- Create: `tests/ml/test_binance_source.py`
- Modify: `scripts/ws_ingest.py:95-133,223-325`

**Interfaces:**
- Produces: `iter_coinbase_trades(session, product, max_pages) -> Iterator[dict]`, `binance_archive_url(market, kind, symbol, date) -> str`, `compact_segments(inputs, output) -> DatasetManifest`.
- Consumes: `atomic_publish`, `DatasetManifest`, and Task 1 storage paths.

- [ ] **Step 1: Write mocked source tests**

```python
from datetime import date

from btcspiker_ml.sources.binance import binance_archive_url
from btcspiker_ml.sources.coinbase import normalize_trade


def test_coinbase_trade_normalization():
    row = normalize_trade({"trade_id": 7, "time": "2026-01-01T00:00:00Z", "price": "90000", "size": "0.1", "side": "sell"})
    assert row == {"source": "coinbase", "product_id": "BTC-USD", "trade_id": 7, "timestamp": "2026-01-01T00:00:00Z", "price": 90000.0, "size": 0.1, "maker_side": "sell"}


def test_official_binance_daily_url():
    assert binance_archive_url("spot", "aggTrades", "BTCUSDT", date(2026, 1, 2)) == "https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2026-01-02.zip"
```

- [ ] **Step 2: Run tests and verify missing modules**

Run: `pytest tests/ml/test_coinbase_source.py tests/ml/test_binance_source.py -q`

Expected: import failures for both source adapters.

- [ ] **Step 3: Implement exact public-source adapters**

```python
# btcspiker_ml/sources/coinbase.py
from collections.abc import Iterator

import httpx


BASE_URL = "https://api.exchange.coinbase.com"


def normalize_trade(raw: dict, product: str = "BTC-USD") -> dict:
    return {
        "source": "coinbase",
        "product_id": product,
        "trade_id": int(raw["trade_id"]),
        "timestamp": raw["time"],
        "price": float(raw["price"]),
        "size": float(raw["size"]),
        "maker_side": raw["side"],
    }


def iter_coinbase_trades(client: httpx.Client, product: str, max_pages: int) -> Iterator[dict]:
    cursor = None
    seen: set[int] = set()
    for _page in range(max_pages):
        params = {"limit": 1000}
        if cursor is not None:
            params["after"] = cursor
        response = client.get(f"{BASE_URL}/products/{product}/trades", params=params)
        response.raise_for_status()
        rows = response.json()
        if not rows:
            return
        for raw in rows:
            row = normalize_trade(raw, product)
            if row["trade_id"] not in seen:
                seen.add(row["trade_id"])
                yield row
        next_cursor = response.headers.get("CB-AFTER")
        if next_cursor is None or next_cursor == cursor:
            return
        cursor = next_cursor
```

```python
# btcspiker_ml/sources/binance.py
from datetime import date


def binance_archive_url(market: str, kind: str, symbol: str, day: date) -> str:
    stamp = day.isoformat()
    return f"https://data.binance.vision/data/{market}/daily/{kind}/{symbol}/{symbol}-{kind}-{stamp}.zip"


def checksum_url(archive_url: str) -> str:
    return f"{archive_url}.CHECKSUM"
```

- [ ] **Step 4: Extend live capture fields and compaction behavior**

In `scripts/ws_ingest.py`, add `best_bid_quantity`, `best_ask_quantity`, `price_percent_chg_24_h`, `source="coinbase"`, `channel="ticker"`, and `sequence_num` to every mirrored payload. Change `MIRROR_FLUSH_SIZE` from `100` to `10000`, write only to local `.cache/btcspiker/capture/`, and make `scripts/collect_dataset.py compact` convert closed NDJSON segments into hourly Zstandard-compressed Parquet before calling `atomic_publish`.

The compacted schema is exactly:

```python
RAW_TICK_COLUMNS = [
    "source", "channel", "product_id", "timestamp", "sequence_num",
    "price", "best_bid", "best_ask", "best_bid_quantity",
    "best_ask_quantity", "volume_24_h", "price_percent_chg_24_h",
]
```

- [ ] **Step 5: Test mocked pagination, checksum rejection, and compaction**

Run: `pytest tests/ml/test_coinbase_source.py tests/ml/test_binance_source.py -q`

Expected: tests prove cursor progression, duplicate trade suppression, exact archive URLs, checksum mismatch rejection, and one immutable Parquet output per hour.

- [ ] **Step 6: Run a bounded live smoke test and commit**

Run: `python scripts/ws_ingest.py --pair BTC-USD --minutes 1`

Expected: local capture contains bid/ask quantities and no file is written directly into iCloud.

Run: `python scripts/collect_dataset.py compact --config experiment.yaml`

Expected: an hourly Parquet partition, manifest, and verified iCloud copy are printed.

```bash
git add btcspiker_ml/sources scripts/collect_dataset.py scripts/ws_ingest.py tests/ml/test_coinbase_source.py tests/ml/test_binance_source.py
git commit -m "feat: collect versioned market data"
```

---

### Task 4: Audit the actual target and produce temporal EDA

**Files:**
- Create: `btcspiker_ml/eda.py`
- Create: `scripts/profile_dataset.py`
- Create: `tests/ml/test_eda.py`
- Modify: `handoff/docs/feature_spec.md:5-20`

**Interfaces:**
- Produces: `profile_dataset(frame, target_column, timestamp_column) -> DataProfile`, `write_profile_artifacts(profile, output_dir) -> list[Path]`.
- Consumes: curated partitions and manifests from Tasks 2-3.

- [ ] **Step 1: Write a failing target-audit test**

```python
import pandas as pd

from btcspiker_ml.eda import profile_dataset


def test_profile_exposes_time_span_duplicates_and_daily_prevalence():
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z", "2026-01-01T00:00:01Z"]),
        "vol_spike": [0, 1, 1],
        "price": [1.0, 1.1, 1.1],
    })
    profile = profile_dataset(frame, "vol_spike", "timestamp")
    assert profile.rows == 3
    assert profile.duplicate_timestamps == 1
    assert profile.positive_rate == 2 / 3
    assert profile.start_time == "2026-01-01T00:00:00+00:00"
```

- [ ] **Step 2: Implement deterministic EDA outputs**

Implement the profile core as:

```python
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataProfile:
    rows: int
    start_time: str
    end_time: str
    duplicate_timestamps: int
    positive_rate: float
    non_finite_by_column: dict[str, int]
    daily_prevalence: dict[str, float]


def profile_dataset(frame: pd.DataFrame, target_column: str, timestamp_column: str) -> DataProfile:
    ordered = frame.sort_values(timestamp_column).copy()
    timestamps = pd.to_datetime(ordered[timestamp_column], utc=True)
    numeric = ordered.select_dtypes(include=[np.number])
    daily = ordered.assign(_day=timestamps.dt.strftime("%Y-%m-%d")).groupby("_day")[target_column].mean()
    return DataProfile(
        rows=len(ordered),
        start_time=timestamps.iloc[0].isoformat(),
        end_time=timestamps.iloc[-1].isoformat(),
        duplicate_timestamps=int(timestamps.duplicated().sum()),
        positive_rate=float(ordered[target_column].mean()),
        non_finite_by_column={column: int((~np.isfinite(numeric[column])).sum()) for column in numeric},
        daily_prevalence={str(day): float(value) for day, value in daily.items()},
    )
```

Extend the report layer with source coverage, duplicate event IDs, gaps by duration, inter-arrival quantiles, feature correlations, autocorrelation at 1/5/60/300 seconds, and drift summaries by day. `scripts/profile_dataset.py` logs `profile.json`, `daily_prevalence.csv`, `correlations.csv`, and PNG plots to a parent MLflow run named `eda-<dataset_id>`.

Use the operational target name `trade_price_future_vol_spike_60s_v1`. Update `handoff/docs/feature_spec.md` to state that v1 uses last-trade `price`, not midprice, because that is what the current code and artifact use.

- [ ] **Step 3: Verify EDA on the handoff sample without making quality claims**

Run: `python scripts/profile_dataset.py --input handoff/data_sample/features_slice.parquet --config experiment.yaml --provisional`

Expected: report labels the sample `provisional`, records its roughly ten-minute span, and refuses to mark it search-eligible.

- [ ] **Step 4: Run tests and commit**

Run: `pytest tests/ml/test_eda.py -q`

Expected: all EDA tests pass.

```bash
git add btcspiker_ml/eda.py scripts/profile_dataset.py tests/ml/test_eda.py handoff/docs/feature_spec.md
git commit -m "feat: audit target and temporal data quality"
```

---

### Task 5: Centralize causal features and prove batch/stream parity

**Files:**
- Create: `btcspiker_ml/features.py`
- Create: `tests/ml/conftest.py`
- Create: `tests/ml/test_feature_engine.py`
- Modify: `features/featurizer.py:61-230`
- Modify: `scripts/replay.py:53-189`
- Modify: `features/feature_funcs.py`

**Interfaces:**
- Produces: `FeatureSet`, `FEATURE_SETS`, `FeatureEngine.ingest(tick) -> list[dict]`, `materialize_features(ticks, feature_set_id) -> DataFrame`.
- Consumes: the frozen target contract and raw canonical schema.

- [ ] **Step 1: Write parity and leakage tests**

```python
# tests/ml/conftest.py
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture
def raw_ticks():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [
        {
            "source": "coinbase", "channel": "ticker", "product_id": "BTC-USD",
            "timestamp": (start + timedelta(seconds=index)).isoformat(), "sequence_num": index,
            "price": 90_000.0 + (index % 17), "best_bid": 89_999.5 + (index % 17),
            "best_ask": 90_000.5 + (index % 17), "best_bid_quantity": 1.0 + index % 3,
            "best_ask_quantity": 1.0 + index % 5, "volume_24_h": 10_000.0,
            "price_percent_chg_24_h": 1.0,
        }
        for index in range(500)
    ]
```

```python
# tests/ml/test_feature_engine.py
import pandas as pd

from btcspiker_ml.features import FeatureEngine, asof_join_external, materialize_features


def test_batch_and_stream_features_match(raw_ticks):
    batch = materialize_features(pd.DataFrame(raw_ticks), "core_v1")
    engine = FeatureEngine("core_v1", horizon_seconds=60, threshold=0.000048)
    streamed = [row for tick in raw_ticks for row in engine.ingest(tick)]
    pd.testing.assert_frame_equal(
        batch.reset_index(drop=True),
        pd.DataFrame(streamed).reset_index(drop=True),
        check_exact=False,
        rtol=1e-10,
        atol=1e-12,
    )


def test_features_do_not_change_when_future_ticks_are_modified(raw_ticks):
    cutoff = len(raw_ticks) // 2
    original = materialize_features(pd.DataFrame(raw_ticks), "multi_window_v1")
    mutated_ticks = [dict(row) for row in raw_ticks]
    for row in mutated_ticks[cutoff + 1:]:
        row["price"] = float(row["price"]) * 10
    mutated = materialize_features(pd.DataFrame(mutated_ticks), "multi_window_v1")
    pd.testing.assert_frame_equal(original.iloc[:cutoff], mutated.iloc[:cutoff])


def test_external_asof_join_never_uses_future_rows():
    coinbase = pd.DataFrame({"timestamp": pd.to_datetime(["2026-01-01T00:00:05Z"])})
    external = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01T00:00:04Z", "2026-01-01T00:00:06Z"]),
        "binance_return_60s": [0.1, 99.0],
    })
    joined = asof_join_external(coinbase, external, tolerance_seconds=5)
    assert joined.loc[0, "binance_return_60s"] == 0.1
```

- [ ] **Step 2: Define stable feature sets**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSet:
    feature_set_id: str
    columns: tuple[str, ...]
    windows_seconds: tuple[int, ...]
    max_lookback_seconds: int
    schema_version: str
    deployable: bool
    required_sources: tuple[str, ...]


FEATURE_SETS = {
    "core_v1": FeatureSet("core_v1", ("log_return", "spread_bps", "vol_60s", "mean_return_60s", "trade_intensity_60s", "n_ticks_60s", "spread_mean_60s"), (60,), 60, "1", True, ("coinbase_ticker",)),
    "multi_window_v1": FeatureSet(
        "multi_window_v1",
        (
            "log_return", "spread_bps",
            "return_5s", "vol_5s", "mean_return_5s", "price_range_bps_5s", "spread_mean_bps_5s", "trade_intensity_5s", "interarrival_mean_ms_5s", "interarrival_std_ms_5s",
            "return_15s", "vol_15s", "mean_return_15s", "price_range_bps_15s", "spread_mean_bps_15s", "trade_intensity_15s", "interarrival_mean_ms_15s", "interarrival_std_ms_15s",
            "return_30s", "vol_30s", "mean_return_30s", "price_range_bps_30s", "spread_mean_bps_30s", "trade_intensity_30s", "interarrival_mean_ms_30s", "interarrival_std_ms_30s",
            "return_60s", "vol_60s", "mean_return_60s", "price_range_bps_60s", "spread_mean_bps_60s", "trade_intensity_60s", "interarrival_mean_ms_60s", "interarrival_std_ms_60s",
            "return_120s", "vol_120s", "mean_return_120s", "price_range_bps_120s", "spread_mean_bps_120s", "trade_intensity_120s", "interarrival_mean_ms_120s", "interarrival_std_ms_120s",
            "return_300s", "vol_300s", "mean_return_300s", "price_range_bps_300s", "spread_mean_bps_300s", "trade_intensity_300s", "interarrival_mean_ms_300s", "interarrival_std_ms_300s",
        ),
        (5, 15, 30, 60, 120, 300), 300, "2", True, ("coinbase_ticker",),
    ),
    "microstructure_v1": FeatureSet(
        "microstructure_v1",
        (
            "log_return", "spread_bps", "return_5s", "vol_5s", "return_15s", "vol_15s", "return_30s", "vol_30s", "return_60s", "vol_60s", "return_120s", "vol_120s", "return_300s", "vol_300s",
            "book_imbalance", "book_imbalance_mean_5s", "book_imbalance_mean_15s", "book_imbalance_mean_60s", "spread_change_5s", "spread_change_60s", "intensity_change_15s", "intensity_change_60s", "ewma_vol_fast", "ewma_vol_slow", "vol_of_vol_60s", "momentum_15s", "momentum_60s", "acceleration_60s",
        ),
        (5, 15, 30, 60, 120, 300), 300, "3", True, ("coinbase_ticker",),
    ),
    "external_context_v1": FeatureSet(
        "external_context_v1",
        ("vol_60s", "return_60s", "book_imbalance", "binance_return_60s", "binance_vol_60s", "coinbase_binance_basis_bps", "binance_funding_rate"),
        (60, 300), 300, "4", False, ("coinbase_ticker", "binance_agg_trades"),
    ),
}
```

- [ ] **Step 3: Move stateful logic into one engine**

Move `ProductState` behavior from `features/featurizer.py` into `FeatureEngine`. Make both `features/featurizer.py` and `scripts/replay.py` import the same class. Include `feature_set_id` and `feature_schema_version` in every emitted row. Preserve `future_vol_60s` and `vol_spike` only on labelled training messages; the API payload excludes labels.

Implement external joins with:

```python
def asof_join_external(coinbase: pd.DataFrame, external: pd.DataFrame, tolerance_seconds: int) -> pd.DataFrame:
    left = coinbase.sort_values("timestamp").copy()
    right = external.sort_values("timestamp").copy()
    left["timestamp"] = pd.to_datetime(left["timestamp"], utc=True)
    right["timestamp"] = pd.to_datetime(right["timestamp"], utc=True)
    return pd.merge_asof(
        left,
        right,
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta(seconds=tolerance_seconds),
    )
```

Reject duplicated external timestamps unless their source event IDs are unique.

- [ ] **Step 4: Run feature tests and existing replay tests**

Run: `pytest tests/ml/test_feature_engine.py tests/test_replay_integration.py -q`

Expected: parity and leakage tests pass; integration either passes against a running stack or skips only through an explicit environment marker.

- [ ] **Step 5: Commit**

```bash
git add btcspiker_ml/features.py features/feature_funcs.py features/featurizer.py scripts/replay.py tests/ml/conftest.py tests/ml/test_feature_engine.py
git commit -m "feat: unify causal feature generation"
```

---

### Task 6: Implement purged walk-forward splits and decision metrics

**Files:**
- Create: `btcspiker_ml/splits.py`
- Create: `btcspiker_ml/metrics.py`
- Create: `tests/ml/test_splits.py`
- Create: `tests/ml/test_metrics.py`

**Interfaces:**
- Produces: `TemporalFold`, `make_temporal_splits(timestamps, config) -> SplitPlan`, `evaluate_predictions(y, scores, timestamps, threshold) -> MetricBundle`, `paired_block_bootstrap(y_true, candidate_scores, baseline_scores, timestamps, block_minutes, resamples, seed) -> ConfidenceInterval`.
- Consumes: `ValidationConfig` from Task 1.

- [ ] **Step 1: Write boundary and event-metric tests**

```python
import numpy as np
import pandas as pd

from btcspiker_ml.metrics import event_metrics
from btcspiker_ml.splits import make_temporal_splits


def test_every_fold_has_required_embargo():
    timestamps = pd.date_range("2026-01-01", periods=10_000, freq="s", tz="UTC")
    plan = make_temporal_splits(timestamps, folds=5, final_holdout_fraction=0.20, embargo_seconds=360)
    assert len(plan.folds) == 5
    for fold in plan.folds:
        assert timestamps[fold.validation[0]] - timestamps[fold.train[-1]] >= pd.Timedelta(seconds=360)
    assert set(plan.final_holdout).isdisjoint(set(index for fold in plan.folds for index in fold.train + fold.validation))


def test_event_metrics_apply_sixty_second_cooldown():
    ts = pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:00:10Z", "2026-01-01T00:02:00Z"])
    result = event_metrics(np.array([1, 1, 0]), np.array([0.9, 0.8, 0.9]), ts, threshold=0.5, cooldown_seconds=60)
    assert result.alerts == 2
```

- [ ] **Step 2: Implement deterministic split and metric types**

Use immutable dataclasses for `TemporalFold(train: list[int], validation: list[int])`, `SplitPlan(folds, final_holdout)`, `MetricBundle`, and `ConfidenceInterval(lower, estimate, upper)`. Sort timestamps once, reject duplicates unless a stable event key is supplied, and reject a split when any fold lacks both target classes.

`evaluate_predictions` must compute PR-AUC, prevalence, `pr_auc / prevalence`, ROC-AUC, log loss, Brier, ECE with ten fixed bins, F1, precision, recall, event F1, alerts per hour, and per-day/per-regime tables.

- [ ] **Step 3: Implement paired temporal bootstrap**

Group paired candidate/baseline predictions into 30-minute UTC blocks. Sample blocks with replacement 2,000 times using `np.random.default_rng(42)` and return the 2.5th, mean, and 97.5th percentiles of candidate-minus-baseline PR-AUC.

- [ ] **Step 4: Run tests and commit**

Run: `pytest tests/ml/test_splits.py tests/ml/test_metrics.py -q`

Expected: all boundary, holdout-isolation, cooldown, calibration, and deterministic-bootstrap tests pass.

```bash
git add btcspiker_ml/splits.py btcspiker_ml/metrics.py tests/ml/test_splits.py tests/ml/test_metrics.py
git commit -m "feat: add temporal validation and alert metrics"
```

---

### Task 7: Build bounded model factories and search spaces

**Files:**
- Create: `btcspiker_ml/models.py`
- Create: `tests/ml/test_models.py`
- Create: `requirements-neural.txt`

**Interfaces:**
- Produces: `build_model(family, params, seed, n_jobs) -> ClassifierMixin`, `suggest_params(trial, family) -> dict`, `model_families(stage) -> tuple[str, ...]`.
- Consumes: ordered feature matrices and fixed seeds.

- [ ] **Step 1: Write model-contract tests**

```python
import numpy as np
import pytest

from btcspiker_ml.models import build_model


@pytest.mark.parametrize("family", ["logistic", "sgd_logistic", "random_forest", "extra_trees", "hist_gradient_boosting", "lightgbm", "xgboost", "catboost"])
def test_every_tabular_model_returns_probabilities(family):
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = build_model(family, {}, seed=42, n_jobs=1)
    model.fit(X, y)
    probabilities = model.predict_proba(X)[:, 1]
    assert probabilities.shape == (4,)
    assert np.all((0 <= probabilities) & (probabilities <= 1))
```

- [ ] **Step 2: Implement factories with bounded defaults**

Every model receives seed 42 where supported. Tree libraries receive `n_jobs=1` inside a parallel Optuna search to avoid nested oversubscription. Linear models wrap scaling in a scikit-learn `Pipeline`. CatBoost defaults to `verbose=False`; LightGBM uses `verbosity=-1`; XGBoost uses `tree_method="hist"`.

Search spaces must be explicit and bounded: regularization strengths `1e-4..1e2` log-uniform; depths `2..10`; leaves `8..256`; learning rates `0.005..0.3`; estimators `100..2000` with early stopping; row/column sampling `0.5..1.0`; minimum child sizes `5..500`.

- [ ] **Step 3: Add gated neural dependency file**

```text
# requirements-neural.txt
torch>=2.3,<3
```

Do not install it in the standard experiment environment. Stage 5 installs it only after the search state confirms 30 days of data and a boosted-tree plateau.

- [ ] **Step 4: Run tests and commit**

Run: `pytest tests/ml/test_models.py -q`

Expected: all installed model families return finite probabilities; unavailable optional libraries produce a clear installation message rather than silent exclusion.

```bash
git add btcspiker_ml/models.py tests/ml/test_models.py requirements-neural.txt
git commit -m "feat: add bounded model tournament factories"
```

---

### Task 8: Make MLflow the complete experiment ledger

**Files:**
- Create: `btcspiker_ml/tracking.py`
- Create: `btcspiker_ml/search.py`
- Create: `scripts/run_experiments.py`
- Create: `tests/ml/test_tracking.py`
- Create: `tests/ml/test_search.py`

**Interfaces:**
- Produces: `ExperimentTracker`, `run_stage(config, dataset_id, feature_set_id, stage) -> StageResult`, `SearchState` persisted under `.experiment-state/<search_id>.json`.
- Consumes: Tasks 1-7.

- [ ] **Step 1: Write a local-file MLflow logging test**

```python
from pathlib import Path

import mlflow

from btcspiker_ml.tracking import ExperimentTracker


def test_tracker_logs_required_lineage(tmp_path: Path):
    mlflow.set_tracking_uri(tmp_path.as_uri())
    tracker = ExperimentTracker("test-experiment")
    run_id = tracker.start_run({"dataset_id": "d1", "feature_set_id": "core_v1", "git_sha": "abc", "model_family": "logistic", "deployable": "true"})
    tracker.log_metrics({"fold_0_pr_auc": 0.2, "aggregate_pr_auc": 0.2})
    tracker.end_run("FINISHED")
    run = mlflow.tracking.MlflowClient().get_run(run_id)
    assert run.data.tags["dataset_id"] == "d1"
    assert run.data.metrics["aggregate_pr_auc"] == 0.2
```

- [ ] **Step 2: Implement required lineage enforcement**

`ExperimentTracker.start_run` must reject missing `dataset_id`, `feature_set_id`, `target_version`, `validation_version`, `git_sha`, `search_id`, `model_family`, or `deployable`. Log exact config YAML, dataset/feature manifests, fold boundaries, out-of-fold predictions, PR/calibration plots, regime tables, model size, peak memory, fit time, inference time, and failure traceback.

- [ ] **Step 3: Implement resumable staged search**

`SearchState` records `search_id`, dataset ID, completed stages, best run IDs, remaining wall-clock seconds, final-holdout-opened flag, and failure counts. `run_stage` creates one parent run and nested child runs for every trial/fold. Catch trial exceptions, tag them `run_status=failed`, log `traceback.txt`, and continue unless failures exceed 20% of the stage.

The CLI accepts only:

```text
python scripts/run_experiments.py --config experiment.yaml --dataset-id <id> --stage baseline|linear|trees|ablation|ensemble|neural --resume
```

Reject `neural` unless the data manifest covers at least 30 days and the completed search state contains `trees` and `ablation`.

- [ ] **Step 4: Test successful, pruned, failed, and resumed trials**

Run: `pytest tests/ml/test_tracking.py tests/ml/test_search.py -q`

Expected: MLflow contains one finished, one pruned, and one failed child run; resume does not duplicate finished trial IDs.

- [ ] **Step 5: Commit**

```bash
git add btcspiker_ml/tracking.py btcspiker_ml/search.py scripts/run_experiments.py tests/ml/test_tracking.py tests/ml/test_search.py
git commit -m "feat: track resumable model tournaments in MLflow"
```

---

### Task 9: Execute the progressive tournament without touching the holdout

**Files:**
- Modify: `experiment.yaml`
- Create: `reports/experiment_summary.md`
- Create: `tests/ml/test_holdout_guard.py`

**Interfaces:**
- Produces: completed MLflow stages and `reports/experiment_summary.md` generated from run IDs.
- Consumes: all experiment framework tasks.

- [ ] **Step 1: Add a failing holdout-access guard test**

```python
import pytest

from btcspiker_ml.search import SearchState


def test_development_stage_cannot_open_final_holdout():
    state = SearchState.new("search-1", "dataset-1", wall_clock_seconds=86400)
    with pytest.raises(PermissionError, match="final holdout is sealed"):
        state.open_final_holdout(requesting_stage="trees")
```

- [ ] **Step 2: Implement the sealed-holdout guard and baseline gate**

Permit final-holdout access only from `scripts/qualify_candidate.py` after baseline, linear, trees, ablation, and ensemble stages are complete. Require shipped pickle-versus-MLflow prediction parity on the handoff sample at absolute tolerance `1e-9`, then re-train the exact current logistic configuration on the new development corpus.

- [ ] **Step 3: Run EDA and staged search in cost order**

```bash
python scripts/profile_dataset.py --dataset-id "$DATASET_ID" --config experiment.yaml
python scripts/run_experiments.py --config experiment.yaml --dataset-id "$DATASET_ID" --stage baseline --resume
python scripts/run_experiments.py --config experiment.yaml --dataset-id "$DATASET_ID" --stage linear --resume
python scripts/run_experiments.py --config experiment.yaml --dataset-id "$DATASET_ID" --stage trees --resume
python scripts/run_experiments.py --config experiment.yaml --dataset-id "$DATASET_ID" --stage ablation --resume
python scripts/run_experiments.py --config experiment.yaml --dataset-id "$DATASET_ID" --stage ensemble --resume
```

Expected: every trial appears in `btc-volatility-tournament`; the final holdout remains sealed; stage summaries name winners and non-winners with fold-level evidence.

- [ ] **Step 4: Apply the data sufficiency checkpoint**

If the target-aligned quote-and-trade manifest covers fewer than 30 calendar days, do not run Stage 5 or qualify a champion. Leave the collector running, tag results `provisional_data=true`, record the resume condition `coverage_days >= 30`, and pause the Codex goal. Historical trade-only experiments remain research-only.

If coverage is sufficient and boosted-tree progress plateaued, install `requirements-neural.txt` in a separate environment and run the bounded neural stage once.

- [ ] **Step 5: Run guard tests and commit the generated summary**

Run: `pytest tests/ml/test_holdout_guard.py -q`

Expected: holdout access is denied to all development stages.

```bash
git add experiment.yaml reports/experiment_summary.md tests/ml/test_holdout_guard.py
git commit -m "exp: record progressive model tournament"
```

---

### Task 10: Qualify candidates and export completed evidence to iCloud

**Files:**
- Create: `btcspiker_ml/qualification.py`
- Create: `btcspiker_ml/export.py`
- Create: `scripts/qualify_candidate.py`
- Create: `tests/ml/test_qualification.py`
- Create: `tests/ml/test_export.py`
- Modify: `scripts/log_model_to_mlflow.py:118-162`

**Interfaces:**
- Produces: `CandidateEvidence`, `QualificationResult`, `qualify(evidence: CandidateEvidence) -> QualificationResult`, `export_run(run_id: str, cloud_root: Path) -> ExportManifest`.
- Consumes: completed development runs, one sealed final holdout, storage publication, and MLflow.

- [ ] **Step 1: Write exact gate tests**

```python
from btcspiker_ml.qualification import CandidateEvidence, qualify


def test_candidate_fails_when_bootstrap_lower_bound_is_not_positive():
    evidence = CandidateEvidence(
        folds_won=5,
        bootstrap_lower=-0.001,
        brier_ratio=0.99,
        event_f1_delta=0.01,
        final_pr_auc_delta=0.02,
        p95_latency_ms=120,
        deployable=True,
        parity_passed=True,
    )
    result = qualify(evidence)
    assert not result.passed
    assert "bootstrap_lower_not_positive" in result.reasons
```

- [ ] **Step 2: Implement all Staging gates as reason-coded predicates**

Require `folds_won >= 4`, `bootstrap_lower > 0`, `brier_ratio <= 1.05`, `event_f1_delta >= 0`, `final_pr_auc_delta > 0`, `p95_latency_ms <= 800`, `deployable is True`, and `parity_passed is True`. Evaluate the final holdout once, store its access timestamp in `SearchState`, and refuse a second evaluation for the same search ID.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateEvidence:
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


def qualify(evidence: CandidateEvidence) -> QualificationResult:
    checks = {
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
```

- [ ] **Step 3: Separate candidate registration from Production promotion**

Remove generic candidate promotion from `scripts/log_model_to_mlflow.py`; leave it responsible only for idempotently bootstrapping `btc-volatility-lr`. `scripts/qualify_candidate.py` registers a passing candidate under `btc-volatility-candidate` and transitions it only to `Staging`. It prints the run ID, model version, gate table, and the explicit message `Production unchanged`.

- [ ] **Step 4: Export immutable run evidence**

`export_run` downloads the model, configs, manifests, predictions, plots, model card, qualification JSON, and dependency freeze into a local temporary directory; creates `export-manifest.json` with SHA-256 per file; atomically publishes the directory contents to `mlflow-exports/run_id=<run_id>/`; then re-hashes the destination. It never deletes local MLflow artifacts.

- [ ] **Step 5: Run tests and commit**

Run: `pytest tests/ml/test_qualification.py tests/ml/test_export.py -q`

Expected: every failing predicate has a stable reason code, final holdout cannot be reopened, Staging succeeds in a temporary MLflow registry, and checksum corruption aborts export.

```bash
git add btcspiker_ml/qualification.py btcspiker_ml/export.py scripts/qualify_candidate.py scripts/log_model_to_mlflow.py tests/ml/test_qualification.py tests/ml/test_export.py
git commit -m "feat: gate and export Staging candidates"
```

---

### Task 11: Make versioned features servable without breaking v1 clients

**Files:**
- Modify: `api/main.py:61-115,172-208,243-247`
- Modify: `scripts/feature_to_predict_bridge.py:41-103`
- Modify: `features/featurizer.py`
- Modify: `tests/test_api.py:24-181`
- Modify: `tests/test_replay_integration.py:34-203`
- Modify: `docker-compose.yaml:106-176`

**Interfaces:**
- Produces: backward-compatible `/predict`, versioned feature messages, and candidate-stage runtime configuration.
- Consumes: MLflow params `feature_cols`, `feature_set_id`, and `feature_schema_version`.

- [ ] **Step 1: Write backward-compatibility and schema-rejection tests**

```python
def test_predict_accepts_legacy_v1_payload(base_url):
    response = requests.post(f"{base_url}/predict", json={"rows": [SAMPLE_ROW]}, timeout=5)
    assert response.status_code == 200


def test_predict_rejects_registered_feature_version_mismatch(base_url):
    payload = {"rows": [{**SAMPLE_ROW, "feature_schema_version": "wrong"}]}
    response = requests.post(f"{base_url}/predict", json=payload, timeout=5)
    assert response.status_code == 422
    assert "feature_schema_version" in response.text
```

- [ ] **Step 2: Make the request model accept registered extras safely**

Keep the seven existing fields required. Add optional `feature_set_id` and `feature_schema_version`, and configure Pydantic to allow extra numeric features. Replace `getattr(row, col)` with a dumped mapping lookup that returns a 422 response listing required missing columns and rejects booleans, strings, NaN, and infinity.

- [ ] **Step 3: Forward complete deployable features from Kafka**

Replace the bridge's hardcoded `FEATURE_COLS` filtering with:

```python
NON_MODEL_FIELDS = {"product_id", "timestamp", "future_vol_60s", "vol_spike"}


def _build_row(message: dict, kafka_timestamp_ms: int | None) -> dict:
    row = {key: value for key, value in message.items() if key not in NON_MODEL_FIELDS}
    row["ts"] = _isoformat_utc(kafka_timestamp_ms) if kafka_timestamp_ms and kafka_timestamp_ms > 0 else message.get("timestamp")
    return row
```

Require the featurizer to emit `feature_set_id` and `feature_schema_version`. Integration tests assert those fields survive Kafka, bridge, and API validation.

- [ ] **Step 4: Keep Production default and make Staging opt-in**

Leave `MODEL_STAGE=${MODEL_STAGE:-Production}` and the current model name as the default. Document the explicit candidate smoke command:

```bash
MODEL_NAME=btc-volatility-candidate MODEL_STAGE=Staging docker compose up -d --build api predict-bridge
```

- [ ] **Step 5: Run API, replay, and load verification**

Run: `pytest tests/test_api.py -q`

Expected: legacy payloads pass, extra registered features pass, mismatched schemas fail with 422.

Run: `REPLAY_TEST_AUTOSTART=1 REPLAY_SPEED=50 pytest tests/test_replay_integration.py -v`

Expected: versioned features traverse the full runtime path.

Run: `python3 tests/load_test.py --url http://127.0.0.1:8000/predict --requests 1000 --concurrency 20`

Expected: 100% success and p95 at or below 800 ms.

- [ ] **Step 6: Commit**

```bash
git add api/main.py scripts/feature_to_predict_bridge.py features/featurizer.py tests/test_api.py tests/test_replay_integration.py docker-compose.yaml
git commit -m "feat: serve versioned candidate feature sets"
```

---

### Task 12: Add the durable `/goal` charter, runbook, and final proof

**Files:**
- Create: `docs/goals/prediction-quality-goal.md`
- Modify: `README.md`
- Modify: `docs/runbook.md`
- Modify: `docs/results.md`
- Modify: `progress.md`

**Interfaces:**
- Produces: the exact `/goal` prompt, pause/resume checkpoint, MLflow/iCloud operating commands, final evidence report, and implementation handoff.
- Consumes: all prior tasks.

- [ ] **Step 1: Write the durable goal charter**

The charter must contain the objective, global constraints from this plan, stage order, MLflow requirements, data threshold, holdout seal, Staging gates, stop conditions, pause conditions, and the exact final report schema. End it with this status checklist:

```markdown
## Goal completion checklist

- [ ] Data manifest and EDA are published.
- [ ] Current artifact and new-data logistic baselines are reproduced.
- [ ] Eligible staged searches are complete or the budget is exhausted.
- [ ] Every run, failure, and pruning decision is visible in MLflow.
- [ ] Final holdout was opened at most once.
- [ ] Qualification reasons are recorded.
- [ ] Passing candidate is Staging only; Production is unchanged.
- [ ] MLflow evidence is checksum-exported to iCloud.
- [ ] Replay, API, rollback, and latency verification results are recorded.
```

- [ ] **Step 2: Add exact operations to the README and runbook**

Document environment setup, `BTCSPIKER_CLOUD_ROOT`, collector start/stop, compaction, dataset profiling, each search stage, MLflow URL, resume state, qualification, Staging smoke test, iCloud export verification, rollback, and the 30-day pause/resume rule.

- [ ] **Step 3: Run the complete verification suite**

```bash
pytest tests/ml -q
pytest tests/test_api.py -q
REPLAY_TEST_AUTOSTART=1 REPLAY_SPEED=50 pytest tests/test_replay_integration.py -v
docker compose ps
curl -s http://127.0.0.1:5001/health
curl -s http://127.0.0.1:8000/version
```

Expected: all unit tests pass; replay integration passes; MLflow and API are reachable; `/version` identifies the intended model and stage; Production remains the legacy champion unless the user separately approves promotion.

- [ ] **Step 4: Verify MLflow and iCloud evidence**

Confirm one successful, one pruned, and one failed run have complete lineage. Recompute every SHA-256 in the chosen run's iCloud `export-manifest.json`. Record dataset ID, search ID, baseline run ID, best candidate run ID, Staging model version, fold metrics, bootstrap interval, final holdout metrics, latency, and export path in `docs/results.md`.

- [ ] **Step 5: Commit final documentation**

```bash
git add docs/goals/prediction-quality-goal.md README.md docs/runbook.md docs/results.md progress.md
git commit -m "docs: operationalize prediction quality goal"
```

## Paste-ready `/goal` command

```text
/goal Improve BTCSpiker's out-of-sample prediction of the existing 60-second Coinbase BTC-USD trade-price volatility-spike target by executing docs/superpowers/plans/2026-07-16-btcspiker-goal-experimentation.md and treating docs/goals/prediction-quality-goal.md as the durable charter. Use the data-first gated tournament, log every successful, pruned, and failed trial to MLflow, preserve the sealed temporal holdout, use only free data and local compute, keep active MLflow state local, export immutable datasets and completed-run evidence to iCloud, and never auto-promote Production. If target-aligned quote-and-trade coverage is under 30 days, finish and verify the framework, leave the collector running, record the exact resume condition, and pause rather than claim a reliable model improvement. Continue until a deployable candidate passes every Staging gate or the justified 24-hour search budget is exhausted, then deliver an evidence-backed comparison and next-step report.
```

## Goal operating commands

```text
/goal                 View current objective and status.
/goal pause           Pause before closing the Mac, losing connectivity, or waiting for data.
/goal resume          Resume after the data manifest reaches 30 days or after a resolved blocker.
/goal edit            Change constraints without discarding the task history.
/goal clear           Remove the goal only after accepting the final report or abandoning the effort.
```

Keep **Prevent sleep while running** enabled during collection, compaction, or a 24-hour search cycle. Use the same Codex task so its goal, MLflow run IDs, dataset IDs, and resume state remain connected.
