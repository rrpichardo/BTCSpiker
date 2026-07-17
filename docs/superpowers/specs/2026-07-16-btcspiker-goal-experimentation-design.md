# BTCSpiker Goal-Driven Prediction Improvement Design

**Status:** Approved through the user's acceptance of all recommended defaults on 2026-07-16.

## Objective

Use Codex `/goal` to run a disciplined, long-running experimentation program that improves BTCSpiker's out-of-sample prediction of the existing 60-second binary volatility-spike target. The program must search broadly across data, deployable features, model families, hyperparameters, calibration, and ensembles while making every result inspectable in MLflow.

The program is for research and alerting, not automated trading. No candidate may replace the Production model automatically.

## Current Baseline and Constraints

- Target: `vol_spike = 1` when future 60-second realized log-return volatility exceeds `0.000048`.
- Current champion: standardized logistic regression on seven 60-second features.
- Reported test PR-AUC: `0.1459`; deterministic baseline: `0.1340`.
- Full historical data is absent from the checkout. The only local sample spans about ten minutes and cannot support credible model selection.
- Prior model evidence used about 65 hours and showed severe regime shift between validation and test.
- Local compute: Apple M3 Pro, 11 cores, 18 GB RAM.
- Local free disk observed during design: about 10 GiB.
- Compute budget: up to 24 hours per major search cycle; no paid APIs or cloud compute.
- Durable storage: iCloud Drive.
- Live MLflow UI: existing local server at `http://localhost:5001`.
- Existing API p95 latency SLO: 800 ms.

## Approaches Considered

### 1. Brute-force the current sample

Run many feature and model combinations immediately against the checked-in handoff sample.

**Advantage:** fastest path to a large MLflow experiment table.

**Rejected because:** the sample contains only minutes of correlated tick data. A broad search would select noise and make the apparent winner less trustworthy as the number of trials increased.

### 2. Data-first gated model tournament — selected

Build a reproducible dataset and immutable temporal evaluation protocol first. Run increasingly expensive feature and model stages only when earlier stages demonstrate real lift. Log every trial, including failures, to MLflow. Qualify deployable winners through offline/online parity, an untouched holdout, latency testing, and Staging registration.

**Advantages:** maximizes credible improvement, controls leakage, uses compute efficiently, preserves reproducibility, and produces candidates that can actually be served.

**Trade-off:** meaningful model improvement may wait on acquiring enough diverse data.

### 3. Deep-learning-first sequence modeling

Begin with GRU, LSTM, TCN, or Transformer models over raw tick sequences.

**Advantage:** potentially learns temporal interactions that fixed feature vectors miss.

**Not selected as the default:** sequence models are data-hungry, slower to tune, harder to calibrate, and harder to serve. They remain a gated late-stage experiment after strong tabular baselines plateau and sufficient data exists.

## Architecture

```mermaid
flowchart LR
    S["Free historical sources and Coinbase live feed"] --> R["Immutable raw Parquet partitions in iCloud"]
    R --> M["Versioned dataset manifest and quality gate"]
    M --> F["Shared causal feature engine"]
    F --> C["Curated versioned feature dataset"]
    C --> V["Purged walk-forward validation"]
    V --> T["Progressive model tournament"]
    T --> L["MLflow runs, metrics, artifacts, and lineage"]
    L --> Q{"Qualification gates"}
    Q -->|pass| G["MLflow Staging candidate"]
    Q -->|fail| N["Documented non-winner"]
    G --> P["Replay, API parity, and latency verification"]
    P --> H["Human decision for Production"]
```

The detailed charter and implementation plan live in the repository. The `/goal` objective stays concise and points Codex to the plan so it remains below the 4,000-character product limit.

## Storage Design

Use the local iCloud Drive mount through a configurable environment variable:

```text
BTCSPIKER_CLOUD_ROOT="$HOME/Library/Mobile Documents/com~apple~CloudDocs/BTCSpiker"
```

Expected layout:

```text
BTCSpiker/
  raw/source=<source>/product=BTC-USD/date=YYYY-MM-DD/part-*.parquet
  curated/dataset_id=<sha256>/part-*.parquet
  manifests/<dataset_id>.json
  mlflow-exports/run_id=<run_id>/
  models/run_id=<run_id>/
  reports/search_id=<search_id>/
```

Rules:

- Raw and curated partitions are immutable after their manifest is published.
- Writes use a local temporary file, checksum verification, and atomic rename into the iCloud directory.
- Active training reads from a bounded local cache under `.cache/btcspiker/`.
- The cache has a configurable size ceiling, default 4 GiB, and evicts least-recently-used partitions.
- The implementation checks both local free space and iCloud availability before acquisition or materialization.
- The active MLflow SQLite database and artifact volume remain local. Completed run artifacts and experiment summaries are exported to iCloud.
- iCloud is not treated as a database, lock manager, or multi-writer filesystem.

## Data Acquisition and Quality

The goal may use free public historical data, the existing Coinbase live feed, and free public multi-exchange or derivatives signals. Paid sources are excluded.

Acquisition priority:

1. Backfill Coinbase BTC-USD trades through the public Exchange endpoint `GET https://api.exchange.coinbase.com/products/BTC-USD/trades?limit=1000`, following the `CB-AFTER` cursor into older pages. This creates target-aligned trade history but not historical quote features.
2. Continue append-only Coinbase collection through the unauthenticated Advanced Trade `ticker`, `market_trades`, and `level2` WebSocket channels. This is the canonical source for a fully deployable quote-and-trade corpus.
3. Download checksum-verified Binance public-data archives for BTCUSDT spot/futures trades and selected derivatives context. Treat them only as external as-of features; never substitute a Binance-derived target for Coinbase BTC-USD.
4. Add another free source only after recording its official documentation, terms, timestamp semantics, and deterministic retrieval procedure in the dataset manifest.

The initial credible corpus target is at least 30 calendar days covering multiple volatility and liquidity regimes. If compatible historical tick data cannot be obtained, the goal must build and start the live collector, then clearly mark model results as provisional until the corpus reaches the minimum.

Each dataset version has a manifest containing:

- dataset ID derived from canonical manifest content;
- source names and retrieval parameters;
- partition paths and SHA-256 hashes;
- event-time range, row count, schema version, and cadence;
- duplicate, gap, null, ordering, and outlier statistics;
- label prevalence overall and by day;
- code commit and feature-engine version;
- creation timestamp and parent dataset ID, when applicable.

Quality gates fail closed on schema mismatch, timestamp regression, overlapping duplicate keys, unexplained gaps, incomplete label horizons, non-finite model inputs, or a source/feature join that uses information unavailable at prediction time.

## Target and Sampling Contract

- Keep the target horizon fixed at 60 seconds.
- Keep the label threshold fixed at `0.000048` for direct comparison with the current model.
- Version the current operational target as trade-price realized volatility because `features/featurizer.py` passes last-trade `price` into `compute_future_vol`. Correct the existing midprice wording in the feature specification; do not silently change the target calculation.
- Do not retune the label definition during the model tournament.
- Compute every feature strictly from data at or before the prediction timestamp.
- Keep future volatility and `vol_spike` out of model inputs.
- Evaluate a controlled cadence grid such as 1-second, 5-second, and event-driven snapshots because adjacent ticks are highly correlated.
- Treat cadence as a logged pipeline parameter, not an untracked preprocessing choice.

## Feature System

Create one shared, causal feature engine used by historical materialization, replay, and streaming. This removes the current duplicated feature logic across the featurizer and replay code.

Feature families are added in stages:

1. **Current contract:** the seven served features and the deterministic volatility rule.
2. **Multi-window state:** returns, realized volatility, mean return, price range, spread statistics, intensity, and tick counts over 5, 15, 30, 60, 120, and 300 seconds.
3. **Dynamics:** EWMA volatility, volatility-of-volatility, momentum, acceleration, range/volatility ratios, spread change, intensity change, inter-arrival statistics, and lagged deltas.
4. **Microstructure:** trade direction, signed volume, bid/ask imbalance, depth, and liquidity shocks when the source provides the required fields.
5. **Regime context:** time-of-day, day-of-week, rolling volatility regime, and volume/liquidity regime.
6. **External as-of features:** cross-venue basis, funding, open interest, and other public signals joined with strict backward-looking as-of semantics.

Every feature set receives a stable ID, explicit column order, source requirements, maximum lookback, schema version, and a `deployable` flag. Research-only features may be evaluated but cannot qualify for Staging until the streaming path computes them with tested numerical parity.

## Temporal Validation

The validation contract is frozen before model search:

1. Sort by event time and partition by calendar time, never by random row shuffle.
2. Reserve the final 20% of time as an untouched holdout. Do not expose its labels or metrics during feature selection, hyperparameter tuning, calibration selection, or ensemble construction.
3. Use five expanding-window walk-forward folds on the remaining development period.
4. Purge and embargo each fold boundary by `max_feature_lookback + 60 seconds`; with the initial 300-second lookback ceiling this is 360 seconds.
5. Fit scalers, imputers, encoders, calibrators, feature selectors, and sampling logic inside each training fold only.
6. Select alert thresholds from out-of-fold development predictions, never from the final holdout.
7. Compare each candidate with the current logistic model using a paired 95% block-bootstrap confidence interval with 30-minute blocks, 2,000 resamples, and seed 42.

Primary model-selection metric:

- walk-forward average precision / PR-AUC, reported per fold and in aggregate.

Required secondary metrics:

- positive prevalence and PR-AUC lift over prevalence;
- ROC-AUC;
- log loss, Brier score, and expected calibration error;
- F1, precision, and recall at the selected threshold;
- event-level precision and recall with a 60-second alert cooldown;
- alerts per hour;
- p50 and p95 inference latency;
- metrics by day and market regime.

## Progressive Model Tournament

Every stage uses fixed seeds, bounded thread counts, early stopping where supported, and nested MLflow runs.

### Stage 0: Reproduce baselines

- prevalence/no-skill predictor;
- deterministic `vol_60s` rule;
- exact current logistic-regression pipeline;
- a duplicate-feature sanity test that should not create artificial lift.

No later search starts until the shipped artifact's predictions are reproduced on the handoff sample and the same logistic configuration is re-trained and evaluated as the baseline on the new versioned dataset. The new-data metrics are expected to differ from the historical `0.1459` score.

### Stage 1: Linear and simple nonlinear models

- logistic regression with L1, L2, and elastic-net regularization;
- SGD logistic classifier;
- linear discriminant variants when numerically valid;
- calibrated shallow decision tree and histogram gradient boosting baseline.

### Stage 2: Tree ensembles and boosting

- Random Forest and Extra Trees;
- scikit-learn HistGradientBoosting;
- LightGBM, XGBoost, and CatBoost;
- class weighting and focal-style objectives only when supported and logged.

Optuna searches use seeded samplers, median/pruning rules, time and trial budgets, and the same temporal folds. Search spaces are explicit artifacts.

### Stage 3: Feature-family ablations

- add one feature family at a time to the best robust model families;
- remove highly redundant or unstable features;
- compare cadence variants;
- measure performance by regime and source availability;
- retain only improvements that repeat across temporal folds.

### Stage 4: Calibration and ensembles

- Platt and isotonic calibration fit only on out-of-fold predictions;
- weighted soft voting over diverse qualified models;
- stacking only from out-of-fold base-model predictions;
- no ensemble may use the final holdout to select members or weights.

### Stage 5: Gated temporal neural models

TCN, GRU, or similarly modest sequence models may run only when:

- at least 30 days of quality-controlled data exists;
- boosted-tree progress has plateaued;
- the remaining compute budget is sufficient;
- the serving path can meet the latency SLO.

Large Transformers and unbounded architecture search are out of scope for the local 24-hour cycle.

## MLflow System of Record

Every attempted run, including pruned and failed runs, appears in MLflow.

Required tags:

- `dataset_id`, `feature_set_id`, `target_version`, `validation_version`;
- `git_sha`, `search_id`, `model_family`, `deployable`;
- `run_status`, `failure_class`, and `candidate_stage`.

Required parameters:

- ordered feature columns and window definitions;
- cadence and source set;
- fold boundaries and embargo duration;
- random seeds, thread limits, model parameters, calibration method, and threshold policy.

Required metrics:

- all fold, aggregate, event, calibration, regime, and latency metrics defined above;
- fit time, inference time, model size, and peak memory where measurable.

Required artifacts:

- exact experiment configuration;
- dataset manifest and feature manifest;
- fold-boundary table;
- out-of-fold predictions in Parquet;
- PR, calibration, and regime plots;
- feature importance or SHAP summary when supported;
- dependency lock/freeze and code SHA;
- failure traceback for failed runs;
- candidate model artifact and model card for qualified runs.

The exporter copies completed-run summaries and immutable artifacts to iCloud and records export checksums. An export failure never deletes the local MLflow run.

## Qualification and Promotion

A candidate may be registered as MLflow `Staging` only when all conditions pass:

1. Data and feature quality gates pass.
2. The candidate is marked deployable and batch/stream feature parity passes within defined numerical tolerances.
3. It beats the reproduced logistic baseline on PR-AUC in at least four of five walk-forward folds.
4. The lower bound of a paired temporal block-bootstrap confidence interval for aggregate PR-AUC improvement is above zero.
5. Brier score is no more than 5% worse than the logistic baseline, event-level F1 is at least as high as the baseline, and the threshold is selected by maximum event-level F1 from development out-of-fold predictions. Alerts per hour is reported for review rather than hidden in the aggregate score.
6. It is then evaluated exactly once on the untouched final holdout and beats the baseline there.
7. Replay integration passes and `/predict` p95 remains at or below 800 ms.
8. A model card, feature contract, dataset lineage, and rollback instructions exist.

Registration to Staging is automatic after these gates. Promotion to Production is always a separate human-approved action.

## Serving Compatibility

The existing seven-field request stays valid. The API and bridge evolve to support versioned additional features without discarding them:

- the feature message includes `feature_schema_version` and the complete deployable feature payload;
- the bridge forwards the versioned payload rather than selecting a hardcoded seven-column list;
- the API loads required columns and schema metadata from the registered model run;
- the API validates missing, unknown, non-finite, and version-mismatched features before scoring;
- the legacy logistic artifact remains a tested rollback target.

Cross-source or sequence features cannot enter a Staging candidate until the runtime pipeline supplies them reliably. Offline-only results remain visible in MLflow with `deployable=false`.

## Goal Orchestration and Checkpoints

The long-running goal follows these checkpoints:

1. Verify storage, disk headroom, dependencies, and baseline reproducibility.
2. Acquire or start collecting data and publish a dataset-quality report.
3. Freeze the dataset, target, feature, and validation manifests.
4. Run progressive search stages, logging everything to MLflow.
5. Stop and summarize after each stage before spending the next budget tier.
6. Qualify at most the strongest evidence-backed deployable candidate.
7. Register as Staging, export artifacts to iCloud, run replay/API verification, and write the final comparison report.

The goal pauses for a user decision if it needs paid data, paid compute, destructive cleanup, Production promotion, or a target-definition change.

If fewer than 30 days of target-aligned quote-and-trade data is available after backfill, the goal completes the framework and provisional trade-only experiments, leaves the Coinbase collector running, records the exact resume condition in MLflow and the final report, and pauses. Resume the same task with `/goal resume` after the manifest reaches the data threshold.

## Stop Conditions

The goal is complete when either:

- a candidate passes every Staging qualification gate and the final report is delivered; or
- the 24-hour major-search budget is exhausted after all cheaper eligible stages, no further justified experiment remains, and the final report identifies the best non-qualifying candidate and the next data or feature dependency.

The goal must not claim improvement when only development metrics improved, when the final holdout regressed, when the candidate uses unavailable runtime features, or when the data corpus is below the credibility threshold.

## Verification

- Unit tests cover manifests, causal joins, label boundaries, split purging, metrics, MLflow logging, cache eviction, and feature parity.
- On the handoff sample, scores from the shipped pickle and the MLflow-loaded copy match within absolute tolerance `1e-9`; on the new corpus, the exact current logistic configuration is re-trained and becomes the comparison baseline.
- Synthetic leakage tests deliberately inject future information and confirm that the pipeline rejects or exposes it.
- Integration tests run a small end-to-end tournament against the handoff sample without treating its score as evidence.
- Replay tests verify feature schema propagation through Kafka, the bridge, and the API.
- Load tests confirm the 800 ms p95 SLO.
- MLflow contains successful, pruned, and intentionally failed example runs with full lineage.
- iCloud exports are checksum-verified and independently readable.
