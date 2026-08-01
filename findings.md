# Findings & Decisions

## Requirements
- Prepare a plan that uses Codex `/goal` for a sustained BTC prediction-quality improvement effort.
- Try diverse feature sets, model families, hyperparameters, and potentially ensembles.
- Log experiments so the user can inspect and compare them in MLflow.
- Optimize prediction quality as much as practical without invalid evaluation or hidden leakage.
- Keep the existing 60-second binary volatility-spike target as the sole primary target.
- Run the complete experimentation framework and `/goal` tournament now using the user's already-collected data.
- Do not generate synthetic market observations.
- Keep all future market-data gathering, provider selection, cloud storage, and accumulation waiting in a separate deferred plan.

## Research Findings
- The active checkout is `/Users/ricopichardo/Claude/BTCSpiker` on branch `codex/v1` at commit `d565e33`; `main` is at `c74ddba`.
- Before this planning session the worktree was clean; the only untracked files now are the three planning artifacts created for this task.
- The repository already contains a real-time ingestion/feature/API stack, a logistic-regression model artifact, EDA notebooks and charts, monitoring assets, and scripts for MLflow model logging and drift reports.
- A fresh Codex manual was fetched to `/var/folders/8p/5vq88y357tn75vl43jsylg200000gn/T/openai-docs-cache/codex-manual.md`; exact `/goal` term coverage still needs to be checked.
- The manual documents `/goal` as a persistent objective attached to the active task. It can be viewed, edited, paused, resumed, or cleared; goal text is limited to 4,000 characters, so a longer plan should live in a repo file referenced by the goal.
- The manual recommends expressing a goal as a verifiable outcome with explicit constraints and verification criteria. Starting a goal does not expand filesystem, network, or approval permissions.
- The current prediction target is binary `vol_spike` over the next 60 seconds, defined from future realized log-return volatility above a fixed `0.000048` threshold (historically selected as P85).
- The shipped model is `StandardScaler -> LogisticRegression(C=0.1, class_weight='balanced')` on seven 60-second features. Its reported time-ordered test PR-AUC is `0.1459` versus `0.1340` for the deterministic baseline, with test F1 `0.1359`.
- The underlying dataset covers only about 65 hours and shows a major prevalence/regime shift: positive rate falls from 15.4% in train and 14.1% in validation to 7.0% in test. The reported validation PR-AUC (`0.3580`) is much higher than test (`0.1459`).
- Existing MLflow code logs a pre-trained pickle plus only a few parameters and two metrics, then promotes it directly to Production. It is registration/bootstrap code, not yet a general training and experiment-tracking framework.
- Existing feature work tested four small logistic-regression feature variants; `spread_mean_60s` helped validation PR-AUC, while `price_range_60s` did not. No tracked broad model-family search is present in the checkout.
- The full `data/raw` and `data/processed` datasets are absent from this checkout and intentionally gitignored. `reports/` is empty except for `.gitkeep`.
- The only local training-like data is `handoff/data_sample/features_slice.csv` with 3,212 labelled rows spanning about ten minutes, plus a 3,578-line raw slice over the same window (originally 6,396/7,156 before a duplicate-tick defect in the capture was fixed — see `handoff/data_sample/manifest.json`). This sample is insufficient for trustworthy broad feature/model selection.
- The repository also contains 122,771 prior test predictions spanning `2026-04-07T00:12:16Z` through `2026-04-07T15:54:58Z`, but those rows contain predictions and labels rather than the full feature matrix.
- Project documentation says the original collected `data/processed/features.parquet` contained about 784,000 rows and 48 MB, but that full table is not present in this checkout. The experimentation plan therefore resolves `BTCSPIKER_EXISTING_DATA` first, then `data/processed/features.parquet`, then the checked-in collected sample as a fallback.
- The current dependency set supports scikit-learn and MLflow but does not include a hyperparameter optimizer or boosted-tree libraries such as Optuna, XGBoost, LightGBM, or CatBoost.
- The user permits storing additional datasets and artifacts with a cloud provider and asked specifically about iCloud.
- iCloud Drive is mounted and syncing locally at `/Users/ricopichardo/Library/Mobile Documents/com~apple~CloudDocs`; the iCloud Drive sync process is running.
- The Mac currently reports only about 10 GiB of free local disk space. Even if the iCloud account has more quota, active experiment data must be staged carefully to avoid exhausting local storage.
- Local compute is an Apple M3 Pro with 11 CPU cores and 18 GB RAM. This is well suited to parallel CPU-based linear, tree, boosting, and modest neural experiments, but the plan should bound memory and concurrency.
- The production feature contract is duplicated across `features/featurizer.py`, `scripts/replay.py`, `scripts/feature_to_predict_bridge.py`, `api/main.py`, and integration tests. Broad feature experimentation will require one versioned feature contract and parity tests or offline wins may be impossible to serve correctly.
- `api/main.py` hardcodes seven request fields even though the model's required columns are fetched from MLflow. New feature sets therefore need backward-compatible acceptance of additional fields plus runtime validation against the registered model contract.
- `scripts/feature_to_predict_bridge.py` currently strips feature messages down to the same seven fields; it must forward a versioned deployable feature set rather than silently discard new features.
- The current MLflow bootstrap script evaluates on the small handoff sample and automatically transitions the registered model to Production. The experiment system must separate run logging, candidate registration, Staging qualification, and explicit Production promotion.
- Docker Compose correctly keeps the live MLflow SQLite database and artifacts together in a local Docker volume. The iCloud design should export snapshots and immutable artifacts from this volume instead of relocating the active database.
- The deployed API has a documented p95 latency SLO of 800 ms and a reference p95 of 106.4 ms. Candidate promotion should preserve the 800 ms SLO and report latency deltas against the current model.
- Official Coinbase documentation confirms that the Exchange market-data API is public, its BTC-USD trades endpoint returns up to 1,000 records per page with `CB-AFTER` cursor pagination, and its public WebSocket feeds provide real-time trades and order-book updates.
- Official Coinbase Advanced Trade WebSocket documentation lists unauthenticated `ticker`, `market_trades`, and `level2` channels. These can supply the full target-aligned live corpus, including bid/ask quantities missing from the current capture.
- Binance's official public-data archive provides checksum-protected daily and monthly BTCUSDT spot/futures trades, aggregate trades, klines, funding, and related files. These are suitable as free external as-of features, but not as a silent replacement for the Coinbase BTC-USD target.
- The current implementation computes `future_vol_60s` from the tick's last-trade `price`, while `handoff/docs/feature_spec.md` describes midprice volatility. The plan must version and preserve the code's actual trade-price target for baseline comparability, then correct the documentation; changing the target requires a separate user decision.
- `scripts/ws_ingest.py` already consumes the public Coinbase ticker and mirrors data atomically, but it creates a new NDJSON segment every 100 lines. Directing that small-file pattern into iCloud would create excessive sync overhead; the plan will stage locally and compact into hourly Parquet partitions before cloud publication.
- The existing WebSocket payload omits bid/ask quantities even though Coinbase's ticker channel provides them. Capturing these fields enables deployable order-book-imbalance features without requiring an authenticated feed.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Establish immutable baselines and temporal validation before broad search | Financial time series are particularly vulnerable to look-ahead leakage and regime overfitting. |
| Put the detailed experiment charter in a repo document and keep `/goal` concise | Codex goal objectives are capped at 4,000 characters and can point to a longer file. |
| Do not expand the primary objective to 5- or 15-minute horizons | The user explicitly selected the existing 60-second horizon. |
| Treat additional historical data as a prerequisite to credible promotion, not to running the goal | Searching many models against a nine-minute sample may select noise, so insufficient coverage forces provisional reporting but does not block framework execution or the tournament. |
| Superseded: use iCloud for immutable dataset and experiment exports | The revised active goal keeps artifacts local and defers all remote storage choices to the separate data-gathering plan. |
| Separate research-only features from deployable features | External or temporal features may improve offline scores but cannot qualify for Staging until the streaming path can compute them with exact parity. |
| Use Coinbase data for the target and Binance archives only for external as-of context | This preserves the BTC-USD prediction contract and avoids training a Coinbase deployment on a substituted BTCUSDT target. |
| Remove acquisition from the active experimentation goal | The user wants the full experiment task completed now and will decide the future data plan independently. |
| Use only resolved existing data in the active goal | The resolver must fail rather than generate, download, stream, or silently combine market observations. |
| Never pause the active goal for data accumulation | Insufficient coverage produces a reason-coded provisional result; a future dataset starts a new immutable search. |

## Issues Encountered
| Issue | Resolution |
|-------|------------|

## Resources
- Project root: `/Users/ricopichardo/Claude/BTCSpiker`
- Fresh Codex manual: `/var/folders/8p/5vq88y357tn75vl43jsylg200000gn/T/openai-docs-cache/codex-manual.md`
- Current feature contract: `handoff/docs/feature_spec.md`
- Current model evidence: `handoff/docs/model_card_v1.md`, `handoff/models/artifacts/metadata.json`, `docs/results.md`
- Available local sample: `handoff/data_sample/features_slice.csv`, `handoff/data_sample/raw_slice.ndjson`
- Coinbase public trades: https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-trades
- Coinbase Exchange pagination: https://docs.cdp.coinbase.com/exchange/rest-api/pagination
- Coinbase Advanced Trade WebSocket channels: https://docs.cdp.coinbase.com/coinbase-app/advanced-trade-apis/websocket/websocket-channels
- Binance public-data archive: https://github.com/binance/binance-public-data

## Visual/Browser Findings
- None.
