# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

BTCSpiker is a real-time Bitcoin volatility-spike detection service: it streams Coinbase-style ticks
through Kafka, computes rolling-window features online, serves predictions via FastAPI, and projects
prediction/outcome events into a disposable SQLite read model for a React UI. It runs end-to-end in
replay mode by default (no external dependencies needed), with Prometheus/Grafana monitoring and an
MLflow-backed model registry supporting instant rollback.

A separate, offline half of the repo (`btcspiker_ml/`, `experiment.yaml`, `scripts/run_experiments.py`,
etc.) runs a resumable MLflow tournament over historical data to search for better models. It is
decoupled from the live service: it can only ever promote a model to **Staging**, never to
**Production** — see "Model tournament & promotion" below.

## Canonical startup

```bash
cp .env.example .env
docker compose up -d --build
# wait ~30s for Kafka + MLflow init
curl http://localhost:8000/health
curl http://localhost:8090/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @handoff/data_sample/sample.json
```

Web UI at http://localhost:3001. Use the root README + root `docker-compose.yaml` as the only startup
guide — `handoff/` is packaged artifacts/docs, not a runtime entrypoint.

To switch to live Coinbase ingestion (mutually exclusive with the replay ingestor — both publish to the
same `ticks.raw` topic):

```bash
docker compose stop ingestor
docker compose --profile live up -d ws-ingestor
```

Rollback the API to the deterministic baseline rule (no model needed):

```bash
MODEL_VARIANT=baseline docker compose up -d api
curl -s http://localhost:8000/version | jq .source   # -> "pickle"
```

## Commands

**Python lint/format** (scope matches CI exactly — `api/`, `materializer/`, `scripts/feature_to_predict_bridge.py`, `tests/`):
```bash
black --check api/ materializer/ scripts/feature_to_predict_bridge.py tests/
ruff check api/ materializer/ scripts/feature_to_predict_bridge.py tests/
```
`btcspiker_ml/`, `btcspiker_data/`, `features/`, and most of `scripts/` are **not** lint-gated in CI.

**Python tests** (pytest, `pythonpath = ["."]` set in `pyproject.toml`, so run from repo root):
```bash
pip install -r requirements-dev.txt -r docker/requirements.api.txt -r materializer/requirements.txt -r requirements-ml.txt
pytest tests --ignore=tests/test_replay_integration.py   # unit tests (no Docker needed)
pytest tests/test_materializer.py -k test_name            # single test
pytest tests/test_replay_integration.py -v --tb=short --retries 2   # needs the full stack up (see CI job below)
```
`tests/ml/` and `tests/data/` cover `btcspiker_ml/` and `btcspiker_data/` respectively and need
`requirements-ml.txt` installed (pulls pandas/pyarrow/mlflow/sklearn/optuna/psutil).

**UI** (in `ui/`):
```bash
npm run dev       # vite dev server
npm run build
npm test           # node --test src/*.test.js
```

**CI** (`.github/workflows/ci.yaml`, on push/PR to `main`): `lint` → `unit` → `integration-replay` (must
pass before merge, no `continue-on-error`). `integration-replay` brings up the full Docker Compose stack
in the runner, runs `tests/test_replay_integration.py`, and asserts the UI's `/api/predictions/recent`
proxy returns real data. Deeper monitoring validation (Grafana panels, Prometheus scrape, Kafka-exporter
lag) is verified locally only — see `docs/runbook.md`.

## Runtime architecture

```
Coinbase/replay -> ticks.raw -> featurizer -> ticks.features (immediate) -> predict-bridge
    -> FastAPI /predict -> ticks.predictions -> materializer -> SQLite read model
                                                       ^              v
                          ticks.outcomes (60s delayed) +      nginx UI on :3001
```

Kafka is the source of truth for prediction events; the SQLite file the materializer owns (on the
`predictions-data` volume) is a **disposable projection** that can be rebuilt by replaying
`ticks.predictions` + `ticks.outcomes`. The UI is same-origin nginx: `/api/predictions/*` → materializer,
other `/api/*` → FastAPI.

**The online/offline split that matters most:** the featurizer publishes each feature row to
`ticks.features` the instant its tick arrives (a genuine online forecast). The true 60-second-forward
label for that same row is computed separately and published ~60s later to `ticks.outcomes`, keyed by a
stable `feature_id`. The materializer joins predictions to outcomes on that id and only grades rows where
the prediction's own scoring timestamp precedes the outcome write — i.e. it proves the model called it
before the answer existed. A `median_lead_seconds` near 0 in `/predictions/performance` means predictions
are arriving already-answered — that's a pipeline bug, not a model problem.

**API prediction boundary is post-featurization**: `/predict` accepts the seven engineered features (see
`handoff/docs/feature_spec.md`), not raw tick fields.

**Model loading in `api/main.py`** (`MODEL_VARIANT=ml|baseline`):
1. `ml` (default): load `models:/<MODEL_NAME>/<MODEL_STAGE>` from the MLflow registry, and require the
   registered run to carry a complete runtime contract (`feature_cols`, `tau`, and — for any model other
   than the legacy `btc-volatility-lr`/`Production` selection — `feature_set_id` and
   `feature_schema_version`). A non-legacy registration with an incomplete contract is a hard failure, not
   a fallback.
2. Only the **legacy** selection (`btc-volatility-lr`/`Production`) falls back to the local pickle at
   `MODEL_PATH` when MLflow is unavailable.
3. `baseline`: skips model loading entirely — a deterministic z-style rule on `vol_60s` vs
   `BASELINE_VOL_THRESHOLD`. This is the rollback path, so it must work even when the ML artifact is
   missing.

### Services (`docker-compose.yaml`)

| Service | Role |
|---|---|
| `kafka` / `kafka-init` | Broker + topic bootstrap (KRaft mode) |
| `mlflow` / `mlflow-init` | Tracking server + one-shot pipeline registration on fresh clone |
| `candidate-publish` | Bridges the tournament's file-store MLflow into the server registry (Staging only) |
| `ingestor` / `ws-ingestor` | Replay (default) vs. live Coinbase WS ingestion — mutually exclusive, same `ticks.raw` topic |
| `featurizer` | Rolling-window features + delayed labels (see above) |
| `predict-bridge` | Consumes `ticks.features`, calls `/predict`, republishes to `ticks.predictions` |
| `api` | FastAPI prediction service |
| `materializer` | Kafka → disposable SQLite read model + Performance-tab grading |
| `ui` | React app + nginx same-origin proxy |
| `kafka-exporter` / `prometheus` / `grafana` | Monitoring stack |

## Repository structure

```
api/             FastAPI prediction service (loads lr_pipeline.pkl or MLflow registry model)
features/        Featurizer Kafka consumer + rolling-window FeatureEngine wrapper
materializer/    ticks.predictions + ticks.outcomes consumer, SQLite read model, Performance grading
ui/              React web UI (Predictions/Settings/System/Performance tabs) + nginx proxy
scripts/         Ingestors, predict-bridge, replay, drift tooling, ML tournament CLIs
btcspiker_ml/    Offline experiment framework: config, features, models, splits, search, qualification
btcspiker_data/  Historical data acquisition/validation pipeline (Hugging Face-backed corpus)
docker/          Dockerfile.api + Dockerfile.worker + their pinned requirements
monitoring/      prometheus.yml + Grafana provisioning + dashboard JSON
docs/            Operational docs — results, SLOs, runbook, drift, architecture diagrams
handoff/         Packaged model artifacts, data samples, reference docs (not a runtime entrypoint)
tests/           tests/ (API/materializer/featurizer smoke+unit), tests/ml/, tests/data/
```

## Model tournament & promotion (`btcspiker_ml/`, `scripts/run_experiments.py` etc.)

This is a **separate, offline** system from the live service, gated by explicit env vars and file-store
MLflow so it never touches Production by accident:

- `scripts/run_experiments.py` runs one resumable tournament stage (`baseline`, `linear`, `trees`,
  `ablation`, `ensemble`, `neural`) against a local file-store MLflow (`experiment.yaml`'s
  `mlflow.tracking_uri`, e.g. `file:.artifacts/btcspiker/mlruns`) — it never opens the final holdout and
  never registers/promotes a model.
- `btcspiker_ml/splits.py` builds leakage-resistant **temporal** folds (expanding-window, with a final
  holdout that's opened at most once) — always use these for any new evaluation, never a random split.
- `scripts/qualify_candidate.py` reads one completed run and can only produce a **Staging** verdict
  (`btcspiker_ml/qualification.py`); Production is never touched by this path.
- `scripts/publish_candidate_to_registry.py` is the file-store → server bridge: it re-reads the
  `qualification.json` verdict already written and copies bytes into the target registry — it does not
  re-implement the gate. It runs alongside the API image (stdlib + mlflow only, no pandas/pyarrow).
- `BTCSPIKER_EXISTING_DATA` must be an absolute path to the real corpus; unset, `bind_existing_dataset.py`
  silently falls back to the tiny `handoff/data_sample/` fixture (a 10-minute schema/smoke sample, not a
  distribution proxy — `drift_report.py` refuses it as a `--reference`).
- `scripts/replay.py --out` defaults to overwriting `config.yaml`'s `data.features_file`
  (`data/processed/features.parquet`) with no confirmation. Always pass an explicit `--out` and never
  point it at a corpus you're relying on for tournament work.

## Conventions

- Feature contract (`FEATURE_COLS`, `feature_set_id`, `feature_schema_version`, `tau`) is duplicated
  across the offline pipeline, the featurizer, the predict-bridge, the API, and tests — when changing the
  feature set, all of these need to move together, and the MLflow-registered run's params are the runtime
  source of truth for the API (see model loading above).
- `scikit-learn` is pinned exactly (`1.6.1`) in `docker/requirements.api.txt` to match the tournament's
  training env, because the non-legacy model-loading path has no pickle fallback and would hard-fail on a
  pickle-protocol mismatch.
- `mlflow` is pinned to `2.12.2` everywhere it's a dependency, to match the tracking-server image in
  `docker-compose.yaml`.
- Timestamps that cross service boundaries (`feature_ts` in the materializer) can arrive as either
  trailing-`Z` or explicit-offset ISO-8601, which do not sort identically as raw strings — always
  normalize before comparing/sorting, following the materializer's `_normalize_ts` pattern.
- Delivery on Kafka topics is at-least-once; consumers dedupe with `INSERT OR IGNORE` keyed on a stable id
  (`event_id` for predictions, `feature_id` for outcomes) rather than assuming exactly-once delivery.
