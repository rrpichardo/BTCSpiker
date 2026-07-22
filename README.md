
# BTCSpiker

**Real-time Bitcoin volatility-spike detection — streaming ML on live market data.**

BTCSpiker started as a school project and has since grown into a polished, production-ready service.

## Quick Start

```bash
cp .env.example .env
docker compose up -d --build
curl http://localhost:8000/health
curl http://localhost:8090/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @handoff/data_sample/sample.json
```

Open the web UI at http://localhost:3001.

BTCSpiker is a real-time service that streams Coinbase style ticks into Kafka, generates rolling window features, and serves predictions via a FastAPI API. Prediction events are published to Kafka and projected into a SQLite read model for the web UI. The system runs end to end in replay mode with Prometheus and Grafana monitoring, and supports rollback using MODEL_VARIANT.

## Canonical startup

### Environment

```bash
cp .env.example .env   # required: mounted read-only by the API settings view
docker compose up -d
# Wait ~30s for Kafka and MLflow init, then:
curl http://localhost:8000/health
curl http://localhost:8090/health
```

Use this root README together with the root `docker-compose.yaml` as the only startup guide for the project.

No manual model registration is required on a fresh clone — a one-shot
`mlflow-init` container logs the trained pipeline to MLflow. The prediction
quality tournament never promotes a candidate automatically: qualifying work
may register **Staging** only, while Production remains unchanged.
See [First-Time Setup in the runbook](docs/runbook.md#first-time-setup)
for the verification commands and notes on the shared `mlflow-data`
volume.

## Quick Test

The prediction API boundary is **post-featurization**: `/predict` accepts the seven engineered 60-second features produced by the featurizer, not raw Coinbase tick messages.

> **API schema note:** The `/predict` endpoint accepts the 7 engineered features the model was trained on (see [`handoff/docs/feature_spec.md`](handoff/docs/feature_spec.md)). This is a post-featurization boundary: the schema matches the trained model's features rather than raw tick fields.

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d @handoff/data_sample/sample.json
```

The `sample.json` payload contains one feature row (row 1 of `handoff/data_sample/features_slice.csv`):

```json
{
  "rows": [{
    "log_return": 0.0,
    "spread_bps": 0.0014345986913609724,
    "vol_60s": 0.0,
    "mean_return_60s": 0.0,
    "trade_intensity_60s": 0.016666666666666666,
    "n_ticks_60s": 1,
    "spread_mean_60s": 0.010000000009313226,
    "ts": "2026-04-06T15:02:34.590029Z"
  }]
}
```

Expected response (`ts` is UTC wall-clock at inference time):

```json
{
  "scores": [0.10401],
  "model_variant": "ml",
  "version": "v1.0",
  "ts": "YYYY-MM-DDTHH:MM:SS.ffffffZ"
}
```

`/version` always returns the same metadata shape, with `stage` and `run_id` set to `null` when the API falls back to the local pickle artifact:

```json
{
  "model": "btc-volatility-lr",
  "version": "v1.0",
  "stage": "Production",
  "source": "mlflow",
  "run_id": "RUN_ID_OR_NULL",
  "sha": "GIT_SHA"
}
```

The system runs fully in replay mode by default, ensuring reproducibility without external dependencies.

## Data Ingestion Modes

The default `docker compose up -d` runs the replay ingestor, which loops a 10 minute Coinbase capture through Kafka at original timestamps. This ensures reproducibility without external dependencies.

To switch to live ingestion from Coinbase public WebSocket:

```bash
docker compose stop ingestor
docker compose --profile live up -d ws-ingestor
```

Both ingestion modes publish to the same `ticks.raw` Kafka topic, so only one should run at a time.

## Runtime Architecture

```text
Coinbase/replay → ticks.raw → featurizer → ticks.features (immediate) → predict-bridge
    → FastAPI /predict → ticks.predictions → materializer → SQLite read model
                                                       ↗            ↘ nginx UI on :3001
                          ticks.outcomes (60s delayed) ┘
```

Kafka is the source of truth for prediction events. The materializer owns a
disposable SQLite read model on the `predictions-data` volume and can rebuild it
by replaying `ticks.predictions` and `ticks.outcomes`. The UI uses same-origin
nginx routes: `/api/predictions/*` reaches the materializer and other `/api/*`
requests reach the FastAPI service.

The featurizer publishes each feature row to `ticks.features` **the instant its
tick arrives** — predictions are genuine online forecasts, not retrodictions.
The exact 60-second-forward label for that same row is computed separately and
published ~60 seconds later to `ticks.outcomes`, keyed by a stable `feature_id`.
The materializer joins predictions to outcomes on that ID and grades only rows
where the prediction's own scoring timestamp precedes the outcome being written
— proof the model called it before the answer existed. See the **Performance**
tab and `docs/runbook.md` for how this is exposed.

## Endpoints and Dashboards

| Service | URL | Notes |
|---|---|---|
| API | http://localhost:8000 | `/health`, `/predict`, `/version`, `/metrics` |
| Web UI | http://localhost:3001 | Predictions, Performance, read-only settings, and system status |
| Materializer | http://localhost:8090 | `/health`, `/predictions/recent`, `/predictions/performance` |
| MLflow | http://localhost:5001 | Training-run tracking |
| Prometheus | http://localhost:9090 | Scrapes API + kafka-exporter |
| Grafana | http://localhost:3000 | Anonymous viewer; dashboard "BTC Volatility Detector — API" |

## Rollback

To switch from the LR model to the deterministic baseline rule:

```bash
MODEL_VARIANT=baseline docker compose up -d api
curl -s http://localhost:8000/version | jq .source   # → "pickle"
```

Roll forward with `MODEL_VARIANT=ml docker compose up -d api`. The Grafana **Active variant** panel reflects the change within ~10 s.

## Repository Structure

```
api/             FastAPI prediction service (loads lr_pipeline.pkl)
features/        Featurizer Kafka consumer + rolling-window functions
materializer/    ticks.predictions + ticks.outcomes consumer, disposable SQLite
                 read model, and the Performance-tab grading endpoint
ui/              React web UI + nginx same-origin proxy
scripts/         Ingestors, prediction bridge, replay, and drift tooling
tests/           Smoke tests (test_api.py) + load test (load_test.py)
docker/          Dockerfile.api + Dockerfile.worker + requirements files
monitoring/      prometheus.yml + Grafana provisioning + dashboard JSON
docs/            Operational docs (see below)
handoff/         Model artifacts, data samples, and reference docs
docker-compose.yaml   Canonical compose stack
config.yaml           Featurizer config
```

## Documentation
| Doc | Purpose |
|---|---|
| [`docs/results.md`](docs/results.md) | Single-page scorecard: latency, success rate, PR-AUC vs baseline, rollback proof |
| [`docs/slo.md`](docs/slo.md) | Service Level Objectives + error budgets |
| [`docs/latency_report.md`](docs/latency_report.md) | Load-test methodology and percentiles |
| [`docs/drift_summary.md`](docs/drift_summary.md) | Evidently train-vs-test drift findings |
| [`docs/runbook.md`](docs/runbook.md) | Cold start, smoke test, rollback, common failures, recovery |
| [`docs/goals/prediction-quality-goal.md`](docs/goals/prediction-quality-goal.md) | Durable existing-data experiment charter and final-report contract |
| [`docs/Architecture Diagram.png`](docs/Architecture%20Diagram.png) | Canonical system architecture diagram |
| [`docs/architecture.svg`](docs/architecture.svg) | Earlier simplified diagram (optional reference) |
| [`docs/grafana_dashboard.png`](docs/grafana_dashboard.png) | Grafana dashboard screenshot |
| [`docs/selection_rationale.md`](docs/selection_rationale.md) | Why this architecture, API boundary, and model were chosen |
| [`handoff/docs/feature_spec.md`](handoff/docs/feature_spec.md) | Feature definitions and ablation notes |
| [`handoff/docs/model_card_v1.md`](handoff/docs/model_card_v1.md) | Model card |

## Notes on `handoff/`

The `handoff/` folder holds the packaged model artifacts, data samples, and reference docs (feature spec, model card). It is not the runtime entrypoint; use this root README and the root `docker-compose.yaml` to launch the system.

## CI

CI runs lint (Black/Ruff) plus a replay integration smoke test that brings up the full Docker Compose stack inside the GitHub Actions runner. Deeper monitoring validation (Grafana panels, Prometheus scrape correctness, Kafka-exporter lag) is verified locally rather than in CI.

Specifically, `.github/workflows/ci.yaml` runs two jobs on every push to `main` and on every pull request:

1. **`lint`** — runs `black --check` and `ruff check` across `api/` and `tests/`.
2. **`integration-replay`** — starts the Docker Compose stack, waits for the API and web UI, runs `pytest tests/test_replay_integration.py`, asserts the UI proxy returns a nonempty `/api/predictions/recent` response, and tears down. This is the smoke-level integration gate; comprehensive multi-scenario load testing is validated locally before merge.

The CI does not attempt to reproduce the full production monitoring stack validation (Grafana panels, Prometheus scrape correctness, Kafka-exporter lag) in the GitHub Actions runner — those are covered by the local smoke test documented in [docs/runbook.md](docs/runbook.md).
