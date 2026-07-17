# Runbook — BTC Volatility Spike Detector

## Monitoring

**Grafana** — http://localhost:3000 (default login: admin / admin, anonymous viewer is also enabled)

The dashboard "BTC Volatility Detector — API" has a **Replay Runtime Path** row that tracks the live replay hop into `/predict`, not just Kafka ingestion:

| Panel | What it shows | Healthy range | Action if unhealthy |
|---|---|---|---|
| **Kafka lag: ticks.raw -> featurizer** | Unprocessed raw ticks waiting on the featurizer consumer group `ticks-featurizer` | ≤ 200 messages | Check `docker compose logs featurizer` — the featurizer may have fallen behind or crashed. Restart with `docker compose restart featurizer`. |
| **Kafka lag: ticks.features -> predict** | Unprocessed feature rows waiting on the runtime bridge consumer group `predict-bridge` | ≤ 200 messages | Check `docker compose logs predict-bridge` and `docker compose logs api` — the bridge may be retrying API calls or the API may be unhealthy. Restart with `docker compose restart predict-bridge api`. |
| **Prediction freshness at API (seconds)** | Age of the Kafka feature-publication timestamp when the runtime bridge reaches `/predict` | ≤ 120 s | Check `docker compose logs ingestor`, `docker compose logs featurizer`, and `docker compose logs predict-bridge` in that order. The bridge stamps requests from Kafka publish time so this gauge tracks the real feature-to-predict hop instead of the archived market timestamp. |

For full SLO thresholds and error budgets see [docs/slo.md](slo.md).

## Drift Detection

The canonical drift analysis is `handoff/reports/train_vs_test.html`. See `docs/drift_summary.md` for the summary and per-feature results.

---

## Startup (cold)

```bash
cp .env.example .env                   # required — mounted read-only by the API settings view
docker compose up -d                   # ~30s for Kafka healthcheck to go green
docker compose ps                      # 11 long-running services should be Up; kafka-init/mlflow-init should show Exited (0)
curl http://localhost:8000/health      # → {"status":"ok"}
curl http://localhost:8090/health      # → materializer status and row/error counts
curl -fsS http://localhost:3001/ >/dev/null && echo "UI reachable"
```

### First-Time Setup

No manual steps are required on a fresh clone. A one-shot `mlflow-init`
service runs automatically during `docker compose up` — it executes
[scripts/log_model_to_mlflow.py](../scripts/log_model_to_mlflow.py) to
log the trained pipeline to MLflow and promote version 1 to stage
`Production`. The `api` service gates on `mlflow-init` completing
successfully (`service_completed_successfully`), so by the time `/health`
returns OK the registry is already populated.

Verify the registration landed:

```bash
curl -s http://localhost:5001/api/2.0/mlflow/registered-models/search | jq .
# → registered_models[0].name == "btc-volatility-lr", latest_versions[0].current_stage == "Production"

curl -s http://localhost:8000/version | jq .
# → "source": "mlflow", non-null "run_id". If source is "pickle" the registry
#   lookup failed — see the MLflow volume notes below.
```

The `mlflow-data` named volume is shared between `mlflow`, `mlflow-init`,
and `api` (all three need filesystem access to `/mlruns/artifacts` because
MLflow's file-based artifact store uploads and downloads go through the
local filesystem, not through the tracking server). If you ever wipe just
part of the volume — e.g. the sqlite `mlflow_ui.db` survives but
`/mlruns/artifacts/` is empty — the bootstrap script now load-probes the
existing Production version and falls through to re-registration, so a
plain `docker compose up -d` recovers without any manual cleanup.

Open dashboards:

- Web UI: http://localhost:3001
- API metrics: http://localhost:8000/metrics
- Materializer health: http://localhost:8090/health
- Prometheus: http://localhost:9090 (Status → Targets should show all `up`)
- Grafana: http://localhost:3000 → dashboard "BTC Volatility Detector — API"
- MLflow: http://localhost:5001

## Prediction Test

```bash
curl -s http://localhost:8000/version | jq .
# → {"model":"btc-volatility-lr","version":"v1.0","stage":"Production","source":"mlflow","run_id":"…","sha":"…"}
#   stage and run_id are null when the API falls back to the local pickle artifact.

curl -X POST http://localhost:8000/predict \
     -H 'Content-Type: application/json' \
     -d @handoff/data_sample/sample.json
# → {"scores":[…],"model_variant":"ml","version":"v1.0", …}

python tests/load_test.py              # 100 burst requests, expect p95 < 800ms
```

`/predict` is a post-featurization boundary. Send the seven engineered features
(`log_return`, `spread_bps`, `vol_60s`, `mean_return_60s`,
`trade_intensity_60s`, `n_ticks_60s`, `spread_mean_60s`) plus optional `ts`,
not raw Coinbase tick messages.

> **Schema note:** The 7-field request body is the schema the model was trained on (see [`handoff/docs/feature_spec.md`](../handoff/docs/feature_spec.md)). The schema matches the trained model's features rather than raw tick fields.

Replay mode is now truly end-to-end inside Compose: `ingestor` produces
`ticks.raw`, `featurizer` produces `ticks.features`, and `predict-bridge`
automatically POSTs each feature row into `/predict` and publishes the result to
`ticks.predictions`. The `materializer` consumes those prediction events into
the SQLite read model served to the UI. Kafka is the source of truth; SQLite is
a disposable projection rebuilt by replaying the topic. You can confirm the
path without the test harness:

```bash
docker compose logs --tail=20 predict-bridge
curl -s http://localhost:8090/health | jq .
curl -s http://localhost:3001/api/predictions/recent | jq '{count}'
curl -s "http://localhost:9090/api/v1/query?query=sum(predict_requests_total)" | jq .
```

## Switch to live ingestion

The default stack runs in **replay mode** (loops a 10-minute Coinbase capture). To stream live ticks from Coinbase's public WebSocket instead:

```bash
docker compose stop ingestor                              # stop the replay source
docker compose --profile live up -d ws-ingestor           # start the live source
docker compose logs -f ws-ingestor                        # confirm "[ticker] subscribed for BTC-USD → topic 'ticks.raw'"
```

Both ingestors publish to `ticks.raw`; run only one at a time. To revert:

```bash
docker compose stop ws-ingestor
docker compose up -d ingestor
```

`ws_ingest.py` has exponential-backoff reconnect, a circuit breaker (exits non-zero after 10 consecutive failures so Compose's `restart: on-failure` rebuilds the connection), and sequence-gap logging for feed-integrity monitoring.

## Rollback Strategy

When the ML variant misbehaves (latency burns budget, error spike, drift alert), fall back to the deterministic baseline:

```bash
# In .env (or one-shot):
MODEL_VARIANT=baseline docker compose up -d api
curl http://localhost:8000/version | jq .source     # "pickle"
```

Roll forward when ready:

```bash
MODEL_VARIANT=ml docker compose up -d api
```

The Grafana **Active variant** stat panel reflects the change within ~10 s of the next Prometheus scrape.

## MLflow model registry

### View registered model versions

Open the MLflow UI at http://localhost:5001, navigate to **Models → btc-volatility-lr**. Each registered version shows its run metrics (PR-AUC, ROC-AUC) and the current stage.

### Promote a prior version to Production

In the MLflow UI: click the version number → **Stage → Transition to → Production**. This archives the current Production version and promotes the selected one. The API will load the new Production version on its next restart.

Alternatively via CLI inside the running stack:

```bash
docker compose run --rm mlflow-init python - <<'EOF'
from mlflow.tracking import MlflowClient
import os
client = MlflowClient("http://mlflow:5000")
# List all versions for the model
for v in client.search_model_versions("name='btc-volatility-lr'"):
    print(v.version, v.current_stage, v.run_id)
# Promote version N (replace 1 with the target version number)
client.transition_model_version_stage(
    "btc-volatility-lr", version="1", stage="Production",
    archive_existing_versions=True
)
EOF
```

Then restart the API to pick up the newly promoted version:

```bash
docker compose restart api
curl http://localhost:8000/version | jq '{source,stage,run_id}'
```

### Force pickle fallback (bypass MLflow)

Set `MLFLOW_TRACKING_URI` to an unreachable address and restart the API.
The startup code will catch the connection error, log a warning, and fall back
to `models/artifacts/lr_pipeline.pkl`:

```bash
MLFLOW_TRACKING_URI=http://invalid:9999 docker compose up -d api
docker compose logs api | grep "falling back to pickle"
curl http://localhost:8000/version | jq '{source,run_id}'
# → {"source": "pickle", "run_id": null}
```

To restore MLflow loading, restart without the override:

```bash
docker compose up -d api
```
## Common Failures and Fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| `kafka` container restarts in a loop | Stale KRaft volume after image upgrade | `docker compose down -v` then `docker compose up -d` (wipes Kafka volume, OK in replay mode) |
| `ingestor` exits with `Kafka bootstrap … not reachable` | Started before `kafka-init` finished | `docker compose restart ingestor` (the service has `restart: on-failure` so it usually self-heals) |
| `featurizer` runs but `ticks.features` offset stays at 0 | First 60 s of ticks are still in the label-delay buffer | Wait — labels emit only after `horizon_sec` (60 s) of future history. Confirm with `docker compose exec -T kafka kafka-run-class kafka.tools.GetOffsetShell --broker-list localhost:9092 --topic ticks.features --time -1` |
| `predict-bridge` logs repeated 5xx / connection errors | API is unhealthy or still starting | `docker compose restart api predict-bridge` and check `curl http://localhost:8000/health` |
| `predict_requests_total` stays flat while `ticks.features` grows | The bridge is not consuming or is stuck on an uncommitted message | Check `docker compose logs predict-bridge`; if needed restart `docker compose restart predict-bridge` |
| Materializer receives fresh events but `last_write_ts` is stale | SQLite writes are stalled or failing | Check `curl -s http://localhost:8090/health | jq .` and `docker compose logs materializer`; restart the service, then use the read-model rebuild procedure below if it remains stalled. |
| UI returns `502 Bad Gateway` | The API or materializer nginx upstream is down/unhealthy | Check `curl -f http://localhost:8000/health`, `curl -f http://localhost:8090/health`, and `docker compose ps`; inspect `docker compose logs ui api materializer` before restarting the unhealthy service. |
| `/predict` returns 500 with `Model not found` | Volume mount didn't pick up `lr_pipeline.pkl` | Rebuild API: `docker compose up -d --build api` |
| Grafana panels say "No data" | Prometheus hasn't scraped yet, or `api` job is `down` | Visit http://localhost:9090/targets and check the `api` row. If `down`, restart with `docker compose restart prometheus` |
| Consumer-lag panel empty | `kafka-exporter` not up | `docker compose up -d kafka-exporter`; check logs |

## Recovery Procedures

**Full reset (loses Kafka data, the SQLite read model, and dashboard state; keeps source code):**

```bash
docker compose down -v
docker compose up -d
```

**Rebuild only the disposable predictions read model:**

Capture the materializer's named volume before removing its container, then
delete only that volume and recreate the service. The new empty projection
causes the materializer to replay retained `ticks.predictions` events from
Kafka.

```bash
PREDICTIONS_VOLUME=$(docker inspect "$(docker compose ps -q materializer)" \
  --format '{{range .Mounts}}{{if eq .Destination "/data"}}{{.Name}}{{end}}{{end}}')
docker compose stop materializer
docker compose rm -f materializer
docker volume rm "$PREDICTIONS_VOLUME"
docker compose up -d materializer
curl -s http://localhost:8090/health | jq .
```

This rebuild cannot recover prediction events that were also removed from
Kafka. Do not treat `predictions.db` as an independent backup or source of
truth.

**Restart one component:**

```bash
docker compose restart <service>       # e.g. featurizer
docker compose logs -f <service>
```

**Inspect Kafka topic offsets:**

```bash
docker compose exec -T kafka kafka-run-class kafka.tools.GetOffsetShell \
    --broker-list localhost:9092 --topic ticks.raw --time -1
docker compose exec -T kafka kafka-run-class kafka.tools.GetOffsetShell \
    --broker-list localhost:9092 --topic ticks.features --time -1
docker compose exec -T kafka kafka-run-class kafka.tools.GetOffsetShell \
    --broker-list localhost:9092 --topic ticks.predictions --time -1
curl -s http://localhost:8090/health | jq .
curl -s http://localhost:3001/api/predictions/recent | jq '{count}'
curl -s "http://localhost:9090/api/v1/query?query=sum(predict_requests_total)" | jq .
```

**Regenerate drift report:**

```bash
python scripts/drift_report.py \
    --reference handoff/data_sample/features_slice.csv \
    --current   data/processed/features.parquet \
    --out       reports/drift_$(date +%Y%m%d).html
```

## Shutdown

```bash
docker compose down                    # keeps named volumes, including predictions-data
docker compose down -v                 # nukes everything
```
