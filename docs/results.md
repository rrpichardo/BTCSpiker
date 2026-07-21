# Results — BTC Volatility Spike Detector

Single-page summary of the production system's measured performance against the SLOs in [`slo.md`](./slo.md).

## Headline numbers

| Dimension | Result | Target | Status |
|---|---:|---:|:---:|
| `/predict` latency p95 (single row) | **106.4 ms** | ≤ 800 ms | PASS |
| `/predict` success rate (100-burst) | **100 %** (100 / 100) | ≥ 99.0 % | PASS |
| Replay mode drives real `/predict` traffic | **verified via `predict_requests_total`** | required | PASS |
| Replay runtime lag panels | **`ticks-featurizer` + `predict-bridge` visible** | required | PASS |
| Held-out test PR-AUC, ML vs baseline | **0.1459 vs 0.1340** | ML > baseline | PASS (+8.9 %) |
| Rollback time, ML → baseline | **< 10 s** | manual, fast | PASS |
| Services reaching healthy state after `docker compose up -d` | **All 9 / 9** | 9 / 9 | PASS |
| Live Coinbase ingestion (`--profile live`) | **verified** | available | PASS |

## Replay mode is now truly end-to-end

The default stack no longer stops at `ticks.features`. The shipped runtime now includes a dedicated `predict-bridge` service that consumes engineered feature rows from Kafka and POSTs them into `/predict`, stamping each request from Kafka publish time so the API's freshness gauge reflects the real feature-to-predict hop instead of a test-only shim or the archived market timestamp.

Operationally, that means replay mode now produces all four artifacts we claimed:

- raw Kafka traffic on `ticks.raw`
- feature Kafka traffic on `ticks.features`
- real API prediction traffic visible in `predict_requests_total`
- Prometheus-visible replay lag on both runtime hops via `ticks-featurizer` and `predict-bridge`

## Live ingestion (verified)

The default stack runs in replay mode for reproducibility, but live Coinbase ingestion is wired and verified. Bringing it up:

```bash
docker compose stop ingestor
docker compose --profile live up -d ws-ingestor
```

End-to-end check observed during testing:

- `ws-ingestor` connected to `wss://advanced-trade-ws.coinbase.com`, subscribed to the `ticker` and `heartbeats` channels for BTC-USD.
- `ticks.raw` Kafka offset advanced from 117,317 → 117,398 in ~10 s (≈ 8 ticks/s, matching Coinbase's published ticker rate).
- Sample message consumed from `ticks.raw`:

  ```json
  {"product_id": "BTC-USD", "price": "75681.74", "best_bid": "75681.74",
   "best_ask": "75681.75", "volume_24_h": "4068.30928598",
   "timestamp": "2026-04-19T04:10:41.328613453Z"}
  ```

- The featurizer, `predict-bridge`, API, and monitoring stack required no changes — same Kafka payload schema as the replay path. The two ingestors are interchangeable behind the `ticks.raw` topic.

## Live dashboard

![Grafana dashboard — BTC Volatility Detector API](./grafana_dashboard.png)

The current dashboard surfaces active variant, p50 / p95 latency, request and error rate, plus a dedicated replay row for `ticks.raw -> featurizer`, `ticks.features -> predict-bridge`, and API freshness. Dashboard JSON is at [`monitoring/grafana/dashboards/api.json`](../monitoring/grafana/dashboards/api.json).

## Latency

Full methodology and percentiles in [`latency_report.md`](./latency_report.md). Highlights:

- Reference verified local run on `2026-04-23` used 100 concurrent requests through `tests/load_test.py` against the running stack (Kafka, ingestor, featurizer, API, MLflow, Prometheus, Grafana, kafka-exporter all live).
- p50 / p95 / p99 = 97.4 / 106.4 / 112.5 ms — comfortably under the 800 ms p95 SLO while replay traffic and the runtime bridge are active.
- The local figures drift depending on concurrent replay activity, so [`latency_report.md`](./latency_report.md) is the canonical reference run for this revision.

## Uptime / availability

All 9 services reach healthy state after `docker compose up -d`. The load test achieves a 100% success rate (100/100 requests) with p95 latency well under the 800 ms SLO.

We do not run a long-horizon uptime measurement (this is a demonstration deployment, not a 24×7 service), so availability is reported as an **SLO with a recovery contract** rather than a measured number:

- **Target:** 99.5 % service availability on `/health` over a 24 h window, with a separate 99 % rolling 5-minute request success-rate SLO for `/predict`.
- **Mechanism:** every container declares `restart: on-failure`; Kafka and the API both have healthchecks; `depends_on … condition: service_healthy` guarantees correct startup ordering.
- **Recovery contract:** documented in [`runbook.md`](./runbook.md) — every common failure mode (Kafka volume corruption, ingestor restart loop, bridge retry loop, missing model artifact, Grafana "No data") has a 1-line recovery command and an expected outcome.
- **Observability hooks:** the Grafana dashboard surfaces error rate per variant, replay lag on both Kafka hops, and API freshness, so the on-call signal arrives before users do.

A continuous-uptime number can be added later by pointing an external prober (e.g. an uptime check) at `/health`; the API and the Prometheus error counters are already wired for it.

## Model performance vs baseline

| Model | Validation PR-AUC | Test PR-AUC | Notes |
|---|---:|---:|---|
| Z-score baseline (`vol_60s > τ`) | n/a (rule-based) | **0.1340** | Deterministic threshold rule on rolling vol, used as both the science baseline and the production rollback target. |
| Logistic Regression, Variant B (7 features) | best of ablation set | **0.1459** | Shipped artifact. Selection rationale in [`docs/selection_rationale.md`](selection_rationale.md). |

The same threshold rule lives in the API as `MODEL_VARIANT=baseline` (`api/main.py::_score_baseline`), so the science baseline and the production rollback path are the *same* code path — the rollback isn't a degraded approximation, it's the documented baseline.

## Rollback verified end-to-end

The `MODEL_VARIANT` toggle was exercised live:

```bash
MODEL_VARIANT=baseline docker compose up -d api
curl -s http://localhost:8000/version | jq '.source'   # → "pickle"

MODEL_VARIANT=ml docker compose up -d api
curl -s http://localhost:8000/version | jq '.source'   # → "mlflow" in the normal stack
```

The Grafana **Active variant** stat panel (top-left of the API dashboard) flips within ~10 s of the next Prometheus scrape, and `predict_requests_total{model_variant=…}` cleanly partitions traffic by variant for post-hoc analysis.

## Drift posture

Full report in [`drift_summary.md`](./drift_summary.md). One-line version: 3 of 7 input features show distribution drift between the training reference and the held-out test slice (`n_ticks_60s`, `trade_intensity_60s`, `spread_mean_60s`), but the model still beats the baseline on PR-AUC because the **rank ordering** the LR coefficients depend on is preserved. The Evidently HTML lives at `handoff/reports/train_vs_test.html` and can be regenerated against fresh production features with `scripts/drift_report.py`.

## What this means for production readiness

- The **performance** budget (latency, success rate) is comfortably met.
- The **reliability** budget is enforced by Compose healthchecks + restart policies + a documented runbook, but does not yet have a long-horizon measured uptime number.
- The **model** earns its keep over the trivial baseline (+8.9 % test PR-AUC) and the rollback to that baseline is a one-environment-variable change with sub-10-second propagation.
- Drift is **monitored**, not yet **alerted on** — a natural next step is tightening manual review cadence and adding Prometheus alert rules on the dashboard's existing panels.

---

## Existing-data tournament — provisional research result (2026-07-21)

This section is the evidence record for the separate existing-data prediction
quality goal. It does not alter the shipped runtime scorecard above and does
not authorize a registry or Production change.

| Field | Evidence |
|---|---|
| Result status | **Provisional, research-only** (`qualification_data=false`) |
| Dataset ID / search ID | `07d95c0d0c8224d8cda43f20122604394694fcc80191cc3b81fd856ec5dbe136` |
| Manifest / source SHA-256 | `.artifacts/btcspiker/manifests/existing-07d95c0d0c8224d8cda43f20122604394694fcc80191cc3b81fd856ec5dbe136.json` / `a72e3062a3e434bf80020b3869679e930f125414ddb5cbaea1e9ec090794fc41` |
| Corpus | 788,465 rows, 2026-04-04 22:54:57Z to 2026-04-15 23:05:52Z (11.0 calendar days) |
| MLflow | local `btc-volatility-tournament`, `file:.artifacts/btcspiker/mlruns` |
| Baseline run | `382f4f034d16479c81603582870d30a5` (parent `2ecf6779ad304290b16674dd121a5614`) |
| Best development candidate | linear logistic `52f8a399422e47e5a352776e22c24363` (parent `979783065a4d4b8f8e8b5c06f3b91446`) |
| Qualification / Staging | Fails the fixed coverage gate; no qualification execution, no candidate registration, and no Staging version |
| Final holdout | **Sealed and unopened**: `final_holdout_opened=false`, `final_holdout_accessed_at=null`; therefore no final-holdout metric exists |
| Latency / runtime proof | No new tournament latency, replay, API, or rollback execution was performed in this final proof; the legacy runtime evidence above remains separate |
| Local export | No `mlflow-exports/run_id=.../export-manifest.json` exists for this provisional candidate, so no production-run checksum claim is made |

The 30-day credibility threshold is a hard Staging requirement. The 11-day
corpus therefore blocks qualification only; it does not pause the completed
research workflow or justify opening the holdout. The development winner may
not be described as out-of-sample improved, deployable, Staging-qualified, or
Production-ready.

### Development fold metrics

All values below are five purged expanding **development** folds, not
final-holdout results.

| Candidate | Run ID | Fold PR-AUC (0–4) | Aggregate PR-AUC |
|---|---|---|---:|
| Development-prevalence baseline | `382f4f034d16479c81603582870d30a5` | 0.078456, 0.173348, 0.211086, 0.135142, 0.074766 | 0.134560 |
| Seven-feature logistic (winner) | `52f8a399422e47e5a352776e22c24363` | 0.081558, 0.328018, 0.357542, 0.361186, 0.205261 | **0.266713** |
| Bounded HistGradientBoosting | `0a6f765cd59e4d8394dee24f4892b7b8` | recorded in MLflow | 0.221710 |
| Logistic without `vol_60s` | `f5de3ecaa43342599fc88be84b870c92` | recorded in MLflow | 0.251999 |
| Mean logistic/tree ensemble | `6c0a158aa2c8461b8f9bc2038cc1f815` | recorded in MLflow | 0.252577 |

No bootstrap interval was produced for the provisional run, and its absence is
another reason no Staging claim is made. The neural parent
`bb22ea9ffa9c4629aaa2fbdfa2b03a07` is a finished, reason-coded skipped run:
bounded tree evidence did not establish a progress plateau.

### MLflow lineage and remaining blockers

The completed stage parents are baseline `2ecf6779ad304290b16674dd121a5614`,
linear `979783065a4d4b8f8e8b5c06f3b91446`, trees
`5b080c0edba0412293fa2684470d323b`, ablation
`beb72d716892450495fb6a10a5f18aa5`, ensemble
`05e9fc2380ea412a98e4e326d7875932`, and skipped neural
`bb22ea9ffa9c4629aaa2fbdfa2b03a07`. The EDA run is
`e0bd525fcbd546dda68778d8d4b77181`.

This bounded tournament contains finished development trials and one skipped
parent; it contains no actual pruned or failed trial lineage. The orchestration
and export tests cover those statuses, but they are not substituted for
production-run evidence. The remaining blockers are: sub-30-day coverage,
no final-holdout qualification (intentionally sealed), no bootstrap/latency or
runtime verification for the candidate, no Staging smoke test, and no local
export manifest. Production remains the legacy champion unless separately
approved.
