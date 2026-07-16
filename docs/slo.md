# Service Level Objectives for BTC Volatility Spike Detector

These SLOs cover the public facing prediction API (`api` service) and the streaming pipeline that feeds it. They are aspirational targets, not strict pass or fail thresholds. Their purpose is to make degradation visible in Grafana and give the on call team a clear signal when intervention is needed.

## SLOs

| # | Objective | Target | Measurement window | Source metric (PromQL) |
|---|---|---|---|---|
| 1 | **Prediction latency p95** | ≤ 800 ms | rolling 5 min | `histogram_quantile(0.95, sum by (le) (rate(predict_latency_seconds_bucket[5m])))` |
| 2 | **Prediction latency p50** | ≤ 100 ms | rolling 5 min | `histogram_quantile(0.50, sum by (le) (rate(predict_latency_seconds_bucket[5m])))` |
| 3 | **Request success rate** | ≥ 99 % | rolling 5 min | `1 - (sum(rate(predict_errors_total[5m])) / clamp_min(sum(rate(predict_requests_total[5m])), 0.001))` |
| 4 | **Replay lag: raw -> featurizer** | ≤ 200 messages on `ticks.raw` | rolling 1 min | `sum(kafka_consumergroup_lag{topic="ticks.raw", consumergroup="ticks-featurizer"})` |
| 5 | **Replay lag: features -> predict** | ≤ 200 messages on `ticks.features` | rolling 1 min | `sum(kafka_consumergroup_lag{topic="ticks.features", consumergroup="predict-bridge"})` |
| 6 | **Prediction freshness at API** | ≤ 120 s | continuous | `feature_freshness_seconds` |
| 7 | **Service availability (`/health`)** | ≥ 99.5 % | 24 h | Compose healthcheck on `api` |

## Error Budget

- **Latency budget:** 5 % of requests / month may exceed 800 ms.
- **Error budget:** 1 % of requests / month may return 5xx.
- **Freshness budget:** either replay hop may exceed 200 queued messages for at most 30 min / day, and `feature_freshness_seconds` may exceed 120 s only during active incident response.

If any budget is consumed by more than 50% within a 24 hour window, the on call response is to switch `MODEL_VARIANT=baseline` as described in `runbook.md` and investigate before re enabling the ML variant..

## Current Measured Baseline

Reference verified local run on `2026-04-23`: 100-request burst load test, one
feature row per request, API running locally on an M2 MacBook with the replay
pipeline and `predict-bridge` active in the background.

| Metric | Value | vs SLO |
|---|---:|:---:|
| p50 latency | 97.4 ms | within target |
| p95 latency | 106.4 ms | within target (target 800 ms) |
| p99 latency | 112.5 ms | n/a |
| Success rate | 100 % | within target |

Full report: [latency_report.md](latency_report.md).

## Alerting approach

Alerting is dashboard-based. Breach visibility is via Grafana panels rather than push notifications. Operators monitor the dashboard during demos and scheduled check-ins.

The Grafana dashboard "BTC Volatility Detector — API" at http://localhost:3000 surfaces all seven SLO signals above as live panels. The on-call response when a panel crosses threshold is documented in the [runbook](runbook.md) — each SLO has a corresponding failure mode entry with a 1-line recovery command. Push-notification alerting (PagerDuty, Slack webhooks, Prometheus Alertmanager rules) is not configured in this deployment.

## Out of scope

- Cold-start latency (model load takes ~2 s; not measured here).
- WebSocket ingestor uptime — the shipped build runs in replay mode by default, with the live WebSocket ingestor available only under the optional `live` profile.
- MLflow tracking server availability beyond startup — non-critical. The `ml`
  variant normally loads from MLflow at startup, and the service falls back to
  the on-disk pickle artifact if the registry is unavailable.
