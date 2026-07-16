# Latency Report

This report summarizes the latency performance of the prediction API under burst load conditions.

## Test setup

- **Tool:** `tests/load_test.py` — 100 concurrent requests via `ThreadPoolExecutor` (max_workers = 100), single-row payload per request.
- **Target:** `http://localhost:8000/predict`
- **Variant:** `MODEL_VARIANT=ml` (sklearn LR pipeline)
- **Environment:** Apple M2 MacBook Pro, Docker Desktop, all 9 long-running services active (Kafka + ingestor + featurizer + predict-bridge + API + MLflow + Prometheus + Grafana + kafka-exporter).
- **Run date:** Verified live run on `2026-04-23`
- **Reproduce:**

  ```bash
  docker compose up -d
  python tests/load_test.py
  ```

## Results

| Metric | Value |
|---|---:|
| Requests sent | 100 |
| Succeeded (HTTP 200) | 100 (100 %) |
| Failed | 0 |
| Latency p50 | 97.4 ms |
| Latency p95 | **106.4 ms** |
| Latency p99 | 112.5 ms |
| Latency max | 112.5 ms |

## SLO Check

p95 SLO target: **≤ 800 ms** → measured **106.4 ms** → **PASS** with comfortable headroom.

## Observations

- The figures above reflect the reference verified local run from `2026-04-23`, so they supersede the older numbers from earlier audit passes.
- The spread between p50 and p99 (~15 ms in this run) reflects a tight local scoring path; the absolute values can still drift depending on how much replay traffic the `predict-bridge` is driving at the same time.
- The featurizer and ingestor were actively running during the test, so this number reflects realistic contention from the streaming workload, not an idle container.
- The baseline variant would likely be slightly faster due to the absence of sklearn inference, but was not evaluated as the SLO applies to the production ML variant.
