# Drift Summary

This document summarizes feature drift between the training and test datasets using Evidently. The goal is to identify distribution shifts that may impact model performance in deployment.

> **Note:** This is a snapshot report, not a live-scheduled monitor. It captures a point-in-time Evidently analysis of the train vs. test split. Continuous drift scheduling is out of scope for this deployment.

## Canonical Drift Report

The authoritative drift analysis for this project is [`handoff/reports/train_vs_test.html`](../handoff/reports/train_vs_test.html), generated during model development.

## Analysis Parameters

- **Reference dataset:** training split (first 60% of collection, ~368K rows) from `data/processed/features.parquet`
- **Current dataset:** test split (last 20% of collection, ~122K rows) from `data/processed/features.parquet`
- **Tool:** Evidently AI — DataDriftPreset (DatasetDriftMetric + DataDriftTable)
- **Drift metric:** Wasserstein distance (normed) for continuous features; Jensen-Shannon distance for the binary `vol_spike` target
- **Features compared:** 8 (7 model input features + `vol_spike` label)

## Key Findings

- **Features with drift:** 3 (`n_ticks_60s`, `spread_mean_60s`, `trade_intensity_60s`)
- **Share of drifted features:** 37.5% (3 of 8)
- **Overall drift verdict at 0.5 threshold:** Not detected — 37.5% of columns drifted, below the 50% dataset-level threshold

### Per-Feature Status

| Feature | Drift Detected | Score | Stat Test |
|---------|:--------------:|------:|-----------|
| `n_ticks_60s` | Yes | 0.4191 | Wasserstein distance (normed) |
| `trade_intensity_60s` | Yes | 0.4191 | Wasserstein distance (normed) |
| `spread_mean_60s` | Yes | 0.1918 | Wasserstein distance (normed) |
| `vol_spike` (target) | No | 0.0974 | Jensen-Shannon distance |
| `spread_bps` | No | 0.0116 | Wasserstein distance (normed) |
| `vol_60s` | No | 0.0041 | Wasserstein distance (normed) |
| `log_return` | No | 0.0010 | Wasserstein distance (normed) |
| `mean_return_60s` | No | 0.0001 | Wasserstein distance (normed) |

## Interpretation

The three drifted features — `n_ticks_60s`, `trade_intensity_60s`, and `spread_mean_60s` — all measure order-book activity and trading pace, indicating the test window (2026-04-07, spike rate 6.9%) was a meaningfully quieter market regime than the training period (2026-04-04 to 2026-04-06, spike rate 15.4%). Price-action features (`log_return`, `mean_return_60s`, `vol_60s`) did not drift, confirming the shift is in market microstructure rather than in price dynamics themselves. The model still outperforms the deterministic baseline on test PR-AUC (0.1459 vs 0.1340) because the rank ordering its LR coefficients rely on is preserved even as the absolute activity distributions shift — but this regime gap is the primary driver of the val→test PR-AUC drop (0.358 → 0.1459) and the canonical trigger for retraining on a more recent window before production promotion.

## Training Reference Dataset

The full training reference dataset (`features.parquet`, 784K rows, 48MB) is not committed to this repo due to GitHub file size limits. It lives in the model training environment. For reproducibility, a 10-minute sample of the feature schema is committed at `handoff/data_sample/features_slice.csv`.

## Drift Monitoring Approach

This report is a **snapshot** taken at model evaluation time (train split vs. test split). A live-scheduled drift monitor is out of scope for this deployment. The Evidently HTML report at [`handoff/reports/train_vs_test.html`](../handoff/reports/train_vs_test.html) is the authoritative artifact, and the drift script at `scripts/drift_report.py` can be re-run against fresh production features to regenerate it on demand.
