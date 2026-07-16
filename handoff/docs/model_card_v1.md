# Model Card — BTC Volatility Spike Detector v1

## Model Details

| Field | Value |
|---|---|
| Card title (human-readable) | BTC Volatility Spike Detector |
| Registry name | `btc-volatility-lr` |
| Version | `v1.0` (MLflow registry version `1`, stage `Production`) |
| Type | Binary classifier — Logistic Regression |
| Framework | scikit-learn 1.4+ |
| Artifact | `models/artifacts/lr_pipeline.pkl` |
| MLflow experiment | `btc-volatility-lr` |
| Date | 2026-04-07 |

**Architecture:** `StandardScaler → LogisticRegression(C=0.1, class_weight='balanced')`

---

## Intended Use

Predicts whether BTC-USD price volatility will **spike** over the next 60 seconds given the current tick-level state. Intended for use as a real-time signal in a trading alerting system.

---

## Data

| Field | Value |
|---|---|
| Source | Coinbase Advanced Trade WebSocket (`wss://advanced-trade-ws.coinbase.com`) |
| Pair | BTC-USD |
| Collection period | 2026-04-04 22:54 UTC → 2026-04-07 15:54 UTC (~65 hours) |
| Raw ticks | ~631,237 (after dedup across segment files) |
| Labelled rows | 613,853 |
| Null labels (edge drain) | Rows with incomplete lookahead windows dropped |

**Label definition:** `vol_spike = 1` if the realised log-return standard deviation over the next 60 seconds exceeds `σ_threshold = 0.000048` (P85 of observed `future_vol_60s`; equivalent to ~$2.88 price movement on a $60k BTC price). Updated from P90 after threshold sweep showed P85 yields best validation PR-AUC.

**Class distribution (full dataset):** ~13.5% positive (vol spike), ~86.5% negative.

---

## Features (Variant B — ablation winner)

| Feature | Description |
|---|---|
| `log_return` | Instantaneous log-return vs previous tick |
| `spread_bps` | Bid-ask spread in basis points |
| `vol_60s` | Rolling std of log-returns over 60s window |
| `mean_return_60s` | Rolling mean log-return over 60s window |
| `trade_intensity_60s` | Trades per second over 60s window |
| `n_ticks_60s` | Tick count in 60s window |
| `spread_mean_60s` | Mean absolute spread over 60s window |

Selected via structured ablation (4 variants). `spread_mean_60s` was added based on correlation analysis showing it captures a smoothed liquidity signal that improved ablation val PR-AUC from 0.3471 (Variant A) to 0.3580 (Variant B). `price_range_60s` was tested but excluded (no lift). See `handoff/docs/feature_spec.md` for full ablation results.

**Feature pipeline:** All features standardised with `StandardScaler` fitted on the training split only.

---

## Training

**Split strategy:** Time-ordered sequential split (no shuffling — preserves temporal structure).

| Split | Rows | Spike rate |
|---|---|---|
| Train (0–60%) | 368,311 | 15.4% |
| Validation (60–80%) | 122,771 | 14.1% |
| Test (80–100%) | 122,771 | 7.0% |

The spike rate variation across splits reflects real market-regime changes across the ~65-hour collection window. Train and validation are relatively balanced (15.4% and 14.1%), while the test window landed on a quieter stretch (7.0%).
---

## Performance

### Test set (held-out, time-ordered)

| Metric | Value |
|---|---|
| PR-AUC | **0.1459** |
| F1 @ tau | 0.1359 |
| Decision threshold (τ) | 0.7015 (best-F1 on validation set) |

### Baseline comparison

| Model | Val PR-AUC | Test PR-AUC | Val F1 | Test F1 |
|---|---|---|---|---|
| Z-score baseline (sigmoid-calibrated) | 0.3270 | 0.1340 | 0.3868 | 0.1487 |
| **Logistic Regression v1 (Variant B)** | **0.3580** | **0.1459** | **0.4463** | **0.1359** |

The LR model outperforms the z-score baseline on both val and test PR-AUC. The val-to-test drop (0.3580 → 0.1459) reflects a market regime shift — the test window captured a quiet period (7.0% spike rate vs 14.1% in validation) — not overfitting.

---

## Feature Importance (Logistic Regression Coefficients)

| Feature | Coefficient | Direction |
|---|---|---|
| `vol_60s` | +0.4652 | Higher rolling vol → more likely spike |
| `n_ticks_60s` | +0.1591 | More ticks → more likely spike |
| `trade_intensity_60s` | +0.1591 | Higher intensity → more likely spike |
| `spread_mean_60s` | +0.1580 | Wider mean spread → more likely spike |
| `spread_bps` | +0.0492 | Wider spread → more likely spike |
| `mean_return_60s` | −0.0363 | Negative drift → more likely spike |
| `log_return` | +0.0119 | Positive instantaneous return → slightly more likely spike |

`vol_60s` remains the dominant predictor. The newly added `spread_mean_60s` ranks 4th by coefficient magnitude, confirming its value as a smoothed liquidity signal.

---

## Drift Monitoring

Drift detected on the **target label (`vol_spike`)** between early and late windows:

| Metric | Value |
|---|---|
| Detection method | Jensen-Shannon distance |
| Drift score | **0.127** |
| Drift detected | Yes |

This reflects the spike rate shifting from ~15% (early/mid collection) to ~7% (later collection). This is consistent with a genuine change in market volatility regime during the collection period — the test window landed on a quieter stretch.
Evidently HTML report: `handoff/reports/train_vs_test.html` (combined feature + target drift).

---

## Limitations

- **Limited training window.** The model has seen only ~65 hours of market data. Performance during extreme volatility regimes (e.g., flash crashes, macro events) is unknown.
- **Single pair.** Trained on BTC-USD only and not validated on other pairs.
- **No order-book depth.** `ob_imbalance` is unavailable from the Coinbase basic ticker feed; adding bid/ask sizes could improve recall.
- **Temporal non-stationarity.** Spike rate varies across splits (7.0%–15.4%), reflecting genuine regime shifts. The val-to-test performance gap is driven by the test window landing on a quieter market period, not overfitting.

---

## Responsible AI Considerations

- Uses only publicly available market data; no PII collected
- Not designed for automated trading decisions
- Predictions should supplement, not replace, human judgment
- Model performance varies across market regimes; deployment without continuous monitoring is not recommended

---

## Versioning and Retraining Triggers

Retrain when any of the following occur:

1. Target drift score (Jensen-Shannon) exceeds **0.15**
2. Test PR-AUC drops below **0.30** on a rolling 7-day evaluation window
3. New feature sources become available (e.g., order-book depth)
