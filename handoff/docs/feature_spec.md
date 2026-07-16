# Feature Specification — BTC Volatility Spike Detector

## Label

| Parameter         | Value                                                        |
|-------------------|--------------------------------------------------------------|
| Target horizon    | 60 seconds                                                   |
| Volatility proxy  | Rolling std of midprice log-returns over the next 60s        |
| Label definition  | `vol_spike = 1` if `σ_future >= τ`; else `0`                |
| Chosen threshold τ | **0.000048**                                                |
| Spike rate at τ   | ~15% of labelled ticks                                       |
| ≈ $1σ move @ $60k | $2.88                                                       |
| Percentile        | P85 — selected via threshold sweep (best val PR-AUC & F1)    |

Threshold was updated from P90 to P85 (0.000048) after a structured
threshold sweep across P85/P90/P95. P85 yielded the best validation PR-AUC (0.3866)
and F1 (0.4524), giving the model sufficient positive examples (~15%) to learn
meaningful patterns.

---

## Features

All features are computed per tick from a lookback window of `window_seconds = 60`.

| Feature               | Formula / description                                            | Unit            |
|-----------------------|------------------------------------------------------------------|-----------------|
| `price`               | Last traded price                                                | USD             |
| `midprice`            | `(best_bid + best_ask) / 2`                                      | USD             |
| `log_return`          | `ln(price_t / price_{t-1})`                                      | dimensionless   |
| `spread_abs`          | `best_ask − best_bid`                                            | USD             |
| `spread_bps`          | `spread_abs / midprice × 10,000`                                 | basis points    |
| `vol_60s`             | Std of log-returns over past 60s                                 | dimensionless   |
| `mean_return_60s`     | Mean log-return over past 60s                                    | dimensionless   |
| `n_ticks_60s`         | Count of ticks in the past 60s                                   | integer         |
| `trade_intensity_60s` | `n_ticks_60s / 60`  (ticks per second)                          | ticks/sec       |
| `spread_mean_60s`     | Mean absolute spread over past 60s                              | USD             |
| `price_range_60s`     | `max(price) − min(price)` over past 60s                         | USD             |

---

## Feature Selection (Ablation Study)

Four feature variants were evaluated using identical LR hyperparameters (C=0.1, balanced, lbfgs).
Threshold selected on validation best-F1; winner picked on validation PR-AUC.

| Variant | Features | val PR-AUC | Outcome |
|---------|----------|-----------|---------|
| A | 6 baseline features | 0.3471 | Control |
| B | A + `spread_mean_60s` | **0.3580** | **Winner** — liquidity signal improves performance |
| C | B + `price_range_60s` | 0.3517 | `price_range_60s` adds noise, slight drop from B |
| D | C − `spread_abs` − `n_ticks_60s` | 0.3517 | Identical to C; confirms dropped features are redundant |

### Correlation analysis results

Highly correlated pairs (|r| > 0.85) informed the ablation design:

| Pair | r | Resolution |
|------|---|------------|
| `n_ticks_60s` ↔ `trade_intensity_60s` | ~1.00 | Linear rescale (`intensity = n_ticks / 60`). Both kept in Variant B since LR handles collinearity; confirmed redundant in D vs C comparison. |
| `spread_abs` ↔ `spread_bps` | ~0.99 | Same signal, different scale. `spread_bps` preferred (unit-free). |
| `spread_abs` ↔ `spread_mean_60s` | high | Instantaneous vs smoothed. `spread_mean_60s` preferred (less noisy, better val PR-AUC). |

### Candidates tested but not included in final model

| Feature | Tested in | Result | Rationale for exclusion |
|---------|-----------|--------|------------------------|
| `price_range_60s` | Variants C, D | No lift over B (val PR-AUC 0.3517 vs 0.3580) | Adds dimensionality without improving discrimination |

### Final model feature set (Variant B — 7 features)

| Feature | Role |
|---------|------|
| `log_return` | Instantaneous price momentum |
| `spread_bps` | Current liquidity (unit-free) |
| `vol_60s` | Rolling volatility (strongest predictor) |
| `mean_return_60s` | Rolling drift direction |
| `trade_intensity_60s` | Market activity rate |
| `n_ticks_60s` | Raw tick count (correlated with intensity but retained) |
| `spread_mean_60s` | Smoothed liquidity signal over 60s window |

---

## Parquet schema (`data/processed/features.parquet`)

| Column                | Type      | Description                              |
|-----------------------|-----------|------------------------------------------|
| `product_id`          | string    | e.g. `BTC-USD`                           |
| `timestamp`           | string    | ISO-8601 source timestamp                |
| `price`               | float64   |                                          |
| `midprice`            | float64   |                                          |
| `log_return`          | float64   |                                          |
| `spread_abs`          | float64   |                                          |
| `spread_bps`          | float64   |                                          |
| `vol_60s`             | float64   |                                          |
| `mean_return_60s`     | float64   |                                          |
| `n_ticks_60s`         | int64     |                                          |
| `trade_intensity_60s` | float64   |                                          |
| `spread_mean_60s`     | float64   | Mean absolute spread over 60s window       |
| `price_range_60s`     | float64   | max − min price over 60s window            |
| `future_vol_60s`      | float64   | σ over next 60s; NaN if horizon not closed |
| `vol_spike`           | int64     | 1 if `future_vol_60s >= 0.000048` else 0 |
