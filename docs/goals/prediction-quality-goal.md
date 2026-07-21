# Prediction-quality goal charter

## Objective

Improve out-of-sample prediction of the existing 60-second Coinbase BTC-USD
trade-price volatility-spike target, using only an already-collected corpus.
This charter is the durable contract for
[`2026-07-16-btcspiker-goal-experimentation.md`](../superpowers/plans/2026-07-16-btcspiker-goal-experimentation.md).

## Global constraints

- Resolve data through `BTCSPIKER_EXISTING_DATA` or the documented local
  fallback. Do not download, generate, collect, or wait for market data.
- Keep source data, MLflow tracking, manifests, EDA, and exports local under
  `BTCSPIKER_ARTIFACT_ROOT` (default `.artifacts/btcspiker`).
- Use the fixed causal `core_v1` features, the 60-second target, five purged
  expanding development folds, and a 20% temporal final holdout. The embargo
  is `max_feature_lookback + 60 seconds`.
- The final holdout must not select features, tune models, calibrate thresholds,
  or choose ensemble members/weights. It may be opened once, only by the
  explicit `qualification` operation after baseline, linear, trees, ablation,
  and ensemble are complete. A second request is an error.
- Run within the justified 24-hour budget. Log successful, failed, pruned, and
  skipped work to the local `btc-volatility-tournament` MLflow experiment.
  Failed trials retain a traceback; a stage stops after failures exceed 20% of
  its fixed trial budget.
- Production is never auto-promoted. Only a candidate that passes every gate
  may be registered in **Staging**; Production remains unchanged.

## Stage order and data rule

Run `baseline`, `linear`, `trees`, `ablation`, `ensemble`, then `neural` only
when its stated data and tree-plateau preconditions hold. The final
`qualification` is post-search and is the sole gate allowed to open the
holdout.

The data credibility threshold is at least 30 calendar days with target-aligned
quote-and-trade coverage and both target classes in every development fold and
the final holdout. A corpus below that threshold is `qualification_data=false`:
complete all statistically eligible work and report a research-only,
provisional result. It does not cause this goal to pause, collect data, or wait
for data.

## Staging gates

All of the following must pass: coverage `>= 30` days, quote/trade coverage,
at least four folds won, positive bootstrap lower bound, Brier ratio `<= 1.05`,
non-negative event-F1 delta, positive final-holdout PR-AUC delta, p95 latency
`<= 800 ms`, deployable features, and offline/online parity. Record every
failed predicate using its stable reason code. A passing candidate is Staging
only and needs a Staging smoke test; no result authorizes Production promotion.

## Stop and pause rules

Stop when every eligible stage is complete or its budget is exhausted, the
single permitted qualification is recorded when eligible, local evidence is
checksum-verified, and the final report is written. Stop early only for a
safety/integrity failure such as an invalid immutable contract, corrupted
evidence, or a sealed-holdout violation.

`/goal pause` is only for an operational interruption or a required user
decision unrelated to data accumulation. `/goal resume` follows resolution of
that interruption or decision. Insufficient data is not a pause condition.
Use the same Codex task and keep **Prevent sleep while running** enabled during
a 24-hour search. Data gathering has no lifecycle dependency on this goal.

## Exact final report schema

```text
result_status: Staging-qualified | provisional-research-only | blocked
dataset_id:
dataset_manifest_path:
source_sha256:
coverage_days:
search_id:
experiment_name:
mlflow_tracking_uri:
stage_runs: {stage: {parent_run_id, trial_run_ids, status, reason}}
baseline_run_id:
best_candidate_run_id:
development_fold_metrics:
bootstrap_interval:
final_holdout: {opened, accessed_at, metrics}
qualification: {qualification_data, passed, reasons, staging_model_version}
latency: {p95_ms, evidence}
runtime_verification: {replay, api, rollback}
local_export: {path, manifest_verified, checksum_result}
remaining_blockers:
production_status:
```

## Paste-ready `/goal` command

```text
/goal Improve BTCSpiker's out-of-sample prediction of the existing 60-second Coinbase BTC-USD trade-price volatility-spike target by executing docs/superpowers/plans/2026-07-16-btcspiker-goal-experimentation.md and treating docs/goals/prediction-quality-goal.md as the durable charter. Use only the user's already-collected dataset resolved by BTCSPIKER_EXISTING_DATA or the documented local fallback; do not generate, download, collect, or wait for market data. Build and verify the complete experimentation framework, run every statistically eligible feature, model, calibration, and ensemble stage within the justified 24-hour budget, log every successful, pruned, failed, and skipped trial to MLflow, preserve the sealed temporal holdout, keep artifacts local, and never auto-promote Production. Finish with the strongest evidence-backed candidate and clearly label it Staging-qualified or provisional according to the fixed gates; insufficient data changes the qualification result but must not pause or leave this goal unfinished.
```

```text
/goal                 View current objective and status.
/goal pause           Pause only for an operational interruption or a required user decision unrelated to data accumulation.
/goal resume          Resume after that operational interruption or user decision is resolved.
/goal edit            Change constraints without discarding the task history.
/goal clear           Remove the goal only after accepting the final report or abandoning the effort.
```

## Goal completion checklist

- [ ] Data manifest and EDA are published.
- [ ] Current artifact and existing-data logistic baselines are reproduced.
- [ ] Eligible staged searches are complete or the budget is exhausted.
- [ ] Every successful, failed, pruned, and skipped run is visible in MLflow.
- [ ] Final holdout was opened at most once.
- [ ] Qualification reasons are recorded.
- [ ] Passing candidate is Staging only; Production is unchanged.
- [ ] MLflow evidence is checksum-exported to the local artifact root.
- [ ] Replay, API, rollback, and latency verification results are recorded.
