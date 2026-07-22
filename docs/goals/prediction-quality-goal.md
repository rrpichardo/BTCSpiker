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

## Dress-rehearsal result (2026-07-22) — not the goal itself

This is a small-scale rehearsal on a 150,000-row / 22h11m local slice, run to
prove the pipeline end-to-end before the real 30-day corpus is available. It
does not count toward the checklist above and does not authorize any registry
or Production change.

```text
result_status: blocked
dataset_id: b0637b4dd26ea9fe45ca12b49b343af882c9f3447f7789c2e265a89ea58dd493
dataset_manifest_path: .artifacts/btcspiker/rehearsal/manifests/existing-b0637b4dd26ea9fe45ca12b49b343af882c9f3447f7789c2e265a89ea58dd493.json
source_sha256: 5470f3b50367a1c0438fd322eb48abef1b68197a0f11bc2dd10725e9eef3d0a4
coverage_days: 2.0 (dataset); holdout slice itself spans ~0.92 days
search_id: rehearsal-2026-07-21
experiment_name: btc-volatility-tournament-rehearsal
mlflow_tracking_uri: file:.artifacts/btcspiker/rehearsal/mlruns
stage_runs:
  baseline: {parent_run_id: 30f834f2b7804ef182ae226e4cf4e5fc, status: completed, best_pr_auc: 0.0792}
  linear: {parent_run_id: 7b5909793d0b4602873e2c2681592d52, status: completed, best_pr_auc: 0.1105}
  trees: {parent_run_id: 30c2b2d1a7484832a2fbccd1bd8fa1b6, status: completed, best_pr_auc: 0.1085}
  ablation: {parent_run_id: 119974ba5dfd4e78a46b00081ee90fca, status: completed, best_pr_auc: 0.1071}
  ensemble: {parent_run_id: 02fbcfc6fd294355ad314bce3d78bf5a, status: completed, best_pr_auc: 0.1053}
  neural: {parent_run_id: f0b6a247bfb94de7aacc059bd53bd74b, status: skipped, reason: "torch not installed in this environment"}
baseline_run_id: 6d016ad4335b4d4091aaa2091f7e3ac1
best_candidate_run_id: 9c9d0db8aa584884979a0b8b30249f8a (linear stage, strongest recorded aggregate_pr_auc)
development_fold_metrics: see stage_runs best_pr_auc above (per-stage best of purged_walkforward_v1 folds)
bootstrap_interval: {lower: 0.0228, note: paired block bootstrap on holdout PR-AUC delta vs baseline}
final_holdout: {opened: true, accessed_at: "2026-07-22T04:19:50.371014+00:00", metrics: {final_pr_auc_delta: 0.1234, folds_won: "2 of 5", brier_ratio: 1.828, event_f1_delta: 0.0102}}
qualification: {qualification_data: false, passed: false, reasons: [coverage_under_thirty_days, fewer_than_four_folds_won, brier_regression_over_five_percent], staging_model_version: not_registered}
latency: {p95_ms: 1.16, evidence: qualification-time inference only; no live serving/SLO test run (Phase 6 blocked, see below)}
runtime_verification: {replay: not run, api: not run, rollback: not run}
local_export: {path: none, manifest_verified: n/a, checksum_result: n/a}
remaining_blockers: >
  Two real gate failures beyond the expected coverage gap: the candidate won
  only 2 of 5 development folds against baseline despite a strong positive
  holdout PR-AUC delta (+0.123), and its Brier score regressed 82.8% vs
  baseline (poor calibration despite good discrimination) — consistent with
  overfitting on a 2-trial linear-stage budget and a small, class-imbalanced
  slice (8.6% positive rate). publish_candidate_to_registry.py's own
  --provisional check requires reasons == [coverage_under_thirty_days]
  exactly, so it refused to run; Phase 6 (staging publish) was correctly
  never attempted. Before trusting a future run's winner, consider a larger
  linear/trees trial budget and a calibration step (e.g. Platt/isotonic) in
  the tournament, not just a bigger dataset.
production_status: unchanged
```
