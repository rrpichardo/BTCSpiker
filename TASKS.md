# Tasks

## Active

### Day 0 — Stop the data bleeding (tonight, ~45 min + overnight run)
- [x] ~~Re-point BTCSPIKER_EXISTING_DATA at a real corpus~~ (2026-08-12) — bound + profiled, dataset_id `85f929e1041284f6e20ce0a86577566545a9fa587ba15279ac9c458da10c4520`, 784,441 rows, 16.44% positive rate, `qualification_data=false` as expected (11 days)
- [ ] **Materialize the 35-day corpus — `core_v1`** - running in background, started 2026-08-12 ~20:52. Manifest cached at `.cache/btcspiker/manifests/manifest-35d.json`, output going to `.artifacts/btcspiker/features35/core_v1`
  - Verify: output parquet exists, ≥30 days coverage, both classes in every fold
- [ ] **Materialize the 35-day corpus — `multi_window_v1` and `microstructure_v1`** - already implemented in `btcspiker_ml/features.py`; never tournament-tested. Queue after `core_v1` finishes (same manifest, `--feature-set` swap, output-root `.artifacts/btcspiker/features35/{multi_window_v1,microstructure_v1}`)
  - Verify: three feature parquets on disk, one per feature set

### Day 1 — EDA that produces ONE number
- [ ] **Reconcile the four conflicting performance numbers** - 0.2667 (tournament dev) vs 0.1459 (docs/results.md) vs 0.7639 (evaluation.py) vs 0% recall (live tab). These cannot all be true. Pick the real one.
  - Verify: one sentence in `docs/eda-35d.md` — "the model is good if X, and today X = Y"
- [ ] **EDA on the 35-day corpus, not the 11-day one** - spike prevalence, clustering by hour/day, feature-vs-label separation per feature set
  - Verify: `docs/eda-35d.md` with one separation table covering all three feature sets
- [ ] **Decide the feature set to carry forward** - `core_v1` (7 feats) vs `multi_window_v1` (50) vs `microstructure_v1` (32, uses order-book depth)
  - Verify: decision written down with the EDA evidence behind it

### Day 2 — One tournament, then the one-shot qualification
- [ ] **Run the tournament on the 35-day corpus** - baseline → linear → trees → ablation → ensemble, per feature set
  - Verify: `reports/experiment_summary.md` regenerated with real run IDs
- [ ] **Open the sealed holdout ONCE** - `final_holdout_opened=false` today. This is the single bullet; do not waste it on a half-configured run.
  - Verify: `qualification.json` verdict written; state file records the timestamp
- [ ] **Publish the winner to Staging if it qualifies** - `scripts/publish_candidate_to_registry.py`. Never Production.
  - Verify: registry shows the candidate in Staging; API `/version` still serves the old model

### Day 3 — Rebuild the UI around the one number
- [ ] **Delete the "adaptive top-15%" mode** - it exists only so the tab has something to show in a quiet window. It is a lie of convenience and it is why the tab feels meaningless.
- [ ] **Rewrite the Performance tab to answer 3 questions above the fold** - Does it work? / How do I know? / What is it doing right now?
  - Verify: someone who has never seen the project can say what the model does and whether it works, in 30 seconds
- [ ] **Make the empty state honest** - "not enough graded predictions yet" beats a fake number

### Day 4 — Freeze
- [ ] **Make README + docs/results.md report the one number** - archive or delete the contradicting ones
- [ ] **Tag a release and stop**

## Waiting On

## Someday
- [ ] **Neural stage** - correctly skipped; no tree plateau established. Do not revisit until trees actually beat linear.
- [ ] **Graph / GNN models** - no graph infrastructure in this repo, and boosted trees already lose to logistic regression. Capacity is not the bottleneck.
- [ ] **Live Coinbase ingestion polish** - already verified working; not on the critical path
- [ ] **More code review / test hardening rounds** - 474 tests pass, CI is green. This is the loop that has been eating the project.

## Done
