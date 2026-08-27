# Finish Plan — 4 days

Written 2026-08-11. Companion checklist: [`TASKS.md`](./TASKS.md).

## Why you're stuck

You have been optimizing the wrong loop. Look at the last three weeks of history:
11 findings fixed, 15 findings fixed, race conditions, CI lint pins, 474 tests
passing. All real work. None of it moves the project toward *done*, because
"done" for this project means two things that have never been closed:

1. **One trustworthy number** that says whether the model works.
2. **A UI that states that number** and nothing that contradicts it.

The UI doesn't convey meaning to you because the project does not currently
have a single meaning to convey. It has four, and they disagree:

| Source | Number | What it measured | Provenance |
|---|---|---|---|
| Tournament dev folds, 11-day corpus | PR-AUC **0.2667** vs 0.1346 baseline | purged walk-forward, ~2× lift | read from `reports/experiment_summary.md` |
| `docs/results.md` | **0.1459** vs 0.1340 | a different held-out eval, +8.9% | read from the file |
| `materializer/evaluation.py` | **0.7639** out-of-sample PR-AUC | different window and definition | session log only — re-verify |
| Live Performance tab | **0% recall** (0/44 spikes) | live replay grading | session log only — re-verify |

The bottom two are the ones you most need to re-derive, since they're the ones
that would most change the story and neither was confirmed against a file.

Those cannot all be true at once. Every InfoIcon, every hedging tooltip, and the
entire "adaptive top-15%" mode in the Performance tab exist to paper over that
disagreement. The tab is honest about being confused — that's the whole problem.
No amount of chart work fixes it. **Pick one number first; the UI job becomes
small and obvious afterward.**

## Two landmines to clear before anything else

**1. Your working dataset is empty.**
`data/processed/features.parquet` in this repo is currently a **9-minute
fixture with a 0.0% positive rate** — 163,086 rows spanning 2026-04-06 15:02 to
15:11. `BTCSPIKER_EXISTING_DATA` is unset, so `experiment.yaml` resolves to it.
This is exactly the `scripts/replay.py --out` footgun documented in
`CLAUDE.md`, and it already fired (file dated Jul 26). If you ran EDA today you
would be doing EDA on nine minutes of nothing.

The real 11-day corpus survived, read-only, with a backup:
`/Users/ricopichardo/Documents/BitcoinProjectTest/data/processed/features.parquet`
— 784,441 rows, 2026-04-04 → 2026-04-15, 16.4% prevalence.

**2. Graph engineering is the wrong call, and here's the evidence.**
There is no graph infrastructure anywhere in this repo, so it's days of new
code. But the stronger argument is in your own tournament results: boosted
trees scored **0.2217** and *lost* to plain logistic regression at **0.2667**.
Removing `vol_60s` entirely only dropped it to **0.2520**. When gradient
boosting can't beat a linear model on your features, the bottleneck is the
**feature set and the target definition — not model capacity.** A graph model
loses the same way, slower and less explainably.

If you meant *feature* engineering, you're right, and it's better news than you
think — see below.

## The good news you already paid for

**The 35-day corpus is acquired and it passes the qualification gate.**
`rrpichardo/btcspiker-coinbase-history` on Hugging Face: 2,522 files, 35 days
(2026-04-24 → 2026-05-28), `coverage_seconds = 2,832,495` against a required
2,592,000. That's ~2.8 days of headroom over the 30-day bar. Staging
qualification is genuinely reachable — the corpus just needs materializing.

**The feature engineering is already written.** `btcspiker_ml/features.py`
defines three feature sets. Your tournament only ever ran the smallest one:

| Set | Features | Windows | Needs |
|---|---:|---|---|
| `core_v1` | 7 | 60s only | ticker | ← the only one ever tournament-tested |
| `multi_window_v1` | 50 | 5s–300s | ticker |
| `microstructure_v1` | 32 | 5s–300s | ticker + **level 2 order book** |

`microstructure_v1` includes `book_imbalance`, EWMA fast/slow volatility,
vol-of-vol, momentum, and acceleration. It needs order-book depth — which the
35-day corpus has (`book_deltas` + `states`) and the 11-day corpus did not.

**So "find the best model" is not a research project. It's three materializations
and one tournament run.** No new code. No new features to invent.

## The plan

### Day 0 — tonight, ~45 min of work then let it run

Point the env var at a real corpus, then start the long pole.

```bash
export BTCSPIKER_EXISTING_DATA=/Users/ricopichardo/Documents/BitcoinProjectTest/data/processed/features.parquet
python scripts/profile_dataset.py
```

Verify: ~784k rows, 11 days, both classes present. If it says 163,086 rows and
0% positives, the env var didn't take.

Then materialize the 35-day corpus. **Time the first one** — 2,522 raw files is
an unknown duration, and every downstream estimate depends on it.

```bash
python scripts/materialize_coinbase_history.py --raw-manifest <path> --feature-set core_v1 --output-root .artifacts/btcspiker/features35
```

Repeat for `multi_window_v1` and `microstructure_v1`. If `core_v1` takes under
an hour, run all three tonight. If it takes six, run them sequentially overnight
and start Day 1 with whatever finished.

Verify: three parquets, each ≥30 days coverage, both classes in every fold.

### Day 1 — EDA that produces ONE number

The deliverable is a single page, `docs/eda-35d.md`, that answers:

- What is a spike, stated once, in plain language.
- How often does one happen, on 35 days rather than 11.
- When do they cluster — hour of day, day of week, in bursts or evenly.
- Which features separate spike from non-spike, **per feature set**, so Day 2
  starts with a decision instead of a guess.
- **The reconciliation.** Take the four numbers in the table above, work out
  what each one actually measured, and delete the three that don't answer the
  question you care about.

The page ends with one sentence you'd be willing to defend: *"This model is good
if X, and today X = Y."* That sentence is the entire Day 3 UI spec.

### Day 2 — one tournament, then the one shot

Run the stages on the 35-day corpus, per feature set:

```bash
python scripts/run_experiments.py --config experiment.yaml --dataset-id <id> --stage linear
```

`baseline` → `linear` → `trees` → `ablation` → `ensemble`. Skip `neural`; it was
correctly skipped before and nothing has changed that.

This is where Codex `/goal` fits if you want it driving — the charter is already
written at [`docs/goals/prediction-quality-goal.md`](./docs/goals/prediction-quality-goal.md)
and the plan at `docs/superpowers/plans/2026-07-16-btcspiker-goal-experimentation.md`.
Note `/goal` is a **Codex CLI** command, not a Claude Code one; it doesn't exist
in this session. The charter's 24-hour budget is the reason Day 2 is a full day.

Then, and only then, the qualification. `final_holdout_opened=false` today —
`SearchState.open_final_holdout()` accepts exactly one call and denies the
second. **Do not spend it on a half-configured run.** Everything above must be
green first.

```bash
python scripts/qualify_candidate.py ...
python scripts/publish_candidate_to_registry.py ...   # Staging only, never Production
```

### Day 3 — rebuild the UI around the one number

Now the UI has something to say. **Cut, don't add.**

Delete the "adaptive top-15%" mode. It was invented so the tab would show
*something* during a quiet replay window. It trains you to distrust the tab,
because a number that changes definition to stay non-empty isn't a measurement.

Rewrite the Performance tab to answer three questions, in this order, above the
fold:

1. **Does it work?** One number, one benchmark, one verdict. Not a table.
2. **How do I know?** The eval that produced it, with data provenance —
   35 days, which feature set, held-out or dev.
3. **What is it doing right now?** Live confusion matrix, and an honest
   "not enough graded predictions yet" when that's the truth.

Verify by handing it to someone who has never seen the project. If they can't
say what the model does and whether it works within 30 seconds, it isn't done.

### Day 4 — freeze

README and `docs/results.md` report the one number. Archive or delete the
contradicting ones. Tag a release. Stop.

## What you are explicitly not doing

Every one of these is defensible work, and every one is why this project is on
week seven instead of week three:

- Neural stage — correctly skipped, no tree plateau established
- Graph / GNN models — see the evidence above
- Live Coinbase ingestion polish — already verified, not on the critical path
- More code review or test hardening — 474 tests pass, CI is green
- Any new UI tab
