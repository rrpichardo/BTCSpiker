# Progressive model tournament summary

## Scope and data sufficiency

- Bound corpus: `07d95c0d0c8224d8cda43f20122604394694fcc80191cc3b81fd856ec5dbe136`
- Source SHA-256: `a72e3062a3e434bf80020b3869679e930f125414ddb5cbaea1e9ec090794fc41`
- Rows: 788,465; event time: 2026-04-04 22:54:57Z through 2026-04-15 23:05:52Z (11.0 calendar days)
- Qualification label: `qualification_data=false`.  This is a research-only result because coverage is below the 30-day requirement.  The tournament continued; it did not fetch or wait for more data.
- MLflow experiment: `btc-volatility-tournament`, local tracking store `.artifacts/btcspiker/mlruns`.

## Development-only results

All scored results use the five purged expanding development folds.  The final 20% temporal holdout was not passed to fit, score, or metric code.

| Stage | Candidate | Aggregate development PR-AUC | MLflow parent | MLflow trial |
| --- | --- | ---: | --- | --- |
| baseline | development prevalence | 0.134560 | `2ecf6779ad304290b16674dd121a5614` | `382f4f034d16479c81603582870d30a5` |
| linear | logistic regression, seven causal core features | **0.266713** | `979783065a4d4b8f8e8b5c06f3b91446` | `52f8a399422e47e5a352776e22c24363` |
| trees | bounded HistGradientBoosting (50 iterations) | 0.221710 | `5b080c0edba0412293fa2684470d323b` | `0a6f765cd59e4d8394dee24f4892b7b8` |
| ablation | logistic without `vol_60s` | 0.251999 | `beb72d716892450495fb6a10a5f18aa5` | `f5de3ecaa43342599fc88be84b870c92` |
| ensemble | mean of bounded logistic and tree probabilities | 0.252577 | `05e9fc2380ea412a98e4e326d7875932` | `6c0a158aa2c8461b8f9bc2038cc1f815` |

The linear candidate is the development winner.  This is not a promotion decision and no Production model was registered or promoted.

### Neural stage

Neural stage parent: `bb22ea9ffa9c4629aaa2fbdfa2b03a07`.

It is a finished, reason-coded skipped MLflow run: `neural stage skipped: boosted-tree progress plateau is not established from one bounded tree trial`.  The row-count precondition is met, but one bounded tree candidate is insufficient evidence of a plateau; no separate neural environment was installed.

## Sealed-holdout and resume state

The state is `.experiment-state/07d95c0d0c8224d8cda43f20122604394694fcc80191cc3b81fd856ec5dbe136.json`.

- Completed development stages: baseline, linear, trees, ablation, ensemble, neural (skipped with reason).
- `final_holdout_opened=false`; `final_holdout_accessed_at=null`.
- `SearchState.open_final_holdout()` accepts only the explicit `qualification` stage after all five development stages, records a timestamp, and denies a second access.
- Resuming requires `--resume` and a matching dataset, feature set, target, validation, and git revision contract.  Failure counts persist across resumes.

## TDD and verification evidence

Red:

```text
tests/ml/test_holdout_guard.py: SearchState.new was missing
AttributeError: type object 'SearchState' has no attribute 'new'
```

Green:

```text
pytest tests/ml/test_search.py tests/ml/test_holdout_guard.py -q
5 passed

pytest tests/ml -q
79 passed, 3 skipped
```

## Concerns and safe follow-up

This is intentionally a bounded, single-candidate-per-stage tournament rather than the configured 24-hour hyperparameter search.  Fold PR-AUC values are logged to the trial runs; the report preserves the run IDs needed for a future expansion.  Before any final-holdout qualification, add broader tree-search evidence and, only if its progress plateaus, run the neural candidate in an isolated environment.  The short corpus coverage remains a hard research-only sufficiency label.
