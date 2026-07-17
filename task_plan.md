# Task Plan: BTCSpiker Prediction Experimentation Goal

## Goal
Produce a repo-specific, reviewable plan for using Codex `/goal` to maximize out-of-sample BTC prediction quality through disciplined feature, model, and ensemble experimentation logged in MLflow.

## Current Phase
Phase 5

## Phases

### Phase 1: Requirements and Discovery
- [x] Inspect repository structure, documentation, current pipeline, tests, and recent commits
- [x] Verify the documented behavior of Codex `/goal`
- [x] Capture the user's objective, constraints, and success criteria
- **Status:** complete

### Phase 2: Design Options
- [x] Compare experimentation strategies
- [x] Recommend a strategy with leakage and overfitting safeguards
- [x] Present the defaults for user approval
- **Status:** complete

### Phase 3: Design Specification
- [x] Write the approved design to `docs/superpowers/specs/`
- [x] Self-review for ambiguity, contradictions, placeholders, and excess scope
- [x] Apply the user's pre-approval to proceed directly to the detailed plan
- **Status:** complete

### Phase 4: Detailed Execution Plan
- [x] Map exact files and interfaces
- [x] Create an implementation plan in `docs/superpowers/plans/`
- [x] Include a paste-ready `/goal` objective with checkpoints and stop conditions
- **Status:** complete

### Phase 5: Delivery
- [x] Verify plan coverage and file references
- [x] Prepare the plan handoff and `/goal` instructions
- **Status:** complete

## Key Questions
1. What exact BTC outcome and prediction horizon is the primary target? **Answered: keep the existing 60-second binary volatility-spike target.**
2. What metric and validation protocol should determine whether an experiment is genuinely better?
   **Answered:** purged walk-forward PR-AUC primary, with calibration, event metrics, latency, and an untouched final holdout.
3. What compute and wall-clock budget should `/goal` respect? **Answered: up to 24 hours per major search cycle on the local M3 Pro; no paid compute.**
4. Which currently available data may be used at prediction time without leakage? **Answered: free public historical and live data, with strict as-of joins and deployability checks.**
5. Is public external market data allowed, and may the plan add new dependencies? **Answered: yes, free sources and the recommended ML libraries are allowed.**
6. Is the intended output an alert signal or an input to automated trading? **Answered: research and alerting, not automated trading.**
7. Should the goal stop at research evidence or integrate a qualified champion into the API? **Answered: register qualified candidates as Staging and integrate only after verification; never auto-promote Production.**

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Plan before starting a long-running goal | The search objective, evaluation protocol, and stop conditions must be fixed before optimization to prevent metric chasing. |
| Treat MLflow as the experiment system of record | The user explicitly wants every feature/model trial visible and comparable. |
| Keep the existing 60-second binary target as the primary objective | The user confirmed this explicitly; it preserves comparability with the shipped baseline and model. |
| Allow durable storage outside the repository | The user confirmed that additional datasets and artifacts may be stored with a cloud provider. |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| Combined verification command exited 1 because `rg` correctly found no forbidden placeholders | 1 | Re-run with an explicit conditional that treats no matches as success. |
| Final progress-update patch used stale checklist text | 1 | Re-read the planning files and applied a patch against the actual current lines. |

## Notes
- No implementation or experiment execution occurs until the design is approved.
- The final goal must optimize held-out performance, not the training score or a repeatedly inspected test set.
