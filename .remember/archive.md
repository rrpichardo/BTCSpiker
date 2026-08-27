# Archive

## Week of 2026-07-28
Fixed 99.5% false-alert bug via canonical dedup in identity/replay layers. Validated /predict endpoint (0.104s) and optimized latency (DB index, guardrails 0.66–1.2s) and accuracy (outcomes JOIN fix). Fixed 15 PR #4 findings (JSX, SQL, atomicity, test dups) and infrastructure tuning (PyTorch NN, CPU cap). Designed verdict badge UX; resolved Codex vulnerabilities. Verified 403 backend + 16 frontend tests passing; merged to main.

## Week of 2026-07-21
Migrated Real-Time-Crypto-ML to public rrpichardo/BTCSpiker repo with comprehensive scrubbing. Designed full-stack prediction-dashboard (Kafka, React UI, read-only settings) with outcome-classification interface (6 states, BTC overlay, dual-axis price+score, 1m–24h range). Codex identified 6 critical bugs; iterated UX (confidence metrics, gridlines). Recovered historical EDA notebooks; initiated 12-task ML pipeline. Phase 2 MLflow bridge complete; Phase 0 verification initiated.

## Week of 2026-07-14
Migrated Real-Time-Crypto-ML to public rrpichardo/BTCSpiker repo with comprehensive school-trace scrubbing. Codex review identified 6 critical bugs; designed full-stack prediction-dashboard architecture (Kafka event-log, React UI, read-only settings) and iterated UX (confidence metrics, BTC overlay, gridlines). Recovered historical EDA notebooks; initiated 12-task ML pipeline foundation. Phase 2 MLflow bridge complete; Phase 0 verification initiated.
```