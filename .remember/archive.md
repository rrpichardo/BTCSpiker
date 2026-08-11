# Archive

## Week of 2026-07-28
Fixed 99.5% false-alert bug via canonical dedup in identity and replay layers. Validated /predict endpoint (0.104s) in 11-service Docker stack. Optimized predictions accuracy (outcomes JOIN fix) and latency (DB index, deadline guardrails: 0.66–1.2s). Designed verdict badge UX (Hit/Miss/False alarm/Correct pass/Pending); verified 79 backend tests passing.

## Week of 2026-07-21
BTCSpiker Phase 2 MLflow bridge complete; Codex/v1 pipeline executed phases 4-8 autonomously (244 tests) with 4 ML tools and fold-0 correction (1.5-3x lift). Linear model hit 7/7 dev gates (0.2701 PR-AUC). Optimized /predictions endpoint (87ms baseline), fixed materializer Kafka stability, verified 11-service Docker E2E. UI refinements (gridlines, verdict badges); scoped BTC overlay; restored PR #1 via integration/web-ui-plus-ml.

## Week of 2026-07-14
Migrated Real-Time-Crypto-ML to public rrpichardo/BTCSpiker repo with comprehensive school-trace scrubbing. Codex review identified 6 critical bugs; designed full-stack prediction-dashboard architecture (Kafka event-log, React UI, read-only settings) and iterated UX (confidence metrics, BTC overlay, gridlines). Recovered historical EDA notebooks; initiated 12-task ML pipeline foundation. Phase 2 MLflow bridge complete; Phase 0 verification initiated.
```