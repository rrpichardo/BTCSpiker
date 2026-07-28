# Archive

## Week of 2026-07-21
BTCSpiker Phase 2 MLflow bridge complete (E2E verified); phase 0 initiated. Codex/v1 pipeline executed phases 4-8 autonomously (244 tests; 0-5✓, 6-7✗, 8✓); installed 4 ML tools (optuna, lightgbm, xgboost, catboost) and corrected fold-0 training, validating 11-day tournament (1.5-3x lift). Diagnosed batch-logging bottleneck; linear model achieved 7/7 dev gates (0.2701 PR-AUC vs 0.1353 baseline); formalized development_gate.py validation (11 tests). Designed rolling-window PyTorch DL phase; prediction-dashboard UI refined (clarity, overlay, gridlines).

## Week of 2026-07-14
Migrated Real-Time-Crypto-ML to public rrpichardo/BTCSpiker repo with comprehensive school-trace scrubbing. Codex review identified 6 critical bugs; designed full-stack prediction-dashboard architecture (Kafka event-log, React UI, read-only settings) and iterated on UX (confidence metrics, BTC overlay, gridlines). Recovered historical EDA notebooks; initiated 12-task ML pipeline foundation. Phase 2 MLflow bridge complete; Phase 0 verification initiated.
```