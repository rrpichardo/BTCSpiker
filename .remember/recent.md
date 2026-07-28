# Recent

```

# Recent

## 2026-07-26
Fixed materializer stuck due to Kafka consumer rebalance (stale UNKNOWN_MEMBER_ID); restarted container. Verified full 11-service stack healthy (API, materializer, UI endpoints operational).

## 2026-07-24
Fixed /predictions endpoint perf hang (missing database index predictions.feature_id causing O(n²) on 100k rows); added index + test achieving 87ms baseline. Verified full Docker stack (11 services, E2E working); UI refinements complete (gridlines, clarity labels); BTC price overlay scoped (3 backend + 3 frontend files). Restored PR #1 (claude/web-ui) via integration/web-ui-plus-ml merge.

## Identity Candidates
- IDENTITY CANDIDATE: End-to-end ML-crypto project ownership—migrated to public repo, designed full-stack prediction system (backend infra, Kafka pipeline, React UI), shipped with comprehensive validation and testing.