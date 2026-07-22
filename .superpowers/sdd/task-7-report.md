# Task 7: Audit Coverage and Enforce the 30-Day Gate

## Status

Complete. The audit is fail-closed and reports only `PASS` or `FAIL`.

## RED evidence

`pytest tests/data/test_quality.py -v` was run before the implementation.
It failed during collection with `ModuleNotFoundError: No module named
'btcspiker_data.quality'`, which is the expected missing-module failure.

## GREEN evidence

`pytest tests/data/test_quality.py tests/data/test_raw_manifest.py
tests/data/test_contracts.py -v` passed: **20 passed**.

## Delivered

- `QualityReport`, `audit_dataset()`, and the exact `2_592_000` second gate.
- Union-of-interval coverage less gaps, invalid intervals, and exclusions.
- Per-UTC-day `trade_pages_complete` enforcement using identity-bound deterministic
  trade completion evidence in `RawDatasetManifest`.
- Strict failures for checksum mismatches, invalid/crossed book evidence,
  out-of-order event time, duplicate IDs, non-positive trade values, and label
  windows intersecting excluded data.
- Machine-readable `quality.json` and human-readable `quality.md` output.

## Concern

The auditor accepts fixture-provided intervals and event records. The production
orchestration layer must pass the persisted partition sidecars/replay incidents
and an output directory beside its manifest when it publishes an audit.

## Review-fix RED evidence

`pytest tests/data/test_quality.py -v` reproduced the review findings with
**8 failed, 7 passed**. The failures proved that missing completion flags were
accepted, joined ticks were unsupported, remote-only partitions bypassed
verification, input order set top-level event bounds, coverage/report assertions
were unmet, and Markdown omitted per-day evidence.

A second focused RED test reproduced a malformed naïve timestamp crashing during
per-day aggregation after it had already been classified as malformed.

## Review-fix GREEN evidence

`pytest tests/data/test_quality.py tests/data/test_raw_manifest.py
tests/data/test_contracts.py -v` passed: **30 passed**.

The final gate requires an explicit boolean `trade_pages_complete is True`,
validates completion identity/day boundaries and optional counters, audits causal
joined-tick timestamps and segments, rejects unverifiable partitions, renders
full per-day Markdown evidence, uses chronological event bounds, and converts
malformed evidence into named failures wherever the report can remain useful.
