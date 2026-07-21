# Review Status — Free Coinbase History Plan

**Plan file:** [2026-07-21-btcspiker-free-coinbase-history.md](2026-07-21-btcspiker-free-coinbase-history.md)
**Status:** Not approved. **Do not execute as written.**
**Reviewed:** 2026-07-21

## Why

An adversarial review flagged five load-bearing assumptions the plan never
verifies before writing ~2,000 lines of code around them. Two were checked
against the codebase and confirmed:

1. **The 30-day gate ignores gaps.** `btcspiker_ml/qualification.py:341`
   computes `coverage_days` as `(end_time - start_time) / 86_400` — pure
   calendar span. Task 4 and Task 7 build an elaborate
   qualified-seconds accounting system that terminates in a JSON file
   nothing downstream reads.

2. **Train/serve skew is structural.** `core_v1` declares
   `required_sources = ("coinbase_ticker",)` — one source, live per-tick
   quotes. The plan trains by joining two sources where every trade in a
   given second shares one frozen ~2-second-stale spread. `spread_bps` and
   `spread_mean_60s` are two of the seven features.

Three further assumptions were plausible but not verified: whether the
Coinbase endpoint honors deep historical windows at all (one `curl` to
check, plan checks it in Task 9); clock alignment between trade timestamps
(exchange-side) and CBB26 book timestamps (third-party collector, likely
receipt-time); and per-tick feature-engine runtime at 14M ticks
(materialize_features is a pure-Python O(N × W) loop).

## The option not on the plan

Point `scripts/ws_ingest.py` at Coinbase and let it run for 30 days. Zero
engineering, zero train/serve skew (it *is* the serving path), $0. Cost:
30 days of calendar. That trade-off is the real decision.

## Do not delete

The plan doc is preserved because it documents (a) the reasoning path that
was rejected and (b) the specific gaps that any future data-acquisition
work has to close. If the collector-forward path is later ruled out, the
first hour of any successor plan should verify the five assumptions
above — see the critique in the session record.
