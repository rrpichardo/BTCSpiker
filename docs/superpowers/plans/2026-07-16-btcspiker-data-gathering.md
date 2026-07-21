# BTCSpiker Deferred Data-Gathering Plan

**Status:** Deferred and independent. Do not execute this plan until the user chooses a data source and storage approach.

**Purpose:** Expand or replace the collected BTCSpiker corpus later without blocking, pausing, or changing the current experimentation `/goal`.

**Relationship to experimentation:** The active experimentation plan is `docs/superpowers/plans/2026-07-16-btcspiker-goal-experimentation.md`. It uses the user's already-collected data and completes on its own. This data-gathering plan has no authority to pause, resume, restart, or modify that goal.

## Fixed Boundary Contract

Any future data pipeline must publish one immutable dataset version that the experimentation system can consume through:

```text
BTCSPIKER_EXISTING_DATA=/absolute/path/to/features.parquet
```

It must also publish a JSON manifest containing:

- `dataset_id`: SHA-256 of canonical manifest content;
- source, exchange, product, and source-license identifiers;
- absolute or object-store partition locations plus SHA-256 per partition;
- schema version and ordered columns;
- UTC event-time start and end;
- row count, duration, cadence, and target prevalence;
- missing, duplicate, non-finite, out-of-order, and gap statistics;
- target version and feature-engine Git SHA;
- creation time and optional parent dataset ID.

The experimentation system must not know whether the files came from Tardis, Coinbase, a VPS collector, R2, iCloud, or a manual transfer. The manifest and feature schema are the only integration boundary.

## Decisions the User Will Make Later

### Acquisition mode

Choose exactly one:

1. **Historical vendor:** Purchase or subscribe to a provider such as Tardis and import existing Coinbase BTC-USD trades, quotes, and order-book updates.
2. **Live collection:** Run the public Coinbase collector continuously and wait for a new contiguous corpus.
3. **Hybrid:** Import vendor history and continue the same normalized schema with a live Coinbase collector.

The default recommendation is hybrid when immediate backtesting and ongoing live validation are both worth the cost. No purchase or subscription is authorized by this document.

### Durable storage

Choose exactly one canonical store:

1. **Cloudflare R2:** recommended for a remote collector because it exposes an S3-compatible API.
2. **Local disk:** acceptable for a temporary manual workflow with verified backups.
3. **Another cloud provider:** acceptable when it supports private object storage, checksums, lifecycle rules, and programmatic credentials.

iCloud may be a secondary personal archive, but it is not required by the experimentation system and should not be the direct multi-writer target of a Linux collector.

### Collection depth

Choose the lowest-cost schema that supports the intended features:

- trades only;
- trades plus top-of-book quotes and quantities;
- trades plus incremental L2 order-book updates.

BTCSpiker's current target can be labelled from trades. Spread and imbalance features require quotes; depth features require L2 data.

## Implementation Tasks After Those Decisions

### Task D1: Record the source and storage decision

Create `docs/data/source-decision.md` containing the chosen acquisition mode, canonical store, retained channels, date range, recurring cost ceiling, license constraints, credential owner, retention policy, and cancellation procedure.

Exit criterion: the document names one source and one canonical store; no alternatives remain active.

### Task D2: Implement a source adapter behind a fixed event schema

The adapter must emit:

```python
MARKET_EVENT_COLUMNS = [
    "source",
    "channel",
    "product_id",
    "timestamp",
    "sequence_num",
    "trade_id",
    "price",
    "size",
    "maker_side",
    "best_bid",
    "best_ask",
    "best_bid_quantity",
    "best_ask_quantity",
]
```

Unavailable fields must be null and documented; fields may not be silently reinterpreted. The adapter must deduplicate stable event IDs, preserve source event time, record receive time separately when available, and fail closed on sequence regression.

Exit criterion: a bounded one-hour import produces normalized Parquet and a quality report without changing the source files.

### Task D3: Publish immutable partitions

Partition by source, product, UTC date, and UTC hour. Write locally to a temporary file, close and checksum it, upload or atomically move it to the canonical store, verify the destination checksum, then publish the manifest entry. Never upload one object per tick.

Expected layout:

```text
raw/source=<source>/product=BTC-USD/date=YYYY-MM-DD/hour=HH/part-<sha256>.parquet
manifests/raw-<dataset_id>.json
```

Exit criterion: re-running the same import is idempotent and produces no duplicate partition or changed dataset ID.

### Task D4: Materialize the experiment-compatible feature dataset

Run the shared BTCSpiker feature engine and fixed 60-second trade-price target over the normalized raw events. Publish immutable feature partitions and the boundary manifest expected by `BTCSPIKER_EXISTING_DATA`.

Exit criterion: `scripts/bind_existing_dataset.py` accepts the produced feature table without a source-specific code path.

### Task D5: Verify continuity and hand off a new dataset ID

Verify schema, UTC ordering, duplicate keys, gaps, source coverage, quote/trade availability, incomplete 60-second label horizons, non-finite values, target prevalence by day, and partition checksums. Record every known incident rather than imputing it silently.

Exit criterion: the handoff contains an absolute local feature path, immutable `dataset_id`, manifest path, coverage report, and exact command for starting a new experimentation search. It does not resume or mutate a completed prior search.

## Explicit Non-Goals

- This plan does not run model experiments.
- It does not modify the 60-second target.
- It does not auto-start a server or collector.
- It does not purchase data or cloud services.
- It does not change MLflow state or promote a model.
- It does not block completion of the current experimentation `/goal`.
