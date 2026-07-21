# BTCSpiker Free Coinbase History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, zero-cash historical-data pipeline that downloads at least 30 qualified day-equivalents of Coinbase BTC-USD trades and L2 order-book data, preserves a feature-flexible raw schema, and publishes a feature Parquet accepted through `BTCSPIKER_EXISTING_DATA`.

**Architecture:** Pin and download 35 UTC days (`2026-04-24` through `2026-05-28`) of CBB26 BTC-USD replay shards, backfill matching Coinbase public trades, normalize both sources into immutable hourly Parquet partitions, and causally join each trade to the last fully observed L2 second. Restore each PostgreSQL custom dump into an ephemeral PostgreSQL 16 staging database, reconstruct BBO quantities from anchors and deltas, and fail closed on source gaps, sequence regressions, checksum mismatches, or fewer than 30 qualified day-equivalents.

**Tech Stack:** Python 3.11+, pandas, PyArrow, requests, psycopg 3, Hugging Face Hub/Xet, PostgreSQL 16 in Docker Compose, pytest.

## Global Constraints

- Spend ceiling is `$0`; CoinAPI, Kaiko, CoinDesk, Tardis trials, and any paid storage require a separate user authorization.
- Pin CBB26 dataset revision `c1e89eded9915e1c75a18911298edfbbbe4050ce`.
- Acquire product `BTC-USD` for UTC dates `2026-04-24` through `2026-05-28`, inclusive.
- Publish only when qualified coverage is at least `2_592_000` seconds (30 days); preserve all incidents and excluded windows in the manifest.
- Preserve source values and semantics. Do not reinterpret Coinbase's reported trade side as maker or aggressor side.
- Do not use the `best_bid` or `best_ask` returned alongside the historical trades response as historical quotes.
- CBB26 rows represent end-of-second book state. A trade at second `S` may only consume book state from `S-1` or earlier to prevent intrasecond look-ahead.
- Source downloads and generated datasets live under `data/` and remain excluded from git.
- Tests use small checked-in fixtures and mocked HTTP/subprocess boundaries; the standard test suite must not download multi-gigabyte files.
- This pipeline remains independent of the current ML tournament and must not mutate MLflow or promote models.
- The CBB26 market-data license remains `other`; manifests must record `usage_scope="research_unverified"` until commercial-use rights are reviewed.

---

## File Structure and Ownership

| Owner | Files | Responsibility |
|---|---|---|
| Main agent | `btcspiker_data/contracts.py`, `tests/data/conftest.py` | Freeze cross-agent types, schemas, fixtures, and invariants before parallel work begins. |
| Agent A | `btcspiker_data/cbb26.py`, `btcspiker_data/book_replay.py`, `btcspiker_data/sql/cbb26_staging.sql`, `docker-compose.data.yaml`, matching tests | Discover/download/restore CBB26 shards and replay L2 books. |
| Agent B | `btcspiker_data/coinbase_trades.py`, `btcspiker_data/materialize.py`, matching tests | Backfill trades, causally join trades and book state, materialize ML inputs. |
| Agent C | `btcspiker_data/storage.py`, `btcspiker_data/quality.py`, `btcspiker_data/raw_manifest.py`, matching tests | Atomic Parquet partitions, checksums, coverage incidents, deterministic raw manifest. |
| Main agent | `scripts/download_coinbase_history.py`, `scripts/materialize_coinbase_history.py`, `btcspiker_ml/datasets.py`, integration tests, docs | Integrate the three workstreams and publish the experiment-compatible dataset. |

Agents must work in separate `codex/` worktrees during implementation. They may not edit another owner's files. The main agent cherry-picks or merges each reviewed task, runs contract tests after every merge, and alone resolves interface changes.

## Parallel Execution Topology

```text
Sequential foundation: Task 1
          |
          +-------------------+-------------------+
          |                   |                   |
Wave 1: Task 2 / Agent A  Task 3 / Agent B  Task 4 / Agent C
          |                   |                   |
          +-------------------+-------------------+
                              |
Wave 2: Task 5 / Agent A  Task 6 / Agent B  Task 7 / Agent C
          |                   |                   |
          +-------------------+-------------------+
                              |
Sequential integration and verification: Tasks 8-9 / Main agent
```

Use the following minimum-strength assignments; do not start every agent at medium reasoning:

| Agent | Tasks | Model | Reasoning effort | Escalation rule |
|---|---|---|---|---|
| Agent A | Tasks 2 and 5: CBB26 download, PostgreSQL restore, L2 replay | `gpt-5.6-terra` | `medium` | Increase to `high` only if a real pinned shard conflicts with its sidecar or documented staging schema. |
| Agent B | Tasks 3 and 6: Coinbase trades, causal join, feature materialization | `gpt-5.6-terra` | `low` | Increase to `medium` only after a focused failing test demonstrates an unresolved pagination or causal-time defect. |
| Agent C | Tasks 4 and 7: storage, manifests, coverage audit | `gpt-5.6-terra` | `low` | Increase to `medium` only after deterministic fixture tests expose a cross-module contract ambiguity. |

The main agent retains its inherited model for source selection, schema changes, causal-time policy, merge review, and completion claims. No stronger subagent is authorized merely because a task is slow or verbose.

### Task 1: Freeze the Shared Raw-Data Contract

**Execution:** Main agent, sequential. All parallel tasks wait for this commit.

**Files:**
- Create: `btcspiker_data/__init__.py`
- Create: `btcspiker_data/contracts.py`
- Create: `tests/data/__init__.py`
- Create: `tests/data/conftest.py`
- Create: `tests/data/test_contracts.py`
- Modify: `requirements-dev.txt`

**Interfaces:**
- Produces: `BookDelta`, `BookState`, `TradeEvent`, `QualityIncident`, `RAW_BOOK_COLUMNS`, `RAW_TRADE_COLUMNS`, `MODEL_TICK_COLUMNS`, `validate_utc_window()`.
- Consumers: Tasks 2-8 import these names without redefining them.

- [ ] **Step 1: Write the contract tests**

```python
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from btcspiker_data.contracts import BookState, TradeEvent, validate_utc_window


def test_book_state_rejects_crossed_market():
    with pytest.raises(ValueError, match="crossed book"):
        BookState(
            product_id="BTC-USD",
            observed_through=datetime(2026, 4, 24, tzinfo=timezone.utc),
            sequence_start=10,
            sequence_end=11,
            best_bid=Decimal("101"),
            bid_size=Decimal("1"),
            best_ask=Decimal("100"),
            ask_size=Decimal("1"),
        )


def test_trade_event_preserves_reported_side():
    event = TradeEvent(
        product_id="BTC-USD",
        trade_id="42",
        event_time=datetime(2026, 4, 24, tzinfo=timezone.utc),
        price=Decimal("90000"),
        size=Decimal("0.01"),
        reported_side="SELL",
        source="coinbase_public_trades",
    )
    assert event.reported_side == "SELL"


def test_window_requires_utc():
    with pytest.raises(ValueError, match="UTC"):
        validate_utc_window(
            datetime(2026, 4, 24),
            datetime(2026, 4, 25, tzinfo=timezone.utc),
        )
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `pytest tests/data/test_contracts.py -v`

Expected: collection fails because `btcspiker_data.contracts` does not exist.

- [ ] **Step 3: Implement the frozen contracts**

```python
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal


RAW_BOOK_COLUMNS = (
    "source", "product_id", "observed_through", "sequence_start",
    "sequence_end", "best_bid", "bid_size", "best_ask", "ask_size",
    "changes_json", "source_revision", "source_date",
)
RAW_TRADE_COLUMNS = (
    "source", "product_id", "trade_id", "event_time", "price", "size",
    "reported_side", "side_semantics", "source_date",
)
MODEL_TICK_COLUMNS = (
    "product_id", "timestamp", "price", "best_bid", "best_ask",
    "bid_size", "ask_size", "trade_id", "trade_size", "reported_side",
    "book_observed_through", "segment_id",
)


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError("timestamp must be UTC")


def validate_utc_window(start: datetime, end: datetime) -> None:
    _require_utc(start)
    _require_utc(end)
    if start >= end:
        raise ValueError("start must be before end")


@dataclass(frozen=True)
class BookState:
    product_id: str
    observed_through: datetime
    sequence_start: int
    sequence_end: int
    best_bid: Decimal
    bid_size: Decimal
    best_ask: Decimal
    ask_size: Decimal
    segment_id: int = 0

    def __post_init__(self) -> None:
        _require_utc(self.observed_through)
        if self.best_bid > self.best_ask:
            raise ValueError("crossed book")
        if self.bid_size < 0 or self.ask_size < 0:
            raise ValueError("book quantities must be non-negative")


@dataclass(frozen=True)
class BookDelta:
    product_id: str
    changed_second: datetime
    sequence_start: int
    sequence_end: int
    best_bid: Decimal
    best_ask: Decimal
    changes: tuple[tuple[str, Decimal, Decimal], ...]

    def __post_init__(self) -> None:
        _require_utc(self.changed_second)
        if self.sequence_start > self.sequence_end:
            raise ValueError("sequence range regressed")
        if any(side not in {"bid", "offer"} or quantity < 0 for side, _, quantity in self.changes):
            raise ValueError("invalid L2 change")


@dataclass(frozen=True)
class TradeEvent:
    product_id: str
    trade_id: str
    event_time: datetime
    price: Decimal
    size: Decimal
    reported_side: str
    source: str
    side_semantics: str = "coinbase_reported_unspecified"

    def __post_init__(self) -> None:
        _require_utc(self.event_time)
        if self.price <= 0 or self.size <= 0:
            raise ValueError("trade price and size must be positive")


@dataclass(frozen=True)
class QualityIncident:
    code: str
    start: datetime
    end: datetime
    severity: str
    detail: str
```

- [ ] **Step 4: Add development dependencies**

Add exactly these lines to `requirements-dev.txt`:

```text
huggingface-hub>=0.34,<2
hf-xet>=1.1,<2
psycopg[binary]>=3.2,<4
pyarrow>=14,<22
```

- [ ] **Step 5: Add deterministic fixtures**

Create fixtures representing one anchor, three delta seconds, four trades, an overlapping trade page, a sequence regression, and a ten-second explicit source gap. Use `Decimal` values and UTC timestamps; no network calls or generated randomness.

- [ ] **Step 6: Run and commit**

Run: `pytest tests/data/test_contracts.py -v`

Expected: all contract tests pass.

Run:

```bash
git add btcspiker_data/__init__.py btcspiker_data/contracts.py tests/data requirements-dev.txt
git commit -m "feat(data): freeze historical market data contracts"
```

### Task 2: Download and Restore Pinned CBB26 Shards

**Execution:** Agent A, parallel Wave 1.

**Files:**
- Create: `btcspiker_data/cbb26.py`
- Create: `btcspiker_data/sql/cbb26_staging.sql`
- Create: `docker-compose.data.yaml`
- Create: `tests/data/test_cbb26.py`

**Interfaces:**
- Consumes: `QualityIncident` and UTC validation from Task 1.
- Produces: `CBB26Shard`, `list_btc_shards()`, `download_shard()`, `restore_shard()`, `read_sidecar()`.

- [ ] **Step 1: Test exact pinned inventory and idempotent downloads**

```python
def test_inventory_requires_every_requested_day(fake_hf_tree):
    shards = list_btc_shards(
        fake_hf_tree,
        start=date(2026, 4, 24),
        end=date(2026, 5, 28),
        product="BTC-USD",
    )
    assert len(shards) == 35


def test_download_reuses_matching_local_file(tmp_path, shard, fake_download):
    target = download_shard(shard, tmp_path, fake_download)
    first_digest = sha256_file(target)
    target_again = download_shard(shard, tmp_path, fake_download)
    assert target_again == target
    assert sha256_file(target_again) == first_digest
    assert fake_download.calls == 1
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `pytest tests/data/test_cbb26.py -v`

Expected: imports fail because `btcspiker_data.cbb26` does not exist.

- [ ] **Step 3: Implement pinned discovery and streaming download**

Define:

```python
CBB26_REPO = "deusmos/cbb26-timeseries-db"
CBB26_REVISION = "c1e89eded9915e1c75a18911298edfbbbe4050ce"
DEFAULT_START = date(2026, 4, 24)
DEFAULT_END = date(2026, 5, 28)

@dataclass(frozen=True)
class CBB26Shard:
    trade_date: date
    product_id: str
    dump_path: str
    sidecar_path: str
    expected_size: int
```

Use `HfApi.list_repo_tree(..., recursive=True, revision=CBB26_REVISION)` and `hf_hub_download()` with the exact revision. Require one `.dump` and one `.json` for every date. Download to `downloads/cbb26/<revision>/<date>/BTC-USD.*`, compare the local byte count with both the Hub tree and sidecar, compute SHA-256, and never replace a matching file.

- [ ] **Step 4: Define a plain PostgreSQL staging schema**

Create schema `cbb26_hf_export_staging` and the four documented tables with the exact columns from the pinned CBB26 migrations. Do not install TimescaleDB; the staging database exists only to restore and read one daily shard at a time.

- [ ] **Step 5: Implement Docker-backed restoration**

`docker-compose.data.yaml` must run `postgres:16-alpine`, bind only `127.0.0.1:55432`, use database/user `btcspiker`, mount the download root read-only at `/imports`, and expose a `pg_isready` health check. `restore_shard()` must truncate staging tables, invoke `pg_restore --data-only --no-owner --no-privileges`, and compare restored table counts to the sidecar before returning.

- [ ] **Step 6: Test restore command construction and count mismatch behavior**

Mock subprocess and psycopg boundaries. Assert that a non-zero `pg_restore` exit, wrong row count, missing anchor, or a row outside the sidecar UTC bounds raises `CBB26IntegrityError` without writing normalized output.

- [ ] **Step 7: Run and commit**

Run: `pytest tests/data/test_cbb26.py -v`

Expected: all CBB26 tests pass.

Run:

```bash
git add btcspiker_data/cbb26.py btcspiker_data/sql/cbb26_staging.sql docker-compose.data.yaml tests/data/test_cbb26.py
git commit -m "feat(data): add pinned CBB26 shard acquisition"
```

### Task 3: Backfill Coinbase Public Trades Safely

**Execution:** Agent B, parallel Wave 1.

**Files:**
- Create: `btcspiker_data/coinbase_trades.py`
- Create: `tests/data/test_coinbase_trades.py`

**Interfaces:**
- Consumes: `TradeEvent` from Task 1.
- Produces: `CoinbaseTradeClient`, `iter_day_trades()`, and `TradePageStalledError`.

- [ ] **Step 1: Write pagination, overlap, and rate-limit tests**

```python
def test_backfill_deduplicates_inclusive_end_overlap(fake_trade_session):
    trades = list(client(fake_trade_session).iter_day_trades(date(2026, 4, 24)))
    assert [trade.trade_id for trade in trades] == ["100", "99", "98", "97"]


def test_backfill_rejects_page_that_cannot_advance(stalled_trade_session):
    with pytest.raises(TradePageStalledError):
        list(client(stalled_trade_session).iter_day_trades(date(2026, 4, 24)))


def test_response_bbo_is_not_copied_into_trade_events(fake_trade_session):
    event = next(client(fake_trade_session).iter_day_trades(date(2026, 4, 24)))
    assert not hasattr(event, "best_bid")
```

- [ ] **Step 2: Verify focused tests fail**

Run: `pytest tests/data/test_coinbase_trades.py -v`

Expected: import failure for `btcspiker_data.coinbase_trades`.

- [ ] **Step 3: Implement a bounded public client**

Use endpoint `https://api.coinbase.com/api/v3/brokerage/market/products/BTC-USD/ticker`, `limit=1000`, integer UTC `start` and `end`, timeout `(5, 30)`, `User-Agent: BTCSpiker-research/1`, and an eight-requests-per-second token bucket. Parse only `trades`; discard response-level BBO.

- [ ] **Step 4: Implement backward pagination**

For each UTC day, request descending pages. Deduplicate by `trade_id`, retain only `day_start <= event_time < day_end`, and set the next inclusive `end` to the epoch second of the oldest returned trade. If two consecutive pages produce no new trade IDs or the oldest event time does not move backward, raise `TradePageStalledError`. Mark the day complete only after a successful page reaches `day_start`; persist that completion evidence in the manifest. Sort final output by `(event_time, trade_id)` before partitioning.

- [ ] **Step 5: Preserve side semantics**

Map the API's `side` to `reported_side` and write constant `side_semantics="coinbase_reported_unspecified"`. Do not populate `maker_side` or `aggressor_side`.

- [ ] **Step 6: Test retries**

Retry `429`, `500`, `502`, `503`, and `504` with capped exponential backoff and `Retry-After`; do not retry other `4xx` responses. After five attempts, raise an exception containing date, page end, and status code.

- [ ] **Step 7: Run and commit**

Run: `pytest tests/data/test_coinbase_trades.py -v`

Expected: all trade-client tests pass.

Run:

```bash
git add btcspiker_data/coinbase_trades.py tests/data/test_coinbase_trades.py
git commit -m "feat(data): add Coinbase historical trade backfill"
```

### Task 4: Write Immutable Partitions and Raw Manifests

**Execution:** Agent C, parallel Wave 1.

**Files:**
- Create: `btcspiker_data/storage.py`
- Create: `btcspiker_data/raw_manifest.py`
- Create: `tests/data/test_storage.py`
- Create: `tests/data/test_raw_manifest.py`

**Interfaces:**
- Consumes: ordered column constants and `QualityIncident` from Task 1.
- Produces: `PartitionRecord`, `write_partition_atomic()`, `RawDatasetManifest`, `raw_manifest_id()`, `publish_raw_manifest()`.

- [ ] **Step 1: Write atomicity and determinism tests**

```python
def test_partition_path_is_content_addressed(tmp_path, trade_table):
    left = write_partition_atomic(trade_table, tmp_path, "trades", "BTC-USD")
    right = write_partition_atomic(trade_table, tmp_path, "trades", "BTC-USD")
    assert left.path == right.path
    assert left.sha256 == right.sha256


def test_manifest_id_ignores_dictionary_order(raw_manifest):
    assert raw_manifest_id(raw_manifest) == raw_manifest_id(reordered(raw_manifest))
```

- [ ] **Step 2: Verify focused tests fail**

Run: `pytest tests/data/test_storage.py tests/data/test_raw_manifest.py -v`

Expected: import failures for the new modules.

- [ ] **Step 3: Implement hourly Parquet publication**

Write to a same-directory temporary file, close and fsync it, compute SHA-256, rename to `part-<sha256>.parquet`, and fsync the parent directory. Use layout:

```text
raw/kind=<book_deltas|book_states|trades>/source=<source>/product=BTC-USD/date=YYYY-MM-DD/hour=HH/part-<sha256>.parquet
```

Refuse unordered columns, naive timestamps, duplicate stable keys, or an existing content-addressed path whose digest does not match its name.

- [ ] **Step 4: Implement a deterministic manifest**

The manifest must record source revision, source URL, `usage_scope`, ordered schemas, every partition path/row count/SHA-256, coverage seconds, missing seconds, duplicate counts, sequence incidents, excluded intervals, and creation metadata outside the hashed identity payload. Hash canonical JSON without `created_at` so identical inputs produce the same dataset ID.

- [ ] **Step 5: Run and commit**

Run: `pytest tests/data/test_storage.py tests/data/test_raw_manifest.py -v`

Expected: all storage and manifest tests pass.

Run:

```bash
git add btcspiker_data/storage.py btcspiker_data/raw_manifest.py tests/data/test_storage.py tests/data/test_raw_manifest.py
git commit -m "feat(data): publish immutable raw data partitions"
```

### Task 5: Replay L2 Books into Causal Per-Second States

**Execution:** Agent A, parallel Wave 2 after Tasks 1, 2, and 4 merge.

**Files:**
- Create: `btcspiker_data/book_replay.py`
- Create: `tests/data/test_book_replay.py`
- Modify: `btcspiker_data/cbb26.py`

**Interfaces:**
- Consumes: restored anchor/checkpoint/delta rows and `write_partition_atomic()`.
- Produces: `replay_day() -> Iterator[BookState]` and book-delta/book-state partitions.

- [ ] **Step 1: Write replay correctness tests**

Test that the baseline anchor initializes both sides, a zero quantity removes a level, the BBO quantity follows the reconstructed dictionary, output timestamps are ordered, and recomputed BBO equals each delta row's `best_bid`/`best_ask`.

- [ ] **Step 2: Verify focused tests fail**

Run: `pytest tests/data/test_book_replay.py -v`

Expected: import failure for `btcspiker_data.book_replay`.

- [ ] **Step 3: Implement deterministic replay**

Select the latest replay anchor at or before `day_start`, load `bid_book` and `ask_book` into `dict[Decimal, Decimal]`, and apply each `[side, price, new_quantity]` change in sequence order. Quantity zero deletes the level. After every changed second, recompute BBO and quantities; fail on mismatch with the source BBO, sequence regression, negative quantity, missing anchor, or crossed book.

- [ ] **Step 4: Carry state only across observed continuity**

Use replay metadata to mark source gap windows. Emit per-second carry-forward states only outside those windows. Do not interpolate through gaps. Assign a new `segment_id` after every excluded interval.

- [ ] **Step 5: Retain both raw and derived forms**

Publish the original `changes` arrays in `book_deltas` partitions and the reconstructed `best_bid`, `bid_size`, `best_ask`, and `ask_size` in `book_states` partitions. This preserves future depth-feature flexibility while giving the current feature engine a compact input.

- [ ] **Step 6: Run and commit**

Run: `pytest tests/data/test_book_replay.py tests/data/test_cbb26.py -v`

Expected: all replay and acquisition tests pass.

Run:

```bash
git add btcspiker_data/book_replay.py btcspiker_data/cbb26.py tests/data/test_book_replay.py
git commit -m "feat(data): reconstruct causal Coinbase L2 states"
```

### Task 6: Join Trades to Books and Materialize Features

**Execution:** Agent B, parallel Wave 2 after Tasks 1, 3, and 4 merge. Use Task 1 fixtures rather than waiting for Task 5's real outputs.

**Files:**
- Create: `btcspiker_data/materialize.py`
- Create: `tests/data/test_materialize_history.py`

**Interfaces:**
- Consumes: trade partitions and Task 1's `BookState` schema.
- Produces: `join_trades_to_books()` and `materialize_segmented_features()`.

- [ ] **Step 1: Write a no-look-ahead test**

```python
def test_trade_uses_last_fully_observed_book_second(book_states, trades):
    joined = join_trades_to_books(trades, book_states)
    row = joined.loc[joined["trade_id"] == "100"].iloc[0]
    assert row["book_observed_through"] < row["timestamp"].floor("s")
```

Also test that trades before the first safe book state are excluded, no join crosses a segment boundary, duplicate trade IDs fail, and output contains every `MODEL_TICK_COLUMNS` field.

- [ ] **Step 2: Verify focused tests fail**

Run: `pytest tests/data/test_materialize_history.py -v`

Expected: import failure for `btcspiker_data.materialize`.

- [ ] **Step 3: Implement the causal join**

Convert each `BookState.observed_through` to `safe_at = observed_through + 1 second`. Use `pandas.merge_asof(..., direction="backward", by=["product_id", "segment_id"], left_on="timestamp", right_on="safe_at", allow_exact_matches=True)`. Assert `book_observed_through < floor(timestamp)` for every joined row.

- [ ] **Step 4: Materialize each continuity segment independently**

For each `segment_id`, rename `trade_size` and BBO fields into the tick contract, call `btcspiker_ml.features.materialize_features()` separately, and concatenate outputs in UTC order. This prevents rolling windows and 60-second labels from crossing a source gap.

- [ ] **Step 5: Produce all three feature sets**

Write immutable feature outputs for `core_v1`, `multi_window_v1`, and `microstructure_v1`. The experiment handoff defaults to `core_v1`; the other tables prove that retained raw fields support the planned optimization work.

- [ ] **Step 6: Run and commit**

Run: `pytest tests/data/test_materialize_history.py tests/ml/test_feature_engine.py -v`

Expected: all historical materialization and existing feature-engine tests pass.

Run:

```bash
git add btcspiker_data/materialize.py tests/data/test_materialize_history.py
git commit -m "feat(data): materialize causal historical features"
```

### Task 7: Audit Coverage and Enforce the 30-Day Gate

**Execution:** Agent C, parallel Wave 2 after Tasks 1 and 4 merge.

**Files:**
- Create: `btcspiker_data/quality.py`
- Create: `tests/data/test_quality.py`
- Modify: `btcspiker_data/raw_manifest.py`

**Interfaces:**
- Consumes: partition records, sidecars, replay incidents, and trade summaries.
- Produces: `QualityReport`, `audit_dataset()`, `QUALIFIED_SECONDS_MIN = 2_592_000`.

- [ ] **Step 1: Write strict quality-gate tests**

Test failures for fewer than 30 qualified day-equivalents, checksum mismatch, out-of-order event time, duplicate trade IDs, non-positive prices/sizes, crossed BBO, book-state leakage, and a 60-second label window that crosses an excluded interval.

- [ ] **Step 2: Verify focused tests fail**

Run: `pytest tests/data/test_quality.py -v`

Expected: import failure for `btcspiker_data.quality`.

- [ ] **Step 3: Implement coverage accounting**

Count the union of valid one-second book intervals, subtract explicit replay gaps and invalid intervals, and require at least `2_592_000` seconds. Report calendar span separately from qualified duration so a 35-day range with gaps cannot masquerade as 35 complete days.

Require `trade_pages_complete=true` for every included UTC date. A date with a stalled, rate-limited, or prematurely terminated trade backfill contributes zero qualified seconds even when its L2 book is present.

- [ ] **Step 4: Emit machine-readable and human-readable reports**

Write `quality.json` and `quality.md` beside the manifest. Include per-day valid seconds, trade count, first/last event, duplicate count, sequence range, gap incidents, exclusions, and a final `PASS` or `FAIL`. Never convert a failure to a warning.

- [ ] **Step 5: Run and commit**

Run: `pytest tests/data/test_quality.py tests/data/test_raw_manifest.py -v`

Expected: all quality tests pass.

Run:

```bash
git add btcspiker_data/quality.py btcspiker_data/raw_manifest.py tests/data/test_quality.py
git commit -m "feat(data): enforce historical data quality gate"
```

### Task 8: Integrate CLIs and the Existing-Dataset Boundary

**Execution:** Main agent, sequential after Wave 2 merges and interface review.

**Files:**
- Create: `scripts/download_coinbase_history.py`
- Create: `scripts/materialize_coinbase_history.py`
- Create: `tests/data/test_history_cli.py`
- Modify: `btcspiker_ml/datasets.py`
- Modify: `tests/ml/test_datasets.py`
- Create: `docs/data/source-decision.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: all Task 2-7 public interfaces.
- Produces: a raw dataset ID, qualified feature Parquet, extended existing-dataset manifest, and the exact `BTCSPIKER_EXISTING_DATA` export command.

- [ ] **Step 1: Write CLI integration tests**

Use mocked source clients and a temporary data root. Assert `download_coinbase_history.py` defaults to the pinned revision/product/35-day window, refuses an unpinned revision, resumes existing verified partitions, and exits non-zero on a failed quality report.

- [ ] **Step 2: Implement the download CLI**

Required arguments:

```text
--data-root PATH
--start 2026-04-24
--end 2026-05-28
--product BTC-USD
--revision c1e89eded9915e1c75a18911298edfbbbe4050ce
--max-rps 8
```

The command prints the raw dataset ID, manifest path, quality-report path, downloaded/reused file counts, bytes, and qualified seconds. It does not start model training.

- [ ] **Step 3: Implement the materialization CLI**

Accept `--raw-manifest`, `--feature-set`, and `--output-root`; verify every input checksum before reading; publish `features.parquet`; invoke `inspect_existing_dataset()`; and print:

```text
export BTCSPIKER_EXISTING_DATA=$(pwd)/data/coinbase_history/features/core_v1/features.parquet
```

- [ ] **Step 4: Extend the feature manifest without breaking current callers**

Add optional `parent_dataset_id`, `source_manifest_path`, `feature_set_id`, `feature_engine_git_sha`, and `excluded_intervals` fields with defaults to `DatasetManifest`. Preserve deterministic hashing and existing sample-manifest tests.

- [ ] **Step 5: Record the source decision**

Document acquisition mode `hybrid-free`, canonical store `local disk under user-selected --data-root`, CBB26 revision/date/license, Coinbase public-trade endpoint, `$0` ceiling, 35-day buffer, 30-day qualified gate, and the rule that commercial use remains unapproved.

- [ ] **Step 6: Run and commit**

Run: `pytest tests/data tests/ml/test_datasets.py tests/ml/test_manifest.py -v`

Expected: all new integration tests and existing dataset/manifest tests pass.

Run:

```bash
git add scripts/download_coinbase_history.py scripts/materialize_coinbase_history.py tests/data/test_history_cli.py btcspiker_ml/datasets.py tests/ml/test_datasets.py docs/data/source-decision.md README.md
git commit -m "feat(data): integrate free Coinbase history pipeline"
```

### Task 9: End-to-End Verification and Handoff

**Execution:** Main agent only. No agent may claim completion before this gate.

**Files:**
- Modify: `docs/runbook.md`
- Create: `docs/data/free-coinbase-history-verification.md`

- [ ] **Step 1: Run the complete unit and integration suite**

Run: `pytest -q`

Expected: zero failures.

- [ ] **Step 2: Run a one-day real-source smoke test**

Run the acquisition command for `2026-04-24` only into a temporary data root. Verify the Hub revision, sidecar, dump size, restore counts, trade pagination, replay, causal join, and manifest. This smoke test is allowed to fail the 30-day publication gate; it must fail for exactly that reason after all one-day integrity checks pass.

- [ ] **Step 3: Run the full 35-day acquisition**

Run:

```bash
python scripts/download_coinbase_history.py \
  --data-root data/coinbase_history \
  --start 2026-04-24 \
  --end 2026-05-28 \
  --product BTC-USD \
  --revision c1e89eded9915e1c75a18911298edfbbbe4050ce \
  --max-rps 8
```

Expected: quality `PASS` with at least `2_592_000` qualified seconds. If it fails, preserve the report and stop; do not purchase gap-fill data or weaken thresholds.

- [ ] **Step 4: Materialize and bind the dataset**

Run `scripts/materialize_coinbase_history.py` for `core_v1`, then run:

```bash
BTCSPIKER_EXISTING_DATA=$(pwd)/data/coinbase_history/features/core_v1/features.parquet \
python scripts/bind_existing_dataset.py --config experiment.yaml
```

Expected: both commands print the same immutable parent/raw lineage and the binder accepts the dataset.

- [ ] **Step 5: Verify future feature readiness**

Materialize `multi_window_v1` and `microstructure_v1` from the same raw manifest. Record row counts, exclusions, and ordered columns; do not start a tournament.

- [ ] **Step 6: Document exact evidence**

Record source revision, manifest IDs, paths, sizes, SHA-256 values, qualified duration, incidents, pytest summary, smoke/full commands, and whether commercial-use review remains outstanding.

- [ ] **Step 7: Commit verification docs**

Run:

```bash
git add docs/runbook.md docs/data/free-coinbase-history-verification.md
git commit -m "docs(data): verify free Coinbase history pipeline"
```

## Agent Review Gates

After each agent task, the main agent must:

1. Read the diff and agent summary.
2. Reject any edit outside assigned ownership.
3. Run that task's focused tests.
4. Run `pytest tests/data/test_contracts.py -q` to detect interface drift.
5. Merge only after both suites pass.
6. Send interface changes back to all dependent agents before Wave 2 starts.

After Wave 2, run the full `tests/data` suite before touching `btcspiker_ml`. This ordering prevents source-specific behavior from leaking into the experiment package.

## Token and Time Controls

- Start Agent A at Terra medium and Agents B-C at Terra low, exactly as specified in the assignment table.
- Spawn every implementation agent with `fork_turns="none"`; construct a self-contained prompt from Task 1's frozen interfaces, the assigned task section, and the owned-file list.
- Give agents only Task 1's contracts, their assigned task text, and owned file list; do not fork the full conversation history.
- Limit each agent to one implementation task per turn and require a concise diff/test summary.
- Use three agents only during the two explicit waves; sequential integration stays with the main agent.
- Do not ask multiple agents to independently inspect the entire repository or redesign the schema.
- If an agent needs an interface change, it reports the proposed signature and stops instead of editing shared contracts.
- Prefer deterministic fixtures and focused tests over repeated live downloads.
- Escalation requires concrete evidence from a failing focused test or real-shard contract mismatch; uncertainty alone is not sufficient.

This arrangement will normally use more aggregate tokens than single-agent execution, but materially less than unrestricted parallel exploration because context duplication and overlapping work are controlled.
