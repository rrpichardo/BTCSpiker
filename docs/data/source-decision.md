# Free Coinbase Historical Data Source Decision

## Decision

Use acquisition mode `hybrid-free` for BTC-USD research history:

- CBB26 provides pinned Coinbase L2 replay dumps for the order book.
- Coinbase Advanced Trade's public market-trades endpoint provides matching trades.
- The canonical normalized store is the authenticated user's private Hugging Face dataset repository, `<authenticated-name>/btcspiker-coinbase-history`.
- Local disk is a transient, one-day working cache. It is not the canonical store.

This design has a hard cash-spend ceiling of `$0`. It does not authorize paid gap filling, storage, API plans, or trials that convert to paid service.

## Frozen source contract

| Item | Frozen value |
|---|---|
| Product | `BTC-USD` |
| Acquisition window | `2026-04-24` through `2026-05-28`, inclusive (35 UTC days) |
| CBB26 repository | `deusmos/cbb26-timeseries-db` |
| CBB26 revision | `c1e89eded9915e1c75a18911298edfbbbe4050ce` |
| CBB26 license metadata | `other` |
| Coinbase endpoint | `/api/v3/brokerage/market/products/{product_id}/ticker` |
| Public request ceiling | 8 requests per second |
| Publication gate | At least `2,592,000` qualified seconds (30 day-equivalents) |
| Usage scope | `research_unverified` |

The 35-day source window provides a five-day buffer. Calendar span does not count as coverage: replay gaps, incomplete trade pagination, invalid sequences, checksum failures, and excluded intervals reduce qualified duration.

## Storage and credentials

Authenticate locally with `hf auth login` using a Hugging Face token with dataset write access. The token is never a command-line argument, manifest field, log field, or chat input. Before every upload, the pipeline verifies that the destination uses the authenticated namespace and remains private.

Each CBB26 day is downloaded and restored separately. The normalized hourly Parquet files are content-addressed and uploaded to the private repository. A daily dump is removed only after all 72 expected hourly artifacts (24 each for deltas, states, and trades) have exact, commit-pinned remote checksum receipts. Upload failures retain the local inputs for recovery.

The raw manifest records source revision, schemas, remote partitions, checksums, trade-pagination completion, coverage, and exclusions. Feature materialization writes an adjacent lineage sidecar; the normal existing-dataset binder carries that lineage into its deterministic dataset manifest.

## Rights boundary

CBB26 currently declares license metadata as `other`. This implementation does not establish commercial redistribution or production-use rights. Commercial use remains unapproved until a separate license review is completed. Keep derived repositories private and treat the corpus as research-only in the meantime.

## Primary references

- [CBB26 dataset](https://huggingface.co/datasets/deusmos/cbb26-timeseries-db)
- [Coinbase public market trades](https://docs.cdp.coinbase.com/api-reference/advanced-trade-api/rest-api/public/get-public-market-trades)
- [Hugging Face upload guidance](https://huggingface.co/docs/huggingface_hub/guides/upload)
