CREATE SCHEMA IF NOT EXISTS cbb26_hf_export_staging;

CREATE TABLE IF NOT EXISTS cbb26_hf_export_staging.orderbook_checkpoints (
    product_id TEXT NOT NULL, checkpoint_hour TIMESTAMPTZ NOT NULL,
    source_sequence_num BIGINT NOT NULL, best_bid NUMERIC(18, 8) NOT NULL,
    best_ask NUMERIC(18, 8) NOT NULL, bid_book JSONB NOT NULL, ask_book JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), PRIMARY KEY (product_id, checkpoint_hour)
);
CREATE TABLE IF NOT EXISTS cbb26_hf_export_staging.orderbook_replay_anchors (
    product_id TEXT NOT NULL, anchor_second TIMESTAMPTZ NOT NULL,
    source_sequence_num BIGINT NOT NULL, best_bid NUMERIC(18, 8) NOT NULL,
    best_ask NUMERIC(18, 8) NOT NULL, bid_book JSONB NOT NULL, ask_book JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), PRIMARY KEY (product_id, anchor_second)
);
CREATE TABLE IF NOT EXISTS cbb26_hf_export_staging.orderbook_second_deltas (
    product_id TEXT NOT NULL, changed_second TIMESTAMPTZ NOT NULL,
    source_sequence_num_start BIGINT NOT NULL, source_sequence_num_end BIGINT NOT NULL,
    best_bid NUMERIC(18, 8) NOT NULL, best_ask NUMERIC(18, 8) NOT NULL,
    changes JSONB NOT NULL, change_count INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), PRIMARY KEY (product_id, changed_second)
);
CREATE TABLE IF NOT EXISTS cbb26_hf_export_staging.orderbook_replay_metadata (
    product_id TEXT NOT NULL, window_start TIMESTAMPTZ NOT NULL, window_end TIMESTAMPTZ NOT NULL,
    checkpoint_hour TIMESTAMPTZ, source_sequence_num_start BIGINT, source_sequence_num_end BIGINT,
    status TEXT NOT NULL, gap_count INTEGER NOT NULL DEFAULT 0, metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), PRIMARY KEY (product_id, window_start, window_end)
);
