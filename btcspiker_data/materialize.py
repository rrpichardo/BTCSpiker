"""Causal historical trade/book joins and segmented feature materialization."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass

import pandas as pd

from btcspiker_data.contracts import BookState, MODEL_TICK_COLUMNS
from btcspiker_ml.features import materialize_features


_FEATURE_SET_IDS = ("core_v1", "multi_window_v1", "microstructure_v1")


def join_trades_to_books(
    trades: pd.DataFrame | Iterable[Mapping[str, object]],
    book_states: pd.DataFrame | Iterable[BookState | Mapping[str, object]],
) -> pd.DataFrame:
    """Attach each trade to a book state known before its UTC second began."""
    trade_frame = _to_frame(trades)
    book_frame = _to_frame(book_states)
    _require_columns(trade_frame, {"product_id", "trade_id", "event_time", "price", "size", "reported_side"}, "trades")
    _require_columns(
        book_frame,
        {"product_id", "observed_through", "best_bid", "bid_size", "best_ask", "ask_size", "segment_id"},
        "book states",
    )
    if "segment_id" not in trade_frame:
        trade_frame["segment_id"] = 0
    if trade_frame["trade_id"].duplicated().any():
        raise ValueError("duplicate trade_id")

    trade_frame = trade_frame.copy()
    book_frame = book_frame.copy()
    trade_frame["timestamp"] = pd.to_datetime(trade_frame.pop("event_time"), utc=True)
    book_frame["book_observed_through"] = pd.to_datetime(book_frame.pop("observed_through"), utc=True)
    book_frame["safe_at"] = book_frame["book_observed_through"] + pd.Timedelta(seconds=1)

    left = trade_frame.sort_values(["timestamp", "product_id", "segment_id"], kind="mergesort")
    right = book_frame.sort_values(["safe_at", "product_id", "segment_id"], kind="mergesort")
    joined = pd.merge_asof(
        left,
        right,
        by=["product_id", "segment_id"],
        left_on="timestamp",
        right_on="safe_at",
        direction="backward",
        allow_exact_matches=True,
        suffixes=("", "_book"),
    )
    joined = joined.dropna(subset=["book_observed_through"])
    if not joined.empty and not (joined["book_observed_through"] < joined["timestamp"].dt.floor("s")).all():
        raise AssertionError("causal join included a book observed during the trade second")

    joined = joined.rename(columns={"size": "trade_size"})
    return joined.loc[:, MODEL_TICK_COLUMNS].sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def materialize_segmented_features(
    trades: pd.DataFrame | Iterable[Mapping[str, object]],
    book_states: pd.DataFrame | Iterable[BookState | Mapping[str, object]],
) -> dict[str, pd.DataFrame]:
    """Materialize every feature set independently inside each continuity segment."""
    ticks = join_trades_to_books(trades, book_states)
    outputs: dict[str, pd.DataFrame] = {}
    for feature_set_id in _FEATURE_SET_IDS:
        pieces = []
        for _, segment in ticks.groupby("segment_id", sort=True):
            features = materialize_features(segment, feature_set_id)
            if not features.empty:
                features["segment_id"] = segment["segment_id"].iloc[0]
                pieces.append(features)
        outputs[feature_set_id] = (
            pd.concat(pieces, ignore_index=True).sort_values("timestamp", kind="mergesort").reset_index(drop=True)
            if pieces
            else pd.DataFrame()
        )
    return outputs


def _to_frame(rows: pd.DataFrame | Iterable[object]) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        return rows.copy()
    return pd.DataFrame([asdict(row) if is_dataclass(row) else dict(row) for row in rows])


def _require_columns(frame: pd.DataFrame, required: set[str], description: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{description} missing required columns: {missing}")
