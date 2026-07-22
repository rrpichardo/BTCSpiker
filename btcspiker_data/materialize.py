"""Causal historical trade/book joins and segmented feature materialization."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
import hashlib
import os
from pathlib import Path
import tempfile

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
        for (_, segment_id), segment in ticks.groupby(
            ["product_id", "segment_id"], sort=True
        ):
            features = materialize_features(segment, feature_set_id)
            if not features.empty:
                features["segment_id"] = segment_id
                pieces.append(features)
        outputs[feature_set_id] = (
            pd.concat(pieces, ignore_index=True)
            .sort_values(
                ["timestamp", "product_id", "segment_id"], kind="mergesort"
            )
            .reset_index(drop=True)
            if pieces
            else pd.DataFrame()
        )
    return outputs


def write_feature_outputs_atomic(
    outputs: Mapping[str, pd.DataFrame], root: str | os.PathLike[str]
) -> dict[str, Path]:
    """Persist the three feature tables as immutable content-addressed Parquet."""
    provided = set(outputs)
    required = set(_FEATURE_SET_IDS)
    if provided != required:
        raise ValueError(
            f"feature outputs must contain exactly {sorted(required)}; got {sorted(provided)}"
        )

    paths: dict[str, Path] = {}
    root_path = Path(root)
    for feature_set_id in _FEATURE_SET_IDS:
        destination_dir = (
            root_path / "features" / f"feature_set={feature_set_id}"
        )
        destination_dir.mkdir(parents=True, exist_ok=True)
        file_descriptor, temp_name = tempfile.mkstemp(
            prefix=".part-", suffix=".parquet.tmp", dir=destination_dir
        )
        os.close(file_descriptor)
        temp_path = Path(temp_name)
        try:
            outputs[feature_set_id].to_parquet(temp_path, index=False)
            _fsync_file(temp_path)
            digest = _sha256(temp_path)
            destination = destination_dir / f"part-{digest}.parquet"
            if destination.exists():
                if _sha256(destination) != digest:
                    raise ValueError(
                        f"content digest mismatch for existing feature output: {destination}"
                    )
                temp_path.unlink()
            else:
                os.replace(temp_path, destination)
                _fsync_directory(destination_dir)
            paths[feature_set_id] = destination
        finally:
            if temp_path.exists():
                temp_path.unlink()
    return paths


def _to_frame(rows: pd.DataFrame | Iterable[object]) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        return rows.copy()
    return pd.DataFrame([asdict(row) if is_dataclass(row) else dict(row) for row in rows])


def _require_columns(frame: pd.DataFrame, required: set[str], description: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{description} missing required columns: {missing}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as file_handle:
        os.fsync(file_handle.fileno())


def _fsync_directory(path: Path) -> None:
    file_descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)
