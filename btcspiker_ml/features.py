"""One causal feature engine for batch materialization and live ingestion."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
import re
from typing import Iterable

import pandas as pd

from features.feature_funcs import (
    compute_future_vol,
    compute_midprice,
    compute_return,
    compute_rolling_stats,
    compute_spread,
    compute_spread_mean,
    compute_trade_intensity,
)


@dataclass(frozen=True)
class FeatureSet:
    feature_set_id: str
    columns: tuple[str, ...]
    windows_seconds: tuple[int, ...]
    max_lookback_seconds: int
    schema_version: str
    deployable: bool
    required_sources: tuple[str, ...]


_CORE_COLUMNS = (
    "log_return", "spread_bps", "vol_60s", "mean_return_60s",
    "trade_intensity_60s", "n_ticks_60s", "spread_mean_60s",
)
_MULTI_COLUMNS = tuple(
    column
    for window in (5, 15, 30, 60, 120, 300)
    for column in (
        f"return_{window}s", f"vol_{window}s", f"mean_return_{window}s",
        f"price_range_bps_{window}s", f"spread_mean_bps_{window}s",
        f"trade_intensity_{window}s", f"interarrival_mean_ms_{window}s",
        f"interarrival_std_ms_{window}s",
    )
)

FEATURE_SETS = {
    "core_v1": FeatureSet("core_v1", _CORE_COLUMNS, (60,), 60, "1", True, ("coinbase_ticker",)),
    "multi_window_v1": FeatureSet(
        "multi_window_v1", ("log_return", "spread_bps", *_MULTI_COLUMNS),
        (5, 15, 30, 60, 120, 300), 300, "2", True, ("coinbase_ticker",),
    ),
    "microstructure_v1": FeatureSet(
        "microstructure_v1",
        ("log_return", "spread_bps", "book_imbalance", "ewma_vol_fast", "ewma_vol_slow"),
        (5, 15, 30, 60, 120, 300), 300, "3", True,
        ("coinbase_ticker", "coinbase_level2"),
    ),
}

_BASE_RAW_COLUMNS = {"product_id", "timestamp", "price", "best_bid", "best_ask"}
_RAW_COLUMNS_BY_SET = {
    "core_v1": _BASE_RAW_COLUMNS,
    "multi_window_v1": _BASE_RAW_COLUMNS,
    "microstructure_v1": _BASE_RAW_COLUMNS | {"book_imbalance"},
}


def parse_timestamp(value: str) -> float:
    """Parse ISO timestamps with the nanosecond precision used by Coinbase."""
    value = re.sub(r"(\.\d{6})\d+", r"\1", value)
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


class FeatureEngine:
    """Stateful, event-time ordered feature and delayed-label generator."""

    def __init__(self, feature_set_id: str, horizon_seconds: float, threshold: float):
        if feature_set_id not in FEATURE_SETS:
            raise ValueError(f"unknown feature set: {feature_set_id}")
        self.feature_set = FEATURE_SETS[feature_set_id]
        self.horizon_seconds = horizon_seconds
        self.threshold = threshold
        self.buffer_max_age = max(600.0, self.feature_set.max_lookback_seconds + horizon_seconds)
        self.price_buffer: deque[dict[str, float]] = deque()
        self.spread_buffer: deque[dict[str, float]] = deque()
        self.timestamp_buffer: deque[float] = deque()
        self.pending: deque[dict[str, object]] = deque()

    def ingest(self, tick: dict) -> list[dict]:
        self._validate_tick(tick)
        timestamp = parse_timestamp(str(tick["timestamp"]))
        if self.timestamp_buffer and timestamp < self.timestamp_buffer[-1]:
            self._reset()

        price = float(tick["price"])
        bid = float(tick["best_bid"])
        ask = float(tick["best_ask"])
        midprice = compute_midprice(bid, ask)
        spread = compute_spread(bid, ask, midprice)
        previous_price = self.price_buffer[-1]["price"] if self.price_buffer else price

        self.price_buffer.append({"price": price, "ts": timestamp})
        self.spread_buffer.append({"spread_abs": spread["spread_abs"], "spread_bps": spread["spread_bps"], "ts": timestamp})
        self.timestamp_buffer.append(timestamp)
        self._prune(timestamp)

        row = self._feature_row(tick, timestamp, price, midprice, spread, previous_price)
        self.pending.append({"row": row, "ts": timestamp})
        return self._drain(timestamp)

    def drain_remaining(self) -> list[dict]:
        if not self.timestamp_buffer:
            return []
        return self._drain(self.timestamp_buffer[-1], force=True)

    def _feature_row(
        self, tick: dict, timestamp: float, price: float, midprice: float,
        spread: dict[str, float], previous_price: float,
    ) -> dict:
        row = {
            "product_id": str(tick["product_id"]),
            "timestamp": str(tick["timestamp"]),
            "price": price,
            "midprice": midprice,
            "log_return": compute_return(price, previous_price),
            "spread_abs": spread["spread_abs"],
            "spread_bps": spread["spread_bps"],
            "feature_set_id": self.feature_set.feature_set_id,
            "feature_schema_version": self.feature_set.schema_version,
        }
        if self.feature_set.feature_set_id == "core_v1":
            row.update(self._core_features())
        elif self.feature_set.feature_set_id == "multi_window_v1":
            row.update(self._multi_window_features(price, timestamp))
        else:
            raise ValueError(
                "microstructure_v1 requires Level 2 book features and cannot be materialized from ticker ticks"
            )
        return row

    def _core_features(self) -> dict[str, float | int]:
        rolling = compute_rolling_stats(self.price_buffer, 60)
        return {
            "vol_60s": rolling["vol"],
            "mean_return_60s": rolling["mean_return"],
            "n_ticks_60s": rolling["n_ticks"],
            "trade_intensity_60s": compute_trade_intensity(self.timestamp_buffer, 60),
            "spread_mean_60s": compute_spread_mean(self.spread_buffer, 60),
            "price_range_60s": rolling["price_range"],
        }

    def _multi_window_features(self, price: float, timestamp: float) -> dict[str, float]:
        result: dict[str, float] = {}
        for window in self.feature_set.windows_seconds:
            rolling = compute_rolling_stats(self.price_buffer, window)
            spread_window = [entry for entry in self.spread_buffer if entry["ts"] >= timestamp - window]
            timestamp_window = [entry for entry in self.timestamp_buffer if entry >= timestamp - window]
            first_price = next(entry["price"] for entry in self.price_buffer if entry["ts"] >= timestamp - window)
            interarrivals = [
                (timestamp_window[index] - timestamp_window[index - 1]) * 1000.0
                for index in range(1, len(timestamp_window))
            ]
            result.update({
                f"return_{window}s": compute_return(price, first_price),
                f"vol_{window}s": rolling["vol"],
                f"mean_return_{window}s": rolling["mean_return"],
                f"price_range_bps_{window}s": rolling["price_range"] / price * 10_000.0 if price else 0.0,
                f"spread_mean_bps_{window}s": sum(entry["spread_bps"] for entry in spread_window) / len(spread_window) if spread_window else 0.0,
                f"trade_intensity_{window}s": compute_trade_intensity(timestamp_window, window),
                f"interarrival_mean_ms_{window}s": sum(interarrivals) / len(interarrivals) if interarrivals else 0.0,
                f"interarrival_std_ms_{window}s": _population_std(interarrivals),
            })
        return result

    def _drain(self, timestamp: float, force: bool = False) -> list[dict]:
        emitted: list[dict] = []
        while self.pending:
            entry = self.pending[0]
            entry_timestamp = float(entry["ts"])
            if not force and timestamp - entry_timestamp < self.horizon_seconds:
                break
            self.pending.popleft()
            future_window = deque(
                value for value in self.price_buffer
                if entry_timestamp <= value["ts"] <= entry_timestamp + self.horizon_seconds
            )
            future_volatility = compute_future_vol(future_window, self.horizon_seconds)
            if future_volatility is None:
                continue
            emitted.append({
                **dict(entry["row"]),
                "future_vol_60s": future_volatility,
                "vol_spike": int(future_volatility > self.threshold),
            })
        return emitted

    def _prune(self, timestamp: float) -> None:
        cutoff = timestamp - self.buffer_max_age
        for buffer in (self.price_buffer, self.spread_buffer, self.timestamp_buffer):
            while buffer and (buffer[0]["ts"] if isinstance(buffer[0], dict) else buffer[0]) < cutoff:
                buffer.popleft()

    def _reset(self) -> None:
        self.price_buffer.clear()
        self.spread_buffer.clear()
        self.timestamp_buffer.clear()
        self.pending.clear()

    def _validate_tick(self, tick: dict) -> None:
        required = _RAW_COLUMNS_BY_SET[self.feature_set.feature_set_id]
        missing = sorted(required - tick.keys())
        if missing:
            raise ValueError(
                f"feature set {self.feature_set.feature_set_id} unavailable; missing raw columns: {missing}"
            )


def materialize_features(ticks: pd.DataFrame, feature_set_id: str) -> pd.DataFrame:
    """Materialize labelled rows from an event-time ordered raw tick frame."""
    required = _RAW_COLUMNS_BY_SET.get(feature_set_id)
    if required is None:
        raise ValueError(f"unknown feature set: {feature_set_id}")
    missing = sorted(required - set(ticks.columns))
    if missing:
        raise ValueError(f"feature set {feature_set_id} unavailable; missing raw columns: {missing}")
    engine = FeatureEngine(feature_set_id, horizon_seconds=60, threshold=0.000048)
    ordered = ticks.sort_values("timestamp", kind="mergesort")
    emitted = [row for tick in ordered.to_dict("records") for row in engine.ingest(tick)]
    return pd.DataFrame(emitted)


def _population_std(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
