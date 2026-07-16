"""
Pure feature functions — no Kafka, no I/O.

All functions are stateless: they take raw scalar values and/or deque buffers
as arguments and return scalars or dicts. Safe to use in featurizer.py,
replay.py, tests, and notebooks.

Buffer convention
-----------------
Buffers are collections.deque of dicts with at minimum:
  {"price": float, "bid": float, "ask": float, "ts": float}   # unix epoch seconds
  {"price": float, "ts": float}                                 # trade-only variant

`window_sec` / `horizon_sec` are durations in seconds.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Sequence


# ---------------------------------------------------------------------------
# Scalar helpers
# ---------------------------------------------------------------------------

def compute_midprice(bid: float, ask: float) -> float:
    """Arithmetic mid-price."""
    return (bid + ask) / 2.0


def compute_return(p_curr: float, p_prev: float) -> float:
    """Log return: ln(p_curr / p_prev).  Returns 0.0 if p_prev is zero."""
    if p_prev <= 0:
        return 0.0
    return math.log(p_curr / p_prev)


def compute_spread(bid: float, ask: float, mid: float) -> dict[str, float]:
    """
    Absolute and relative (basis-point) bid-ask spread.

    Returns
    -------
    {"spread_abs": float, "spread_bps": float}
    """
    spread_abs = ask - bid
    spread_bps = (spread_abs / mid * 10_000) if mid > 0 else 0.0
    return {"spread_abs": spread_abs, "spread_bps": spread_bps}


def compute_ob_imbalance(bid_size: float, ask_size: float) -> float:
    """
    Order-book imbalance in [-1, 1].
      +1  →  all size on the bid (buying pressure)
      -1  →  all size on the ask (selling pressure)
    Returns 0.0 when both sides are zero.
    """
    total = bid_size + ask_size
    if total <= 0:
        return 0.0
    return (bid_size - ask_size) / total


# ---------------------------------------------------------------------------
# Rolling-window stats
# ---------------------------------------------------------------------------

def _window_slice(buffer: deque, window_sec: float) -> list[dict]:
    """Return buffer entries whose 'ts' falls within the last `window_sec`."""
    if not buffer:
        return []
    cutoff = buffer[-1]["ts"] - window_sec
    return [e for e in buffer if e["ts"] >= cutoff]


def compute_rolling_stats(buffer: deque, window_sec: float) -> dict[str, float]:
    """
    Compute rolling volatility and mean return over `window_sec`.

    Requires buffer entries with keys: {"price": float, "ts": float}.

    Returns
    -------
    {
        "vol":         float,   # std of log-returns (0.0 if < 2 prices)
        "mean_return": float,   # mean log-return
        "n_ticks":     int,     # number of ticks in window
        "price_range": float,   # max(price) - min(price) in window
    }
    """
    window = _window_slice(buffer, window_sec)
    n = len(window)

    if n < 2:
        return {"vol": 0.0, "mean_return": 0.0, "n_ticks": n, "price_range": 0.0}

    returns = [
        compute_return(window[i]["price"], window[i - 1]["price"])
        for i in range(1, n)
    ]

    mean_r = sum(returns) / len(returns)
    # Population variance (divides by n, not n-1). This is intentional: with
    # ~240 ticks in a typical 60s window the bias is negligible, and using
    # population variance keeps the metric stable for very short windows (n=2).
    variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)

    prices = [e["price"] for e in window]
    return {
        "vol": math.sqrt(variance),
        "mean_return": mean_r,
        "n_ticks": n,
        "price_range": max(prices) - min(prices),
    }


def compute_spread_mean(buffer: deque, window_sec: float) -> float:
    """
    Mean absolute spread over the last `window_sec`.

    Requires buffer entries with keys: {"spread_abs": float, "ts": float}.
    Returns 0.0 when the window is empty.
    """
    window = _window_slice(buffer, window_sec)
    if not window:
        return 0.0
    return sum(e["spread_abs"] for e in window) / len(window)


def compute_trade_intensity(timestamps: Sequence[float], window_sec: float) -> float:
    """
    Trades per second within the last `window_sec`.

    Parameters
    ----------
    timestamps : sequence of unix epoch floats (newest last)
    window_sec : look-back window in seconds
    """
    if not timestamps:
        return 0.0
    cutoff = timestamps[-1] - window_sec
    count = sum(1 for t in timestamps if t >= cutoff)
    return count / window_sec if window_sec > 0 else 0.0


def compute_future_vol(buffer: deque, horizon_sec: float) -> float | None:
    """
    Realised volatility over the *next* `horizon_sec` starting from the
    oldest entry in `buffer`.

    Used for label generation during replay / batch feature building.
    Returns None when the buffer does not span the full horizon.

    Requires buffer entries with keys: {"price": float, "ts": float}.
    """
    if len(buffer) < 2:
        return None

    t_start = buffer[0]["ts"]
    t_end   = t_start + horizon_sec

    window = [e for e in buffer if t_start <= e["ts"] <= t_end]
    if len(window) < 2:
        return None

    # Check we actually cover the horizon. Allow up to 2s of slack at the
    # boundary — ticks arrive at irregular intervals (~250ms) so the last
    # tick in the slice will typically land just short of t_end.
    if window[-1]["ts"] < t_end - 2.0:
        return None

    returns = [
        compute_return(window[i]["price"], window[i - 1]["price"])
        for i in range(1, len(window))
    ]
    mean_r = sum(returns) / len(returns)
    # Population variance — consistent with compute_rolling_stats (see note there)
    variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
    return math.sqrt(variance)
