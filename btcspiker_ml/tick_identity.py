"""Canonical tick-redelivery identity, shared by every dedupe site.

One key, used identically by the offline replay builder (``scripts/replay.py``),
the Kafka replay producer (``scripts/replay_to_kafka.py``), and the live
serving guard (``features/featurizer.py``) — so "is this tick a repeat" means
the same thing everywhere instead of three independently-drifting definitions.

``volume_24_h`` is deliberately excluded: it is a rolling 24-hour ticker field
that changes on nearly every message and is not a model feature (absent from
``FEATURE_COLS``). Including it would let a redelivered tick that differs only
in ``volume_24_h`` slip past the guard and still produce a duplicate feature
row — the exact failure this key exists to catch.
"""
from __future__ import annotations

TickKey = tuple[object, object, object, object, object]


def tick_dedupe_key(tick: dict) -> TickKey:
    """(product_id, timestamp, price, best_bid, best_ask) — the fields that
    affect feature computation, and only those."""
    return (
        tick.get("product_id"),
        tick.get("timestamp"),
        tick.get("price"),
        tick.get("best_bid"),
        tick.get("best_ask"),
    )
