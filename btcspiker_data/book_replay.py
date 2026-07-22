"""Deterministic reconstruction of CBB26 L2 books into causal second states."""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
from typing import Any

import pyarrow as pa

from .contracts import BookState, RAW_BOOK_COLUMNS
from .storage import PartitionRecord, write_partition_atomic


class BookReplayError(ValueError):
    """A source replay cannot produce trustworthy causal book states."""


def _field(row: object, name: str) -> Any:
    return row.get(name) if isinstance(row, Mapping) else getattr(row, name, None)


def _decimal(value: object, name: str) -> Decimal:
    try:
        return value if isinstance(value, Decimal) else Decimal(str(value))
    except Exception as error:  # source values are untrusted JSON/SQL values
        raise BookReplayError(f"invalid {name}") from error


def _utc(value: object, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
        raise BookReplayError(f"{name} must be UTC")
    return value


def _book(value: object, name: str) -> dict[Decimal, Decimal]:
    pairs = value.items() if isinstance(value, Mapping) else value
    if not isinstance(pairs, Iterable):
        raise BookReplayError(f"invalid {name}")
    result: dict[Decimal, Decimal] = {}
    try:
        for price, quantity in pairs:
            parsed_price, parsed_quantity = _decimal(price, "price"), _decimal(quantity, "quantity")
            if parsed_price <= 0 or parsed_quantity <= 0:
                raise BookReplayError("book levels must be positive")
            result[parsed_price] = parsed_quantity
    except (TypeError, ValueError) as error:
        raise BookReplayError(f"invalid {name}") from error
    if not result:
        raise BookReplayError(f"{name} is empty")
    return result


def _bbo(bids: Mapping[Decimal, Decimal], asks: Mapping[Decimal, Decimal]) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    if not bids or not asks:
        raise BookReplayError("book side is empty")
    best_bid, best_ask = max(bids), min(asks)
    if best_bid >= best_ask:
        raise BookReplayError("crossed book")
    return best_bid, bids[best_bid], best_ask, asks[best_ask]


def _is_gap(row: object) -> bool:
    status = str(_field(row, "status") or "").lower()
    return status in {"gap", "excluded", "incomplete", "missing", "error"} or int(_field(row, "gap_count") or 0) > 0


def _gap_windows(metadata: Iterable[object], start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    excluded: list[tuple[datetime, datetime]] = []
    for row in metadata:
        if not _is_gap(row):
            continue
        window_start = _utc(_field(row, "window_start"), "window_start")
        window_end = _utc(_field(row, "window_end"), "window_end")
        if window_end < window_start:
            raise BookReplayError("gap window regressed")
        if window_end > start and window_start <= end:
            excluded.append((max(start, window_start), min(end + timedelta(seconds=1), window_end)))
    return sorted(excluded)


def _anchor_time(row: object) -> datetime:
    value = _field(row, "anchor_second")
    if value is None:
        value = _field(row, "checkpoint_hour")
    return _utc(value, "anchor_second")


def replay_day(
    *,
    anchors: Iterable[object],
    deltas: Iterable[object],
    metadata: Iterable[object] = (),
    day_start: datetime,
    day_end: datetime | None = None,
    product_id: str | None = None,
) -> Iterator[BookState]:
    """Yield end-of-second states without manufacturing states across source gaps."""
    start = _utc(day_start, "day_start")
    all_anchors = list(anchors)
    product = product_id or next((_field(row, "product_id") for row in all_anchors if _field(row, "product_id")), None)
    if not isinstance(product, str) or not product:
        raise BookReplayError("missing product_id")
    anchor_rows = [row for row in all_anchors if _field(row, "product_id") == product]
    for row in anchor_rows:
        _anchor_time(row)
    initial_anchors = [row for row in anchor_rows if _anchor_time(row) <= start]
    if not initial_anchors:
        raise BookReplayError("missing anchor at or before day start")
    initial_anchor = max(initial_anchors, key=_anchor_time)

    delta_rows = [row for row in deltas if _field(row, "product_id") == product]
    for row in delta_rows:
        _utc(_field(row, "changed_second"), "changed_second")
    delta_rows.sort(key=lambda row: (_field(row, "changed_second"), _field(row, "source_sequence_num_start")))
    final = _utc(day_end, "day_end") if day_end is not None else max(
        [start, *(_field(row, "changed_second") for row in delta_rows), *(_anchor_time(row) for row in anchor_rows)]
    )
    if final < start:
        raise BookReplayError("day_end precedes day_start")
    anchor_events: dict[datetime, list[object]] = {}
    for row in anchor_rows:
        when = _anchor_time(row)
        if _anchor_time(initial_anchor) <= when <= final:
            anchor_events.setdefault(when, []).append(row)
    delta_events: dict[datetime, list[object]] = {}
    for row in delta_rows:
        when = _field(row, "changed_second")
        if _anchor_time(initial_anchor) < when <= final:
            delta_events.setdefault(when, []).append(row)
    gaps = _gap_windows(metadata, _anchor_time(initial_anchor), final)

    bids: dict[Decimal, Decimal] = {}
    asks: dict[Decimal, Decimal] = {}
    current: BookState | None = None
    last_sequence: int | None = None
    segment = 0
    needs_new_segment = False
    second = _anchor_time(initial_anchor)
    while second <= final:
        if any(window_start <= second < window_end for window_start, window_end in gaps):
            needs_new_segment = True
            if current is not None:
                current = None
                bids, asks = {}, {}
                last_sequence = None
            second += timedelta(seconds=1)
            continue

        anchors_at_second = anchor_events.get(second, ())
        if anchors_at_second:
            anchor = max(anchors_at_second, key=lambda row: int(_field(row, "source_sequence_num")))
            bids = _book(_field(anchor, "bid_book"), "bid_book")
            asks = _book(_field(anchor, "ask_book"), "ask_book")
            best_bid, bid_size, best_ask, ask_size = _bbo(bids, asks)
            if best_bid != _decimal(_field(anchor, "best_bid"), "best_bid") or best_ask != _decimal(_field(anchor, "best_ask"), "best_ask"):
                raise BookReplayError("source BBO mismatch")
            last_sequence = int(_field(anchor, "source_sequence_num"))
            if needs_new_segment:
                segment += 1
                needs_new_segment = False
            current = BookState(
                product, second, last_sequence, last_sequence, best_bid, bid_size, best_ask, ask_size, segment
            )

        for row in delta_events.get(second, ()):
            if current is None or last_sequence is None:
                continue
            sequence_start = int(_field(row, "source_sequence_num_start"))
            sequence_end = int(_field(row, "source_sequence_num_end"))
            if sequence_start > sequence_end or sequence_start <= last_sequence:
                raise BookReplayError("sequence regression")
            changes = _field(row, "changes")
            if not isinstance(changes, Sequence):
                raise BookReplayError("invalid changes")
            for change in changes:
                try:
                    side, raw_price, raw_quantity = change
                except (TypeError, ValueError) as error:
                    raise BookReplayError("invalid L2 change") from error
                price = _decimal(raw_price, "price")
                quantity = _decimal(raw_quantity, "quantity")
                if side not in {"bid", "offer"} or price <= 0 or quantity < 0:
                    raise BookReplayError("invalid L2 change")
                book = bids if side == "bid" else asks
                if quantity == 0:
                    book.pop(price, None)
                else:
                    book[price] = quantity
            best_bid, bid_size, best_ask, ask_size = _bbo(bids, asks)
            if best_bid != _decimal(_field(row, "best_bid"), "best_bid") or best_ask != _decimal(_field(row, "best_ask"), "best_ask"):
                raise BookReplayError("source BBO mismatch")
            last_sequence = sequence_end
            current = BookState(
                product, second, sequence_start, sequence_end, best_bid, bid_size, best_ask, ask_size, segment
            )

        if second >= start and current is not None:
            yield BookState(
                current.product_id, second, current.sequence_start, current.sequence_end,
                current.best_bid, current.bid_size, current.best_ask, current.ask_size, segment,
            )
        second += timedelta(seconds=1)


def publish_replay_day(
    states: Iterable[BookState], *, root: str | Path, source_revision: str, source_date: str, source: str = "cbb26"
) -> list[PartitionRecord]:
    """Publish derived L2 states as immutable hourly partitions.

    Raw deltas are intentionally published by the caller that retains the original
    change arrays; this function only receives the causal derived states.
    """
    grouped: dict[datetime, list[BookState]] = {}
    for state in states:
        hour = state.observed_through.replace(minute=0, second=0, microsecond=0)
        grouped.setdefault(hour, []).append(state)
    records: list[PartitionRecord] = []
    for hour, rows in sorted(grouped.items()):
        table = pa.table({
            "source": [source] * len(rows), "product_id": [row.product_id for row in rows],
            "observed_through": [row.observed_through for row in rows], "sequence_start": [row.sequence_start for row in rows],
            "sequence_end": [row.sequence_end for row in rows], "best_bid": [str(row.best_bid) for row in rows],
            "bid_size": [str(row.bid_size) for row in rows], "best_ask": [str(row.best_ask) for row in rows],
            "ask_size": [str(row.ask_size) for row in rows], "segment_id": [row.segment_id for row in rows],
            "changes_json": [json.dumps([])] * len(rows), "source_revision": [source_revision] * len(rows),
            "source_date": [source_date] * len(rows),
        })
        records.append(write_partition_atomic(table, root, "book_states", rows[0].product_id))
    return records


def publish_replay_partitions(
    *,
    deltas: Iterable[object],
    states: Iterable[BookState],
    root: str | Path,
    source_revision: str,
    source_date: str,
    source: str = "cbb26",
) -> list[PartitionRecord]:
    """Publish retained source changes and their causal reconstructed states.

    A delta cannot be published without its reconstructed quantities: accepting one
    would make the frozen raw-book schema misleading, so this deliberately fails
    closed when the caller supplied an excluded or otherwise unreplayed delta.
    """
    state_rows = list(states)
    records = publish_replay_day(
        state_rows, root=root, source_revision=source_revision, source_date=source_date, source=source
    )
    by_second = {state.observed_through: state for state in state_rows}
    grouped: dict[datetime, list[dict[str, object]]] = {}
    for delta in deltas:
        changed = _utc(_field(delta, "changed_second"), "changed_second")
        state = by_second.get(changed)
        changes = _field(delta, "changes")
        if not isinstance(changes, Sequence):
            raise BookReplayError("invalid changes")
        hour = changed.replace(minute=0, second=0, microsecond=0)
        grouped.setdefault(hour, []).append({
            "source": source, "product_id": state.product_id if state else _field(delta, "product_id"), "observed_through": changed,
            "sequence_start": int(_field(delta, "source_sequence_num_start")),
            "sequence_end": int(_field(delta, "source_sequence_num_end")),
            "best_bid": str(_field(delta, "best_bid")), "bid_size": str(state.bid_size) if state else None,
            "best_ask": str(_field(delta, "best_ask")), "ask_size": str(state.ask_size) if state else None,
            "segment_id": state.segment_id if state else None,
            "changes_json": json.dumps(changes, default=str, separators=(",", ":")),
            "source_revision": source_revision, "source_date": source_date,
        })
    for _, rows in sorted(grouped.items()):
        table = pa.table({name: [row[name] for row in rows] for name in RAW_BOOK_COLUMNS})
        records.append(write_partition_atomic(table, root, "book_deltas", rows[0]["product_id"]))
    return records
