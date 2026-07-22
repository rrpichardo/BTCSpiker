"""Fail-closed audit for public historical Coinbase data."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


QUALIFIED_SECONDS_MIN = 2_592_000
UTC = timezone.utc


@dataclass(frozen=True)
class QualityReport:
    status: str
    qualified_seconds: int
    calendar_span_seconds: int
    per_day: dict[str, dict[str, Any]]
    failures: tuple[str, ...]
    duplicate_count: int
    first_event: datetime | None
    last_event: datetime | None
    sequence_range: tuple[int, int] | None
    gap_incidents: tuple[dict[str, Any], ...]
    exclusions: tuple[dict[str, Any], ...]

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def _value(item: Any, name: str) -> Any:
    return item.get(name) if isinstance(item, dict) else getattr(item, name)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("audit timestamps must be UTC")
    return value


def _interval(value: Any) -> tuple[datetime, datetime]:
    if isinstance(value, dict):
        start, end = value["start"], value["end"]
    else:
        start, end = value
    _utc(start); _utc(end)
    if start >= end:
        raise ValueError("interval must be non-empty")
    return start, end


def _union(intervals: Iterable[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    merged: list[tuple[datetime, datetime]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _subtract(base: list[tuple[datetime, datetime]], cuts: list[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    result = base
    for cut_start, cut_end in _union(cuts):
        next_result = []
        for start, end in result:
            if cut_end <= start or end <= cut_start:
                next_result.append((start, end)); continue
            if start < cut_start:
                next_result.append((start, cut_start))
            if cut_end < end:
                next_result.append((cut_end, end))
        result = next_result
    return result


def _seconds(intervals: Iterable[tuple[datetime, datetime]]) -> int:
    return sum(int((end - start).total_seconds()) for start, end in intervals)


def _completion_days(manifest: Any) -> set[date]:
    complete = set()
    for item in getattr(manifest, "trade_day_completions", ()):
        item = item if isinstance(item, dict) else vars(item)
        if not item.get("trade_pages_complete", True):
            continue
        source_date = item.get("source_date")
        parsed = date.fromisoformat(source_date) if isinstance(source_date, str) else source_date
        start = datetime.combine(parsed, time.min, tzinfo=UTC)
        if item.get("day_start_epoch") != int(start.timestamp()) or item.get("day_end_epoch") != int((start + timedelta(days=1)).timestamp()):
            continue
        complete.add(parsed)
    return complete


def _write_report(report: QualityReport, directory: str | Path) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report.payload(), default=lambda value: value.isoformat(), sort_keys=True, indent=2)
    (path / "quality.json").write_text(payload + "\n", encoding="utf-8")
    lines = [f"# Historical Data Quality: {report.status}", "", f"Qualified seconds: {report.qualified_seconds}",
             f"Calendar span seconds: {report.calendar_span_seconds}", "", "## Failures", *[f"- {item}" for item in report.failures]]
    (path / "quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_dataset(manifest: Any, *, book_intervals: Iterable[Any] = (), trades: Iterable[Any] = (),
                  book_states: Iterable[Any] = (), replay_incidents: Iterable[Any] = (),
                  excluded_intervals: Iterable[Any] = (), invalid_intervals: Iterable[Any] = (),
                  label_windows: Iterable[Any] = (), output_dir: str | Path | None = None) -> QualityReport:
    """Return a strict PASS/FAIL report; invalid evidence never becomes a warning."""
    failures: set[str] = set()
    intervals = [_interval(value) for value in book_intervals]
    excluded = [_interval(value) for value in excluded_intervals]
    excluded += [_interval(value) for value in getattr(manifest, "excluded_intervals", ())]
    gaps = [_interval(value) for value in replay_incidents] + [_interval(value) for value in invalid_intervals]
    gaps += [_interval(value) for value in getattr(manifest, "sequence_incidents", ())]
    valid = _subtract(_union(intervals), excluded + gaps)

    for partition in getattr(manifest, "partitions", ()):
        path = partition.get("path") or partition.get("local_path")
        expected = partition.get("sha256")
        if path and expected:
            digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if digest != expected:
                failures.add("checksum_mismatch")

    state_rows = list(book_states)
    sequence_values: list[int] = []
    previous_state_time: datetime | None = None
    for state in state_rows:
        observed = _value(state, "observed_through")
        bid, ask = _value(state, "best_bid"), _value(state, "best_ask")
        if bid > ask:
            failures.add("crossed_bbo")
        if _value(state, "sequence_start") > _value(state, "sequence_end"):
            failures.add("invalid_book_state")
        if previous_state_time is not None and observed < previous_state_time:
            failures.add("book_state_leakage")
        previous_state_time = observed
        sequence_values.extend((_value(state, "sequence_start"), _value(state, "sequence_end")))

    trade_rows = list(trades)
    ids: set[str] = set(); duplicate_count = 0; previous_time: datetime | None = None
    for trade in trade_rows:
        event_time = _value(trade, "event_time")
        if previous_time is not None and event_time < previous_time:
            failures.add("out_of_order_event_time")
        previous_time = event_time
        trade_id = str(_value(trade, "trade_id"))
        if trade_id in ids:
            duplicate_count += 1
        ids.add(trade_id)
        if _value(trade, "price") <= 0 or _value(trade, "size") <= 0:
            failures.add("non_positive_trade")
    if duplicate_count:
        failures.add("duplicate_trade_ids")

    for window in label_windows:
        start, end = _interval(window)
        if any(start < gap_end and gap_start < end for gap_start, gap_end in excluded + gaps):
            failures.add("label_window_crosses_excluded_interval")

    calendar_days = {start.date() + timedelta(days=n) for start, end in intervals
                     for n in range((end.date() - start.date()).days + 1)
                     if start.date() + timedelta(days=n) < end.date() or end.time() != time.min}
    completions = _completion_days(manifest)
    per_day: dict[str, dict[str, Any]] = {}
    qualified = 0
    for day in sorted(calendar_days):
        start = datetime.combine(day, time.min, tzinfo=UTC); end = start + timedelta(days=1)
        seconds = _seconds(_subtract(valid, [(datetime.min.replace(tzinfo=UTC), start), (end, datetime.max.replace(tzinfo=UTC))]))
        complete = day in completions
        if not complete:
            seconds = 0; failures.add("trade_pages_complete")
        day_trades = [item for item in trade_rows if start <= _value(item, "event_time") < end]
        day_ids = [str(_value(item, "trade_id")) for item in day_trades]
        day_sequences = [value for state in state_rows if start <= _value(state, "observed_through") < end
                         for value in (_value(state, "sequence_start"), _value(state, "sequence_end"))]
        per_day[day.isoformat()] = {
            "valid_seconds": seconds, "trade_pages_complete": complete, "trade_count": len(day_trades),
            "first_event": min((_value(item, "event_time") for item in day_trades), default=None),
            "last_event": max((_value(item, "event_time") for item in day_trades), default=None),
            "duplicate_count": len(day_ids) - len(set(day_ids)),
            "sequence_range": [min(day_sequences), max(day_sequences)] if day_sequences else None,
            "gap_incidents": [{"start": a, "end": b} for a, b in gaps if a < end and start < b],
            "exclusions": [{"start": a, "end": b} for a, b in excluded if a < end and start < b],
        }
        qualified += seconds
    if qualified < QUALIFIED_SECONDS_MIN:
        failures.add("qualified_seconds_below_minimum")
    first_event = _value(trade_rows[0], "event_time") if trade_rows else None
    last_event = _value(trade_rows[-1], "event_time") if trade_rows else None
    calendar_span = int((max(end for _, end in intervals) - min(start for start, _ in intervals)).total_seconds()) if intervals else 0
    report = QualityReport("PASS" if not failures else "FAIL", qualified, calendar_span, per_day,
                           tuple(sorted(failures)), duplicate_count, first_event, last_event,
                           (min(sequence_values), max(sequence_values)) if sequence_values else None,
                           tuple({"start": a, "end": b} for a, b in gaps), tuple({"start": a, "end": b} for a, b in excluded))
    if output_dir is not None:
        _write_report(report, output_dir)
    return report
