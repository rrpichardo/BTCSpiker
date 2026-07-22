"""Fail-closed audit for public historical Coinbase data."""
from __future__ import annotations

import hashlib
import json
import re
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


def _optional_value(item: Any, name: str, default: Any = None) -> Any:
    return item.get(name, default) if isinstance(item, dict) else getattr(item, name, default)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("audit timestamps must be UTC")
    return value


def _is_utc_datetime(value: Any) -> bool:
    return isinstance(value, datetime) and value.tzinfo is not None and value.utcoffset() == UTC.utcoffset(value)


def _interval(value: Any) -> tuple[datetime, datetime]:
    if isinstance(value, dict):
        start, end = value["start"], value["end"]
    else:
        start, end = value
    _utc(start); _utc(end)
    if start >= end:
        raise ValueError("interval must be non-empty")
    return start, end


def _collect_intervals(values: Iterable[Any], failures: set[str]) -> list[tuple[datetime, datetime]]:
    intervals = []
    for value in values:
        try:
            intervals.append(_interval(value))
        except (AttributeError, KeyError, TypeError, ValueError):
            failures.add("malformed_interval_evidence")
    return intervals


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


def _completion_days(manifest: Any, failures: set[str]) -> set[date]:
    complete = set()
    for item in getattr(manifest, "trade_day_completions", ()):
        try:
            item = item if isinstance(item, dict) else vars(item)
        except TypeError:
            failures.add("invalid_trade_completion_evidence")
            continue
        try:
            source_date = item["source_date"]
            parsed = date.fromisoformat(source_date) if isinstance(source_date, str) else source_date
            if not isinstance(parsed, date) or isinstance(parsed, datetime):
                raise ValueError("source_date")
            start = datetime.combine(parsed, time.min, tzinfo=UTC)
            end = start + timedelta(days=1)
            valid = (
                item.get("trade_pages_complete") is True
                and isinstance(item.get("product_id"), str)
                and bool(item["product_id"])
                and type(item.get("day_start_epoch")) is int
                and type(item.get("day_end_epoch")) is int
                and item["day_start_epoch"] == int(start.timestamp())
                and item["day_end_epoch"] == int(end.timestamp())
            )
            for count_name in ("page_count", "trade_count"):
                if count_name in item:
                    valid = valid and type(item[count_name]) is int and item[count_name] >= 0
            if "completed_through" in item:
                completed = item["completed_through"]
                if isinstance(completed, str):
                    completed = datetime.fromisoformat(completed.replace("Z", "+00:00"))
                elif type(completed) is int:
                    completed = datetime.fromtimestamp(completed, UTC)
                valid = valid and _is_utc_datetime(completed) and completed == end
        except (KeyError, TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            failures.add("invalid_trade_completion_evidence")
            continue
        complete.add(parsed)
    return complete


def _verified_partition(partition: Any, failures: set[str]) -> None:
    if not isinstance(partition, dict):
        failures.add("partition_unverified")
        return
    expected = partition.get("sha256")
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-fA-F]{64}", expected) is None:
        failures.add("partition_unverified")
        return
    local = partition.get("local_path") or partition.get("path")
    if local:
        try:
            local_path = Path(local)
            if not local_path.is_file():
                failures.add("partition_unverified")
                return
            if hashlib.sha256(local_path.read_bytes()).hexdigest() != expected.lower():
                failures.add("checksum_mismatch")
        except (OSError, TypeError, ValueError):
            failures.add("partition_unverified")
        return
    receipt = partition.get("verified_receipt")
    if not isinstance(receipt, dict):
        failures.add("partition_unverified")
        return
    remote_path = partition.get("remote_path")
    revision = receipt.get("revision")
    if not (
        receipt.get("success") is True
        and isinstance(remote_path, str)
        and receipt.get("remote_path") == remote_path
        and receipt.get("sha256") == expected
        and isinstance(revision, str)
        and re.fullmatch(r"[0-9a-fA-F]{40}", revision) is not None
    ):
        failures.add("partition_unverified")


def _write_report(report: QualityReport, directory: str | Path) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report.payload(), default=lambda value: value.isoformat(), sort_keys=True, indent=2)
    (path / "quality.json").write_text(payload + "\n", encoding="utf-8")
    lines = [
        f"# Historical Data Quality: {report.status}",
        "",
        f"Qualified seconds: {report.qualified_seconds}",
        f"Calendar span seconds: {report.calendar_span_seconds}",
        "",
        "## Per-day evidence",
    ]
    for source_date, evidence in sorted(report.per_day.items()):
        lines.extend(("", f"### {source_date}", ""))
        for name, value in evidence.items():
            rendered = json.dumps(value, default=lambda item: item.isoformat(), sort_keys=True)
            lines.append(f"- {name}: {rendered}")
    lines.extend(("", "## Failures"))
    lines.extend(f"- {item}" for item in report.failures)
    lines.extend(("", f"Final result: {report.status}"))
    (path / "quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_dataset(manifest: Any, *, book_intervals: Iterable[Any] = (), trades: Iterable[Any] = (),
                  book_states: Iterable[Any] = (), replay_incidents: Iterable[Any] = (),
                  excluded_intervals: Iterable[Any] = (), invalid_intervals: Iterable[Any] = (),
                  label_windows: Iterable[Any] = (), joined_ticks: Iterable[Any] = (),
                  output_dir: str | Path | None = None) -> QualityReport:
    """Return a strict PASS/FAIL report; invalid evidence never becomes a warning."""
    failures: set[str] = set()
    intervals = _collect_intervals(book_intervals, failures)
    excluded = _collect_intervals(excluded_intervals, failures)
    excluded += _collect_intervals(getattr(manifest, "excluded_intervals", ()), failures)
    gaps = _collect_intervals(replay_incidents, failures)
    gaps += _collect_intervals(invalid_intervals, failures)
    gaps += _collect_intervals(getattr(manifest, "sequence_incidents", ()), failures)
    valid = _subtract(_union(intervals), excluded + gaps)

    for partition in getattr(manifest, "partitions", ()):
        _verified_partition(partition, failures)

    state_rows = list(book_states)
    sequence_values: list[int] = []
    previous_state_time: datetime | None = None
    for state in state_rows:
        try:
            observed = _utc(_value(state, "observed_through"))
            bid, ask = _value(state, "best_bid"), _value(state, "best_ask")
            if bid > ask:
                failures.add("crossed_bbo")
            if _value(state, "sequence_start") > _value(state, "sequence_end"):
                failures.add("invalid_book_state")
            if previous_state_time is not None and observed < previous_state_time:
                failures.add("out_of_order_book_state")
            previous_state_time = observed
            sequence_values.extend((_value(state, "sequence_start"), _value(state, "sequence_end")))
        except (AttributeError, KeyError, TypeError, ValueError):
            failures.add("malformed_book_evidence")

    trade_rows = list(trades)
    ids: set[str] = set(); duplicate_count = 0; previous_time: datetime | None = None
    for trade in trade_rows:
        try:
            event_time = _utc(_value(trade, "event_time"))
            if previous_time is not None and event_time < previous_time:
                failures.add("out_of_order_event_time")
            previous_time = event_time
            trade_id = str(_value(trade, "trade_id"))
            if trade_id in ids:
                duplicate_count += 1
            ids.add(trade_id)
            if _value(trade, "price") <= 0 or _value(trade, "size") <= 0:
                failures.add("non_positive_trade")
        except (AttributeError, KeyError, TypeError, ValueError):
            failures.add("malformed_trade_evidence")
    if duplicate_count:
        failures.add("duplicate_trade_ids")

    for tick in joined_ticks:
        try:
            timestamp = _utc(_value(tick, "timestamp"))
            observed = _optional_value(tick, "book_observed_through")
            if observed is None:
                failures.add("missing_causal_join_evidence")
                continue
            observed = _utc(observed)
            second = timestamp.replace(microsecond=0)
            if observed >= second:
                failures.add("book_state_leakage")
            tick_segment = _optional_value(tick, "segment_id")
            book_segment = _optional_value(tick, "book_segment_id")
            if tick_segment is not None and book_segment is not None and tick_segment != book_segment:
                failures.add("causal_join_segment_mismatch")
            if any(observed < cut_end and cut_start < timestamp for cut_start, cut_end in gaps + excluded):
                failures.add("causal_join_crosses_gap")
        except (AttributeError, KeyError, TypeError, ValueError):
            failures.add("missing_causal_join_evidence")

    for window in label_windows:
        try:
            start, end = _interval(window)
            if any(start < gap_end and gap_start < end for gap_start, gap_end in excluded + gaps):
                failures.add("label_window_crosses_excluded_interval")
        except (AttributeError, KeyError, TypeError, ValueError):
            failures.add("malformed_label_window")

    calendar_days = {start.date() + timedelta(days=n) for start, end in intervals
                     for n in range((end.date() - start.date()).days + 1)
                     if start.date() + timedelta(days=n) < end.date() or end.time() != time.min}
    completions = _completion_days(manifest, failures)
    per_day: dict[str, dict[str, Any]] = {}
    qualified = 0
    for day in sorted(calendar_days):
        start = datetime.combine(day, time.min, tzinfo=UTC); end = start + timedelta(days=1)
        seconds = _seconds(_subtract(valid, [(datetime.min.replace(tzinfo=UTC), start), (end, datetime.max.replace(tzinfo=UTC))]))
        complete = day in completions
        if not complete:
            seconds = 0; failures.add("trade_pages_complete")
        day_trades = [item for item in trade_rows if _is_utc_datetime(_optional_value(item, "event_time"))
                      and start <= _optional_value(item, "event_time") < end]
        day_ids = [str(_optional_value(item, "trade_id")) for item in day_trades]
        day_states = [state for state in state_rows if _is_utc_datetime(_optional_value(state, "observed_through"))
                      and start <= _optional_value(state, "observed_through") < end]
        day_sequences = [value for state in day_states
                         for value in (_optional_value(state, "sequence_start"), _optional_value(state, "sequence_end"))
                         if isinstance(value, int)]
        per_day[day.isoformat()] = {
            "valid_seconds": seconds, "trade_pages_complete": complete, "trade_count": len(day_trades),
            "first_event": min((_optional_value(item, "event_time") for item in day_trades), default=None),
            "last_event": max((_optional_value(item, "event_time") for item in day_trades), default=None),
            "duplicate_count": len(day_ids) - len(set(day_ids)),
            "sequence_range": [min(day_sequences), max(day_sequences)] if day_sequences else None,
            "gap_incidents": [{"start": a, "end": b} for a, b in gaps if a < end and start < b],
            "exclusions": [{"start": a, "end": b} for a, b in excluded if a < end and start < b],
        }
        qualified += seconds
    if qualified < QUALIFIED_SECONDS_MIN:
        failures.add("qualified_seconds_below_minimum")
    event_times = [_optional_value(item, "event_time") for item in trade_rows
                   if _is_utc_datetime(_optional_value(item, "event_time"))]
    first_event = min(event_times, default=None)
    last_event = max(event_times, default=None)
    calendar_span = int((max(end for _, end in intervals) - min(start for start, _ in intervals)).total_seconds()) if intervals else 0
    report = QualityReport("PASS" if not failures else "FAIL", qualified, calendar_span, per_day,
                           tuple(sorted(failures)), duplicate_count, first_event, last_event,
                           (min(sequence_values), max(sequence_values)) if sequence_values else None,
                           tuple({"start": a, "end": b} for a, b in gaps), tuple({"start": a, "end": b} for a, b in excluded))
    if output_dir is not None:
        _write_report(report, output_dir)
    return report
