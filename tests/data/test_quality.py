from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

from btcspiker_data.contracts import BookState, TradeEvent
from btcspiker_data.quality import QUALIFIED_SECONDS_MIN, audit_dataset
from btcspiker_data.raw_manifest import RawDatasetManifest


UTC = timezone.utc


def manifest(completions=()):
    return RawDatasetManifest(
        source_revision="r1", source_url="https://example.test", repo_id="user/data",
        revision="abc", usage_scope="research_unverified", schemas={}, partitions=[],
        coverage_seconds=0, missing_seconds=0, duplicate_counts={}, sequence_incidents=[],
        excluded_intervals=[], created_at=datetime(2026, 1, 1, tzinfo=UTC),
        trade_day_completions=list(completions),
    )


def completion(day):
    start = datetime.combine(day, datetime.min.time(), tzinfo=UTC)
    return {"product_id": "BTC-USD", "source_date": day.isoformat(),
            "day_start_epoch": int(start.timestamp()), "day_end_epoch": int((start + timedelta(days=1)).timestamp()),
            "trade_pages_complete": True}


def intervals(days=30):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return [(start + timedelta(days=n), start + timedelta(days=n + 1)) for n in range(days)]


def test_gate_fails_below_exact_thirty_day_qualified_seconds():
    report = audit_dataset(manifest([completion(date(2026, 1, 1))]), book_intervals=intervals(1))
    assert report.status == "FAIL"
    assert report.qualified_seconds == 86_400
    assert QUALIFIED_SECONDS_MIN == 2_592_000


def test_completion_evidence_is_required_for_each_included_date():
    report = audit_dataset(manifest([]), book_intervals=intervals(1))
    assert report.per_day["2026-01-01"]["valid_seconds"] == 0
    assert "trade_pages_complete" in report.failures


def test_excluded_interval_and_label_window_fail_strictly():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    data = manifest([completion(date(2026, 1, 1))])
    report = audit_dataset(data, book_intervals=[(start, start + timedelta(days=1))],
                           excluded_intervals=[(start + timedelta(seconds=30), start + timedelta(seconds=40))],
                           label_windows=[(start, start + timedelta(seconds=60))])
    assert "label_window_crosses_excluded_interval" in report.failures


def test_trade_order_and_duplicate_ids_fail():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    trades = [
        TradeEvent("BTC-USD", "same", start + timedelta(seconds=2), Decimal("1"), Decimal("1"), "BUY", "x"),
        TradeEvent("BTC-USD", "same", start + timedelta(seconds=1), Decimal("1"), Decimal("1"), "BUY", "x"),
    ]
    report = audit_dataset(manifest([completion(date(2026, 1, 1))]), book_intervals=intervals(1), trades=trades)
    assert "out_of_order_event_time" in report.failures
    assert "duplicate_trade_ids" in report.failures


def test_checksum_mismatch_and_invalid_book_evidence_fail(tmp_path):
    payload = tmp_path / "part.parquet"
    payload.write_bytes(b"different")
    data = RawDatasetManifest(**{**manifest([completion(date(2026, 1, 1))]).__dict__, "partitions": [
        {"path": str(payload), "sha256": "0" * 64}
    ]})
    start = datetime(2026, 1, 1, tzinfo=UTC)
    bad_book = {"observed_through": start, "best_bid": Decimal("2"), "best_ask": Decimal("1"),
                "sequence_start": 3, "sequence_end": 2}
    report = audit_dataset(data, book_intervals=intervals(1), book_states=[bad_book])
    assert "checksum_mismatch" in report.failures
    assert "crossed_bbo" in report.failures


def test_report_files_are_machine_and_human_readable(tmp_path):
    report = audit_dataset(manifest([]), book_intervals=intervals(1), output_dir=tmp_path)
    assert report.status == "FAIL"
    assert '"status": "FAIL"' in (tmp_path / "quality.json").read_text()
    assert "Historical Data Quality: FAIL" in (tmp_path / "quality.md").read_text()
