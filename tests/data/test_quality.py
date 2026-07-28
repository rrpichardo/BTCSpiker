import json
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from btcspiker_data.contracts import BookState, TradeEvent
from btcspiker_data.quality import QUALIFIED_SECONDS_MIN, audit_dataset
from btcspiker_data.raw_manifest import RawDatasetManifest


UTC = timezone.utc


def test_audit_rejects_negative_minimum_threshold():
    with pytest.raises(ValueError, match="non-negative"):
        audit_dataset(manifest(), minimum_qualified_seconds=-1)


def manifest(completions=()):
    return RawDatasetManifest(
        source_revision="r1",
        source_url="https://example.test",
        repo_id="user/data",
        revision="abc",
        usage_scope="research_unverified",
        schemas={},
        partitions=[],
        coverage_seconds=0,
        missing_seconds=0,
        duplicate_counts={},
        sequence_incidents=[],
        excluded_intervals=[],
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        trade_day_completions=list(completions),
    )


def completion(day):
    start = datetime.combine(day, datetime.min.time(), tzinfo=UTC)
    return {
        "product_id": "BTC-USD",
        "source_date": day.isoformat(),
        "day_start_epoch": int(start.timestamp()),
        "day_end_epoch": int((start + timedelta(days=1)).timestamp()),
        "trade_pages_complete": True,
    }


def completions(days=30):
    return [completion(date(2026, 1, 1) + timedelta(days=n)) for n in range(days)]


def intervals(days=30):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        (start + timedelta(days=n), start + timedelta(days=n + 1)) for n in range(days)
    ]


def test_gate_fails_below_exact_thirty_day_qualified_seconds():
    report = audit_dataset(
        manifest([completion(date(2026, 1, 1))]), book_intervals=intervals(1)
    )
    assert report.status == "FAIL"
    assert report.qualified_seconds == 86_400
    assert QUALIFIED_SECONDS_MIN == 2_592_000


def test_completion_evidence_is_required_for_each_included_date():
    report = audit_dataset(manifest([]), book_intervals=intervals(1))
    assert report.per_day["2026-01-01"]["valid_seconds"] == 0
    assert "trade_pages_complete" in report.failures


def test_completion_flag_must_be_present_and_exactly_true():
    for flag in (None, False, 1, "true"):
        evidence = completion(date(2026, 1, 1))
        if flag is None:
            evidence.pop("trade_pages_complete")
        else:
            evidence["trade_pages_complete"] = flag
        report = audit_dataset(manifest([evidence]), book_intervals=intervals(1))
        assert report.per_day["2026-01-01"]["valid_seconds"] == 0
        assert "invalid_trade_completion_evidence" in report.failures


def test_completion_identity_and_day_boundary_must_match():
    evidence = completion(date(2026, 1, 1))
    evidence.update(product_id="", day_end_epoch=evidence["day_end_epoch"] - 1)
    report = audit_dataset(manifest([evidence]), book_intervals=intervals(1))
    assert report.status == "FAIL"
    assert "invalid_trade_completion_evidence" in report.failures


def test_excluded_interval_and_label_window_fail_strictly():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    data = manifest([completion(date(2026, 1, 1))])
    report = audit_dataset(
        data,
        book_intervals=[(start, start + timedelta(days=1))],
        excluded_intervals=[
            (start + timedelta(seconds=30), start + timedelta(seconds=40))
        ],
        label_windows=[(start, start + timedelta(seconds=60))],
    )
    assert "label_window_crosses_excluded_interval" in report.failures


def test_trade_order_and_duplicate_ids_fail():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    trades = [
        TradeEvent(
            "BTC-USD",
            "same",
            start + timedelta(seconds=2),
            Decimal("1"),
            Decimal("1"),
            "BUY",
            "x",
        ),
        TradeEvent(
            "BTC-USD",
            "same",
            start + timedelta(seconds=1),
            Decimal("1"),
            Decimal("1"),
            "BUY",
            "x",
        ),
    ]
    report = audit_dataset(
        manifest([completion(date(2026, 1, 1))]),
        book_intervals=intervals(1),
        trades=trades,
    )
    assert "out_of_order_event_time" in report.failures
    assert "duplicate_trade_ids" in report.failures
    assert report.first_event == start + timedelta(seconds=1)
    assert report.last_event == start + timedelta(seconds=2)


def test_nonpositive_trade_is_a_named_failure_instead_of_crash():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    bad_trade = {
        "trade_id": "1",
        "event_time": start,
        "price": Decimal("0"),
        "size": Decimal("1"),
    }
    report = audit_dataset(
        manifest([completion(start.date())]),
        book_intervals=intervals(1),
        trades=[bad_trade],
    )
    assert "non_positive_trade" in report.failures


def test_causal_join_rejects_current_second_and_missing_book_evidence():
    timestamp = datetime(2026, 1, 1, 0, 0, 1, 500000, tzinfo=UTC)
    report = audit_dataset(
        manifest([completion(timestamp.date())]),
        book_intervals=intervals(1),
        joined_ticks=[
            {
                "timestamp": timestamp,
                "book_observed_through": timestamp.replace(microsecond=0),
            },
            {"timestamp": timestamp, "book_observed_through": None},
        ],
    )
    assert "book_state_leakage" in report.failures
    assert "missing_causal_join_evidence" in report.failures


def test_causal_join_rejects_segment_mismatch_and_crossed_gap():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    report = audit_dataset(
        manifest([completion(start.date())]),
        book_intervals=intervals(1),
        replay_incidents=[(start + timedelta(seconds=1), start + timedelta(seconds=2))],
        joined_ticks=[
            {
                "timestamp": start + timedelta(seconds=3),
                "book_observed_through": start,
                "segment_id": 2,
                "book_segment_id": 1,
            }
        ],
    )
    assert "causal_join_segment_mismatch" in report.failures
    assert "causal_join_crosses_gap" in report.failures


def test_checksum_mismatch_and_invalid_book_evidence_fail(tmp_path):
    payload = tmp_path / "part.parquet"
    payload.write_bytes(b"different")
    data = RawDatasetManifest(
        **{
            **manifest([completion(date(2026, 1, 1))]).__dict__,
            "partitions": [{"path": str(payload), "sha256": "0" * 64}],
        }
    )
    start = datetime(2026, 1, 1, tzinfo=UTC)
    bad_book = {
        "observed_through": start,
        "best_bid": Decimal("2"),
        "best_ask": Decimal("1"),
        "sequence_start": 3,
        "sequence_end": 2,
    }
    report = audit_dataset(data, book_intervals=intervals(1), book_states=[bad_book])
    assert "checksum_mismatch" in report.failures
    assert "crossed_bbo" in report.failures


def test_remote_manifest_partition_without_verified_receipt_fails_closed():
    data = RawDatasetManifest(
        **{
            **manifest([]).__dict__,
            "partitions": [
                {
                    "remote_path": "raw/kind=trades/part.parquet",
                    "sha256": "0" * 64,
                    "row_count": 1,
                }
            ],
        }
    )
    report = audit_dataset(data)
    assert "partition_unverified" in report.failures


def test_exact_commit_verified_remote_receipt_is_accepted():
    digest = "a" * 64
    remote_path = "raw/kind=trades/part.parquet"
    data = RawDatasetManifest(
        **{
            **manifest([]).__dict__,
            "partitions": [
                {
                    "remote_path": remote_path,
                    "sha256": digest,
                    "verified_receipt": {
                        "remote_path": remote_path,
                        "sha256": digest,
                        "revision": "b" * 40,
                        "success": True,
                    },
                }
            ],
        }
    )
    report = audit_dataset(data)
    assert "partition_unverified" not in report.failures
    assert "checksum_mismatch" not in report.failures


def test_overlap_union_and_all_cut_types_are_counted_once():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    report = audit_dataset(
        manifest([completion(start.date())]),
        book_intervals=[
            (start, start + timedelta(seconds=100)),
            (start + timedelta(seconds=50), start + timedelta(seconds=150)),
        ],
        replay_incidents=[
            (start + timedelta(seconds=10), start + timedelta(seconds=20))
        ],
        invalid_intervals=[
            (start + timedelta(seconds=30), start + timedelta(seconds=40))
        ],
        excluded_intervals=[
            (start + timedelta(seconds=35), start + timedelta(seconds=55))
        ],
    )
    assert report.qualified_seconds == 115


def test_exact_thirty_day_threshold_passes():
    report = audit_dataset(manifest(completions()), book_intervals=intervals())
    assert report.qualified_seconds == QUALIFIED_SECONDS_MIN
    assert report.status == "PASS"


def test_report_files_are_machine_and_human_readable(tmp_path):
    start = datetime(2026, 1, 1, tzinfo=UTC)
    trade = TradeEvent("BTC-USD", "1", start, Decimal("1"), Decimal("1"), "BUY", "x")
    state = BookState(
        "BTC-USD", start, 1, 2, Decimal("1"), Decimal("1"), Decimal("2"), Decimal("1")
    )
    report = audit_dataset(
        manifest([]),
        book_intervals=intervals(1),
        trades=[trade],
        book_states=[state],
        replay_incidents=[(start + timedelta(seconds=2), start + timedelta(seconds=3))],
        excluded_intervals=[
            (start + timedelta(seconds=4), start + timedelta(seconds=5))
        ],
        output_dir=tmp_path,
    )
    assert report.status == "FAIL"
    assert json.loads((tmp_path / "quality.json").read_text())["status"] == "FAIL"
    markdown = (tmp_path / "quality.md").read_text()
    for field in (
        "valid_seconds",
        "trade_count",
        "first_event",
        "last_event",
        "duplicate_count",
        "sequence_range",
        "gap_incidents",
        "exclusions",
    ):
        assert field in markdown
    assert "Historical Data Quality: FAIL" in markdown


def test_malformed_rows_still_emit_an_inspectable_fail_report(tmp_path):
    naive = datetime(2026, 1, 1)
    report = audit_dataset(
        manifest(completions()),
        book_intervals=intervals(),
        trades=[{"event_time": naive}],
        book_states=[{"observed_through": None}],
        label_windows=[{"start": naive}],
        output_dir=tmp_path,
    )
    assert report.status == "FAIL"
    assert "malformed_trade_evidence" in report.failures
    assert "malformed_book_evidence" in report.failures
    assert "malformed_label_window" in report.failures
    assert json.loads((tmp_path / "quality.json").read_text())["status"] == "FAIL"
