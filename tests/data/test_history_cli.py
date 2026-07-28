from datetime import date
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import pytest
import pandas as pd

from btcspiker_data.cbb26 import CBB26_REVISION
from btcspiker_data.history_pipeline import (
    DownloadSummary,
    HistoryDownloadConfig,
    raw_manifest_payload_id,
    materialize_history,
    _hourly_trade_partitions,
)
from scripts import download_coinbase_history, materialize_coinbase_history


def _summary(tmp_path: Path, *, status: str = "PASS") -> DownloadSummary:
    manifest = tmp_path / "raw-manifest.json"
    quality = tmp_path / "quality.json"
    manifest.write_text("{}")
    quality.write_text(json.dumps({"status": status}))
    return DownloadSummary(
        dataset_id="dataset-1",
        repo_id="alice/btcspiker-coinbase-history",
        revision="a" * 40,
        manifest_path=manifest,
        quality_report_path=quality,
        quality_status=status,
        qualified_seconds=2_592_000 if status == "PASS" else 0,
        downloaded_files=70,
        uploaded_files=72,
        reused_files=0,
        bytes_downloaded=123,
    )


def _manifest_partition(kind: str, path: Path, digest: str | None = None):
    digest = digest or hashlib.sha256(path.read_bytes()).hexdigest()
    source = "coinbase_public_trades" if kind == "trades" else "cbb26"
    remote_path = (
        f"raw/kind={kind}/source={source}/product=BTC-USD/"
        f"date=2026-04-24/hour=00/part-{digest}.parquet"
    )
    return {
        "kind": kind,
        "local_path": str(path),
        "remote_path": remote_path,
        "sha256": digest,
        "verified_receipt": {
            "repo_id": "alice/btcspiker-coinbase-history",
            "revision": "a" * 40,
            "remote_path": remote_path,
            "sha256": digest,
            "size_bytes": path.stat().st_size,
            "success": True,
        },
    }


def _raw_manifest(partitions):
    return {
        "source_revision": CBB26_REVISION,
        "source_url": "https://huggingface.co/datasets/deusmos/cbb26-timeseries-db",
        "repo_id": "alice/btcspiker-coinbase-history",
        "revision": "b" * 40,
        "usage_scope": "research_unverified",
        "schemas": {},
        "partitions": partitions,
        "coverage_seconds": 0,
        "missing_seconds": 0,
        "duplicate_counts": {},
        "sequence_incidents": [],
        "excluded_intervals": [],
        "created_at": "2026-07-22T00:00:00+00:00",
        "trade_day_completions": [],
    }


def test_download_cli_uses_pinned_defaults_and_prints_safe_summary(tmp_path, capsys):
    observed = []

    def runner(config):
        observed.append(config)
        return _summary(tmp_path)

    code = download_coinbase_history.main(
        ["--cache-root", str(tmp_path / "cache")], runner=runner
    )

    assert code == 0
    config = observed[0]
    assert config == HistoryDownloadConfig(
        cache_root=(tmp_path / "cache").resolve(),
        start=date(2026, 4, 24),
        end=date(2026, 5, 28),
        product="BTC-USD",
        revision=CBB26_REVISION,
        max_rps=8,
    )
    output = capsys.readouterr().out
    assert "alice/btcspiker-coinbase-history" in output
    assert "dataset-1" in output
    assert "token" not in output.lower()


def test_download_cli_refuses_unpinned_revision(tmp_path):
    with pytest.raises(SystemExit):
        download_coinbase_history.main(
            [
                "--cache-root",
                str(tmp_path),
                "--revision",
                "main",
            ],
            runner=lambda _: pytest.fail("runner must not start"),
        )


def test_download_cli_returns_nonzero_for_failed_quality(tmp_path, capsys):
    code = download_coinbase_history.main(
        ["--cache-root", str(tmp_path / "cache")],
        runner=lambda _: _summary(tmp_path, status="FAIL"),
    )
    assert code == 1
    assert "quality_status: FAIL" in capsys.readouterr().out


def test_materialize_rejects_manifest_partition_checksum_mismatch(tmp_path):
    partition = tmp_path / "trades.parquet"
    partition.write_bytes(b"not-the-declared-content")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(_raw_manifest([_manifest_partition("trades", partition, "0" * 64)]))
    )

    with pytest.raises(ValueError, match="checksum"):
        materialize_coinbase_history.load_verified_manifest(manifest)


def test_manifest_rejects_receipt_that_does_not_match_partition(tmp_path):
    partition = tmp_path / "trades.parquet"
    partition.write_bytes(b"content")
    item = _manifest_partition("trades", partition)
    item["verified_receipt"]["sha256"] = "0" * 64
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_raw_manifest([item])))

    with pytest.raises(ValueError, match="receipt"):
        materialize_coinbase_history.load_verified_manifest(manifest)


def test_remote_manifest_verifies_every_kind_at_exact_revision(tmp_path):
    partitions = []
    for kind in ("book_deltas", "book_states", "trades"):
        local = tmp_path / f"{kind}.parquet"
        local.write_bytes(kind.encode())
        item = _manifest_partition(kind, local)
        item.pop("local_path")
        partitions.append(item)

    class Api:
        def get_paths_info(self, *, paths: list[str], **kwargs):
            assert kwargs["repo_id"] == "alice/btcspiker-coinbase-history"
            assert kwargs["revision"] == "b" * 40
            paths_seen = list(paths)
            paths.extend([])
            return [
                type(
                    "Info",
                    (),
                    {
                        "path": remote_path,
                        "lfs": type(
                            "Lfs",
                            (),
                            {
                                "sha256": next(
                                    item["sha256"]
                                    for item in partitions
                                    if item["remote_path"] == remote_path
                                )
                            },
                        )(),
                    },
                )()
                for remote_path in paths_seen
            ]

    api = Api()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_raw_manifest(partitions)))
    payload = materialize_coinbase_history.load_verified_manifest(manifest, api=api)

    assert {item["kind"] for item in payload["partitions"]} == {
        "book_deltas",
        "book_states",
        "trades",
    }


def test_materialize_cli_prints_existing_dataset_export(tmp_path, capsys):
    output = tmp_path / "features.parquet"
    output.write_bytes(b"fixture")

    def runner(raw_manifest, feature_set, output_root):
        assert raw_manifest == (tmp_path / "raw.json").resolve()
        assert feature_set == "core_v1"
        return output

    (tmp_path / "raw.json").write_text('{"partitions": []}')
    code = materialize_coinbase_history.main(
        [
            "--raw-manifest",
            str(tmp_path / "raw.json"),
            "--feature-set",
            "core_v1",
            "--output-root",
            str(tmp_path),
        ],
        runner=runner,
        inspector=lambda path: object(),
        verifier=lambda path: {"partitions": []},
    )
    assert code == 0
    assert (
        f"export BTCSPIKER_EXISTING_DATA={output.resolve()}" in capsys.readouterr().out
    )


def test_materialize_cli_verifies_manifest_before_custom_runner(tmp_path):
    manifest = tmp_path / "raw.json"
    manifest.write_text("{}")
    order = []

    def verifier(path):
        order.append(("verify", path))
        return {}

    def runner(path, feature_set, output_root):
        order.append(("run", path))
        output = output_root / "features.parquet"
        output.write_bytes(b"fixture")
        return output

    materialize_coinbase_history.main(
        ["--raw-manifest", str(manifest), "--output-root", str(tmp_path)],
        verifier=verifier,
        runner=runner,
        inspector=lambda path: object(),
    )

    assert [name for name, _ in order] == ["verify", "run"]


def test_raw_manifest_payload_id_excludes_only_creation_time():
    base = {
        "source_revision": "pinned",
        "partitions": [{"sha256": "a" * 64}],
        "created_at": "2026-07-22T00:00:00+00:00",
    }
    changed_time = {**base, "created_at": "2026-07-23T00:00:00+00:00"}
    changed_data = {**base, "partitions": [{"sha256": "b" * 64}]}

    assert raw_manifest_payload_id(base) == raw_manifest_payload_id(changed_time)
    assert raw_manifest_payload_id(base) != raw_manifest_payload_id(changed_data)


def test_trade_writer_emits_all_24_hours_including_empty_hours(tmp_path):
    records = _hourly_trade_partitions([], tmp_path, date(2026, 4, 24), "BTC-USD")

    assert len(records) == 24
    assert all(record.row_count == 0 for record in records)
    assert {record.path.parent.name for record in records} == {
        f"hour={hour:02d}" for hour in range(24)
    }


def test_materialize_history_verifies_raw_files_and_writes_lineage(tmp_path):
    start = datetime(2026, 4, 24, tzinfo=timezone.utc)
    books = pd.DataFrame(
        [
            {
                "product_id": "BTC-USD",
                "observed_through": start + timedelta(seconds=second),
                "best_bid": "89999.9",
                "bid_size": "2",
                "best_ask": "90000.1",
                "ask_size": "1",
                "segment_id": 0,
                "source_date": "2026-04-24",
            }
            for second in range(62)
        ]
    )
    trades = pd.DataFrame(
        [
            {
                "product_id": "BTC-USD",
                "trade_id": str(second),
                "event_time": start + timedelta(seconds=second, microseconds=500_000),
                "price": str(90_000 + second / 100),
                "size": "0.01",
                "reported_side": "BUY",
                "source_date": "2026-04-24",
            }
            for second in range(1, 62)
        ]
    )
    book_path = tmp_path / "books.parquet"
    trade_path = tmp_path / "trades.parquet"
    books.to_parquet(book_path, index=False)
    trades.to_parquet(trade_path, index=False)
    partitions = []
    for kind, path in (("book_states", book_path), ("trades", trade_path)):
        partitions.append(
            {
                "kind": kind,
                "local_path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest_partitions = [
        _manifest_partition(item["kind"], Path(item["local_path"]))
        for item in partitions
    ]
    manifest_payload = _raw_manifest(manifest_partitions)
    manifest_path = tmp_path / "raw.json"
    manifest_path.write_text(json.dumps(manifest_payload))

    output = materialize_history(manifest_path, "core_v1", tmp_path / "output")

    assert output.is_file()
    assert not pd.read_parquet(output).empty
    lineage = json.loads(output.with_suffix(".parquet.lineage.json").read_text())
    assert lineage["parent_dataset_id"] == raw_manifest_payload_id(manifest_payload)
    assert lineage["source_manifest_path"] == str(manifest_path.resolve())
    assert lineage["feature_set_id"] == "core_v1"
    assert len(lineage["feature_engine_git_sha"]) == 40
