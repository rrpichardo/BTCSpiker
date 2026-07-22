"""Download and verify the pinned free Coinbase BTC-USD history window."""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys
from typing import Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from btcspiker_data.cbb26 import CBB26_REVISION  # noqa: E402
from btcspiker_data.history_pipeline import (  # noqa: E402
    DownloadSummary,
    HistoryDownloadConfig,
    run_history_download,
)


def _date(value: str) -> date:
    return date.fromisoformat(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--start", type=_date, default=date(2026, 4, 24))
    parser.add_argument("--end", type=_date, default=date(2026, 5, 28))
    parser.add_argument("--product", default="BTC-USD")
    parser.add_argument("--revision", default=CBB26_REVISION)
    parser.add_argument(
        "--max-rps",
        type=int,
        default=8,
        help="Frozen reviewed public request ceiling (must remain 8)",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[[HistoryDownloadConfig], DownloadSummary] = run_history_download,
) -> int:
    args = _parser().parse_args(argv)
    try:
        config = HistoryDownloadConfig(
            cache_root=args.cache_root,
            start=args.start,
            end=args.end,
            product=args.product,
            revision=args.revision,
            max_rps=args.max_rps,
        )
    except ValueError as error:
        _parser().error(str(error))
    summary = runner(config)
    for name, value in (
        ("dataset_id", summary.dataset_id),
        ("repo_id", summary.repo_id),
        ("revision", summary.revision),
        ("manifest", summary.manifest_path),
        ("quality_report", summary.quality_report_path),
        ("quality_status", summary.quality_status),
        ("qualified_seconds", summary.qualified_seconds),
        ("downloaded_files", summary.downloaded_files),
        ("uploaded_files", summary.uploaded_files),
        ("reused_files", summary.reused_files),
        ("bytes_downloaded", summary.bytes_downloaded),
    ):
        print(f"{name}: {value}")
    return 0 if summary.quality_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
