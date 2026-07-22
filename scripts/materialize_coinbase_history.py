"""Materialize an immutable raw manifest into an experiment feature Parquet."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from btcspiker_data.history_pipeline import (  # noqa: E402
    load_verified_manifest,
    materialize_history,
)
from btcspiker_ml.datasets import inspect_existing_dataset  # noqa: E402


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[[Path, str, Path], Path] = materialize_history,
    inspector: Callable[[Path], object] = inspect_existing_dataset,
    verifier: Callable[[Path], object] = load_verified_manifest,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument(
        "--feature-set",
        choices=("core_v1", "multi_window_v1", "microstructure_v1"),
        default="core_v1",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    raw_manifest = args.raw_manifest.expanduser().resolve()
    verifier(raw_manifest)
    output = runner(raw_manifest, args.feature_set, args.output_root)
    if args.feature_set == "core_v1":
        inspector(output)
    print(f"features: {output.resolve()}")
    print(f"export BTCSPIKER_EXISTING_DATA={output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
