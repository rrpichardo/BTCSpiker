"""Bind the previously collected corpus and record a manifest.

Purely descriptive: no network calls, no collector, no writes outside the
configured artifact root. Prints the bound dataset's identity and provenance.
"""
import argparse
import sys
from pathlib import Path

# Allow `python scripts/bind_existing_dataset.py ...` to import btcspiker_ml
# without requiring PYTHONPATH=. or an editable install. The pyproject.toml
# pythonpath entry only affects pytest.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from btcspiker_ml.config import load_experiment_config  # noqa: E402
from btcspiker_ml.datasets import (  # noqa: E402
    inspect_existing_dataset,
    publish_existing_manifest,
    resolve_existing_dataset,
)


def _format_duration(start_iso: str, end_iso: str) -> str:
    import pandas as pd

    delta = pd.Timestamp(end_iso) - pd.Timestamp(start_iso)
    total_seconds = int(delta.total_seconds())
    days, remainder = divmod(total_seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, _ = divmod(remainder, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiment.yaml"),
        help="Path to the frozen experiment.yaml",
    )
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    resolved_path = resolve_existing_dataset(cfg.storage.existing_data)
    dataset = inspect_existing_dataset(resolved_path)
    dataset_id, manifest_path = publish_existing_manifest(
        dataset, cfg.storage.artifact_root
    )

    print(f"dataset_id: {dataset_id}")
    print(f"source: {dataset.path}")
    print(f"sha256: {dataset.sha256}")
    print(f"rows: {dataset.rows}")
    print(f"event_time: {dataset.start_time} -> {dataset.end_time}")
    print(f"duration: {_format_duration(dataset.start_time, dataset.end_time)}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
