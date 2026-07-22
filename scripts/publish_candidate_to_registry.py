"""Publish a qualified tournament candidate into the target MLflow registry.

This is the file-store -> server bridge referenced in
btcspiker_ml/qualification.py and scripts/qualify_candidate.py: those write
model versions into a local file store the API cannot see. This script never
re-implements a Staging gate -- it only reads the qualification.json verdict
that qualify_candidate.py already wrote, then copies bytes into the target
registry. Stdlib + mlflow only: this runs alongside the API image, which has
no pandas/pyarrow/yaml installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


CONTRACT_PARAMS = ("feature_cols", "tau", "feature_set_id", "feature_schema_version")
ALLOWED_STAGE = "Staging"
PROVISIONAL_REASON = "coverage_under_thirty_days"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-store-root", required=True, type=Path)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--qualification-json", required=True, type=Path)
    parser.add_argument("--target-tracking-uri", required=True)
    parser.add_argument("--model-name", default="btc-volatility-candidate")
    parser.add_argument("--stage", default=ALLOWED_STAGE)
    parser.add_argument("--expect-feature-set-id", required=True)
    parser.add_argument("--expect-feature-schema-version", required=True)
    parser.add_argument("--provisional", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _read_source_run(store_root: Path, run_id: str):
    client = MlflowClient(f"file:{store_root}")
    run = client.get_run(run_id)
    if run.info.status != "FINISHED":
        raise ValueError(f"source run {run_id} is not FINISHED (status={run.info.status})")
    return run


def _validate_contract(
    params: dict, expect_feature_set_id: str, expect_feature_schema_version: str
) -> dict:
    missing = [key for key in CONTRACT_PARAMS if not params.get(key)]
    if missing:
        raise ValueError(f"source run lacks runtime contract parameters: {', '.join(missing)}")
    if not [column for column in params["feature_cols"].split(",") if column]:
        raise ValueError("source run feature_cols is empty")
    tau = float(params["tau"])
    if not 0 <= tau <= 1:
        raise ValueError(f"source run tau {tau} is not in [0, 1]")
    if params["feature_set_id"] != expect_feature_set_id:
        raise ValueError(
            f"feature_set_id mismatch: expected {expect_feature_set_id!r}, "
            f"got {params['feature_set_id']!r}"
        )
    if params["feature_schema_version"] != expect_feature_schema_version:
        raise ValueError(
            "feature_schema_version mismatch: expected "
            f"{expect_feature_schema_version!r}, got {params['feature_schema_version']!r}"
        )
    return {key: params[key] for key in CONTRACT_PARAMS}


def _read_qualification(path: Path, *, provisional: bool) -> dict:
    payload = json.loads(path.read_text())
    reasons = list(payload.get("reasons", []))
    if provisional:
        if reasons != [PROVISIONAL_REASON]:
            raise ValueError(
                "provisional publish requires exactly one failing gate "
                f"({PROVISIONAL_REASON!r}); qualification reasons were {reasons!r}"
            )
    elif not payload.get("passed"):
        raise ValueError(f"qualification did not pass; reasons: {reasons!r}")
    return payload


def _model_directory(store_root: Path, experiment_id: str, run_id: str) -> Path:
    # Deliberately not client.download_artifacts(): the file store's recorded
    # artifact_uri is an absolute *host* path from wherever the tournament
    # ran, which does not resolve inside this container. The on-disk layout
    # under store_root is stable, so build the path directly instead.
    model_dir = store_root / experiment_id / run_id / "artifacts" / "model"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model artifact directory not found: {model_dir}")
    return model_dir


def _existing_version_for_source(client: MlflowClient, model_name: str, source_run_id: str):
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception:
        return None
    for version in versions:
        if version.tags.get("source_run_id") == source_run_id:
            return version
    return None


def main() -> int:
    args = _arguments()
    if args.stage != ALLOWED_STAGE:
        raise ValueError(
            f"refusing --stage {args.stage!r}; this bridge only ever publishes to "
            f"{ALLOWED_STAGE!r} -- Production promotion is a separate human decision"
        )

    source_run = _read_source_run(args.source_store_root, args.source_run_id)
    contract = _validate_contract(
        source_run.data.params, args.expect_feature_set_id, args.expect_feature_schema_version
    )
    qualification = _read_qualification(args.qualification_json, provisional=args.provisional)
    model_dir = _model_directory(
        args.source_store_root, source_run.info.experiment_id, args.source_run_id
    )

    if args.dry_run:
        print(f"model_dir: {model_dir}")
        for key, value in contract.items():
            print(f"{key}: {value}")
        print("dry-run: no changes made; Production unchanged")
        return 0

    mlflow.set_tracking_uri(args.target_tracking_uri)
    target_client = MlflowClient(args.target_tracking_uri)

    existing = _existing_version_for_source(target_client, args.model_name, args.source_run_id)
    if existing is not None:
        print(
            f"model_version: {existing.version} already published for "
            f"source_run_id={args.source_run_id}; skipping"
        )
        print("Production unchanged")
        return 0

    coverage_days = qualification.get("evidence", {}).get("coverage_days")
    dataset_id = source_run.data.tags.get("dataset_id", "unknown")
    search_id = source_run.data.tags.get("search_id", "unknown")
    git_sha = source_run.data.tags.get("git_sha", "unknown")
    qualification_sha256 = hashlib.sha256(args.qualification_json.read_bytes()).hexdigest()

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_params(contract)
        mlflow.set_tags(
            {
                "source_run_id": args.source_run_id,
                "dataset_id": dataset_id,
                "search_id": search_id,
                "git_sha": git_sha,
                "coverage_days": str(coverage_days),
                "qualification_sha256": qualification_sha256,
            }
        )
        # Copy bytes, don't deserialize -- sidesteps sklearn skew here and
        # leaves the real load/compatibility test to the API, where it belongs.
        mlflow.log_artifacts(str(model_dir), artifact_path="model")

    version = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=args.model_name)
    target_client.transition_model_version_stage(
        args.model_name, version.version, stage=ALLOWED_STAGE, archive_existing_versions=False
    )
    target_client.set_model_version_tag(
        args.model_name, version.version, "provisional", str(args.provisional).lower()
    )
    if args.provisional:
        target_client.set_model_version_tag(
            args.model_name, version.version, "provisional_reason", PROVISIONAL_REASON
        )
    target_client.set_model_version_tag(
        args.model_name, version.version, "coverage_days", str(coverage_days)
    )
    target_client.set_model_version_tag(args.model_name, version.version, "not_for_production", "true")
    target_client.set_model_version_tag(
        args.model_name, version.version, "source_run_id", args.source_run_id
    )

    print(f"model_version: {version.version}")
    if args.provisional:
        print("PROVISIONAL — RESEARCH ONLY — NOT FOR PRODUCTION")
    print("Production unchanged")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
