"""Qualify one completed MLflow run for Staging; Production is never changed."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

# Direct ``python scripts/qualify_candidate.py`` execution needs the repository
# root on the import path; module execution already has it.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import mlflow
from mlflow.tracking import MlflowClient

from btcspiker_ml.qualification import CandidateEvidence, qualify
from btcspiker_ml.search import SearchState


CANDIDATE_MODEL_NAME = "btc-volatility-candidate"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("evidence_json", type=Path)
    parser.add_argument("--search-state", type=Path, required=True)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    return parser.parse_args()


def _write_qualification_artifact(
    artifact_root: Path, run_id: str, payload: dict[str, object]
) -> Path:
    directory = artifact_root / "qualifications" / f"run_id={run_id}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "qualification.json"
    if path.exists():
        raise FileExistsError(f"qualification record already exists: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def main() -> int:
    args = _arguments()
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(args.tracking_uri)
    # Require a real completed candidate, never invent a registry source.
    run = client.get_run(args.run_id)
    if run.info.status != "FINISHED":
        raise ValueError(f"run {args.run_id} is not completed")

    evidence = CandidateEvidence(**json.loads(args.evidence_json.read_text()))
    state = SearchState.load(args.search_state)
    # Task 9 persists this timestamp and refuses any reopened search id. Save
    # immediately after opening, before the gate decision or registry action.
    state.open_final_holdout(requesting_stage="qualification")
    state.save(args.search_state)
    result = qualify(evidence)
    payload: dict[str, object] = {
        "run_id": args.run_id,
        "evidence": asdict(evidence),
        "passed": result.passed,
        "reasons": list(result.reasons),
        "final_holdout_accessed_at": state.final_holdout_accessed_at,
    }
    qualification_path = _write_qualification_artifact(
        args.artifact_root, args.run_id, payload
    )

    print(f"run_id: {args.run_id}")
    print("gates:")
    for reason in result.reasons:
        print(f"  FAIL {reason}")
    if not result.reasons:
        print("  PASS all_staging_gates")
    if result.passed:
        version = mlflow.register_model(
            model_uri=f"runs:/{args.run_id}/model",
            name=CANDIDATE_MODEL_NAME,
        )
        client.transition_model_version_stage(
            CANDIDATE_MODEL_NAME,
            version.version,
            stage="Staging",
            archive_existing_versions=False,
        )
        print(f"model_version: {version.version}")
    else:
        print("model_version: not_registered")
    print(f"qualification_json: {qualification_path}")
    print("Production unchanged")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
