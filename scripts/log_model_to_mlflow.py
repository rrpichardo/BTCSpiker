"""
Log the trained LR pipeline to MLflow and register it under stage Production.

Reads env vars:
  MLFLOW_TRACKING_URI  (default: http://localhost:5001)
  MODEL_NAME           (default: btc-volatility-lr)
  MODEL_ARTIFACT_PATH  (default: handoff/models/artifacts/lr_pipeline.pkl)
  FEATURES_PATH        (default: handoff/data_sample/features_slice.csv)
  BASELINE_VOL_THRESHOLD (default: 0.000048)

Idempotent: if a Production version already tagged with the same pickle SHA256
exists, the script skips logging and exits cleanly.
"""

import hashlib
import os
import pickle
import sys

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import average_precision_score, roc_auc_score

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MODEL_NAME = os.getenv("MODEL_NAME", "btc-volatility-lr")
MODEL_ARTIFACT_PATH = os.getenv(
    "MODEL_ARTIFACT_PATH", "handoff/models/artifacts/lr_pipeline.pkl"
)
FEATURES_PATH = os.getenv(
    "FEATURES_PATH", "handoff/data_sample/features_slice.csv"
)
BASELINE_VOL_THRESHOLD = float(os.getenv("BASELINE_VOL_THRESHOLD", "0.000048"))

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(MLFLOW_TRACKING_URI)


def _sha256_file(path: str) -> str:
    # Compute SHA256 of the pickle file for idempotency checks
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_production_version_with_hash(model_name: str, pickle_hash: str):
    # Return the first Production model version whose pickle_hash tag matches
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
    except Exception:
        # Model not registered yet
        return None
    for v in versions:
        if v.tags.get("pickle_hash") == pickle_hash:
            return v
    return None


def main():
    # ------------------------------------------------------------------
    # 1. Load pickle bundle and compute hash for idempotency
    # ------------------------------------------------------------------
    # Compute hash of the source pickle for matching against registry tags
    pickle_hash = _sha256_file(MODEL_ARTIFACT_PATH)

    # Look for an existing Production version tagged with this exact pickle hash
    existing = _get_production_version_with_hash(MODEL_NAME, pickle_hash)
    if existing is not None:
        # Registry remembers this version, but its artifact directory may have
        # been lost (e.g. a partial volume reset that kept the sqlite db but
        # dropped /mlruns/artifacts/). Probe by actually loading the model —
        # if the artifact download fails, treat it as not-registered and fall
        # through to the re-registration path below so cold-start recovers.
        try:
            mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
        except Exception as exc:
            print(
                f"Production version {existing.version} exists in the registry "
                f"but its artifacts are not loadable ({exc}); re-registering."
            )
        else:
            # Artifacts are present AND loadable — safe to skip re-upload
            print(
                f"Production version already exists with same pickle hash "
                f"(run_id={existing.run_id}). Skipping."
            )
            print(existing.run_id)
            sys.exit(0)

    with open(MODEL_ARTIFACT_PATH, "rb") as f:
        bundle = pickle.load(f)

    pipeline = bundle["pipeline"]
    feature_cols = bundle["feature_cols"]
    tau = bundle["tau"]

    # ------------------------------------------------------------------
    # 2. Load evaluation data — use only columns the model actually needs
    # ------------------------------------------------------------------
    df = pd.read_csv(FEATURES_PATH)
    # Verify all model feature columns exist in the CSV
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns in features CSV: {missing}", file=sys.stderr)
        sys.exit(1)

    X = df[feature_cols].to_numpy()
    y = df["vol_spike"].to_numpy()

    # ------------------------------------------------------------------
    # 3. Start MLflow run, log params and metrics, log the sklearn model
    # ------------------------------------------------------------------
    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # Log model parameters
        mlflow.log_param("feature_cols", ",".join(feature_cols))
        mlflow.log_param("tau", tau)
        mlflow.log_param("BASELINE_VOL_THRESHOLD", BASELINE_VOL_THRESHOLD)

        # Evaluate pipeline on the feature slice and log metrics
        y_prob = pipeline.predict_proba(X)[:, 1]
        pr_auc = average_precision_score(y, y_prob)
        roc_auc = roc_auc_score(y, y_prob)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("roc_auc", roc_auc)

        # Log the sklearn pipeline as a model artifact
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

    # ------------------------------------------------------------------
    # 4. Promote the just-registered version to Production
    # ------------------------------------------------------------------
    # get_latest_versions returns the newest version just registered above
    versions = client.get_latest_versions(MODEL_NAME, stages=["None"])
    # Sort by version number descending to get the one we just created
    versions_sorted = sorted(versions, key=lambda v: int(v.version), reverse=True)
    new_version = versions_sorted[0]

    # Tag it with the pickle hash so future runs can detect duplicates
    client.set_model_version_tag(
        MODEL_NAME, new_version.version, "pickle_hash", pickle_hash
    )

    # Transition to Production (archive any existing Production versions first)
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=new_version.version,
        stage="Production",
        archive_existing_versions=True,
    )

    print(run_id)


if __name__ == "__main__":
    main()
