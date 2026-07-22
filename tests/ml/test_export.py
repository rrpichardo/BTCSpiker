import json
from pathlib import Path

import mlflow
import pytest

from btcspiker_ml.export import export_run


def _logged_run(tmp_path: Path) -> str:
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    mlflow.set_experiment("qualification-export")
    with mlflow.start_run() as run:
        for relative_path, content in {
            "model/model.bin": "model bytes",
            "configs/config.json": "{}",
            "manifests/dataset.json": "{}",
            "predictions/final.csv": "score",
            "plots/pr_auc.txt": "plot",
            "model-card.md": "# Model card",
            "qualification.json": '{"passed": true}',
            "dependencies.txt": "package==1.0",
        }.items():
            path = tmp_path / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            parent = Path(relative_path).parent
            mlflow.log_artifact(
                str(path), artifact_path=None if parent == Path(".") else str(parent)
            )
        return run.info.run_id


def test_export_run_copies_mlflow_artifacts_with_manifest_and_verified_checksums(
    tmp_path: Path,
):
    run_id = _logged_run(tmp_path)

    manifest = export_run(run_id, tmp_path / "exports")

    destination = tmp_path / "exports" / "mlflow-exports" / f"run_id={run_id}"
    assert manifest.destination == destination
    assert (destination / "export-manifest.json").exists()
    entries = json.loads((destination / "export-manifest.json").read_text())["files"]
    assert set(entries) >= {
        "model/model.bin",
        "configs/config.json",
        "manifests/dataset.json",
        "predictions/final.csv",
        "plots/pr_auc.txt",
        "model-card.md",
        "qualification.json",
        "dependencies.txt",
    }
    assert manifest.verify()


def test_export_run_refuses_to_replace_an_immutable_existing_export(tmp_path: Path):
    run_id = _logged_run(tmp_path)
    export_run(run_id, tmp_path / "exports")

    with pytest.raises(FileExistsError, match="immutable"):
        export_run(run_id, tmp_path / "exports")


def test_export_run_aborts_when_published_destination_checksum_is_corrupt(
    tmp_path: Path, monkeypatch
):
    run_id = _logged_run(tmp_path)
    import btcspiker_ml.export as exporter

    original = exporter._verify_export

    def corrupt_after_publish(destination, files):
        (destination / "model" / "model.bin").write_text("corrupted")
        return original(destination, files)

    monkeypatch.setattr(exporter, "_verify_export", corrupt_after_publish)

    with pytest.raises(OSError, match="checksum verification failed"):
        export_run(run_id, tmp_path / "exports")
