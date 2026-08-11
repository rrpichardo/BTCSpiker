"""Read-only projection of the offline tournament's MLflow file store.

`scripts/run_experiments.py` writes to a local file-store MLflow, deliberately
separate from the serving registry so the tournament can never promote to
Production (see CLAUDE.md, "Model tournament & promotion"). This module reads
that directory so the UI's Tournament tab can show what was tried and why one
candidate beat another.

Read-only by construction: nothing here writes, and the directory is mounted
`:ro` into the container. Stdlib only, matching the materializer's existing
dependency footprint (same rationale as `evaluation.py`).

MLflow's file-store layout is stable and flat:

    <root>/<experiment_id>/meta.yaml               experiment metadata
    <root>/<experiment_id>/<run_id>/meta.yaml      flat `key: value`
    <root>/<experiment_id>/<run_id>/params/<name>  one raw value per file
    <root>/<experiment_id>/<run_id>/metrics/<name> "<ts_ms> <value> <step>" lines
    <root>/<experiment_id>/<run_id>/tags/<name>    one raw value per file
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# mlflow.entities.RunStatus, which the file store persists as a bare integer.
_RUN_STATUS = {
    1: "RUNNING",
    2: "SCHEDULED",
    3: "FINISHED",
    4: "FAILED",
    5: "KILLED",
}

_FOLD_METRIC = re.compile(r"^fold_(\d+)_pr_auc$")

# A run_id reaches get_run() straight from the URL path, so it is restricted to
# the shape MLflow actually generates rather than sanitized after the fact.
_RUN_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class TournamentUnavailable(RuntimeError):
    """The tournament store is absent, or present but holding nothing.

    Both map to the same 'not wired up' signal: Docker Compose auto-creates
    a bind-mount source directory that doesn't exist on the host as an empty
    directory before the container starts, so on a fresh clone (no tournament
    ever run) the mounted root exists and is merely empty rather than
    missing. Treating "exists, zero entries" as a distinct state from
    "missing" would silently turn the intended 'run scripts/run_experiments.py'
    message into a misleading 'ran, found nothing' one on every fresh clone.
    """


def _read_flat_yaml(path: Path) -> dict[str, str]:
    """Parse MLflow's flat `key: value` meta.yaml.

    Deliberately not a general YAML parser. Values routinely contain colons
    (`artifact_uri: file:///...`), so the split is bounded to the first one,
    and MLflow quotes numeric-looking strings (`experiment_id: '123'`).
    """
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, _, raw = line.partition(":")
        value = raw.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def _read_metric(path: Path) -> float | None:
    """Last recorded value of one metric, or None if unreadable.

    Each line is "<timestamp_ms> <value> <step>"; the tournament logs each
    metric once, but taking the last line is correct either way.
    """
    try:
        lines = [line for line in path.read_text().splitlines() if line.strip()]
    except OSError:
        return None
    if not lines:
        return None
    parts = lines[-1].split()
    if len(parts) < 2:
        return None
    try:
        return float(parts[1])
    except ValueError:
        return None


def _read_dir(directory: Path) -> dict[str, str]:
    """Every file in a params/ or tags/ directory as {name: contents}."""
    if not directory.is_dir():
        return {}
    out: dict[str, str] = {}
    for entry in sorted(directory.iterdir()):
        if entry.is_file():
            try:
                out[entry.name] = entry.read_text().strip()
            except OSError:
                continue
    return out


def _read_metrics(directory: Path) -> dict[str, float]:
    if not directory.is_dir():
        return {}
    out: dict[str, float] = {}
    for entry in sorted(directory.iterdir()):
        if not entry.is_file():
            continue
        value = _read_metric(entry)
        if value is not None:
            out[entry.name] = value
    return out


def _iter_run_dirs(root: Path):
    """Yield every run directory under every experiment in the store."""
    for experiment in sorted(root.iterdir()):
        if not experiment.is_dir() or experiment.name.startswith("."):
            continue
        for run in sorted(experiment.iterdir()):
            # A run always carries meta.yaml; this also skips the experiment's
            # own meta.yaml and MLflow's sibling bookkeeping directories.
            if run.is_dir() and (run / "meta.yaml").is_file():
                yield run


def _summarize(run_dir: Path) -> dict:
    meta = _read_flat_yaml(run_dir / "meta.yaml")
    params = _read_dir(run_dir / "params")
    tags = _read_dir(run_dir / "tags")
    metrics = _read_metrics(run_dir / "metrics")

    try:
        status = _RUN_STATUS.get(int(meta.get("status", "")), "UNKNOWN")
    except ValueError:
        status = "UNKNOWN"

    duration_seconds = None
    try:
        start, end = int(meta.get("start_time", 0)), int(meta.get("end_time", 0))
        if start and end >= start:
            duration_seconds = (end - start) / 1000.0
    except ValueError:
        duration_seconds = None

    return {
        "run_id": run_dir.name,
        "run_name": meta.get("run_name") or tags.get("mlflow.runName"),
        "stage": tags.get("candidate_stage"),
        "model_family": params.get("model_family"),
        "aggregate_pr_auc": metrics.get("aggregate_pr_auc"),
        "folds_won": metrics.get("development_folds_won"),
        "deployable": tags.get("deployable", "").lower() == "true",
        "status": status,
        "duration_seconds": duration_seconds,
        "_dir": run_dir,
        "_params": params,
        "_tags": tags,
        "_metrics": metrics,
    }


def _public(summary: dict) -> dict:
    return {k: v for k, v in summary.items() if not k.startswith("_")}


def list_runs(root: Path) -> list[dict]:
    """Leaderboard: every recorded run, best aggregate PR-AUC first.

    Runs with no PR-AUC yet (still running, or failed before scoring) sort
    last rather than being hidden — a failed stage is information.
    """
    root = Path(root)
    if not root.is_dir() or not any(root.iterdir()):
        raise TournamentUnavailable(f"no tournament store at {root}")

    summaries = [_summarize(run_dir) for run_dir in _iter_run_dirs(root)]
    summaries.sort(
        key=lambda s: (
            s["aggregate_pr_auc"] is None,
            -(s["aggregate_pr_auc"] or 0.0),
            s["run_id"],
        )
    )
    return [_public(s) for s in summaries]


def get_run(root: Path, run_id: str) -> dict:
    """Full detail for one run: settings chosen, per-fold scores, artifacts."""
    root = Path(root)
    if not root.is_dir() or not any(root.iterdir()):
        raise TournamentUnavailable(f"no tournament store at {root}")
    if not _RUN_ID.match(run_id or ""):
        raise KeyError(run_id)

    for run_dir in _iter_run_dirs(root):
        if run_dir.name == run_id:
            break
    else:
        raise KeyError(run_id)

    summary = _summarize(run_dir)
    params, metrics = summary["_params"], summary["_metrics"]

    model_params = None
    if "model_params" in params:
        try:
            model_params = json.loads(params["model_params"])
        except json.JSONDecodeError:
            # Keep the raw string visible rather than dropping it silently.
            model_params = params["model_params"]

    folds = []
    for name, value in metrics.items():
        match = _FOLD_METRIC.match(name)
        if match:
            folds.append({"fold": int(match.group(1)), "pr_auc": value})
    folds.sort(key=lambda f: f["fold"])

    artifacts_dir = run_dir / "artifacts"
    artifacts = (
        sorted(entry.name for entry in artifacts_dir.iterdir())
        if artifacts_dir.is_dir()
        else []
    )

    detail = _public(summary)
    detail.update(
        {
            "model_params": model_params,
            "fold_pr_aucs": folds,
            "params": params,
            "tags": summary["_tags"],
            "metrics": metrics,
            "artifacts": artifacts,
        }
    )
    return detail
