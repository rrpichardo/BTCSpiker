"""
Deterministic end-to-end verification for the corrupt-fixture repair (see
handoff/data_sample/manifest.json and materializer/evaluation.py).

Brings up an ISOLATED Compose project (distinct --project-name, so Docker
allocates fresh, non-colliding named volumes automatically), replays the
now-deduplicated fixture through the real pipeline exactly ONCE (no
`--loop`, so "one complete replay pass" is just "the replay process
exited"), and asserts the numbers the duplicated-tick defect corrupted are
now correct: alert rate no longer ~100%, base rate no longer 0%, PR-AUC no
longer suppressed, and one pass's rows are not double-counted.

Isolated by project name, NOT by port remapping -- docker-compose.yaml's
host ports (8000, 8090, 5001, 9092) are fixed, so this script refuses to
run if a live `btcspiker` project already has containers on those ports,
rather than silently colliding with it.

Every wait is condition-based (materializer /health, the replay subprocess
exiting, /predictions/performance's own counters), never a fixed sleep -- a
prior draft of this check used `docker compose up && sleep 90`, which
cannot produce its expected numbers: the fixture spans 600s and
REPLAY_SPEED defaults to 1.0, so 90s yields ~150 ticks, not the full 3578.

Usage:
    python scripts/verify_e2e.py
    python scripts/verify_e2e.py --keep-up   # skip teardown, for debugging
"""

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("verify_e2e")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPOSE_FILE = PROJECT_ROOT / "docker-compose.yaml"
PROJECT_NAME = "btcspiker-verify"

# Everything the pipeline needs EXCEPT the replay ingestor, which is run
# separately as a one-shot `docker compose run` (see main()) so "one replay
# pass" is an exact, waitable event rather than a log line to scrape.
LONG_RUNNING_SERVICES = [
    "kafka", "kafka-init", "mlflow", "mlflow-init",
    "api", "featurizer", "predict-bridge", "materializer",
]
HOST_PORTS = [8000, 8090, 5001, 9092]  # api, materializer, mlflow, kafka

MATERIALIZER_URL = "http://localhost:8090"
API_URL = "http://localhost:8000"

REPLAY_SPEED = "50"
STARTUP_WAIT_TIMEOUT_SECONDS = 300
REPLAY_TIMEOUT_SECONDS = 120  # 600s fixture / 50x speed = 12s, generous slack for build/boot
GRADING_TIMEOUT_SECONDS = 240  # 60s label horizon + slack for the pipeline to fully drain

MANIFEST_PATH = PROJECT_ROOT / "handoff" / "data_sample" / "manifest.json"


class VerificationFailed(RuntimeError):
    pass


def _compose(*args: str, extra_env: dict | None = None) -> subprocess.CompletedProcess:
    cmd = ["docker", "compose", "-p", PROJECT_NAME, "-f", str(COMPOSE_FILE), *args]
    log.info("$ %s", " ".join(cmd))
    env = {**os.environ, **(extra_env or {})}
    return subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def _refuse_if_ports_busy() -> None:
    busy = [p for p in HOST_PORTS if not _port_free(p)]
    if busy:
        raise VerificationFailed(
            f"host port(s) {busy} already in use. docker-compose.yaml's ports are "
            "fixed (not parameterized), so this isolated project cannot run "
            "alongside a live `btcspiker` stack. Stop it first: "
            "`docker compose down` (from the checkout that has it up), then retry."
        )


def _ensure_env_file() -> None:
    # api mounts ./.env read-only (docker-compose.yaml); a missing file
    # becomes an empty directory under Docker's default bind-mount behavior,
    # which 500s /settings -- irrelevant to this check, but cheap to avoid.
    env_path = PROJECT_ROOT / ".env"
    example_path = PROJECT_ROOT / ".env.example"
    if not env_path.exists() and example_path.exists():
        log.warning("%s missing; copying %s per docker-compose.yaml's setup note", env_path, example_path)
        env_path.write_text(example_path.read_text())


def _get_json(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def _wait_until(description: str, check, timeout_seconds: float, interval: float = 3.0):
    deadline = time.monotonic() + timeout_seconds
    last_exc = None
    while time.monotonic() < deadline:
        try:
            result = check()
            if result:
                log.info("ok: %s", description)
                return result
        except Exception as exc:  # noqa: BLE001 - surfaced only on final timeout
            last_exc = exc
        time.sleep(interval)
    detail = f" (last error: {last_exc})" if last_exc else ""
    raise VerificationFailed(f"timed out after {timeout_seconds:.0f}s waiting for: {description}{detail}")


def _materializer_healthy() -> bool:
    try:
        return _get_json(f"{MATERIALIZER_URL}/health").get("ok") is True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _performance(window_minutes: int = 30) -> dict:
    return _get_json(f"{MATERIALIZER_URL}/predictions/performance?window_minutes={window_minutes}")


def _expected_source_row_count() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text())
    return manifest["artifacts"]["raw_slice.ndjson"]["rows"]


def _dump_logs(*services: str) -> None:
    for service in services:
        log.error("--- docker compose logs %s (tail) ---", service)
        subprocess.run(
            ["docker", "compose", "-p", PROJECT_NAME, "-f", str(COMPOSE_FILE),
             "logs", "--tail=50", service],
            cwd=PROJECT_ROOT, check=False,
        )


def _run_verification() -> dict:
    """Runs the pipeline and returns the final /predictions/performance body.
    Raises VerificationFailed on any assertion failure."""
    expected_rows = _expected_source_row_count()
    log.info("Expecting %d source rows (one deduplicated replay pass).", expected_rows)

    log.info("Building and starting %s...", LONG_RUNNING_SERVICES)
    _compose(
        "up", "-d", "--build", "--wait",
        "--wait-timeout", str(STARTUP_WAIT_TIMEOUT_SECONDS),
        *LONG_RUNNING_SERVICES,
    )
    # Belt-and-suspenders on top of --wait: materializer's own healthcheck
    # already asserts ok==True, but confirm directly before triggering replay.
    _wait_until("materializer reports ok=true", _materializer_healthy, STARTUP_WAIT_TIMEOUT_SECONDS)

    log.info("Running one replay pass at %sx speed (no --loop)...", REPLAY_SPEED)
    replay = subprocess.run(
        ["docker", "compose", "-p", PROJECT_NAME, "-f", str(COMPOSE_FILE),
         "run", "--rm", "-e", f"REPLAY_SPEED={REPLAY_SPEED}", "ingestor",
         "python", "-u", "scripts/replay_to_kafka.py",
         "--file", "/app/data_sample/raw_slice.ndjson", "--speed", REPLAY_SPEED],
        cwd=PROJECT_ROOT, timeout=REPLAY_TIMEOUT_SECONDS,
    )
    if replay.returncode != 0:
        raise VerificationFailed(f"replay process exited {replay.returncode}")
    log.info("ok: one complete replay pass observed (process exited 0)")

    def _fully_graded():
        body = _performance(window_minutes=30)
        window = body["window"]
        if window["n_joined"] < expected_rows:
            return None  # still draining ticks.predictions
        if window["n_graded"] <= 0:
            return None  # outcome horizon hasn't closed / graded yet
        if window["stream_id"] is None:
            return None  # lineage not yet visible (shouldn't happen post-A4)
        return body

    body = _wait_until(
        "pipeline drains, outcome horizon closes, and an active stream_id is selected",
        _fully_graded, GRADING_TIMEOUT_SECONDS,
    )
    return body


def _assert_expected_numbers(body: dict, expected_rows: int) -> list[str]:
    """Returns a list of human-readable PASS/FAIL lines; raises on any FAIL."""
    window = body["window"]
    official = body["modes"]["official"]
    ml_series = next((s for s in official["series"] if s["model_variant"] == "ml"), None)
    lines = []
    failures = []

    def check(label, condition, detail):
        status = "PASS" if condition else "FAIL"
        lines.append(f"[{status}] {label}: {detail}")
        if not condition:
            failures.append(label)

    check(
        "source_row_count (no double-counting)",
        window["n_joined"] == expected_rows,
        f"n_joined={window['n_joined']} expected={expected_rows}",
    )
    check(
        "active segment selected",
        window["stream_id"] is not None,
        f"stream_id={window['stream_id']!r}",
    )
    check(
        "no silently-dropped legacy rows",
        window["n_unknown_lineage"] == 0,
        f"n_unknown_lineage={window['n_unknown_lineage']}",
    )
    check(
        "grading occurred",
        window["n_graded"] > 0,
        f"n_graded={window['n_graded']}",
    )
    check(
        "base rate is not the corrupted 0.0",
        official["base_rate"] is not None and 0.10 < official["base_rate"] < 0.40,
        f"base_rate={official['base_rate']} (manifest prevalence ~0.254; corrupt-fixture reading was 0.0)",
    )
    if ml_series is not None:
        alert_rate = ml_series["fp"] / ml_series["n"] if ml_series["n"] else None
        check(
            "ML alert rate is not the corrupted ~100%",
            alert_rate is not None and alert_rate < 0.5,
            f"fp/n={ml_series['fp']}/{ml_series['n']} = {alert_rate} (corrupt-fixture reading was 0.995)",
        )
        check(
            "ML PR-AUC is not suppressed",
            ml_series["pr_auc"] is not None,
            f"pr_auc={ml_series['pr_auc']} (corrupt-fixture reading was null)",
        )
    else:
        check("ML series present", False, "no ml-variant series in modes.official.series")

    chart_epoch = body["chart"]["stream_epoch"]
    window_epoch = int(window["stream_id"].split(":")[-1]) if window["stream_id"] else None
    check(
        "metrics and chart describe the same segment",
        chart_epoch == window_epoch,
        f"chart.stream_epoch={chart_epoch} window.stream_id's epoch={window_epoch}",
    )
    check(
        "chart downsample bounded by source population",
        body["chart"]["rendered_row_count"] <= window["n_joined"],
        f"rendered_row_count={body['chart']['rendered_row_count']} n_joined={window['n_joined']}",
    )

    return lines, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-up", action="store_true", help="skip teardown, for debugging")
    args = parser.parse_args()

    _refuse_if_ports_busy()
    _ensure_env_file()
    expected_rows = _expected_source_row_count()

    try:
        body = _run_verification()
        lines, failures = _assert_expected_numbers(body, expected_rows)
    except VerificationFailed as exc:
        log.error("Verification failed: %s", exc)
        _dump_logs("kafka", "mlflow-init", "api", "featurizer", "predict-bridge", "materializer")
        if not args.keep_up:
            _compose("down", "-v")
        return 1
    except Exception:
        log.exception("Unexpected error during verification")
        _dump_logs("kafka", "mlflow-init", "api", "featurizer", "predict-bridge", "materializer")
        if not args.keep_up:
            _compose("down", "-v")
        return 1

    print("\n".join(lines))

    if not args.keep_up:
        _compose("down", "-v")
    else:
        log.info("--keep-up passed; stack left running under project %r", PROJECT_NAME)
        log.info("Tear down manually with: docker compose -p %s down -v", PROJECT_NAME)

    if failures:
        log.error("%d check(s) failed: %s", len(failures), failures)
        return 1
    log.info("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
