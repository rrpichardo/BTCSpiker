"""
GET /system/status — read-only health rollup across the whole stack.

Probes every service's health endpoint over plain HTTP (stdlib urllib, no
new dependency) from a single shared thread pool, and caches the combined
result briefly so several browser tabs polling this endpoint don't multiply
the number of probes hitting the stack.
"""

import concurrent.futures
import json
import os
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from fastapi import APIRouter

router = APIRouter()

SERVICE_NAMES = ["api", "grafana", "prometheus", "mlflow", "materializer"]

_DEFAULT_PROBE_URLS = {
    "api": "http://localhost:8000/health",
    "grafana": "http://grafana:3000/api/health",
    "prometheus": "http://prometheus:9090/-/healthy",
    "mlflow": "http://mlflow:5000/health",
    "materializer": "http://materializer:8090/health",
}

_DEFAULT_OPEN_URLS = {
    "api": "http://localhost:8000/docs",
    "grafana": "http://localhost:3000",
    "prometheus": "http://localhost:9090",
    "mlflow": "http://localhost:5001",
    "materializer": None,
}


def _probe_url(name: str) -> str:
    return os.environ.get(f"SYSTEM_PROBE_{name.upper()}_URL", _DEFAULT_PROBE_URLS[name])


def _open_url(name: str) -> str | None:
    return os.environ.get(f"SYSTEM_OPEN_{name.upper()}_URL", _DEFAULT_OPEN_URLS[name])


def _probe_timeout() -> float:
    return float(os.environ.get("SYSTEM_PROBE_TIMEOUT", "2"))


def _cache_seconds() -> float:
    return float(os.environ.get("SYSTEM_STATUS_CACHE_SECONDS", "5"))


def _materializer_stale_seconds() -> float:
    return float(os.environ.get("SYSTEM_MATERIALIZER_STALE_SECONDS", "60"))


# One shared pool for the process (not created per request) so concurrent
# requests reuse the same worker threads instead of spawning a fresh pool.
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=len(SERVICE_NAMES), thread_name_prefix="system-probe"
)

_cache_lock = threading.Lock()
_cache: dict | None = None
_cache_ts: float = 0.0


def _reset_cache() -> None:
    """Test seam: clear the cached /system/status response."""
    global _cache, _cache_ts
    with _cache_lock:
        _cache = None
        _cache_ts = 0.0


def _do_probe(url: str, timeout: float) -> tuple[bool, float | None, str, bytes | None]:
    """Perform one HTTP GET. Returns (responding, latency_ms, detail, body).
    `responding` is True only on HTTP 200; any other status or exception is
    treated as not responding, with the error described in `detail`."""
    start = time.monotonic()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read()
            status = resp.getcode()
        latency_ms = (time.monotonic() - start) * 1000
        if status == 200:
            return True, latency_ms, "ok", body
        return False, latency_ms, f"HTTP {status}", body
    except Exception as exc:
        latency_ms = (time.monotonic() - start) * 1000
        return False, latency_ms, str(exc), None


def _check_materializer(body: bytes | None) -> tuple[bool, bool, str]:
    """Interpret the materializer's /health payload (pinned contract):
    {"ok": bool, "last_event_ts": str|null, "last_write_ts": str|null,
     "rows_total": int, "consume_errors": int, "write_errors": int}

    Returns (ok, degraded, detail). `degraded` means the service is up but
    the prediction log has stalled while fresh events show the pipeline is running.
    """
    if body is None:
        return False, False, "no response body"
    try:
        payload = json.loads(body)
    except Exception as exc:
        return False, False, f"invalid JSON: {exc}"

    payload_ok = bool(payload.get("ok"))
    detail = "ok" if payload_ok else "materializer reports ok=false"

    def _age_seconds(raw_ts) -> float | None:
        if not raw_ts:
            return None
        try:
            ts = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - ts).total_seconds()
        except (ValueError, TypeError):
            return None

    stale_seconds = _materializer_stale_seconds()
    event_age = _age_seconds(payload.get("last_event_ts"))
    write_age = _age_seconds(payload.get("last_write_ts"))
    pipeline_running = (
        payload_ok and event_age is not None and event_age <= stale_seconds
    )
    degraded = pipeline_running and (write_age is None or write_age > stale_seconds)
    if degraded:
        write_detail = (
            "has never written"
            if write_age is None
            else f"last wrote {write_age:.0f}s ago"
        )
        detail = (
            f"materializer is receiving events but {write_detail} "
            f"(stale after {stale_seconds:.0f}s) — the prediction log appears stalled"
        )

    return payload_ok, degraded, detail


def _check_service(name: str) -> dict:
    try:
        responding, latency_ms, detail, body = _do_probe(
            _probe_url(name), _probe_timeout()
        )
        ok = responding
        degraded = False

        if name == "materializer" and responding:
            ok, degraded, detail = _check_materializer(body)

        return {
            "name": name,
            "ok": ok,
            "degraded": degraded,
            "latency_ms": None if latency_ms is None else round(latency_ms, 1),
            "detail": detail,
            "open_url": _open_url(name),
        }
    except Exception as exc:
        # Belt-and-suspenders: a probe failure must never take down the
        # whole status response — the other services are unaffected.
        return {
            "name": name,
            "ok": False,
            "degraded": False,
            "latency_ms": None,
            "detail": str(exc),
            "open_url": _open_url(name),
        }


def _compute_status() -> dict:
    futures = {name: _EXECUTOR.submit(_check_service, name) for name in SERVICE_NAMES}
    services = [futures[name].result() for name in SERVICE_NAMES]
    return {
        "services": services,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/system/status")
def get_system_status():
    """Return cached (or freshly probed) health for every stack service."""
    global _cache, _cache_ts
    with _cache_lock:
        now = time.monotonic()
        if _cache is not None and (now - _cache_ts) < _cache_seconds():
            return _cache
        # Held across the probe round on purpose: concurrent requests during
        # a cache miss wait here and then get the same freshly-cached result
        # instead of each triggering their own probe round.
        result = _compute_status()
        _cache = result
        _cache_ts = time.monotonic()
        return result
