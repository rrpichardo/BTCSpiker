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
MAX_RESPONSE_BYTES = 64 * 1024

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
            body = resp.read(MAX_RESPONSE_BYTES + 1)
            status = resp.getcode()
        latency_ms = (time.monotonic() - start) * 1000
        if len(body) > MAX_RESPONSE_BYTES:
            return (
                False,
                latency_ms,
                f"response too large (limit {MAX_RESPONSE_BYTES} bytes)",
                None,
            )
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
    the log has stalled while fresh events show the pipeline is running.
    """
    if body is None:
        return False, False, "no response body"
    try:
        payload = json.loads(body)
    except Exception as exc:
        return False, False, f"invalid JSON: {exc}"

    if not isinstance(payload, dict):
        return False, False, "invalid materializer health: expected JSON object"

    required_fields = {
        "ok",
        "last_event_ts",
        "last_write_ts",
        "rows_total",
        "consume_errors",
        "write_errors",
    }
    missing = required_fields - payload.keys()
    if missing:
        return (
            False,
            False,
            f"invalid materializer health: missing fields {sorted(missing)}",
        )
    if type(payload["ok"]) is not bool:
        return False, False, "invalid materializer health: ok must be a boolean"
    for field in ("rows_total", "consume_errors", "write_errors"):
        value = payload[field]
        if type(value) is not int:
            return (
                False,
                False,
                f"invalid materializer health: {field} must be an integer",
            )
        if value < 0:
            return (
                False,
                False,
                f"invalid materializer health: {field} must be non-negative",
            )

    parsed_timestamps: dict[str, datetime | None] = {}
    for field in ("last_event_ts", "last_write_ts"):
        raw_ts = payload[field]
        if raw_ts is None:
            parsed_timestamps[field] = None
            continue
        if not isinstance(raw_ts, str):
            return (
                False,
                False,
                f"invalid materializer health: {field} must be a timestamp or null",
            )
        try:
            parsed = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
        except ValueError:
            return False, False, f"invalid materializer health: invalid {field}"
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return (
                False,
                False,
                f"invalid materializer health: {field} must include a timezone",
            )
        parsed_timestamps[field] = parsed

    payload_ok = payload["ok"]
    detail = "ok" if payload_ok else "materializer reports ok=false"

    def _age_seconds(timestamp: datetime | None) -> float | None:
        if timestamp is None:
            return None
        return (datetime.now(timezone.utc) - timestamp).total_seconds()

    stale_seconds = _materializer_stale_seconds()
    event_age = _age_seconds(parsed_timestamps["last_event_ts"])
    write_age = _age_seconds(parsed_timestamps["last_write_ts"])
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
    timeout = _probe_timeout()
    futures = {name: _EXECUTOR.submit(_check_service, name) for name in SERVICE_NAMES}
    done, _ = concurrent.futures.wait(futures.values(), timeout=timeout)

    services = []
    for name in SERVICE_NAMES:
        future = futures[name]
        if future in done:
            services.append(future.result())
            continue
        future.cancel()
        services.append(
            {
                "name": name,
                "ok": False,
                "degraded": False,
                "latency_ms": round(timeout * 1000, 1),
                "detail": f"probe round timed out after {timeout:g}s",
                "open_url": _open_url(name),
            }
        )
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
