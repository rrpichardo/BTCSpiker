"""Unit tests for api/system.py.

Calls the route handler function directly and monkeypatches the probe seam
(`system._do_probe`) instead of hitting real network addresses. Never
imports api.main.
"""

import json
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from api import system

_DEFAULT_TS = object()


def _fresh_ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stale_ts() -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()


def _materializer_body(
    ok: bool = True,
    last_event_ts: str | None | object = _DEFAULT_TS,
    last_write_ts: str | None | object = _DEFAULT_TS,
) -> bytes:
    if last_event_ts is _DEFAULT_TS:
        last_event_ts = _fresh_ts()
    if last_write_ts is _DEFAULT_TS:
        last_write_ts = _fresh_ts()
    return json.dumps(
        {
            "ok": ok,
            "last_event_ts": last_event_ts,
            "last_write_ts": last_write_ts,
            "rows_total": 10,
            "consume_errors": 0,
            "write_errors": 0,
        }
    ).encode()


def _make_fake_do_probe(materializer_body=None, fail_for=None, call_log=None):
    """Build a fake replacing system._do_probe: (url, timeout) -> tuple."""

    def _fake(url, timeout):
        if call_log is not None:
            call_log.append(url)
        if fail_for and fail_for in url:
            raise RuntimeError("boom")
        if "materializer" in url:
            body = (
                materializer_body
                if materializer_body is not None
                else _materializer_body()
            )
            return True, 1.0, "ok", body
        return True, 1.0, "ok", b"{}"

    return _fake


@pytest.fixture(autouse=True)
def _reset_between_tests():
    system._reset_cache()
    yield
    system._reset_cache()


def test_all_probes_ok_all_green(monkeypatch):
    monkeypatch.setattr(system, "_do_probe", _make_fake_do_probe())

    body = system.get_system_status()

    assert len(body["services"]) == 5
    for svc in body["services"]:
        assert set(svc) == {
            "name",
            "ok",
            "degraded",
            "latency_ms",
            "detail",
            "open_url",
        }
        assert svc["ok"] is True
        assert svc["degraded"] is False
    assert "checked_at" in body


def test_one_probe_raising_does_not_affect_others(monkeypatch):
    monkeypatch.setattr(system, "_do_probe", _make_fake_do_probe(fail_for="prometheus"))

    body = system.get_system_status()
    by_name = {s["name"]: s for s in body["services"]}

    assert by_name["prometheus"]["ok"] is False
    assert "boom" in by_name["prometheus"]["detail"]
    for name in ("api", "grafana", "mlflow", "materializer"):
        assert by_name[name]["ok"] is True


def test_materializer_degraded_when_last_write_stale(monkeypatch):
    monkeypatch.setattr(
        system,
        "_do_probe",
        _make_fake_do_probe(
            materializer_body=_materializer_body(
                last_event_ts=_fresh_ts(), last_write_ts=_stale_ts()
            )
        ),
    )

    body = system.get_system_status()
    by_name = {s["name"]: s for s in body["services"]}

    assert by_name["materializer"]["ok"] is True
    assert by_name["materializer"]["degraded"] is True
    assert "stalled" in by_name["materializer"]["detail"]


def test_materializer_not_degraded_when_pipeline_is_stale(monkeypatch):
    monkeypatch.setattr(
        system,
        "_do_probe",
        _make_fake_do_probe(
            materializer_body=_materializer_body(
                last_event_ts=_stale_ts(), last_write_ts=_stale_ts()
            )
        ),
    )

    body = system.get_system_status()
    materializer = {s["name"]: s for s in body["services"]}["materializer"]

    assert materializer["ok"] is True
    assert materializer["degraded"] is False


def test_materializer_degraded_when_events_flow_before_first_write(monkeypatch):
    monkeypatch.setattr(
        system,
        "_do_probe",
        _make_fake_do_probe(
            materializer_body=_materializer_body(
                last_event_ts=_fresh_ts(), last_write_ts=None
            )
        ),
    )

    body = system.get_system_status()
    materializer = {s["name"]: s for s in body["services"]}["materializer"]

    assert materializer["ok"] is True
    assert materializer["degraded"] is True


def test_materializer_not_degraded_when_health_reports_pipeline_down(monkeypatch):
    monkeypatch.setattr(
        system,
        "_do_probe",
        _make_fake_do_probe(
            materializer_body=_materializer_body(
                ok=False, last_event_ts=_fresh_ts(), last_write_ts=_stale_ts()
            )
        ),
    )

    body = system.get_system_status()
    materializer = {s["name"]: s for s in body["services"]}["materializer"]

    assert materializer["ok"] is False
    assert materializer["degraded"] is False


def test_other_services_never_degraded(monkeypatch):
    monkeypatch.setattr(system, "_do_probe", _make_fake_do_probe())

    body = system.get_system_status()

    for svc in body["services"]:
        if svc["name"] != "materializer":
            assert svc["degraded"] is False


def test_cache_avoids_duplicate_probes_within_window(monkeypatch):
    call_log: list[str] = []
    monkeypatch.setattr(system, "_do_probe", _make_fake_do_probe(call_log=call_log))

    system.get_system_status()
    first_count = len(call_log)
    assert first_count == 5  # one probe per service

    system.get_system_status()
    assert len(call_log) == first_count  # second call served from cache

    system._reset_cache()
    system.get_system_status()
    assert len(call_log) == first_count * 2  # cache cleared -> probed again


def test_cache_expires_after_ttl(monkeypatch):
    call_log: list[str] = []
    clock = [100.0]
    monkeypatch.setattr(system, "_do_probe", _make_fake_do_probe(call_log=call_log))
    monkeypatch.setattr(system.time, "monotonic", lambda: clock[0])

    system.get_system_status()
    clock[0] = 104.9
    system.get_system_status()
    assert len(call_log) == 5

    clock[0] = 105.1
    system.get_system_status()
    assert len(call_log) == 10


def test_probes_run_concurrently(monkeypatch):
    rendezvous = threading.Barrier(len(system.SERVICE_NAMES))

    def _concurrent_probe(url, timeout):
        rendezvous.wait(timeout=1)
        body = _materializer_body() if "materializer" in url else b"{}"
        return True, 1.0, "ok", body

    monkeypatch.setattr(system, "_do_probe", _concurrent_probe)

    body = system.get_system_status()

    assert all(service["ok"] for service in body["services"])


def test_probe_round_returns_timeout_result_at_overall_deadline(monkeypatch):
    release = threading.Event()

    def _one_hung_probe(url, timeout):
        if "prometheus" in url:
            release.wait(timeout=1)
        body = _materializer_body() if "materializer" in url else b"{}"
        return True, 1.0, "ok", body

    monkeypatch.setattr(system, "_do_probe", _one_hung_probe)
    monkeypatch.setenv("SYSTEM_PROBE_TIMEOUT", "0.05")

    started = time.monotonic()
    try:
        body = system.get_system_status()
    finally:
        release.set()
    elapsed = time.monotonic() - started
    prometheus = {s["name"]: s for s in body["services"]}["prometheus"]

    assert elapsed < 0.5
    assert prometheus["ok"] is False
    assert prometheus["degraded"] is False
    assert "timed out" in prometheus["detail"]


def test_probe_reads_are_bounded(monkeypatch):
    read_limits = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, limit):
            read_limits.append(limit)
            return b"x" * limit

        def getcode(self):
            return 200

    monkeypatch.setattr(
        system.urllib.request, "urlopen", lambda url, timeout: _Response()
    )

    ok, _, detail, body = system._do_probe("http://example.test", 2)

    assert read_limits == [system.MAX_RESPONSE_BYTES + 1]
    assert ok is False
    assert "too large" in detail
    assert body is None


@pytest.mark.parametrize(
    ("payload", "detail_fragment"),
    [
        (["not", "an", "object"], "JSON object"),
        (
            {
                "ok": "false",
                "last_event_ts": None,
                "last_write_ts": None,
                "rows_total": 0,
                "consume_errors": 0,
                "write_errors": 0,
            },
            "ok must be a boolean",
        ),
        (
            {
                "ok": True,
                "last_event_ts": "not-a-timestamp",
                "last_write_ts": None,
                "rows_total": 0,
                "consume_errors": 0,
                "write_errors": 0,
            },
            "last_event_ts",
        ),
        (
            {
                "ok": True,
                "last_event_ts": None,
                "last_write_ts": None,
                "rows_total": "0",
                "consume_errors": 0,
                "write_errors": 0,
            },
            "rows_total must be an integer",
        ),
    ],
)
def test_materializer_health_contract_is_strict(payload, detail_fragment, monkeypatch):
    monkeypatch.setattr(
        system,
        "_do_probe",
        _make_fake_do_probe(materializer_body=json.dumps(payload).encode()),
    )

    body = system.get_system_status()
    materializer = {s["name"]: s for s in body["services"]}["materializer"]

    assert materializer["ok"] is False
    assert materializer["degraded"] is False
    assert detail_fragment in materializer["detail"]


def test_env_url_override_respected(monkeypatch):
    monkeypatch.setenv("SYSTEM_PROBE_API_URL", "http://example.test/health")
    monkeypatch.setenv("SYSTEM_OPEN_API_URL", "http://example.test/")

    seen_urls = []

    def _fake(url, timeout):
        seen_urls.append(url)
        return True, 1.0, "ok", b"{}"

    monkeypatch.setattr(system, "_do_probe", _fake)

    body = system.get_system_status()
    by_name = {s["name"]: s for s in body["services"]}

    assert "http://example.test/health" in seen_urls
    assert by_name["api"]["open_url"] == "http://example.test/"
