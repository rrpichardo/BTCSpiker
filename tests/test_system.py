"""Unit tests for api/system.py.

Calls the route handler function directly and monkeypatches the probe seam
(`system._do_probe`) instead of hitting real network addresses. Never
imports api.main.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from api import system

FRESH_TS = datetime.now(timezone.utc).isoformat()
STALE_TS = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()


def _materializer_body(
    ok: bool = True,
    last_event_ts: str | None = FRESH_TS,
    last_write_ts: str | None = FRESH_TS,
) -> bytes:
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
                last_event_ts=FRESH_TS, last_write_ts=STALE_TS
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
                last_event_ts=STALE_TS, last_write_ts=STALE_TS
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
                last_event_ts=FRESH_TS, last_write_ts=None
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
                ok=False, last_event_ts=FRESH_TS, last_write_ts=STALE_TS
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
