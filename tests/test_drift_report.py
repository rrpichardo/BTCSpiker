import logging

import pytest

from scripts.drift_report import main


def test_refuses_smoke_fixture_as_reference(monkeypatch, caplog):
    monkeypatch.setattr(
        "sys.argv",
        [
            "drift_report.py",
            "--reference",
            "handoff/data_sample/features_slice.csv",
            "--current",
            "handoff/data_sample/features_slice.csv",
            "--out",
            "/tmp/should-not-be-written.html",
        ],
    )
    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1
    assert "schema/smoke fixture" in caplog.text
