import subprocess
import sys
from pathlib import Path


def test_load_test_cli_exposes_configurable_request_options():
    script = Path("tests/load_test.py")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--url" in result.stdout
    assert "--requests" in result.stdout
    assert "--concurrency" in result.stdout
