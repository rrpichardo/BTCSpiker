import json
from pathlib import Path

import pytest


@pytest.fixture
def raw_ticks():
    rows = []
    with Path("handoff/data_sample/raw_slice.ndjson").open() as handle:
        for line in handle:
            rows.append(json.loads(line))
            if len(rows) == 1500:
                break
    # The 60-second label contract needs more than the first 500 ticks in this
    # capture, which cover only about 40 seconds of event time.
    assert len(rows) == 1500
    return rows
