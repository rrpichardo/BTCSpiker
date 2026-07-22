from datetime import timedelta
from decimal import Decimal

import pytest

from btcspiker_data.book_replay import BookReplayError, publish_replay_partitions, replay_day


def test_replay_reconstructs_bbo_and_removes_zero_quantity_level(
    replay_anchor, replay_delta_rows, explicit_source_gap
):
    replay_delta_rows[0]["changes"] = (
        ("bid", Decimal("89999.90"), Decimal("2")),
        ("bid", Decimal("90000.00"), Decimal("0")),
    )
    replay_delta_rows[0]["best_bid"] = Decimal("89999.90")

    states = list(
        replay_day(
            anchors=[replay_anchor],
            deltas=replay_delta_rows,
            metadata=[],
            day_start=replay_anchor["anchor_second"],
            day_end=replay_anchor["anchor_second"] + timedelta(seconds=3),
        )
    )

    assert [state.observed_through for state in states] == sorted(
        state.observed_through for state in states
    )
    assert states[0].best_bid == Decimal("90000.00")
    assert states[0].bid_size == Decimal("1.25")
    assert states[1].best_bid == Decimal("89999.90")
    assert states[1].bid_size == Decimal("2")
    assert states[1].best_ask == Decimal("90000.10")
    assert states[1].ask_size == Decimal("0.75")


def test_replay_skips_explicit_gaps_and_starts_new_segment(replay_anchor, replay_delta_rows, explicit_source_gap):
    states = list(
        replay_day(
            anchors=[replay_anchor],
            deltas=replay_delta_rows,
            metadata=[{"window_start": explicit_source_gap[0], "window_end": explicit_source_gap[1], "status": "gap"}],
            day_start=replay_anchor["anchor_second"],
            day_end=explicit_source_gap[1] + timedelta(seconds=1),
        )
    )

    assert explicit_source_gap[0] not in {state.observed_through for state in states}
    resumed = next(state for state in states if state.observed_through == explicit_source_gap[1] + timedelta(seconds=1))
    assert resumed.segment_id == 1


def test_replay_fails_closed_when_source_bbo_disagrees(replay_anchor, replay_delta_rows):
    replay_delta_rows[0]["best_bid"] = Decimal("99999")

    with pytest.raises(BookReplayError, match="source BBO mismatch"):
        list(replay_day(anchors=[replay_anchor], deltas=replay_delta_rows, metadata=[], day_start=replay_anchor["anchor_second"]))


def test_replay_fails_closed_on_sequence_regression(replay_anchor, replay_delta_rows, sequence_regression):
    with pytest.raises(BookReplayError, match="sequence regression"):
        list(replay_day(anchors=[replay_anchor], deltas=[*replay_delta_rows, sequence_regression], metadata=[], day_start=replay_anchor["anchor_second"]))


def test_replay_applies_deltas_between_anchor_and_day_start(replay_anchor, replay_delta_rows):
    states = list(
        replay_day(
            anchors=[replay_anchor],
            deltas=replay_delta_rows[:2],
            metadata=[],
            day_start=replay_anchor["anchor_second"] + timedelta(seconds=3),
            day_end=replay_anchor["anchor_second"] + timedelta(seconds=3),
        )
    )

    assert states[0].bid_size == Decimal("1.45")


def test_replay_publishes_raw_deltas_and_derived_states(tmp_path, replay_anchor, replay_delta_rows):
    states = list(replay_day(anchors=[replay_anchor], deltas=replay_delta_rows, metadata=[], day_start=replay_anchor["anchor_second"]))

    records = publish_replay_partitions(
        deltas=replay_delta_rows, states=states, root=tmp_path, source_revision="pinned", source_date="2026-04-24"
    )

    assert {"book_deltas", "book_states"} == {
        next(part.split("=", 1)[1] for part in record.path.parts if part.startswith("kind="))
        for record in records
    }
