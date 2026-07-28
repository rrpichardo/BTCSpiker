from datetime import timedelta
from decimal import Decimal
import json

import pyarrow.parquet as pq
import pytest

from btcspiker_data.book_replay import (
    BookReplayError,
    publish_replay_partitions,
    replay_day,
)
from btcspiker_data.contracts import RAW_BOOK_COLUMNS


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


def test_replay_does_not_carry_state_after_gap_without_post_gap_anchor(
    replay_anchor, replay_delta_rows, explicit_source_gap
):
    states = list(
        replay_day(
            anchors=[replay_anchor],
            deltas=replay_delta_rows,
            metadata=[
                {
                    "window_start": explicit_source_gap[0],
                    "window_end": explicit_source_gap[1],
                    "status": "gap",
                }
            ],
            day_start=replay_anchor["anchor_second"],
            day_end=explicit_source_gap[1] + timedelta(seconds=1),
        )
    )

    assert explicit_source_gap[0] not in {state.observed_through for state in states}
    assert not [
        state for state in states if state.observed_through >= explicit_source_gap[1]
    ]


def test_replay_does_not_apply_post_gap_delta_without_full_anchor(
    replay_anchor, replay_delta_rows, explicit_source_gap
):
    post_gap_delta = {
        **replay_delta_rows[-1],
        "changed_second": explicit_source_gap[1] + timedelta(seconds=1),
        "source_sequence_num_start": 500,
        "source_sequence_num_end": 500,
        "changes": (("bid", Decimal("90000.00"), Decimal("9")),),
    }

    states = list(
        replay_day(
            anchors=[replay_anchor],
            deltas=[*replay_delta_rows, post_gap_delta],
            metadata=[
                {
                    "window_start": explicit_source_gap[0],
                    "window_end": explicit_source_gap[1],
                    "status": "gap",
                }
            ],
            day_start=replay_anchor["anchor_second"],
            day_end=post_gap_delta["changed_second"],
        )
    )

    assert not [
        state for state in states if state.observed_through >= explicit_source_gap[1]
    ]


def test_replay_resumes_from_post_gap_anchor_in_new_segment(
    replay_anchor, replay_delta_rows, explicit_source_gap
):
    resumed_at = explicit_source_gap[1]
    post_gap_anchor = {
        **replay_anchor,
        "anchor_second": resumed_at,
        "source_sequence_num": 500,
        "best_bid": Decimal("91000.00"),
        "best_ask": Decimal("91000.10"),
        "bid_book": {Decimal("91000.00"): Decimal("1")},
        "ask_book": {Decimal("91000.10"): Decimal("0.5")},
    }
    post_gap_delta = {
        **replay_delta_rows[-1],
        "changed_second": resumed_at + timedelta(seconds=1),
        "source_sequence_num_start": 501,
        "source_sequence_num_end": 501,
        "best_bid": Decimal("91000.00"),
        "best_ask": Decimal("91000.10"),
        "changes": (("bid", Decimal("91000.00"), Decimal("2")),),
    }

    states = list(
        replay_day(
            anchors=[replay_anchor, post_gap_anchor],
            deltas=[*replay_delta_rows, post_gap_delta],
            metadata=[
                {
                    "window_start": explicit_source_gap[0],
                    "window_end": explicit_source_gap[1],
                    "status": "gap",
                }
            ],
            day_start=replay_anchor["anchor_second"],
            day_end=post_gap_delta["changed_second"],
        )
    )

    resumed = [state for state in states if state.observed_through >= resumed_at]
    assert [(state.observed_through, state.segment_id) for state in resumed] == [
        (resumed_at, 1),
        (resumed_at + timedelta(seconds=1), 1),
    ]
    assert resumed[0].bid_size == Decimal("1")
    assert resumed[1].bid_size == Decimal("2")


def test_replay_increments_segment_when_day_starts_inside_gap(replay_anchor):
    gap_end = replay_anchor["anchor_second"] + timedelta(seconds=2)
    post_gap_anchor = {
        **replay_anchor,
        "anchor_second": gap_end,
        "source_sequence_num": 200,
    }

    states = list(
        replay_day(
            anchors=[replay_anchor, post_gap_anchor],
            deltas=[],
            metadata=[
                {
                    "window_start": replay_anchor["anchor_second"],
                    "window_end": gap_end,
                    "status": "gap",
                }
            ],
            day_start=replay_anchor["anchor_second"],
            day_end=gap_end,
        )
    )

    assert states[0].observed_through == gap_end
    assert states[0].segment_id == 1


@pytest.mark.parametrize("reverse", [False, True])
def test_replay_chooses_highest_sequence_same_second_anchor_independent_of_order(
    replay_anchor, reverse
):
    higher_sequence = {
        **replay_anchor,
        "source_sequence_num": 200,
        "bid_book": {Decimal("90000.00"): Decimal("2")},
    }
    anchors = [replay_anchor, higher_sequence]
    if reverse:
        anchors.reverse()

    states = list(
        replay_day(
            anchors=anchors,
            deltas=[],
            metadata=[],
            day_start=replay_anchor["anchor_second"],
            day_end=replay_anchor["anchor_second"],
        )
    )

    assert states[0].sequence_end == 200
    assert states[0].bid_size == Decimal("2")


@pytest.mark.parametrize("row_index", [0, 1, 2])
def test_replay_fails_closed_when_any_source_bbo_disagrees(
    replay_anchor, replay_delta_rows, row_index
):
    replay_delta_rows[row_index]["best_bid"] = Decimal("99999")

    with pytest.raises(BookReplayError, match="source BBO mismatch"):
        list(
            replay_day(
                anchors=[replay_anchor],
                deltas=replay_delta_rows,
                metadata=[],
                day_start=replay_anchor["anchor_second"],
            )
        )


def test_replay_fails_closed_on_sequence_regression(
    replay_anchor, replay_delta_rows, sequence_regression
):
    with pytest.raises(BookReplayError, match="sequence regression"):
        list(
            replay_day(
                anchors=[replay_anchor],
                deltas=[*replay_delta_rows, sequence_regression],
                metadata=[],
                day_start=replay_anchor["anchor_second"],
            )
        )


def test_replay_applies_deltas_between_anchor_and_day_start(
    replay_anchor, replay_delta_rows
):
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


def test_replay_publishes_raw_deltas_and_derived_states(
    tmp_path, replay_anchor, replay_delta_rows
):
    states = list(
        replay_day(
            anchors=[replay_anchor],
            deltas=replay_delta_rows,
            metadata=[],
            day_start=replay_anchor["anchor_second"],
        )
    )

    records = publish_replay_partitions(
        deltas=replay_delta_rows,
        states=states,
        root=tmp_path,
        source_revision="pinned",
        source_date="2026-04-24",
    )

    by_kind = {
        next(
            part.split("=", 1)[1]
            for part in record.path.parts
            if part.startswith("kind=")
        ): record
        for record in records
        if record.row_count
    }
    assert len(records) == 48
    assert set(by_kind) == {"book_deltas", "book_states"}
    delta_table = pq.ParquetFile(by_kind["book_deltas"].path).read()
    state_table = pq.ParquetFile(by_kind["book_states"].path).read()
    assert tuple(delta_table.column_names) == RAW_BOOK_COLUMNS
    assert tuple(state_table.column_names) == RAW_BOOK_COLUMNS
    assert json.loads(delta_table["changes_json"][0].as_py()) == [
        ["bid", "90000.00", "1.35"]
    ]
    assert state_table["best_bid"].to_pylist()[:2] == ["90000.00", "90000.00"]
    assert state_table["bid_size"].to_pylist()[:2] == ["1.25", "1.35"]
    assert state_table["best_ask"].to_pylist()[:2] == ["90000.10", "90000.10"]
    assert state_table["ask_size"].to_pylist()[:2] == ["0.75", "0.75"]
