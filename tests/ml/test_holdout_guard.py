import pytest

from btcspiker_ml.search import SearchState


def test_development_stage_cannot_open_final_holdout():
    state = SearchState.new("search-1", "dataset-1", wall_clock_seconds=86400)
    with pytest.raises(PermissionError, match="final holdout is sealed"):
        state.open_final_holdout(requesting_stage="trees")


def test_final_holdout_requires_qualification_and_all_development_stages():
    state = SearchState.new("search-1", "dataset-1", wall_clock_seconds=86400)
    with pytest.raises(PermissionError, match="final holdout is sealed"):
        state.open_final_holdout(requesting_stage="qualification")

    state.completed_stages = ["baseline", "linear", "trees", "ablation", "ensemble"]
    state.open_final_holdout(requesting_stage="qualification")
    assert state.final_holdout_opened is True
    assert state.final_holdout_accessed_at is not None
    with pytest.raises(PermissionError, match="final holdout is sealed"):
        state.open_final_holdout(requesting_stage="qualification")
