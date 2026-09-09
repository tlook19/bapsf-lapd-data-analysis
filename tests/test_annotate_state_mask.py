"""Projection of a registered shot range onto the product's averaging cells."""

import numpy as np
import pytest

from scripts.annotate_state_mask import STATE_REGISTRY, state_mask_for_run

N_POSITIONS = 51
N_WINDOWS = 20
N_SHOTS = 20


def _mask(shot_first, shot_last):
    return state_mask_for_run(
        N_POSITIONS, N_WINDOWS, N_SHOTS, shot_first, shot_last
    )


def test_range_on_position_boundaries_covers_whole_positions():
    mask, covered = _mask(240, 695)

    # 240 is the first shot of position 12; 695 is the sixteenth of position 34.
    assert np.array_equal(np.flatnonzero(covered), np.arange(12, 35))
    assert mask[12:35].all()
    assert not mask[:12].any()
    assert not mask[35:].any()
    assert mask.sum() == 23 * N_WINDOWS


def test_every_dead_time_window_of_a_covered_position_is_masked():
    mask, _ = _mask(240, 695)
    per_position = mask.sum(axis=1)
    assert set(np.unique(per_position)) == {0, N_WINDOWS}


def test_a_position_only_partly_inside_the_range_is_still_covered():
    # One shot of position 34 inside the range is enough: the product averages
    # a position's shots together, so the cell carries the two states mixed.
    mask, covered = _mask(240, 680)
    assert covered[34]
    assert mask[34].all()


def test_a_single_shot_range_covers_exactly_its_position():
    mask, covered = _mask(695, 695)
    assert np.array_equal(np.flatnonzero(covered), np.array([34]))
    assert mask.sum() == N_WINDOWS


def test_run_43_isat_is_the_registered_state():
    entry = STATE_REGISTRY[("43", "isat")]
    assert (entry["shot_first"], entry["shot_last"]) == (240, 695)
    assert entry["factor"] == pytest.approx(1.76)
