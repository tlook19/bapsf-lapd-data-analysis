"""Projection of a registration's shot ranges onto the product's averaging cells."""

import numpy as np
import pytest

from scripts.annotate_state_mask import (
    STATE_REGISTRY,
    contiguous_ranges,
    ranges_text,
    state_mask_for_run,
)

N_POSITIONS = 51
N_WINDOWS = 20
N_SHOTS = 20


def _mask(*shot_ranges):
    return state_mask_for_run(N_POSITIONS, N_WINDOWS, N_SHOTS, shot_ranges)


def test_range_on_position_boundaries_covers_whole_positions():
    mask, covered = _mask((240, 695))

    # 240 is the first shot of position 12; 695 is the sixteenth of position 34.
    assert np.array_equal(np.flatnonzero(covered), np.arange(12, 35))
    assert mask[12:35].all()
    assert not mask[:12].any()
    assert not mask[35:].any()
    assert mask.sum() == 23 * N_WINDOWS


def test_every_dead_time_window_of_a_covered_position_is_masked():
    mask, _ = _mask((240, 695))
    per_position = mask.sum(axis=1)
    assert set(np.unique(per_position)) == {0, N_WINDOWS}


def test_a_position_only_partly_inside_the_range_is_still_covered():
    # One shot of position 34 inside the range is enough: the product averages
    # a position's shots together, so the cell carries the two states mixed.
    mask, covered = _mask((240, 680))
    assert covered[34]
    assert mask[34].all()


def test_a_single_shot_range_covers_exactly_its_position():
    mask, covered = _mask((695, 695))
    assert np.array_equal(np.flatnonzero(covered), np.array([34]))
    assert mask.sum() == N_WINDOWS


def test_several_ranges_union_their_coverage_and_leave_the_gap_unmasked():
    """Two entries of the same state, and the positions between them survive.

    The union is what a registration with more than one range means: a position
    is covered when ANY range reaches it.  The positions no range reaches must
    come back unmasked, or the second entry would silently swallow the cells
    between the two.
    """
    mask, covered = _mask((240, 695), (760, 767), (781, 1019))

    assert np.array_equal(
        np.flatnonzero(covered),
        np.concatenate([np.arange(12, 35), np.arange(38, 51)]),
    )
    # The gap: positions 35, 36 and 37 are x = +10, +11 and +12 cm.
    assert not covered[35:38].any()
    assert not mask[35:38].any()
    # Every partly-covered position of either entry is covered whole.
    assert mask[34].all() and mask[38].all() and mask[39].all()
    assert mask.sum() == (23 + 13) * N_WINDOWS

    # Each range on its own covers only its own share, so the union is not an
    # artefact of the ranges being read as one span from 240 to 1019.
    _, first_only = _mask((240, 695))
    _, second_only = _mask((760, 767), (781, 1019))
    assert np.array_equal(covered, first_only | second_only)
    assert not (first_only & second_only).any()


def test_gapped_coverage_is_written_as_ranges_not_as_a_span():
    """A first-to-last span would claim the gap; the recorded text must not."""
    _, covered = _mask((240, 695), (760, 767), (781, 1019))
    text = ranges_text(contiguous_ranges(np.flatnonzero(covered)))

    assert text == "12-34,38-50"
    assert text != "12-50"
    assert ranges_text(((240, 695), (760, 767), (781, 1019))) == (
        "240-695,760-767,781-1019"
    )
    assert contiguous_ranges([5]) == [(5, 5)]
    assert contiguous_ranges([3, 1, 2, 9]) == [(1, 3), (9, 9)]


def test_run_43_isat_registers_both_entries_of_the_high_state():
    entry = STATE_REGISTRY[("43", "isat")]
    assert entry["shot_ranges"] == ((240, 695), (760, 767), (781, 1019))
    assert entry["factor"] == pytest.approx(1.76)
    # The registration text has to name the second entry's own transitions and
    # the rejection tell, not only the first entry's steps.
    for token in ("759|760", "767->768", "780->781", "1.976", "2.074", "162"):
        assert token in entry["evidence"]
