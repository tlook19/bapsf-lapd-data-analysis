"""The two step-screen legs, their eligibility floor, and what each can reach."""

import numpy as np
import pytest

from scripts.screen_consecutive_shot_steps import (
    BLOCK_SHOTS,
    LN_RATIO_FLAG,
    SIGNIFICANCE_FACTOR,
    boundary_leg,
    eligible_samples,
    fixed_position_leg,
)

N_POSITIONS = 6
N_SHOTS = 20
NOISE = 0.001
LEVEL = 1.0
STEP = 1.76


def _flat(level=LEVEL):
    return np.full((N_POSITIONS, N_SHOTS), level)


def test_eligibility_floor_is_five_noise_levels():
    signal = np.array([[SIGNIFICANCE_FACTOR * NOISE * 1.01,
                        SIGNIFICANCE_FACTOR * NOISE * 0.99]])
    eligible = eligible_samples(signal, NOISE)
    assert eligible[0, 0]
    assert not eligible[0, 1]


def test_a_flat_run_flags_nothing():
    _, _, flagged = fixed_position_leg(_flat(), NOISE)
    assert not flagged.any()


def test_a_step_inside_a_position_is_flagged_at_its_own_pair():
    signal = _flat()
    signal[2, 16:] = LEVEL / STEP
    ratios, _, flagged = fixed_position_leg(signal, NOISE)

    assert flagged.sum() == 1
    assert flagged[2, 15]
    assert ratios[2, 15] == np.log(1.0 / STEP)


def test_a_pair_below_the_eligibility_floor_is_not_flagged():
    # The same ratio, at a level the record cannot resolve, is a ratio of noise.
    signal = np.full((N_POSITIONS, N_SHOTS), SIGNIFICANCE_FACTOR * NOISE * 0.5)
    signal[2, 16:] /= STEP
    _, pair_eligible, flagged = fixed_position_leg(signal, NOISE)

    assert not pair_eligible.any()
    assert not flagged.any()


def test_a_step_below_the_threshold_is_not_flagged():
    signal = _flat()
    signal[1, 10:] = LEVEL * LN_RATIO_FLAG * 0.99
    _, _, flagged = fixed_position_leg(signal, NOISE)
    assert not flagged.any()


def test_a_step_on_a_position_boundary_is_invisible_to_the_fixed_position_leg():
    signal = _flat()
    signal[3:] = LEVEL * STEP
    _, _, flagged = fixed_position_leg(signal, NOISE)
    assert not flagged.any()


def _boundary_case(profile_move, isat_step):
    """Both channels carry the profile move; only ISAT carries the level step."""
    isat = _flat()
    isweep = _flat(0.5)
    isat[3:] *= profile_move * isat_step
    isweep[3:] *= profile_move
    return isat, isweep


def test_the_profile_move_cancels_in_the_two_channel_ratio():
    isat, isweep = _boundary_case(profile_move=0.93, isat_step=1.0)
    two_channel, _, _, eligible, flagged, _ = boundary_leg(
        isat, NOISE, isweep, NOISE
    )
    assert eligible.all()
    assert not flagged.any()
    assert two_channel[2] == pytest.approx(0.0, abs=1e-15)


def test_a_one_channel_step_on_a_boundary_is_flagged_and_attributed():
    isat, isweep = _boundary_case(profile_move=0.93, isat_step=STEP)
    two_channel, isat_ratio, isweep_ratio, _, flagged, attributed = boundary_leg(
        isat, NOISE, isweep, NOISE
    )

    assert flagged.sum() == 1
    assert flagged[2]
    assert two_channel[2] == pytest.approx(np.log(STEP))
    # The moved channel's own ratio carries move x step; the other carries move.
    assert isat_ratio[2] == pytest.approx(np.log(0.93 * STEP))
    assert isweep_ratio[2] == pytest.approx(np.log(0.93))
    assert attributed[2] == "isat"


def test_a_step_on_the_other_channel_is_attributed_to_it():
    isat = _flat()
    isweep = _flat(0.5)
    isweep[3:] *= STEP
    _, _, _, _, flagged, attributed = boundary_leg(isat, NOISE, isweep, NOISE)
    assert flagged[2]
    assert attributed[2] == "i_sweep"


def test_a_boundary_below_the_eligibility_floor_is_not_flagged():
    isat, isweep = _boundary_case(profile_move=0.93, isat_step=STEP)
    huge_noise = LEVEL  # every block mean now sits under 5 x noise
    _, _, _, eligible, flagged, _ = boundary_leg(
        isat, huge_noise, isweep, huge_noise
    )
    assert not eligible.any()
    assert not flagged.any()


def test_the_boundary_blocks_are_the_last_and_first_shots():
    isat = _flat()
    isweep = _flat(0.5)
    # A step confined to the shots the blocks do NOT read leaves the ratio flat:
    # the boundaries below and above position 2 read its first and last blocks.
    isat[2, BLOCK_SHOTS : N_SHOTS - BLOCK_SHOTS] *= STEP
    _, _, _, _, flagged, _ = boundary_leg(isat, NOISE, isweep, NOISE)
    assert not flagged.any()
