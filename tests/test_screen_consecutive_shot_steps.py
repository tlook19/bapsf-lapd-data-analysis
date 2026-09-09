"""The consecutive-shot step statistic and what its threshold can reach."""

import numpy as np

from scripts.screen_consecutive_shot_steps import LN_RATIO_FLAG, screen_run_channel

N_POSITIONS = 4
N_SHOTS = 20


def _flat_signal(level=1.0):
    return np.full((N_POSITIONS, N_SHOTS), level)


def test_a_flat_run_flags_nothing():
    _, _, flagged = screen_run_channel(_flat_signal())
    assert not flagged.any()


def test_a_step_inside_a_position_is_flagged_at_its_own_pair():
    signal = _flat_signal()
    signal[2, 16:] = 1.0 / 1.76
    within, _, flagged = screen_run_channel(signal)

    assert flagged.sum() == 1
    assert flagged[2, 15]
    assert within[2, 15] == np.log(1.0 / 1.76)


def test_a_step_on_a_position_boundary_is_invisible_to_the_flag():
    # No pair spans two positions, so a step that lands exactly on a boundary
    # leaves every position-internal ratio at one.  It surfaces only in the
    # boundary column, which is not flagged because it also carries the
    # profile move between the two positions.
    signal = _flat_signal()
    signal[2:] = 1.76
    _, across, flagged = screen_run_channel(signal)

    assert not flagged.any()
    assert across[1] == np.log(1.76)


def test_a_step_below_the_threshold_is_not_flagged():
    signal = _flat_signal()
    signal[1, 10:] = LN_RATIO_FLAG * 0.99
    _, _, flagged = screen_run_channel(signal)
    assert not flagged.any()


def test_a_sign_change_gives_no_finite_ratio_and_is_not_flagged():
    signal = _flat_signal()
    signal[0, 5] = 0.0
    within, _, flagged = screen_run_channel(signal)
    assert not np.isfinite(within[0, 4])
    assert not np.isfinite(within[0, 5])
    assert not flagged[0, 4]
    assert not flagged[0, 5]
