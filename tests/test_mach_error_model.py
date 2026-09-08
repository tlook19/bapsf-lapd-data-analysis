"""The three declared Mach systematics, and the refusal that guards them."""

import pytest

import bapsf_lapd
from bapsf_lapd.density import (
    MACH_FACE_ASYMMETRY_M_RANGE,
    MACH_FACE_ASYMMETRY_M_RMS,
    MACH_K,
    MACH_K_BRACKET,
    MACH_SHADOW_BRACKET_M,
    mach_shadow_bracket_m,
)


def test_the_area_ratio_shadow_offset_is_gone_from_the_package():
    with pytest.raises(AttributeError):
        bapsf_lapd.MACH_SHADOW_OFFSET


def test_the_calibration_constant_is_a_convention_inside_its_bracket():
    assert MACH_K == 1.66
    assert MACH_K_BRACKET == (1.34, 1.74)
    low, high = MACH_K_BRACKET
    assert low < MACH_K < high


def test_the_measured_face_asymmetry_is_declared():
    assert MACH_FACE_ASYMMETRY_M_RMS == 0.082
    assert MACH_FACE_ASYMMETRY_M_RANGE == (0.005, 0.146)
    smallest, largest = MACH_FACE_ASYMMETRY_M_RANGE
    assert 0.0 <= smallest <= MACH_FACE_ASYMMETRY_M_RMS <= largest


def test_the_shadow_bracket_holds_only_the_estimated_points():
    assert set(MACH_SHADOW_BRACKET_M) == {(1, 21), (1, 29), (1, 50)}
    assert mach_shadow_bracket_m(1, 50) == MACH_SHADOW_BRACKET_M[(1, 50)]
    assert (mach_shadow_bracket_m(1, 50).bohm, mach_shadow_bracket_m(1, 50).classical,
            mach_shadow_bracket_m(1, 50).broadened) == (0.05, 0.23, 0.67)
    assert (mach_shadow_bracket_m(1, 21).bohm, mach_shadow_bracket_m(1, 21).classical,
            mach_shadow_bracket_m(1, 21).broadened) == (0.01, 0.08, 0.23)
    assert mach_shadow_bracket_m(1, 29) == mach_shadow_bracket_m(1, 21)


@pytest.mark.parametrize("point", [(1, 11), (1, 41), (2, 21), (3, 50), (4, 29)])
def test_an_unestimated_point_is_refused_by_name(point):
    experiment_set, port = point
    with pytest.raises(KeyError) as excinfo:
        mach_shadow_bracket_m(experiment_set, port)
    message = str(excinfo.value)
    assert f"experiment set {experiment_set} port {port}" in message
    assert "ES1 p21, ES1 p29, ES1 p50" in message


@pytest.mark.parametrize("point", sorted(MACH_SHADOW_BRACKET_M))
def test_each_populated_bracket_is_monotone_and_one_sided(point):
    bracket = MACH_SHADOW_BRACKET_M[point]
    assert 0.0 <= bracket.bohm <= bracket.classical <= bracket.broadened
