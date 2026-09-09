"""Rest-bias factor extrapolation arithmetic against closed-form branches.

A synthetic ion branch ``|I| = A (V_f - V) ** 3/4`` has an exactly representable
extrapolated ratio: between two biases it is ``((V_f - V_deep) / (V_f -
V_shallow)) ** 3/4``, independent of ``A``.  The numbers below are chosen so
that ratio can be written down and compared with what the fitted objects
return, rather than merely bracketed.  A straight branch gets the same
treatment, with a zero crossing deliberately away from ``V_FLOAT`` so its ratio
cannot coincide with the power law's.

One test carries the docstring's disclosure rather than the arithmetic: a
shallow, straight branch of the size the real data shows is fitted with the
fixed 3/4 form, and the fit is shown to buy its shape by throwing ``V_f`` far
above the swept range and landing just under the linear answer.
"""

import numpy as np
import pytest

from scripts.es4_sweep_rest_bias_factor import (
    FORBIDDEN_OUTPUT_DIR,
    LinearFit,
    PowerLawFit,
    bracket_contains,
    bracket_position,
    checked_output_path,
    fit_linear,
    fit_power_law,
    uniform_ramp_length,
)

AMPLITUDE = 2.0e-3
V_FLOAT = 5.0
EXPONENT = 0.75

# The two biases the factor is quoted between, and the window the fit sees.
V_DEEP = -74.0
V_SHALLOW = -19.0
V_WINDOW = np.linspace(-19.0, -10.0, 64)

# ((5 + 74) / (5 + 19)) ** 0.75
EXPECTED_POWER_RATIO = (79.0 / 24.0) ** EXPONENT

# A straight branch whose zero sits at V = 4, not at V_FLOAT, so that its ratio
# 78/23 is independent of the power law's 79/24.
LINE_INTERCEPT = 1.2e-3
LINE_SLOPE = -3.0e-4


def _power_branch(v):
    return AMPLITUDE * (V_FLOAT - np.asarray(v)) ** EXPONENT


def test_fixed_exponent_fit_recovers_the_synthetic_branch():
    fit = fit_power_law(V_WINDOW, _power_branch(V_WINDOW), exponent=EXPONENT)

    assert fit.exponent == EXPONENT
    assert fit.v_float == pytest.approx(V_FLOAT, rel=1e-6)
    assert fit.amplitude == pytest.approx(AMPLITUDE, rel=1e-6)
    assert fit.rms == pytest.approx(0.0, abs=1e-12)


def test_fixed_exponent_ratio_matches_the_closed_form():
    fit = fit_power_law(V_WINDOW, _power_branch(V_WINDOW), exponent=EXPONENT)

    assert fit.ratio(V_DEEP, V_SHALLOW) == pytest.approx(EXPECTED_POWER_RATIO, rel=1e-6)


def test_ratio_is_independent_of_amplitude():
    scaled = PowerLawFit(7.0 * AMPLITUDE, V_FLOAT, EXPONENT, 0.0)
    unscaled = PowerLawFit(AMPLITUDE, V_FLOAT, EXPONENT, 0.0)

    assert scaled.ratio(V_DEEP, V_SHALLOW) == pytest.approx(unscaled.ratio(V_DEEP, V_SHALLOW))
    assert unscaled.ratio(V_DEEP, V_SHALLOW) == pytest.approx(EXPECTED_POWER_RATIO, rel=1e-12)


def test_free_exponent_fit_recovers_three_quarters():
    fit = fit_power_law(V_WINDOW, _power_branch(V_WINDOW), exponent=None)

    assert fit.exponent == pytest.approx(EXPONENT, rel=1e-3)
    assert fit.v_float == pytest.approx(V_FLOAT, rel=1e-3)
    assert fit.ratio(V_DEEP, V_SHALLOW) == pytest.approx(EXPECTED_POWER_RATIO, rel=1e-3)


def test_three_quarter_fit_of_a_shallow_line_throws_v_float_out_and_undercuts_it():
    # A straight branch rising 7 percent across the window, the size the ES4
    # data shows.  The fixed 3/4 form can only follow it by placing V_f far
    # above the swept range, and its extrapolation then lands just below the
    # line's own -- which is why the gate cannot discriminate between them.
    slope, intercept = -1.5e-4, 1.7786e-2
    branch = intercept + slope * V_WINDOW
    assert branch[0] / branch[-1] == pytest.approx(1.07, rel=1e-3)

    power = fit_power_law(V_WINDOW, branch, exponent=EXPONENT)
    line = fit_linear(V_WINDOW, branch)

    assert power.v_float > 50.0
    assert power.ratio(V_DEEP, V_SHALLOW) < line.ratio(V_DEEP, V_SHALLOW)
    assert power.ratio(V_DEEP, V_SHALLOW) == pytest.approx(
        line.ratio(V_DEEP, V_SHALLOW), rel=0.05
    )


def test_linear_fit_recovers_a_straight_branch_and_its_ratio():
    fit = fit_linear(V_WINDOW, LINE_INTERCEPT + LINE_SLOPE * V_WINDOW)

    assert fit.intercept == pytest.approx(LINE_INTERCEPT, rel=1e-9)
    assert fit.slope == pytest.approx(LINE_SLOPE, rel=1e-9)
    assert fit.rms == pytest.approx(0.0, abs=1e-15)
    expected = (LINE_INTERCEPT + LINE_SLOPE * V_DEEP) / (LINE_INTERCEPT + LINE_SLOPE * V_SHALLOW)
    assert fit.ratio(V_DEEP, V_SHALLOW) == pytest.approx(expected, rel=1e-9)


def test_linear_ratio_is_the_quotient_of_two_line_values():
    fit = LinearFit(LINE_INTERCEPT, LINE_SLOPE, 0.0)

    # |I| at -74 V is 1.2e-3 + 2.22e-2 = 2.34e-2; at -19 V it is 6.9e-3.
    assert fit.evaluate(V_DEEP) == pytest.approx(2.34e-2, rel=1e-12)
    assert fit.evaluate(V_SHALLOW) == pytest.approx(6.9e-3, rel=1e-12)
    assert fit.ratio(V_DEEP, V_SHALLOW) == pytest.approx(78.0 / 23.0, rel=1e-12)
    # The straight branch's ratio is not the power law's.
    assert fit.ratio(V_DEEP, V_SHALLOW) != pytest.approx(EXPECTED_POWER_RATIO, rel=1e-3)


def test_bracket_contains_is_end_order_independent():
    assert bracket_contains(1.1, 1.5, 1.3)
    assert bracket_contains(1.5, 1.1, 1.3)
    assert not bracket_contains(1.1, 1.5, 1.6)
    assert not bracket_contains(1.5, 1.1, 1.0)


def test_bracket_position_is_zero_at_direct_and_one_at_linear():
    assert bracket_position(1.1, 1.5, 1.1) == pytest.approx(0.0)
    assert bracket_position(1.1, 1.5, 1.5) == pytest.approx(1.0)
    assert bracket_position(1.1, 1.5, 1.4) == pytest.approx(0.75)
    assert np.isnan(bracket_position(1.2, 1.2, 1.2))


def test_ragged_ramp_windows_are_refused_by_length():
    uniform = [slice(0, 3125), slice(6250, 9375)]
    assert uniform_ramp_length(uniform, "32") == 3125

    ragged = [slice(0, 1562), slice(3125, 4688)]
    with pytest.raises(ValueError, match=r"\[1562, 1563\]"):
        uniform_ramp_length(ragged, "02")


def test_output_inside_the_product_directory_is_refused(tmp_path):
    allowed = tmp_path / "factor.csv"
    assert checked_output_path(allowed) == allowed.resolve()

    forbidden = tmp_path / FORBIDDEN_OUTPUT_DIR / "factor.csv"
    with pytest.raises(ValueError, match=FORBIDDEN_OUTPUT_DIR):
        checked_output_path(forbidden)

    # A path that only reaches the product directory after resolution is caught too.
    (tmp_path / FORBIDDEN_OUTPUT_DIR).mkdir()
    with pytest.raises(ValueError, match=FORBIDDEN_OUTPUT_DIR):
        checked_output_path(tmp_path / FORBIDDEN_OUTPUT_DIR / ".." / FORBIDDEN_OUTPUT_DIR / "f.csv")
