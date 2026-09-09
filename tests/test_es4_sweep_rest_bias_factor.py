"""Rest-bias factor extrapolation arithmetic against closed-form branches.

A synthetic ion branch ``|I| = A (V_f - V) ** 3/4`` has an exactly representable
extrapolated ratio: between two biases it is ``((V_f - V_deep) / (V_f -
V_shallow)) ** 3/4``, independent of ``A``.  The numbers below are chosen so
that ratio can be written down and compared with what the fitted objects
return, rather than merely bracketed.  A straight branch gets the same
treatment, where the ratio is the quotient of two line values.
"""

import numpy as np
import pytest

from scripts.es4_sweep_rest_bias_factor import (
    LinearFit,
    PowerLawFit,
    bracket_contains,
    fit_linear,
    fit_power_law,
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


def test_v_float_is_bounded_above_the_fit_window():
    # A branch that keeps falling toward the top of the window cannot be fitted
    # with V_f inside it; the bound keeps the power-law base positive.
    fit = fit_power_law(V_WINDOW, np.full(V_WINDOW.shape, 1.0e-3), exponent=EXPONENT)

    assert fit.v_float > V_WINDOW.max()
    assert np.isfinite(fit.evaluate(V_DEEP))


def test_linear_fit_recovers_a_straight_branch_and_its_ratio():
    intercept, slope = 1.0e-3, -2.0e-4
    fit = fit_linear(V_WINDOW, intercept + slope * V_WINDOW)

    assert fit.intercept == pytest.approx(intercept, rel=1e-9)
    assert fit.slope == pytest.approx(slope, rel=1e-9)
    assert fit.rms == pytest.approx(0.0, abs=1e-15)
    expected = (intercept + slope * V_DEEP) / (intercept + slope * V_SHALLOW)
    assert fit.ratio(V_DEEP, V_SHALLOW) == pytest.approx(expected, rel=1e-9)


def test_linear_ratio_is_the_quotient_of_two_line_values():
    fit = LinearFit(1.0e-3, -2.0e-4, 0.0)

    # |I| at -74 V is 1e-3 + 1.48e-2 = 1.58e-2; at -19 V it is 4.8e-3.
    assert fit.evaluate(V_DEEP) == pytest.approx(1.58e-2, rel=1e-12)
    assert fit.evaluate(V_SHALLOW) == pytest.approx(4.8e-3, rel=1e-12)
    assert fit.ratio(V_DEEP, V_SHALLOW) == pytest.approx(1.58e-2 / 4.8e-3, rel=1e-12)


def test_bracket_contains_is_end_order_independent():
    assert bracket_contains(1.1, 1.5, 1.3)
    assert bracket_contains(1.5, 1.1, 1.3)
    assert not bracket_contains(1.1, 1.5, 1.6)
    assert not bracket_contains(1.5, 1.1, 1.0)


def test_bracket_contains_includes_its_endpoints():
    assert bracket_contains(1.1, 1.5, 1.1)
    assert bracket_contains(1.1, 1.5, 1.5)
