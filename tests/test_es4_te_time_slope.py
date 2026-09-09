"""Arithmetic and gate coverage for the ES4 p21 T_e(t) plateau-slope instrument.

The raw-sweep path (``cell_te_estimates`` / ``run_plateau_series``) needs real
digitizer data and is exercised by running the script, not here. What is unit
tested is the closed-form arithmetic underneath it: the OLS slope/SE, its
percent-of-plateau-mean conversion, the pre-registered gate, and the output
refusal.
"""

import math

import numpy as np
import pytest

from scripts.es4_te_time_slope import (
    FORBIDDEN_OUTPUT_DIR,
    checked_output_path,
    gate_verdict,
    ols_slope_se,
    pct_per_ms,
)


def test_ols_slope_se_recovers_a_known_linear_trend():
    # T_e(t) = 0.80 - 0.01 * t (eV, t in ms): an exact -1.25 %/ms slope of the
    # plateau mean at t = 0.  No noise, so SE must read exactly zero.
    t = np.linspace(12.0, 19.5, 8)
    y = 0.80 - 0.01 * t
    slope, se, intercept, n = ols_slope_se(t, y)
    assert n == 8
    assert slope == pytest.approx(-0.01, abs=1e-12)
    assert se == pytest.approx(0.0, abs=1e-9)
    assert intercept == pytest.approx(0.80, abs=1e-9)


def test_ols_slope_se_arithmetic_against_a_hand_worked_example():
    # Four points with a known residual pattern: y = 1 + 2t plus residuals
    # [+1, -1, -1, +1] at t = [0, 1, 2, 3].  The least-squares slope is exactly
    # 2 (the residual pattern is orthogonal to t after centering), and the
    # standard error is worked by hand from the standard OLS formula
    # se = sqrt(s^2 / sum((t - tbar)^2)), s^2 = sum(resid^2) / (n - 2).
    t = np.array([0.0, 1.0, 2.0, 3.0])
    baseline = 1.0 + 2.0 * t
    residual_pattern = np.array([1.0, -1.0, -1.0, 1.0])
    y = baseline + residual_pattern
    slope, se, intercept, n = ols_slope_se(t, y)
    assert n == 4
    assert slope == pytest.approx(2.0, abs=1e-12)
    assert intercept == pytest.approx(1.0, abs=1e-12)
    resid = y - (slope * t + intercept)
    expected_s2 = np.sum(resid**2) / (n - 2)
    expected_ss_t = np.sum((t - t.mean()) ** 2)
    expected_se = math.sqrt(expected_s2 / expected_ss_t)
    assert se == pytest.approx(expected_se, rel=1e-12)


def test_ols_slope_se_refuses_fewer_than_three_finite_points():
    t = np.array([12.0, 13.0, 14.0, 15.0])
    y = np.array([0.7, 0.68, np.nan, np.nan])
    slope, se, intercept, n = ols_slope_se(t, y)
    assert n == 2
    assert math.isnan(slope)
    assert math.isnan(se)
    assert math.isnan(intercept)


def test_pct_per_ms_conversion_and_nonfinite_mean():
    # slope -0.01 eV/ms, SE 0.002 eV/ms, mean 0.80 eV -> -1.25 %/ms, 0.25 %/ms.
    slope_pct, se_pct = pct_per_ms(-0.01, 0.002, 0.80)
    assert slope_pct == pytest.approx(-1.25, abs=1e-9)
    assert se_pct == pytest.approx(0.25, abs=1e-9)
    slope_pct, se_pct = pct_per_ms(-0.01, 0.002, 0.0)
    assert math.isnan(slope_pct)
    assert math.isnan(se_pct)


@pytest.mark.parametrize(
    "slope_pct, n_finite, n_total, expected",
    [
        (-1.0, 8, 8, "inside"),
        (-1.5, 8, 8, "inside"),
        (-0.5, 8, 8, "inside"),
        (-0.49, 8, 8, "outside (-0.49 %/ms)"),
        (-1.51, 8, 8, "outside (-1.51 %/ms)"),
        (0.30, 8, 8, "outside (+0.30 %/ms)"),
    ],
)
def test_gate_verdict_bins_the_slope(slope_pct, n_finite, n_total, expected):
    assert gate_verdict(slope_pct, n_finite, n_total) == expected


def test_gate_verdict_refuses_fewer_than_three_finite_cycles():
    assert gate_verdict(-1.0, 2, 8) == "REFUSED (2 of 8 cycles)"
    assert gate_verdict(-1.0, 0, 8) == "REFUSED (0 of 8 cycles)"


def test_gate_verdict_never_reports_a_bare_nan():
    # A non-finite slope with enough finite cycles (degenerate design, e.g.
    # every t equal) must still resolve to a REFUSED bin, not "nan".
    verdict = gate_verdict(float("nan"), 5, 8)
    assert verdict == "REFUSED (5 of 8 cycles)"
    assert "nan" not in verdict.lower()


def test_output_inside_the_product_directory_is_refused(tmp_path):
    allowed = tmp_path / "es4_te_time_slope.csv"
    assert checked_output_path(allowed) == allowed.resolve()

    forbidden = tmp_path / FORBIDDEN_OUTPUT_DIR / "out.csv"
    with pytest.raises(ValueError, match=FORBIDDEN_OUTPUT_DIR):
        checked_output_path(forbidden)

    (tmp_path / FORBIDDEN_OUTPUT_DIR).mkdir()
    with pytest.raises(ValueError, match=FORBIDDEN_OUTPUT_DIR):
        checked_output_path(tmp_path / FORBIDDEN_OUTPUT_DIR / ".." / FORBIDDEN_OUTPUT_DIR / "f.csv")
