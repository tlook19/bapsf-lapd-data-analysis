"""Arithmetic and gate coverage for the ES4 T_e(t) plateau-slope instrument.

The raw-sweep path (``cell_te_estimates`` / ``run_plateau_series``) needs real
digitizer data and is exercised by running the script, not here. What is unit
tested is the closed-form arithmetic underneath it: the OLS slope/SE, its
percent-of-plateau-mean conversion, the pre-registered gates (slope and
excess bin), the port/run lookup and its output-path guard, the windowed
mean, and the overlay-prior read -- the last two need only the processed
fixtures already provisioned for the raw-sweep path (``langmuir_sweeps.hdf5``,
``es4_sim1d_overlay.npz``), not the raw digitizer files.
"""

import math

import numpy as np
import pytest

from scripts.es4_te_time_slope import (
    DEFAULT_PORT,
    FORBIDDEN_OUTPUT_DIR,
    SWEEPS_H5,
    WINDOW_MATCHED_MS,
    OVERLAY_NPZ,
    checked_output_path,
    excess_bin_verdict,
    gate_verdict,
    ols_slope_se,
    overlay_prior_te_ev,
    pct_per_ms,
    runs_for_port,
    windowed_mean_ev,
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


# The output-path guard's trigger set: every new run-selecting flag
# (--port, --runs, --plateau-window-ms) feeds only *args.output* into
# checked_output_path, so a census of what triggers/does not trigger the
# guard is unaffected by which port or runs were selected -- exercised here
# with filenames that plausibly vary by port (the shape a --port-aware
# default might have taken) to make that explicit.
@pytest.mark.parametrize(
    "relative, triggers",
    [
        ("processed/es4_te_time_slope_p21.csv", True),
        ("processed/es4_te_time_slope_p29.csv", True),
        ("sub/processed/out.csv", True),
        ("processed_nearby/out.csv", False),  # "processed" substring, not a path part
        ("p29_out.csv", False),
        ("out/p21_p29_bins.csv", False),
    ],
)
def test_output_guard_trigger_set(tmp_path, relative, triggers):
    path = tmp_path / relative
    if triggers:
        with pytest.raises(ValueError, match=FORBIDDEN_OUTPUT_DIR):
            checked_output_path(path)
    else:
        assert checked_output_path(path) == path.resolve()


@pytest.mark.parametrize(
    "value_ev, n_finite, n_total, expected",
    [
        (0.55, 10, 10, "<= 0.6 eV"),
        (0.6, 10, 10, "<= 0.6 eV"),
        (1.0, 10, 10, ">= 1 eV"),
        (1.75, 10, 10, ">= 1 eV"),
        (0.8, 10, 10, "between (undetermined)"),
    ],
)
def test_excess_bin_verdict_bins_the_value(value_ev, n_finite, n_total, expected):
    assert excess_bin_verdict(value_ev, n_finite, n_total) == expected


def test_excess_bin_verdict_refuses_fewer_than_three_finite_cycles():
    assert excess_bin_verdict(0.5, 2, 8) == "REFUSED (2 of 8 cycles)"
    assert excess_bin_verdict(float("nan"), 5, 8) == "REFUSED (5 of 8 cycles)"


def test_windowed_mean_ev_selects_the_window_and_refuses_empty():
    cycle_ms = np.array([10.0, 12.0, 15.0, 17.0, 19.0])
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert windowed_mean_ev(cycle_ms, values, (15.0, 19.5)) == pytest.approx(4.0)
    assert windowed_mean_ev(cycle_ms, values, (10.0, 19.5)) == pytest.approx(3.0)
    assert math.isnan(windowed_mean_ev(cycle_ms, values, (100.0, 200.0)))


def test_runs_for_port_default_reproduces_the_original_p21_pair():
    specs = runs_for_port(SWEEPS_H5, DEFAULT_PORT)
    assert [s["run_id"] for s in specs] == ["42", "43"]
    assert [s["port"] for s in specs] == [21, 21]
    assert [s["rotation_deg"] for s in specs] == [0.0, 180.0]


def test_runs_for_port_p29_gives_runs_44_and_45():
    specs = runs_for_port(SWEEPS_H5, 29)
    assert [s["run_id"] for s in specs] == ["44", "45"]
    assert [s["port"] for s in specs] == [29, 29]
    assert [s["rotation_deg"] for s in specs] == [0.0, 180.0]


def test_runs_for_port_refuses_a_port_with_no_runs():
    with pytest.raises(ValueError, match="port 9999"):
        runs_for_port(SWEEPS_H5, 9999)


def test_runs_for_port_override_uses_the_named_runs_and_their_own_attrs():
    specs = runs_for_port(SWEEPS_H5, 21, override_run_ids=("45", "44"))
    # Order follows the override, not a re-sort; port/rotation come from the
    # file's own attrs for whatever run id was named, not the --port value.
    assert [s["run_id"] for s in specs] == ["45", "44"]
    assert [s["port"] for s in specs] == [29, 29]


def test_runs_for_port_override_refuses_an_absent_run_id():
    with pytest.raises(ValueError, match="99"):
        runs_for_port(SWEEPS_H5, 21, override_run_ids=("99",))


def test_overlay_prior_te_ev_reads_the_p29_window_matched_row():
    prior = overlay_prior_te_ev(OVERLAY_NPZ, 29, WINDOW_MATCHED_MS)
    # On record: 1.58-1.74 eV for the ES4 overlay's p29 plateau.
    assert 1.5 <= prior <= 1.8


def test_overlay_prior_te_ev_nan_for_a_port_the_overlay_does_not_carry():
    assert math.isnan(overlay_prior_te_ev(OVERLAY_NPZ, 9999, WINDOW_MATCHED_MS))
