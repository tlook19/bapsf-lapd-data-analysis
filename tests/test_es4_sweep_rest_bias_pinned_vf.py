"""The V_f-pinned sheath-expansion mode of the rest-bias factor instrument.

The free-``V_f`` power law has no discrimination on a branch this shallow: the
base potential runs wherever the shape demands and the form ends up fitting
itself.  ``--pin-vf measured`` anchors that base to a plasma potential the
Langmuir product already measured for the same run, leaving the amplitude and
the power free, and the tests below pin down three separate things.

First, that the pin is HONOURED and is the whole story: a synthetic
``A (V_f - V) ** 3/4`` branch pinned at its own ``V_f`` returns ``A`` and 3/4
with a vanishing residual, and the same branch pinned ten volts away does not
-- it is forced onto a different power, its residual rises by orders of
magnitude, and its extrapolated ratio moves.  That contrast is what makes the
mode an instrument rather than a relabelling.

Second, that the DEFAULT invocation is untouched: ``analyse_pair`` without a
product returns exactly the 55 columns it returned before, in order, and the
pinned columns are appended only when the mode is on.

Third, that the new mode cannot escape the instrument's standing refusal to
write into the product directory, and that every way of failing to read a
measured potential is a refusal naming what was missing rather than a silent
fallback to a guessed one.
"""

import h5py
import numpy as np
import pytest

from scripts.es4_sweep_rest_bias_factor import (
    FORBIDDEN_OUTPUT_DIR,
    PINNED_GATE_TOLERANCE,
    VP_ESTIMATOR,
    VP_ESTIMATORS_REPORTED,
    RunSweep,
    analyse_pair,
    fit_power_law,
    main,
    pinned_gate_band,
    pinned_gate_verdict,
    read_pinned_v_float,
)

AMPLITUDE = 2.0e-3
V_FLOAT = 5.0
EXPONENT = 0.75

V_DEEP = -74.0
V_SHALLOW = -19.3
V_WINDOW = np.linspace(V_SHALLOW, -10.0, 64)

# ((5 + 74) / (5 + 19.3)) ** 0.75, the ratio the pinned fit must return when it
# is pinned at the branch's own base potential.
EXPECTED_RATIO = ((V_FLOAT - V_DEEP) / (V_FLOAT - V_SHALLOW)) ** EXPONENT

# The ES3 ramp the direct read is taken off belongs to a DIFFERENT plasma -- the
# instrument says so itself -- so the synthetic pair gives it its own base
# potential, chosen so the pinned extrapolation lands a few percent above the
# direct factor: strictly inside the gate band and not on either edge of it.
V_FLOAT_DEEP = 8.247
EXPECTED_DIRECT = ((V_FLOAT_DEEP - V_DEEP) / (V_FLOAT_DEEP - V_SHALLOW)) ** EXPONENT

# The columns the default invocation writes, in order, as captured from a
# re-run of the unmodified instrument.  The pinned mode may only APPEND.
BASE_FIELDNAMES = (
    "port",
    "rotation_deg",
    "run_deep",
    "run_shallow",
    "experiment_set_deep",
    "experiment_set_shallow",
    "v_parked_deep_v",
    "v_offset_applied_deep_v",
    "v_departure_deep_v",
    "v_rest_deep_v",
    "v_rest_deep_std_v",
    "v_rest_deep_raw_frame_v",
    "v_parked_shallow_v",
    "v_offset_applied_shallow_v",
    "v_departure_shallow_v",
    "v_rest_shallow_v",
    "v_rest_shallow_std_v",
    "v_rest_shallow_raw_frame_v",
    "ramp_min_deep_v",
    "ramp_max_deep_v",
    "ramp_min_shallow_v",
    "ramp_max_shallow_v",
    "flyback_us_deep",
    "flyback_us_shallow",
    "fit_v_min",
    "fit_v_max",
    "n_fit_samples",
    "n_ramps_deep",
    "n_ramps_shallow",
    "n_dead_windows_deep",
    "n_dead_windows_shallow",
    "n_cells",
    "n_shots_per_cell_deep",
    "n_shots_per_cell_shallow",
    "f_direct_up",
    "f_direct_up_std",
    "inv_f_direct_up",
    "f_eta34",
    "f_eta34_std",
    "v_float_eta34",
    "rms_eta34_a",
    "f_free",
    "f_free_std",
    "v_float_free",
    "exponent_free",
    "rms_free_a",
    "f_linear",
    "f_linear_std",
    "linear_slope_a_per_v",
    "rms_linear_a",
    "bracket_low",
    "bracket_high",
    "eta34_bracket_position",
    "n_cells_eta34_below_linear",
    "eta34_in_bracket",
)

# A 7-position profile whose middle five cells clear the 0.5-of-peak core cut.
CELL_WEIGHTS = np.array([0.2, 0.6, 1.0, 1.0, 1.0, 0.6, 0.2])
CORE_CELLS = np.array([1, 2, 3, 4, 5])

# Cycle-time bins in seconds: two lie inside the 15.0-19.5 ms plateau.
CYCLE_TIME_S = np.array([0.010, 0.015, 0.018, 0.021])
PLATEAU_BINS = np.array([1, 2])
POISON_V = 999.0


def _branch(v, v_float=V_FLOAT):
    return AMPLITUDE * (v_float - np.asarray(v, dtype=float)) ** EXPONENT


def _run_sweep(run_id, experiment_set, v_ramp, v_rest, v_float=V_FLOAT):
    """One synthetic run whose every cell is the same closed-form ion branch."""
    profile = _branch(v_ramp, v_float)[None, :] * CELL_WEIGHTS[:, None]
    return RunSweep(
        run_id=run_id,
        port=21,
        rotation_deg=0.0,
        experiment_set=experiment_set,
        v_ramp=np.asarray(v_ramp, dtype=float),
        i_ramp_abs=profile,
        v_rest=float(v_rest),
        v_rest_std=0.0,
        v_parked=float(round(v_rest)),
        v_offset_applied=0.0,
        n_ramps=5,
        n_dead_windows=4,
        n_shots_per_position=20,
        flyback_us=5.0,
    )


def _pair():
    # The deep ramp must sweep THROUGH the shallow rest bias for the direct read
    # to interpolate rather than clamp, and it is sampled finely enough that the
    # linear interpolation of a curved branch is not what the test measures.
    deep = _run_sweep("32", 3, np.linspace(V_DEEP, 0.0, 4096), V_DEEP, V_FLOAT_DEEP)
    shallow = _run_sweep("42", 4, np.linspace(V_SHALLOW, -1.0, 256), V_SHALLOW)
    return deep, shallow


def _write_product(path, run_id="42", experiment_set=4, n_positions=7, values=None):
    """A minimal Langmuir product carrying all three plasma-potential estimators.

    Everything outside the plateau window and outside the core cells is written
    as POISON_V, so a reader that widens either selection cannot pass.
    """
    values = {VP_ESTIMATOR: V_FLOAT} if values is None else values
    with h5py.File(path, "w") as handle:
        handle.create_dataset("x_cm", data=np.linspace(-3.0, 3.0, n_positions))
        node = handle.create_group(f"experiment_sets/{experiment_set}/{run_id}")
        node.create_dataset("cycle_time_s", data=CYCLE_TIME_S)
        for name in VP_ESTIMATORS_REPORTED:
            grid = np.full((n_positions, CYCLE_TIME_S.size), POISON_V)
            grid[np.ix_(CORE_CELLS, PLATEAU_BINS)] = values.get(name, V_FLOAT + 1.0)
            node.create_dataset(name, data=grid)
    return path


# --------------------------------------------------------------- the pin itself


def test_the_pin_is_honoured_and_not_refitted():
    branch = _branch(V_WINDOW)

    free_power = fit_power_law(V_WINDOW, branch, v_float=V_FLOAT + 30.0)
    fixed_power = fit_power_law(
        V_WINDOW, branch, exponent=EXPONENT, v_float=V_FLOAT + 30.0
    )

    assert free_power.v_float == V_FLOAT + 30.0
    assert fixed_power.v_float == V_FLOAT + 30.0
    assert fixed_power.exponent == EXPONENT
    # The unpinned call still fits its own base potential and is unaffected.
    assert fit_power_law(V_WINDOW, branch, exponent=EXPONENT).v_float == pytest.approx(
        V_FLOAT, rel=1e-6
    )


def test_pinned_at_the_true_base_recovers_the_amplitude_and_the_slope():
    fit = fit_power_law(V_WINDOW, _branch(V_WINDOW), v_float=V_FLOAT)

    assert fit.amplitude == pytest.approx(AMPLITUDE, rel=1e-6)
    assert fit.exponent == pytest.approx(EXPONENT, rel=1e-6)
    assert fit.rms == pytest.approx(0.0, abs=1e-12)
    assert fit.ratio(V_DEEP, V_SHALLOW) == pytest.approx(EXPECTED_RATIO, rel=1e-6)


def test_pinned_ten_volts_off_misfits_the_same_branch():
    branch = _branch(V_WINDOW)
    truth = fit_power_law(V_WINDOW, branch, v_float=V_FLOAT)
    displaced = fit_power_law(V_WINDOW, branch, v_float=V_FLOAT + 10.0)

    # The wrong base cannot be absorbed: amplitude and power both move, the
    # residual stops vanishing, and the extrapolated ratio moves with them.
    assert displaced.amplitude != pytest.approx(AMPLITUDE, rel=0.05)
    assert displaced.exponent != pytest.approx(EXPONENT, rel=0.05)
    assert displaced.rms > 1.0e4 * max(truth.rms, 1.0e-16)
    assert displaced.ratio(V_DEEP, V_SHALLOW) != pytest.approx(EXPECTED_RATIO, rel=0.02)


def test_a_pin_inside_the_fit_window_is_refused():
    branch = _branch(V_WINDOW)

    with pytest.raises(ValueError, match="not above the top of the fit window"):
        fit_power_law(V_WINDOW, branch, v_float=float(V_WINDOW.max()))


# ------------------------------------------------------------- the gate itself


def test_pinned_gate_arithmetic():
    direct = 1.2
    low, high = pinned_gate_band(direct)

    assert low == pytest.approx(1.2)
    assert high == pytest.approx(1.2 * (1.0 + PINNED_GATE_TOLERANCE))
    assert pinned_gate_verdict(direct, low).startswith("inside")
    assert pinned_gate_verdict(direct, high).startswith("inside")
    assert pinned_gate_verdict(direct, 1.3).startswith("inside")
    assert pinned_gate_verdict(direct, 1.19).startswith("outside")
    assert pinned_gate_verdict(direct, 1.4).startswith("outside")
    assert "1.4000" in pinned_gate_verdict(direct, 1.4)
    assert pinned_gate_verdict(direct, None) == "REFUSED (no fit)"
    assert pinned_gate_verdict(direct, float("nan")) == "REFUSED (no fit)"
    # No verdict is ever blank or a bare NaN.
    for candidate in (low, high, 1.4, None, float("nan")):
        assert pinned_gate_verdict(direct, candidate).strip()
        assert "nan" not in pinned_gate_verdict(direct, candidate)


# ------------------------------------------------------ reading the potential


def test_the_measured_potential_is_read_over_the_plateau_and_the_core_cells(tmp_path):
    product = _write_product(
        tmp_path / "langmuir_sweeps.hdf5",
        values={"vp_derivative_v": V_FLOAT, "vp_log_v": 6.0, "vp_exp_v": 7.0},
    )

    pin = read_pinned_v_float(product, "42", CORE_CELLS, n_positions=7)

    assert pin.estimator == VP_ESTIMATOR
    assert pin.v_float == pytest.approx(V_FLOAT)
    assert pin.n_cells == CORE_CELLS.size
    assert pin.n_time_bins == PLATEAU_BINS.size
    assert pin.cell_std == pytest.approx(0.0)
    # Every estimator the product carries is reported beside the pinned one.
    assert set(pin.by_estimator) == set(VP_ESTIMATORS_REPORTED)
    assert pin.by_estimator["vp_log_v"] == pytest.approx(6.0)
    assert pin.by_estimator["vp_exp_v"] == pytest.approx(7.0)


def test_every_unreadable_potential_is_a_refusal_and_never_a_fallback(tmp_path):
    with pytest.raises(ValueError, match="not in this checkout"):
        read_pinned_v_float(tmp_path / "absent.hdf5", "42", CORE_CELLS, n_positions=7)

    product = _write_product(tmp_path / "langmuir_sweeps.hdf5")
    with pytest.raises(ValueError, match="has no experiment_sets/4/99"):
        read_pinned_v_float(product, "99", CORE_CELLS, n_positions=7)
    with pytest.raises(ValueError, match="do not transfer"):
        read_pinned_v_float(product, "42", CORE_CELLS, n_positions=51)
    with pytest.raises(ValueError, match="no core cells"):
        read_pinned_v_float(product, "42", np.array([], dtype=int), n_positions=7)

    blank = _write_product(
        tmp_path / "blank.hdf5", values=dict.fromkeys(VP_ESTIMATORS_REPORTED, np.nan)
    )
    with pytest.raises(ValueError, match="is NaN"):
        read_pinned_v_float(blank, "42", CORE_CELLS, n_positions=7)


# ------------------------------------------------- default mode stays untouched


def test_default_mode_writes_exactly_the_columns_it_always_wrote():
    deep, shallow = _pair()

    row = analyse_pair(deep, shallow, None)

    assert tuple(row) == BASE_FIELDNAMES


def test_the_pinned_mode_only_appends_columns_and_lands_on_the_direct_factor(tmp_path):
    deep, shallow = _pair()
    product = _write_product(tmp_path / "langmuir_sweeps.hdf5")

    row = analyse_pair(deep, shallow, None, product)

    # The base columns are untouched, in order, and the pinned ones follow.
    assert tuple(row)[: len(BASE_FIELDNAMES)] == BASE_FIELDNAMES
    added = tuple(row)[len(BASE_FIELDNAMES) :]
    assert "f_pinned" in added and "pinned_gate_verdict" in added
    assert row["vp_pin_v"] == pytest.approx(V_FLOAT)
    assert row["vp_pin_estimator"] == VP_ESTIMATOR
    assert row["n_cells_pinned_fit"] == CORE_CELLS.size
    assert row["n_cells_pinned_fit_failed"] == 0
    # The shallow branch is a closed-form 3/4 law pinned at its own base, so the
    # pinned fit returns that law exactly; the ES3 branch it is compared with is
    # a different plasma, and the two land a few percent apart, inside the band.
    assert row["f_pinned"] == pytest.approx(EXPECTED_RATIO, rel=1e-6)
    assert row["f_direct_up"] == pytest.approx(EXPECTED_DIRECT, rel=1e-6)
    assert row["exponent_pinned"] == pytest.approx(EXPONENT, rel=1e-4)
    assert row["f_pinned_p075"] == pytest.approx(EXPECTED_RATIO, rel=1e-9)
    assert 1.05 < row["f_pinned"] / row["f_direct_up"] < 1.10
    assert row["pinned_in_gate"]
    assert row["pinned_gate_verdict"].startswith("inside")
    # The pinned fit costs the branch nothing against the free-fit control.
    assert row["rms_pinned_over_rms_free"] < 1.0


# ---------------------------------------------------- the standing refusal holds


def test_the_pinned_mode_still_goes_through_the_output_refusal(tmp_path):
    forbidden = tmp_path / FORBIDDEN_OUTPUT_DIR / "factor.csv"

    with pytest.raises(ValueError, match=FORBIDDEN_OUTPUT_DIR):
        main([str(tmp_path), "--pin-vf", "measured", "--output", str(forbidden)])

    # And the default mode refuses the same path, so the mode is not the reason.
    with pytest.raises(ValueError, match=FORBIDDEN_OUTPUT_DIR):
        main([str(tmp_path), "--output", str(forbidden)])
