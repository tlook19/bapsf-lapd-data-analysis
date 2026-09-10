"""The ``--rest-bias-frame raw`` mode of the ES4/ES3 rest-bias factor instrument.

The default mode anchors F at each run's OWN zero-offset-corrected rest bias,
read as a coordinate directly on the deep (ES3) run's own offset-frame ramp
axis.  Because ES3 and ES4 carry DIFFERENT zero-offset corrections (about
-16.55 V and -5.07 V respectively -- see the module docstring), that is not
the same physical voltage scale, which is exactly what the raw-frame mode
below is built to avoid: the raw digitizer frame has no per-set convention, so
a raw-frame coordinate means one physical voltage on every run.

The tests below check three things: that ``v_ramp_raw_frame`` and
``direct_ratio``'s override are pure, correctly-signed translations; that the
translation is what actually MOVES F between the two frames (a synthetic pair
built with EQUAL per-run offsets must give identical F in both frames, since
the frame is then a shared additive constant and cancels in every ratio; a
pair with DIFFERENT offsets must not); and, against live data when it is
present, that the raw-frame rest levels this instrument reads reproduce the
advisor's raw-frame numbers for run 32 (ES3) and run 42 (ES4).
"""

import numpy as np
import pytest

from bapsf_lapd import LapdDataset
from bapsf_lapd.manifest import load_run_manifest
from scripts.es4_sweep_rest_bias_factor import (
    RunSweep,
    analyse_pair,
    analyse_pair_raw_frame,
    direct_ratio,
    parse_args,
    read_run_sweep,
)

AMPLITUDE = 2.0e-3
EXPONENT = 0.75


def _power_branch(v, v_float, amplitude=AMPLITUDE):
    return amplitude * (v_float - np.asarray(v, dtype=float)) ** EXPONENT


def _run_sweep(run_id, experiment_set, v_ramp, v_rest, v_float, v_offset_applied):
    """One synthetic run: a single closed-form ion branch on every cell."""
    weights = np.array([0.2, 0.6, 1.0, 1.0, 1.0, 0.6, 0.2])
    profile = _power_branch(v_ramp, v_float)[None, :] * weights[:, None]
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
        v_offset_applied=float(v_offset_applied),
        n_ramps=5,
        n_dead_windows=4,
        n_shots_per_position=20,
        flyback_us=5.0,
    )


# ----------------------------------------------------------- pure arithmetic


def test_v_ramp_raw_frame_is_the_offset_axis_translated_by_v_offset_applied():
    sweep = _run_sweep("42", 4, np.linspace(-25.0, -10.0, 16), -19.3, 40.0, 5.07)

    np.testing.assert_allclose(sweep.v_ramp_raw_frame, sweep.v_ramp + 5.07)
    assert sweep.v_rest_raw_frame == pytest.approx(-19.3 + 5.07)


def test_direct_ratio_v_ramp_override_reads_the_supplied_axis():
    # A linear branch so the two interpolations are exactly checkable.
    v = np.linspace(-30.0, 0.0, 31)
    i_abs = 1.0 - 0.01 * v  # positive, decreasing toward V=0
    sweep = _run_sweep("32", 3, v, -20.0, 999.0, 16.55)
    sweep.i_ramp_abs[0, :] = i_abs

    default = direct_ratio(sweep, 0, -20.0, -10.0)
    # Reading the SAME two coordinates on a raw-frame axis (here the offset
    # axis shifted by a constant with i_ramp_abs unchanged) must land on the
    # samples shifted by that same constant, not on the same numeric result,
    # unless the two query points are shifted the same way.
    shifted = direct_ratio(
        sweep, 0, -20.0 + 16.55, -10.0 + 16.55, v_ramp=sweep.v_ramp + 16.55
    )
    assert shifted == pytest.approx(default, rel=1e-12)

    # Omitting v_ramp is exactly the sweep.v_ramp path (existing behaviour).
    assert direct_ratio(sweep, 0, -20.0, -10.0, v_ramp=None) == pytest.approx(default)


def test_equal_offsets_leave_f_unchanged_between_frames():
    # When deep and shallow share ONE offset, that offset is a common additive
    # constant across v_deep, v_shallow AND the fitted branch, and every ratio
    # this instrument computes is translation-invariant under a COMMON shift.
    v_float_deep, v_float_shallow = 60.0, 30.0
    common_offset = 8.0
    deep = _run_sweep(
        "32", 3, np.linspace(-80.0, 0.0, 4096), -74.0, v_float_deep, common_offset
    )
    shallow = _run_sweep(
        "42", 4, np.linspace(-19.3, -1.0, 256), -19.3, v_float_shallow, common_offset
    )

    offset_row = analyse_pair(deep, shallow, None)
    raw_row = analyse_pair_raw_frame(deep, shallow, None)

    assert raw_row["f_direct_up"] == pytest.approx(offset_row["f_direct_up"], rel=1e-9)
    assert raw_row["f_linear"] == pytest.approx(offset_row["f_linear"], rel=1e-6)
    assert raw_row["f_eta34"] == pytest.approx(offset_row["f_eta34"], rel=1e-6)


def test_unequal_offsets_are_exactly_what_moves_f_between_frames():
    # Same branches as above, but now deep and shallow carry DIFFERENT
    # offsets (as ES3 and ES4 genuinely do): the frame is no longer a shared
    # constant and F must move between the two modes.
    v_float_deep, v_float_shallow = 60.0, 30.0
    deep = _run_sweep(
        "32", 3, np.linspace(-80.0, 0.0, 4096), -74.0, v_float_deep, 16.55
    )
    shallow = _run_sweep(
        "42", 4, np.linspace(-19.3, -1.0, 256), -19.3, v_float_shallow, 5.07
    )

    offset_row = analyse_pair(deep, shallow, None)
    raw_row = analyse_pair_raw_frame(deep, shallow, None)

    assert raw_row["f_direct_up"] != pytest.approx(offset_row["f_direct_up"], rel=1e-6)


def test_raw_frame_mode_selects_the_identical_samples_as_the_default_mode():
    # in_window / core_cells are decided from the offset-frame ramp values in
    # BOTH modes; n_cells and n_fit_samples must therefore agree exactly.
    deep = _run_sweep("32", 3, np.linspace(-80.0, 0.0, 4096), -74.0, 60.0, 16.55)
    shallow = _run_sweep("42", 4, np.linspace(-19.3, -1.0, 256), -19.3, 30.0, 5.07)

    offset_row = analyse_pair(deep, shallow, None)
    raw_row = analyse_pair_raw_frame(deep, shallow, None)

    assert raw_row["n_cells"] == offset_row["n_cells"]
    assert raw_row["n_fit_samples"] == offset_row["n_fit_samples"]
    assert raw_row["fit_v_min"] == offset_row["fit_v_min"]
    assert raw_row["fit_v_max"] == offset_row["fit_v_max"]


def test_raw_frame_and_pin_vf_measured_are_refused_together():
    with pytest.raises(SystemExit):
        parse_args([".", "--rest-bias-frame", "raw", "--pin-vf", "measured"])


# ---------------------------------------------------------------- live data

DATASET = LapdDataset.from_directory("data/may2026")


@pytest.mark.skipif(len(DATASET) == 0, reason="local HDF5 data files are not present")
def test_raw_frame_rest_levels_reproduce_the_advisor_reading():
    # advisor consult isweep-tail-zero-offset 2026-09-10, vsweep_frame_check.out:
    # run 32 (ES3 p21 rot0) dead-time rest -90.9 V raw, run 42 (ES4 p21 rot0) -24.1 V.
    configs = load_run_manifest("config/may2026_run_manifest.toml", data_dir="data/may2026")
    es3 = read_run_sweep(configs["32"])
    es4 = read_run_sweep(configs["42"])

    assert es3.v_rest_raw_frame == pytest.approx(-90.8, abs=0.5)
    assert es4.v_rest_raw_frame == pytest.approx(-24.4, abs=0.5)
