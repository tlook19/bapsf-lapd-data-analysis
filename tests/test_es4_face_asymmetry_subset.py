"""Rest-bias correction of a rotation-pair half-difference, on known numbers.

The arithmetic under test is small and exactly checkable, so every case here
writes down the answer independently of the code: a synthetic rotation pair is
built with currents chosen so that ``ln R`` is a round number, the factor is
applied by hand to the face the rotation puts it on, and the half-difference is
compared with the value that follows from the algebra rather than with a
re-run of the same expression.

The one case that cannot be synthetic is the control: the nine ES1-ES3
half-differences are a property of the shipped Mach product, so that case reads
the product when it is present and skips when the checkout has no products.
"""

import csv
import math

import h5py
import numpy as np
import pytest

from bapsf_lapd.density import MACH_FACE_ASYMMETRY_M_RANGE, MACH_K
from scripts.es4_face_asymmetry_subset import (
    CONTROL_HALF_DIFFERENCES_M,
    CONTROL_TOL_M,
    EDGE_CELLS_LABEL,
    EDGE_CUT_CM,
    FORBIDDEN_OUTPUT_DIR,
    GATE_HIGH_M,
    X0_LABEL,
    FaceFactor,
    PairRefused,
    analyse_es4_pair,
    checked_output_path,
    choose_reduction,
    corrected_ln_ratio,
    f_half_bracket_m,
    gate_verdict,
    half_difference_ln,
    plateau_ln_ratio,
    read_rest_bias_factors,
    refusal_verdict,
    run_control,
    sweep_face_is_upstream,
)

# A three-cell transverse grid whose middle cell is the declared x = 0.
X_CM = np.array([-1.0, 0.0, 1.0])
# Two plateau samples and one sample outside the window, so the window itself
# is exercised rather than assumed.
TIMES_S = np.array([13.0e-3, 15.0e-3, 17.0e-3])
IN_WINDOW = np.array([False, True, True])

# Upstream/downstream current densities chosen so ln R is exactly ln 2 at
# rot 0 and exactly ln 4 at rot 180, on every admitted cell.
ROT0_UP, ROT0_DOWN = 2.0, 1.0
ROT180_UP, ROT180_DOWN = 4.0, 1.0
RAW_HALF_DIFFERENCE_LN = (math.log(2.0) - math.log(4.0)) / 2.0

SWEEP_FILE = "processed/isweep_rot180_deadtime_profiles.hdf5"
ISAT_FILE = "processed/isat_rot180_deadtime_profiles.hdf5"


def _write_run(group, *, port, rotation_deg, upstream, downstream,
               upstream_file, downstream_file, mask=None):
    """One run group of the Mach product, masked cells written as NaN."""
    up = np.full((X_CM.size, TIMES_S.size), float(upstream))
    down = np.full((X_CM.size, TIMES_S.size), float(downstream))
    if mask is not None:
        up[mask] = np.nan
        down[mask] = np.nan
    group.create_dataset("inter_sweep_time_s", data=TIMES_S)
    group.create_dataset("upstream_current_density_a_m2", data=up)
    group.create_dataset("downstream_current_density_a_m2", data=down)
    group.attrs["port"] = port
    group.attrs["rotation_deg"] = float(rotation_deg)
    group.attrs["run_id"] = f"{port}{rotation_deg}"
    group.attrs["upstream_source_file"] = upstream_file
    group.attrs["downstream_source_file"] = downstream_file


def _synthetic_product(tmp_path, *, rot180_mask=None):
    """A one-port ES4 rotation pair: swept face upstream at rot 0 only."""
    path = tmp_path / "mach_velocity.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("x_cm", data=X_CM)
        sets = handle.create_group("experiment_sets")
        runs = sets.create_group("4")
        _write_run(
            runs.create_group("42"),
            port=21, rotation_deg=0,
            upstream=ROT0_UP, downstream=ROT0_DOWN,
            upstream_file=SWEEP_FILE, downstream_file=ISAT_FILE,
        )
        _write_run(
            runs.create_group("43"),
            port=21, rotation_deg=180,
            upstream=ROT180_UP, downstream=ROT180_DOWN,
            upstream_file=ISAT_FILE, downstream_file=SWEEP_FILE,
            mask=rot180_mask,
        )
    return path


def _factors(central_0, spread_0, central_180, spread_180):
    return {
        (21, 0): FaceFactor(21, 0, central_0, spread_0),
        (21, 180): FaceFactor(21, 180, central_180, spread_180),
    }


def test_the_rotation_decides_which_face_the_factor_lands_on(tmp_path):
    """The swept face is upstream at rot 0 and downstream at rot 180."""
    path = _synthetic_product(tmp_path)
    with h5py.File(path, "r") as handle:
        runs = handle["experiment_sets"]["4"]
        assert sweep_face_is_upstream(runs["42"]) is True
        assert sweep_face_is_upstream(runs["43"]) is False


def test_a_pair_whose_faces_cannot_be_told_apart_is_refused(tmp_path):
    """Two sources naming the same channel are refused, never guessed."""
    path = tmp_path / "ambiguous.hdf5"
    with h5py.File(path, "w") as handle:
        _write_run(
            handle.create_group("run"),
            port=21, rotation_deg=0,
            upstream=ROT0_UP, downstream=ROT0_DOWN,
            upstream_file=SWEEP_FILE, downstream_file=SWEEP_FILE,
        )
        with pytest.raises(PairRefused, match="cannot tell the swept face"):
            sweep_face_is_upstream(handle["run"])


def test_the_factor_moves_the_half_difference_by_the_mean_log_factor(tmp_path):
    """``+ln F`` on the upstream face, ``-ln F`` on the downstream one.

    With the swept face upstream at rot 0 and downstream at rot 180 the two
    shifts add in the half-difference, so the corrected value must be the raw
    one plus ``(ln F_rot0 + ln F_rot180) / 2`` exactly.
    """
    f_rot0, f_rot180 = 1.25, 1.60
    path = _synthetic_product(tmp_path)
    with h5py.File(path, "r") as handle:
        runs = handle["experiment_sets"]["4"]
        ln0 = plateau_ln_ratio(runs["42"])
        ln180 = plateau_ln_ratio(runs["43"])
        reduction = choose_reduction(ln0, ln180, X_CM)
        raw = half_difference_ln(ln0, ln180, reduction)
        corrected = half_difference_ln(
            corrected_ln_ratio(ln0, f_rot0, True),
            corrected_ln_ratio(ln180, f_rot180, False),
            reduction,
        )

    assert raw == pytest.approx(RAW_HALF_DIFFERENCE_LN)
    expected_shift = (math.log(f_rot0) + math.log(f_rot180)) / 2.0
    assert corrected == pytest.approx(raw + expected_shift)
    # Landing the factor on the wrong face of one rotation would cancel the
    # shift instead of doubling it, so the two cases must not coincide.
    wrong_face = half_difference_ln(
        corrected_ln_ratio(ln0, f_rot0, True),
        corrected_ln_ratio(ln180, f_rot180, True),
        reduction,
    )
    assert wrong_face == pytest.approx(raw + (math.log(f_rot0) - math.log(f_rot180)) / 2.0)
    assert wrong_face != pytest.approx(corrected)


def test_the_three_bracket_points_are_evaluated_and_ordered(tmp_path):
    """Central, low and high F give three half-differences, low the smallest.

    The correction is monotone increasing in both factors, so the low end of
    the spread must give the smallest corrected half-difference and the high
    end the largest, with the central value strictly between them.
    """
    path = _synthetic_product(tmp_path)
    factors = _factors(1.20, 0.10, 1.50, 0.20)
    with h5py.File(path, "r") as handle:
        row = analyse_es4_pair(handle["experiment_sets"], 21, factors, X_CM)

    assert row["reduction"] == X0_LABEL
    assert row["sweep_face_rot0"] == "upstream"
    assert row["sweep_face_rot180"] == "downstream"
    assert set(row["points"]) == {"central", "low", "high"}
    for point, (f0, f180) in {
        "central": (1.20, 1.50),
        "low": (1.10, 1.30),
        "high": (1.30, 1.70),
    }.items():
        entry = row["points"][point]
        assert entry["f_rot0"] == pytest.approx(f0)
        assert entry["f_rot180"] == pytest.approx(f180)
        expected = RAW_HALF_DIFFERENCE_LN + (math.log(f0) + math.log(f180)) / 2.0
        assert entry["half_difference_ln"] == pytest.approx(expected)
        assert entry["half_difference_m"] == pytest.approx(expected / MACH_K)

    values = [row["points"][p]["half_difference_ln"] for p in ("low", "central", "high")]
    assert values[0] < values[1] < values[2]


def test_a_spread_reaching_zero_is_refused_rather_than_clipped():
    """A current ratio cannot be evaluated at a non-positive factor."""
    factor = FaceFactor(21, 0, 1.05, 1.05)
    assert factor.at("high") == pytest.approx(2.10)
    with pytest.raises(ValueError, match="must be positive"):
        factor.at("low")
    with pytest.raises(ValueError, match="unknown bracket point"):
        factor.at("middle")


def test_the_gate_reads_magnitudes_against_the_upper_edge_only():
    """The largest of the nine is the only edge, and it is inclusive."""
    low, high = MACH_FACE_ASYMMETRY_M_RANGE
    assert GATE_HIGH_M == high
    joins = "joins as a labelled ES4 member"
    assert gate_verdict(high) == joins
    assert gate_verdict(-high) == joins
    assert gate_verdict(low) == joins
    assert gate_verdict(-(low + high) / 2) == joins
    above = gate_verdict(-2 * high)
    assert above.startswith("outside the range") and f"{2 * high:.4f}" in above
    assert f"{high:.3f}" in above
    assert refusal_verdict(255) == "REFUSED (0 of 255)"


def test_a_value_indistinguishable_from_zero_is_inside_not_below():
    """There is no lower edge: the nine pairs' smallest value is not a floor.

    The previous form gated on ``low <= |M| <= high`` and put anything under the
    smallest observed ES1-ES3 asymmetry OUTSIDE, which excluded a pair for
    agreeing with the convention better than any of the nine did.
    """
    low, _ = MACH_FACE_ASYMMETRY_M_RANGE
    joins = "joins as a labelled ES4 member"
    assert gate_verdict(0.0) == joins
    assert gate_verdict(-0.0) == joins
    assert gate_verdict(low / 100) == joins
    # The value that made the old form fire: p29 at the high end of F.
    assert gate_verdict(-0.0013) == joins
    assert low / 2 < low
    assert gate_verdict(low / 2) == joins


def test_the_f_half_bracket_is_half_the_low_to_high_spread(tmp_path):
    """The printed bar is half the distance between the two F end readings."""
    path = _synthetic_product(tmp_path)
    factors = _factors(1.20, 0.10, 1.50, 0.20)
    with h5py.File(path, "r") as handle:
        row = analyse_es4_pair(handle["experiment_sets"], 21, factors, X_CM)

    low = row["points"]["low"]["half_difference_m"]
    high = row["points"]["high"]["half_difference_m"]
    bar = f_half_bracket_m(row)
    assert bar == pytest.approx((high - low) / 2.0)
    # The correction is additive in ln, so the bar follows from the factors
    # alone and does not depend on the raw half-difference at all.
    expected = (
        (math.log(1.30) + math.log(1.70)) - (math.log(1.10) + math.log(1.30))
    ) / 4.0 / MACH_K
    assert bar == pytest.approx(expected)
    assert bar > 0.0


def test_a_masked_core_falls_back_to_the_surviving_cells(tmp_path):
    """With x = 0 masked the pair is read on the kept cells and says so."""
    mask = np.zeros((X_CM.size, TIMES_S.size), dtype=bool)
    mask[1, :] = True  # the x = 0 row, every time sample
    path = _synthetic_product(tmp_path, rot180_mask=mask)
    with h5py.File(path, "r") as handle:
        runs = handle["experiment_sets"]["4"]
        ln0 = plateau_ln_ratio(runs["42"])
        ln180 = plateau_ln_ratio(runs["43"])
        reduction = choose_reduction(ln0, ln180, X_CM)

    assert reduction.label == EDGE_CELLS_LABEL
    # Two surviving x rows over the two in-window samples.
    assert reduction.n_admitted == 2 * int(IN_WINDOW.sum())
    assert reduction.n_candidate == X_CM.size * int(IN_WINDOW.sum())
    assert not reduction.mask[1].any()

    # Masking every cell instead leaves nothing to read.
    all_masked = np.ones((X_CM.size, TIMES_S.size), dtype=bool)
    everything = tmp_path / "all"
    everything.mkdir()
    path = _synthetic_product(everything, rot180_mask=all_masked)
    with h5py.File(path, "r") as handle:
        runs = handle["experiment_sets"]["4"]
        empty = choose_reduction(
            plateau_ln_ratio(runs["42"]), plateau_ln_ratio(runs["43"]), X_CM
        )
    assert empty is None


def test_the_edge_reduction_is_exactly_the_cells_outside_the_mask(tmp_path):
    """Run 43's shape: core and one whole side masked, one edge left.

    The extended registration masks the core AND the far-positive side, so the
    cells the pair is read on are the low-state ones that remain.  The reduction
    must be exactly those and nothing else -- a cell the product marked unusable
    must not come back in through the fallback, and the half-difference must be
    the one the surviving cells give.
    """
    mask = np.zeros((X_CM.size, TIMES_S.size), dtype=bool)
    mask[1, :] = True   # the core, x = 0
    mask[2, :] = True   # one whole side, x = +1
    path = _synthetic_product(tmp_path, rot180_mask=mask)
    with h5py.File(path, "r") as handle:
        runs = handle["experiment_sets"]["4"]
        ln0 = plateau_ln_ratio(runs["42"])
        ln180 = plateau_ln_ratio(runs["43"])
        reduction = choose_reduction(ln0, ln180, X_CM)
        value = half_difference_ln(ln0, ln180, reduction)

    assert reduction.label == EDGE_CELLS_LABEL
    # The reduction mask is already restricted to the plateau window, so it has
    # one column per in-window sample.
    survivors = np.zeros_like(reduction.mask)
    survivors[0, :] = True
    assert survivors.shape == (X_CM.size, int(IN_WINDOW.sum()))
    assert np.array_equal(reduction.mask, survivors)
    assert reduction.n_admitted == int(IN_WINDOW.sum())
    assert reduction.n_candidate == X_CM.size * int(IN_WINDOW.sum())
    assert not reduction.mask[1].any() and not reduction.mask[2].any()
    assert value == pytest.approx(RAW_HALF_DIFFERENCE_LN)


def test_the_registered_edge_cut_removes_exactly_the_outer_positions():
    """``|x| < EDGE_CUT_CM`` on the edge-cell reduction, and nothing else.

    The grid and the surviving positions are run 43's: the extended mask leaves
    x = -25 ... -14 and +10 ... +12, and the cut must take the three positions
    at |x| >= 23 cm and leave the other twelve untouched.  It is checked as a
    set difference rather than by a count, so a cut that removed the wrong
    positions could not pass by removing the right number of them.
    """
    x_cm = np.linspace(-25.0, 25.0, 51)
    survivors = ((x_cm >= -25) & (x_cm <= -14)) | ((x_cm >= 10) & (x_cm <= 12))
    ln = np.where(survivors[:, None], 0.5, np.nan) * np.ones((x_cm.size, 5))

    reduction = choose_reduction(ln, ln, x_cm)

    assert reduction.label == EDGE_CELLS_LABEL
    assert f"{EDGE_CUT_CM:.0f}" in EDGE_CELLS_LABEL
    admitted = set(x_cm[reduction.mask.any(axis=1)])
    offered = set(x_cm[survivors])
    assert offered - admitted == {-25.0, -24.0, -23.0}
    assert admitted == {x for x in offered if abs(x) < EDGE_CUT_CM}
    assert reduction.n_admitted == 12 * 5
    # The cut narrows what is READ, never what was on offer.
    assert reduction.n_candidate == x_cm.size * 5


def test_the_edge_cut_does_not_touch_the_declared_reduction():
    """A cell at x = 0 is admitted whatever the cut is; only the fallback cuts."""
    x_cm = np.linspace(-25.0, 25.0, 51)
    ln = np.full((x_cm.size, 5), 0.5)

    reduction = choose_reduction(ln, ln, x_cm)

    assert reduction.label == X0_LABEL
    assert set(x_cm[reduction.mask.any(axis=1)]) == {0.0}


def test_an_output_inside_the_product_directory_is_refused(tmp_path):
    """This instrument writes measurements, never into the product chain."""
    good = checked_output_path(tmp_path / "es4_face_asymmetry_subset.csv")
    assert good.name == "es4_face_asymmetry_subset.csv"
    with pytest.raises(ValueError, match=FORBIDDEN_OUTPUT_DIR):
        checked_output_path(tmp_path / FORBIDDEN_OUTPUT_DIR / "out.csv")


def test_the_rest_bias_csv_is_read_as_the_direct_factor(tmp_path):
    """The direct column and its spread are the ones read; duplicates refuse."""
    path = tmp_path / "rest_bias.csv"
    fields = ["port", "rotation_deg", "f_direct_up", "f_direct_up_std", "f_linear"]
    rows = [
        {"port": 21, "rotation_deg": 0, "f_direct_up": 1.1, "f_direct_up_std": 0.04,
         "f_linear": 1.35},
        {"port": 21, "rotation_deg": 180, "f_direct_up": 1.2, "f_direct_up_std": 0.05,
         "f_linear": 1.29},
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    factors = read_rest_bias_factors(path)
    assert factors[(21, 0)].central == pytest.approx(1.1)
    assert factors[(21, 0)].spread == pytest.approx(0.04)
    assert factors[(21, 180)].central == pytest.approx(1.2)

    with path.open("a", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fields).writerow(rows[0])
    with pytest.raises(ValueError, match="twice"):
        read_rest_bias_factors(path)


def test_the_nine_es1_es3_half_differences_reproduce():
    """The control: the declared statistic, off the shipped Mach product."""
    from pathlib import Path

    product = Path("processed/mach_velocity.hdf5")
    if not product.exists():
        pytest.skip("the rotation-pair Mach product is not in this checkout")

    with h5py.File(product, "r") as handle:
        rows, ok = run_control(handle["experiment_sets"], handle["x_cm"][()])

    assert len(rows) == len(CONTROL_HALF_DIFFERENCES_M) == 9
    for row in rows:
        expected = CONTROL_HALF_DIFFERENCES_M[(row["experiment_set"], row["port"])]
        assert row["reduction"] == X0_LABEL
        assert row["measured_m"] == pytest.approx(expected, abs=CONTROL_TOL_M)
    assert ok
