"""The rot-180 pair's rail attrs, read back across the two placed products.

The pair's two products are annotated by two different scripts, and each one
records BOTH its own face's dead-time rail fraction and the opposite face's.
So for a given run the ISAT face is measured twice -- once as the isat
product's ``rail_fraction_deadtime`` and once as the isweep product's
``opposite_face_rail_fraction_deadtime`` -- and the I_SWEEP face likewise.
Those two readings have to be the same number, bitwise: they are the same
measurement of the same raw record, and a Mach pair built on a disagreement
between them is contaminated on a side nobody named.  The dead-time fraction is
the one shared-named rail attr with an opposite-face counterpart written into
the sibling; the rest describe only the product's own face, so for those the
cross-product check is that both products carry the same shared names -- an
eighth one has to be brought in here deliberately -- and stamp the same screen
constants.

This reads the products as placed, so it also covers the writers agreeing on
the screen constants they stamp at the root.  It skips when the products are
not in the checkout, and it refuses to pass vacuously: the railed run must be
present and read non-zero on the face that rails.
"""
from pathlib import Path

import h5py
import numpy as np
import pytest

ISAT_PRODUCT = Path("processed/isat_rot180_deadtime_profiles.hdf5")
ISWEEP_PRODUCT = Path("processed/isweep_rot180_deadtime_profiles.hdf5")

# Written by both annotators onto their own product, from the same measurement.
SHARED_RUN_ATTRS = (
    "rail_fraction_full_record",
    "rail_fraction_deadtime",
    "rail_fraction_plateau",
    "raw_code_min",
    "raw_code_max",
    "saturation_excluded",
    "opposite_face_rail_fraction_deadtime",
)
# Screen constants both annotators stamp at the product root.
SHARED_ROOT_ATTRS = (
    "saturation_screen_method",
    "saturation_screen_rail_codes",
    "saturation_screen_clip_s",
    "saturation_screen_plateau_ms",
)


def _read(path):
    """{run_id: {attr: value}} for every run group, plus the root attrs."""
    runs, root = {}, {}
    with h5py.File(path, "r") as hf:
        root = dict(hf.attrs)
        for set_id in sorted(hf["experiment_sets"], key=int):
            for run_id in sorted(hf[f"experiment_sets/{set_id}"]):
                runs[run_id] = dict(hf[f"experiment_sets/{set_id}/{run_id}"].attrs)
    return runs, root


def _pair_or_skip():
    if not (ISAT_PRODUCT.exists() and ISWEEP_PRODUCT.exists()):
        pytest.skip("the rot-180 product pair is not in this checkout")
    isat_runs, isat_root = _read(ISAT_PRODUCT)
    isweep_runs, isweep_root = _read(ISWEEP_PRODUCT)
    for runs in (isat_runs, isweep_runs):
        if not all("rail_fraction_deadtime" in attrs for attrs in runs.values()):
            pytest.skip("a product predates the rail attrs")
    return isat_runs, isat_root, isweep_runs, isweep_root


def _shared_named(attrs):
    """The rail attrs of one run group that both writers write under one name."""
    return {key for key in attrs
            if key.startswith(("rail_fraction", "opposite_face_rail", "raw_code"))
            or key == "saturation_excluded"}


def _bits(value):
    return np.float64(value).tobytes()


def test_the_two_products_carry_the_same_runs():
    isat_runs, _, isweep_runs, _ = _pair_or_skip()

    assert set(isat_runs) == set(isweep_runs)
    assert isat_runs


def test_both_products_carry_the_shared_rail_attrs_this_test_covers():
    isat_runs, _, isweep_runs, _ = _pair_or_skip()

    for runs in (isat_runs, isweep_runs):
        for run_id, attrs in runs.items():
            assert _shared_named(attrs) == set(SHARED_RUN_ATTRS), run_id


def test_the_products_agree_bitwise_on_the_isat_face_fraction():
    isat_runs, _, isweep_runs, _ = _pair_or_skip()

    # The isat product measures this face as its own; the isweep product
    # measures the same face as the opposite one.
    for run_id, attrs in isat_runs.items():
        assert _bits(attrs["rail_fraction_deadtime"]) == _bits(
            isweep_runs[run_id]["opposite_face_rail_fraction_deadtime"]
        ), run_id


def test_the_products_agree_bitwise_on_the_isweep_face_fraction():
    isat_runs, _, isweep_runs, _ = _pair_or_skip()

    for run_id, attrs in isweep_runs.items():
        assert _bits(attrs["rail_fraction_deadtime"]) == _bits(
            isat_runs[run_id]["opposite_face_rail_fraction_deadtime"]
        ), run_id


def test_a_railed_run_is_present_and_reads_the_same_from_both_products():
    isat_runs, _, isweep_runs, _ = _pair_or_skip()

    railed = {run_id for run_id, attrs in isat_runs.items()
              if attrs["rail_fraction_deadtime"] > 0}
    assert railed, "no railed run in the pair: the cross-product check is vacuous"
    for run_id in railed:
        own = isat_runs[run_id]["rail_fraction_deadtime"]
        assert _bits(own) == _bits(
            isweep_runs[run_id]["opposite_face_rail_fraction_deadtime"]
        )
        assert own <= 1.0
        assert bool(isat_runs[run_id]["saturation_excluded"]) is True


def test_each_products_verdict_matches_the_siblings_reading_of_that_face():
    isat_runs, _, isweep_runs, _ = _pair_or_skip()

    for own_runs, other_runs in ((isat_runs, isweep_runs), (isweep_runs, isat_runs)):
        for run_id, attrs in own_runs.items():
            verdict = bool(attrs["saturation_excluded"])
            sibling = other_runs[run_id]["opposite_face_rail_fraction_deadtime"]
            assert verdict == (sibling > 0), run_id


def test_the_products_stamp_the_same_screen_constants():
    _, isat_root, _, isweep_root = _pair_or_skip()

    for key in SHARED_ROOT_ATTRS:
        assert isat_root[key] == isweep_root[key], key


def test_the_products_agree_on_which_face_each_run_railed():
    _, isat_root, _, isweep_root = _pair_or_skip()

    assert (isat_root["saturation_screen_runs_railed_this_face"]
            == isweep_root["saturation_screen_runs_railed_opposite_face"])
    assert (isweep_root["saturation_screen_runs_railed_this_face"]
            == isat_root["saturation_screen_runs_railed_opposite_face"])
