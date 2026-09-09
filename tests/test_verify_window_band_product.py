"""Coverage of the band product's tolerance-mode regeneration check.

The four cases are the ones the check exists to tell apart: a regeneration that
reproduces the placed product exactly, one that moves it by less than the
library-vintage drift, one that moves it by far more, and one that changes
which cells were fittable at all.  None of them runs the pass itself -- the
products are written by ``write_product`` from synthetic records, so what is
pinned here is the comparison and its verdict, not the fitting.
"""
from pathlib import Path

import numpy as np

from scripts.refit_window_band import write_product
from scripts.verify_window_band_product import compare_band_products, exit_code

RTOL = 1.0e-6

X_CM = np.array([-2.0, -1.0, 0.0, 1.0])


def _record(dln):
    """A four-cell band record carrying the given per-cell window spreads."""
    return dict(
        cells=np.array([0, 1, 2, 3]),
        x_cm=X_CM,
        is_x0=np.array([False, False, True, False]),
        dln_te_window=np.asarray(dln, dtype=float),
        te_default_med=np.array([0.70, 0.72, 0.74, 0.71]),
        n_sweeps=np.array([200, 200, 200, 200]),
        med_grids=np.full((4, 5, 5), 0.66),
        run_id="42",
        sid=1,
        port=21,
        wall_s=1.0,
        n_plateau=10,
    )


def _write(tmp_path: Path, label: str, dln) -> tuple[Path, Path]:
    """Write a band product under ``label``; return its HDF5 and CSV paths."""
    hdf5_path = tmp_path / f"{label}.hdf5"
    summary_path = tmp_path / f"{label}.csv"
    write_product(
        [_record(dln)],
        X_CM,
        hdf5_path=hdf5_path,
        summary_path=summary_path,
        metadata_path=tmp_path / f"{label}.json",
    )
    return hdf5_path, summary_path


BASE_DLN = [0.30, 0.44, 0.35, 0.40]


def _compare(tmp_path, new_dln):
    placed_hdf5, placed_summary = _write(tmp_path, "placed", BASE_DLN)
    new_hdf5, new_summary = _write(tmp_path, "new", new_dln)
    return compare_band_products(
        placed_hdf5=placed_hdf5,
        placed_summary=placed_summary,
        new_hdf5=new_hdf5,
        new_summary=new_summary,
        rtol=RTOL,
    )


def test_an_identical_regeneration_is_byte_exact(tmp_path):
    result = _compare(tmp_path, BASE_DLN)

    assert exit_code(result) == 0
    assert result.byte_exact
    assert result.max_rel == 0.0
    assert all(row.byte_identical for row in result.rows)


def test_a_vintage_sized_move_passes_within_tolerance(tmp_path):
    # A relative move of 1e-9 is two orders below the tolerance and one order
    # below the library-vintage drift this product actually shows.
    moved = [value * (1.0 + 1.0e-9) for value in BASE_DLN]
    result = _compare(tmp_path, moved)

    assert exit_code(result) == 0
    assert not result.byte_exact
    assert 0.0 < result.max_rel <= RTOL


def test_a_move_far_outside_the_tolerance_fails_and_names_the_dataset(tmp_path):
    moved = [value * (1.0 + 1.0e-3) for value in BASE_DLN]
    result = _compare(tmp_path, moved)

    assert exit_code(result) == 1
    assert not result.byte_exact
    assert result.failure is not None
    assert "dln_te_window" in result.failure
    assert "rtol" in result.failure


def test_a_flipped_nan_mask_fails_however_small_the_tolerance_would_allow(tmp_path):
    # Cell 1 was unfittable in the placed product and fittable in the
    # regeneration.  No tolerance covers that: it is a different measurement.
    placed = [0.30, float("nan"), 0.35, 0.40]
    result = _compare_asymmetric(tmp_path, placed, BASE_DLN)

    assert exit_code(result) == 1
    assert not result.byte_exact
    assert result.failure is not None
    assert "dln_te_window" in result.failure
    assert "NaN mask" in result.failure


def _compare_asymmetric(tmp_path, placed_dln, new_dln):
    placed_hdf5, placed_summary = _write(tmp_path, "placed", placed_dln)
    new_hdf5, new_summary = _write(tmp_path, "new", new_dln)
    return compare_band_products(
        placed_hdf5=placed_hdf5,
        placed_summary=placed_summary,
        new_hdf5=new_hdf5,
        new_summary=new_summary,
        rtol=RTOL,
    )
