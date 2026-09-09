"""Coverage of the band pass's window statistics and its run resolver.

The pass itself reads raw digitizer traces; what is pinned here is the two
per-cell statistics the product reports and which runs each cell set and
rotation walks, none of which depends on the measured values.
"""
import h5py
import numpy as np

from scripts.refit_window_band import (
    CORE_MAX_CM,
    core_cell_indices,
    set_runs,
    window_family_median,
    window_spread_dln,
)


def _write_sweeps(path, runs, *, x_cm):
    """Write a minimal ``langmuir_sweeps``-shaped source for the resolver."""
    with h5py.File(path, "w") as hdf:
        hdf.create_dataset("x_cm", data=np.asarray(x_cm, dtype=np.float64))
        for set_id, run_id, port, rotation_deg in runs:
            group = hdf.create_group(f"experiment_sets/{set_id}/{run_id}")
            group.attrs["port"] = np.int64(port)
            group.attrs["rotation_deg"] = np.float64(rotation_deg)
    return path


# --------------------------------------------------------------------------
# window-spread statistic
# --------------------------------------------------------------------------


def test_window_spread_is_the_log_ratio_of_the_family_extremes():
    grid = np.array([[1.0, 2.0], [4.0, 3.0]])
    assert window_spread_dln(grid) == np.log(4.0)


def test_window_spread_ignores_unfittable_windows():
    grid = np.array([[np.nan, 2.0], [4.0, np.nan]])
    assert window_spread_dln(grid) == np.log(2.0)


def test_window_spread_of_a_flat_family_is_zero():
    assert window_spread_dln(np.full((5, 5), 1.7)) == 0.0


def test_window_spread_is_nan_when_no_window_fitted():
    assert np.isnan(window_spread_dln(np.full((5, 5), np.nan)))


def test_window_spread_is_nan_rather_than_a_spread_at_a_non_positive_minimum():
    # A non-positive T_e is unphysical, so the cell reports no spread rather
    # than a ratio that would read as stability or as a finite excursion.
    assert np.isnan(window_spread_dln(np.array([[0.0, 2.0], [4.0, 3.0]])))
    assert np.isnan(window_spread_dln(np.array([[-1.0, 2.0], [4.0, 3.0]])))


def test_window_family_median_is_the_median_over_every_window():
    grid = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, np.nan]])
    assert window_family_median(grid) == 3.0


# --------------------------------------------------------------------------
# core cell set
# --------------------------------------------------------------------------


def test_core_cells_are_the_cells_within_the_core_radius():
    x_cm = np.arange(-25.0, 25.5, 1.0)
    cells, index_x0 = core_cell_indices(x_cm)
    assert np.all(np.abs(x_cm[cells]) <= CORE_MAX_CM)
    assert cells.size == int((np.abs(x_cm) <= CORE_MAX_CM).sum())
    # x = 0 belongs to the core, so it is not appended as a separate control
    # the way the band pass appends it.
    assert x_cm[index_x0] == 0.0
    assert index_x0 in cells.tolist()
    assert cells.tolist() == sorted(set(cells.tolist()))


def test_core_radius_is_the_radius_the_te_row_is_built_from():
    from scripts.fit_te_spatial import X_CORE_CM

    assert CORE_MAX_CM == X_CORE_CM


# --------------------------------------------------------------------------
# set-4 port/run resolver
# --------------------------------------------------------------------------


ES4_RUNS = [
    (4, "41", 11, 0.0),
    (4, "42", 21, 0.0),
    (4, "43", 21, 180.0),
    (4, "44", 29, 0.0),
    (4, "45", 29, 180.0),
    (4, "46", 41, 0.0),
    (4, "47", 41, 180.0),
    (4, "48", 50, 0.0),
    (1, "02", 21, 0.0),
    (1, "03", 21, 180.0),
]


def _resolver_source(monkeypatch, tmp_path):
    import scripts.refit_window_band as module

    sweeps = _write_sweeps(
        tmp_path / "langmuir_sweeps.hdf5",
        ES4_RUNS,
        x_cm=np.arange(-25.0, 25.5, 1.0),
    )
    monkeypatch.setattr(module._rw, "SWEEPS_H5", sweeps)
    return module


def test_set_four_rot0_resolves_to_the_rot0_runs(monkeypatch, tmp_path):
    _resolver_source(monkeypatch, tmp_path)
    walked = list(set_runs((4,), 0.0))
    assert walked == [
        (4, "41", 11),
        (4, "42", 21),
        (4, "44", 29),
        (4, "46", 41),
        (4, "48", 50),
    ]


def test_set_four_rot180_resolves_to_the_reversed_probe_runs(monkeypatch, tmp_path):
    _resolver_source(monkeypatch, tmp_path)
    walked = list(set_runs((4,), 180.0))
    assert walked == [(4, "43", 21), (4, "45", 29), (4, "47", 41)]


def test_the_resolver_keeps_the_sets_apart(monkeypatch, tmp_path):
    _resolver_source(monkeypatch, tmp_path)
    assert all(set_id == 1 for set_id, _, _ in set_runs((1,), 0.0))
    assert list(set_runs((4,), 90.0)) == []


def test_rot0_resolution_is_the_published_walk(monkeypatch, tmp_path):
    # The rot-0 case must stay the x = 0 product's own walk, not a re-derived
    # one, so the band product cannot drift away from it.
    module = _resolver_source(monkeypatch, tmp_path)
    assert list(set_runs((1, 2, 4), 0.0)) == list(
        module._rw.rot0_runs(sets=(1, 2, 4))
    )
