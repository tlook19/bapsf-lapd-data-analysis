import h5py
import numpy as np
import pytest

from scripts.export_es1_sim1d_overlay import (
    DESPIKE_MIN_PEAK_FRACTION,
    FLUX_TUBE_RADIUS_CM,
    PORTS,
    X_MAX_CM,
    X_MIN_CM,
    _despike_profile,
    _flux_tube_profile_stats,
    _flux_tube_weights,
    _rot0_isat_profiles,
    _subtract_background,
    _te_window_spread_frac,
)
from scripts.plot_core_density_temperature_timeseries import _nan_core_stats

X_CM = np.linspace(-25.0, 25.0, 51)


def _write_refits(path, sets):
    with h5py.File(path, "w") as hdf:
        for set_id, ports in sets.items():
            group = hdf.create_group(f"set{set_id}")
            for port, grid in ports.items():
                group.create_group(f"port{port}").create_dataset(
                    "te_window_ev",
                    data=np.asarray(grid, dtype=np.float64),
                )


def test_window_spread_is_peak_to_peak_over_the_mean(tmp_path):
    refits = tmp_path / "sweep_window_refits.hdf5"
    _write_refits(refits, {1: {11: np.full((5, 5), 4.0)}})
    with h5py.File(refits, "r+") as hdf:
        hdf["set1/port11/te_window_ev"][0, 0] = 3.0
        hdf["set1/port11/te_window_ev"][0, 1] = 5.0

    spread = _te_window_spread_frac(refits, 1, np.array([11], dtype=np.int16))

    grid = np.full((5, 5), 4.0)
    grid[0, 0] = 3.0
    grid[0, 1] = 5.0
    assert spread[0] == pytest.approx(np.ptp(grid) / np.mean(grid))


def test_window_spread_is_aligned_with_the_port_axis(tmp_path):
    refits = tmp_path / "sweep_window_refits.hdf5"
    _write_refits(
        refits,
        {
            1: {
                int(port): np.full((5, 5), 2.0) + index
                for index, port in enumerate(PORTS)
            }
        },
    )
    with h5py.File(refits, "r+") as hdf:
        hdf["set1/port29/te_window_ev"][2, 3] = 8.0

    spread = _te_window_spread_frac(refits, 1, PORTS)

    assert spread.shape == PORTS.shape
    uniform = np.array([i != 2 for i in range(len(PORTS))])
    assert np.all(spread[uniform] == 0.0)
    assert spread[2] > 0.0


def test_window_spread_is_nan_without_a_refit_group(tmp_path):
    refits = tmp_path / "sweep_window_refits.hdf5"
    _write_refits(refits, {1: {11: np.full((5, 5), 4.0)}})

    partial = _te_window_spread_frac(refits, 1, np.array([11, 21], dtype=np.int16))
    missing_set = _te_window_spread_frac(refits, 4, PORTS)

    assert partial[0] == 0.0
    assert np.isnan(partial[1])
    assert np.all(np.isnan(missing_set))


def test_window_spread_is_nan_when_a_window_refit_failed(tmp_path):
    refits = tmp_path / "sweep_window_refits.hdf5"
    _write_refits(refits, {3: {50: np.full((5, 5), 1.0)}})
    with h5py.File(refits, "r+") as hdf:
        hdf["set3/port50/te_window_ev"][4, 4] = np.nan

    spread = _te_window_spread_frac(refits, 3, np.array([50], dtype=np.int16))

    assert np.isnan(spread[0])


def test_flux_tube_weights_average_a_uniform_profile_to_its_own_level():
    for centroid in (0.0, 1.7, -3.25):
        weights = _flux_tube_weights(np.abs(X_CM - centroid), FLUX_TUBE_RADIUS_CM)
        assert weights.sum() == pytest.approx(1.0, abs=1e-12)


def test_flux_tube_weights_match_the_analytic_cone_average():
    radius = np.abs(X_CM)
    cone = np.clip(1.0 - radius / FLUX_TUBE_RADIUS_CM, 0.0, None)
    average = _flux_tube_weights(radius, FLUX_TUBE_RADIUS_CM) @ cone
    # 2/R^2 * int_0^R r (1 - r/R) dr = 1/3, up to the 1 cm sampling.
    assert average == pytest.approx(1.0 / 3.0, rel=5e-3)


def test_flux_tube_weights_refuse_a_profile_that_stops_short_of_the_radius():
    short = np.linspace(0.0, FLUX_TUBE_RADIUS_CM - 1.0, 20)
    with pytest.raises(ValueError, match="short of the flux-tube radius"):
        _flux_tube_weights(short, FLUX_TUBE_RADIUS_CM)


def test_flux_tube_average_of_a_flat_column_equals_its_core_band_mean():
    # Flat across the whole integration disc, tapering outside it so the
    # background is well defined and no step triggers the despike gate.
    taper = {21: 2.0, 22: 1.0, 23: 0.5, 24: 0.25, 25: 0.0}
    flat = np.array(
        [taper.get(int(abs(position)), 3.0) for position in X_CM],
        dtype=np.float64,
    )
    stats = _flux_tube_profile_stats(flat, X_CM)
    assert stats["n_despiked"] == 0
    level = 3.0 - 0.25  # the outer-three median baseline is subtracted
    assert stats["ftavg"] == pytest.approx(level, rel=1e-12)
    assert stats["core"] == pytest.approx(level, rel=1e-12)
    assert stats["centroid"] == pytest.approx(0.0, abs=1e-12)


def test_flux_tube_average_weights_the_outer_radii_more_than_the_core_mean():
    peaked = np.exp(-((X_CM / 8.0) ** 2))
    stats = _flux_tube_profile_stats(peaked, X_CM)
    assert stats["core"] > stats["ftavg"]


def test_despike_repairs_an_isolated_bracketed_spike():
    profile = np.full(51, 2.0)
    profile[25] = 8.0
    repaired, n_replaced = _despike_profile(profile)
    assert n_replaced == 1
    assert repaired[25] == pytest.approx(2.0)
    assert np.array_equal(np.delete(repaired, 25), np.delete(profile, 25))


def test_despike_leaves_a_monotone_skirt_alone():
    profile = np.exp(-((X_CM / 9.0) ** 2))
    repaired, n_replaced = _despike_profile(profile)
    assert n_replaced == 0
    assert np.array_equal(repaired, profile)


def test_despike_does_not_judge_an_unbracketed_endpoint():
    profile = np.full(51, 2.0)
    profile[0] = 8.0
    repaired, n_replaced = _despike_profile(profile)
    assert n_replaced == 0
    assert repaired[0] == 8.0


def test_despike_does_not_judge_cells_below_the_amplitude_floor():
    profile = np.full(51, 100.0)
    profile[25] = 0.25 * DESPIKE_MIN_PEAK_FRACTION * 100.0
    repaired, n_replaced = _despike_profile(profile)
    assert n_replaced == 0
    assert np.array_equal(repaired, profile)


def test_background_is_the_smaller_outer_three_median_clipped_at_zero():
    profile = np.full(51, 5.0)
    profile[:3] = [1.0, 2.0, 3.0]
    profile[-3:] = [7.0, 8.0, 9.0]
    assert _subtract_background(profile)[25] == pytest.approx(3.0)

    negative = np.full(51, 5.0)
    negative[:3] = -1.0
    assert _subtract_background(negative)[25] == pytest.approx(5.0)


def test_legacy_core_band_mean_stays_an_unweighted_line_cut_mean():
    values = np.arange(51, dtype=np.float64).reshape(1, 51, 1)
    mean, _, _, count = _nan_core_stats(values, X_CM, X_MIN_CM, X_MAX_CM)
    band = (X_CM >= X_MIN_CM) & (X_CM <= X_MAX_CM)
    assert mean[0, 0] == pytest.approx(float(np.mean(values[0, band, 0])))
    assert count[0, 0] == int(band.sum())


def _write_line_scans(path, runs, *, rotation_deg=0.0, n_t=4):
    """Minimal stand-in for a dead-time line-scan product."""
    with h5py.File(path, "w") as hdf:
        hdf.attrs["rotation_filter_deg"] = rotation_deg
        hdf.create_dataset("x_cm", data=X_CM)
        group = hdf.create_group("experiment_sets/1")
        for run_id, (port, z_cm, channel) in runs.items():
            run = group.create_group(run_id)
            run.attrs["port"] = port
            run.attrs["z_cm"] = z_cm
            run.attrs["deadtime_source_channel"] = channel
            run.create_dataset("inter_sweep_time_s", data=np.arange(n_t) * 1e-3)
            run.create_dataset("isat_a", data=np.full((51, n_t), float(port)))
            run.create_dataset("isat_a_std", data=np.full((51, n_t), 2.0))
            run.create_dataset("n_shots_used", data=np.full((51, n_t), 4))


def test_line_scans_are_ordered_onto_the_overlay_z_axis(tmp_path):
    path = tmp_path / "scans.hdf5"
    _write_line_scans(
        path,
        {
            "08": (50, 1716.1, "i_sweep"),
            "01": (11, 470.05, "i_sweep"),
            "04": (29, 1045.15, "i_sweep"),
        },
    )
    z_cm = np.array([470.05, 1045.15, 1716.1])

    scans = _rot0_isat_profiles(path, 1, z_cm)

    assert list(scans["run_id"]) == ["01", "04", "08"]
    assert list(scans["port"]) == [11, 29, 50]
    assert scans["isat_a"][:, 0, 0].tolist() == [11.0, 29.0, 50.0]
    # SEM is the per-cell shot SEM, std / sqrt(n_shots_used).
    assert scans["sem_a"][0, 0, 0] == pytest.approx(1.0)


def test_line_scans_keep_a_mixed_channel_set_and_disclose_it(tmp_path):
    path = tmp_path / "scans.hdf5"
    _write_line_scans(
        path,
        {"01": (11, 470.05, "i_sweep"), "04": (29, 1045.15, "isat")},
    )

    scans = _rot0_isat_profiles(path, 1, np.array([470.05, 1045.15]))

    assert list(scans["source_channel"]) == ["i_sweep", "isat"]


def test_line_scans_refuse_a_rotated_product(tmp_path):
    path = tmp_path / "scans.hdf5"
    _write_line_scans(path, {"03": (21, 789.55, "isat")}, rotation_deg=180.0)

    with pytest.raises(ValueError, match="rot-180 product"):
        _rot0_isat_profiles(path, 1, np.array([789.55]))


def test_line_scans_refuse_a_mismatched_z_grid(tmp_path):
    path = tmp_path / "scans.hdf5"
    _write_line_scans(path, {"01": (11, 470.05, "i_sweep")})

    with pytest.raises(ValueError, match="does not match the overlay z grid"):
        _rot0_isat_profiles(path, 1, np.array([999.0]))
