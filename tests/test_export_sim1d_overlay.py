from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.export_es1_sim1d_overlay import (
    DESPIKE_MIN_PEAK_FRACTION,
    FLUX_TUBE_RADIUS_CM,
    PLASMA_DIAMETER_CM,
    PORTS,
    RAW_PLATEAU_WINDOW_MS,
    X_MAX_CM,
    X_MIN_CM,
    _check_density_convention_pair,
    _despike_profile,
    _discharge_stats,
    _flux_tube_profile_stats,
    _flow_symmetrized_profiles,
    _flux_tube_weights,
    _interferometer_decay_stats,
    _isat_decay_geomean,
    _plateau_current_a,
    _rot0_isat_profiles,
    _subtract_background,
    _t_half_level_ms,
    _te_trust_records,
    _te_window_spread_frac,
    export_overlay,
    REQUIRED_TE_RECORDS,
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


def _convention_pair(mean, ftavg):
    """One port row of the two density conventions, on a 3-sample time base."""
    return (
        np.array([mean], dtype=np.float64),
        np.array([ftavg], dtype=np.float64),
        np.array([29], dtype=np.int16),
        np.array([0.5, 1.5, 2.5], dtype=np.float64),
    )


def test_a_zero_core_band_mean_under_a_finite_flux_tube_average_is_refused():
    # The density grid holds strictly positive cells or NaN, so a zero
    # core-band mean means the grid has begun carrying zero-filled cells --
    # and the flux-tube row's error is transferred from that mean.
    args = _convention_pair([4.0, 0.0, 6.0], [3.0, 5.0, 5.5])

    with pytest.raises(ValueError, match="finite flux-tube average") as excinfo:
        _check_density_convention_pair(*args)

    message = str(excinfo.value)
    assert "port 29" in message
    assert "t = 1.5 ms" in message
    assert "density_total_sem_cm3 / density_mean_cm3" in message


def test_a_non_finite_core_band_mean_under_a_finite_flux_tube_average_is_refused():
    args = _convention_pair([4.0, np.nan, 6.0], [3.0, 5.0, 5.5])

    with pytest.raises(ValueError, match="zero or non-finite") as excinfo:
        _check_density_convention_pair(*args)

    assert "t = 1.5 ms" in str(excinfo.value)


def test_an_unusable_core_band_mean_passes_when_the_flux_tube_side_is_nan():
    # The other direction: the guard judges the PAIR, so a sample the
    # flux-tube reduction already refused is not a defect, and neither is a
    # fully usable one.  Nothing here raises.
    _check_density_convention_pair(*_convention_pair([4.0, 5.0, 6.0], [3.0, 4.0, 5.0]))
    _check_density_convention_pair(*_convention_pair([4.0, 0.0, 6.0], [3.0, np.nan, 5.0]))
    _check_density_convention_pair(
        *_convention_pair([4.0, np.nan, 6.0], [3.0, np.nan, 5.0])
    )


def test_the_two_conventions_must_share_one_port_sample_grid():
    with pytest.raises(ValueError, match="same \\(port, sample\\) grid"):
        _check_density_convention_pair(
            np.zeros((5, 40)),
            np.zeros((5, 20)),
            PORTS,
            np.arange(40, dtype=np.float64),
        )


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


def _face(channel, current, sem=1.0, run_id="01", port=11):
    """One face of a two-face pair, in the loader's return shape."""
    return {
        "x_cm": X_CM,
        "time_ms": np.zeros(1),
        "isat_a": np.full((1, 51, 1), float(current)),
        "sem_a": np.full((1, 51, 1), float(sem)),
        "port": np.array([port], dtype=np.int16),
        "run_id": np.asarray([run_id]),
        "source_channel": np.asarray([channel]),
    }


AREAS = {
    "01": {"ap_L_cm2": 2.0, "ap_R_cm2": 4.0},
    "31": {"ap_L_cm2": 2.0, "ap_R_cm2": 4.0},
}


def test_geomean_normalizes_each_face_by_its_own_channel_area():
    combined = _flow_symmetrized_profiles(
        _face("i_sweep", 6.0), _face("isat", 8.0), AREAS
    )
    # sqrt((6/2) * (8/4)) = sqrt(6)
    assert combined["profiles"][0, 0, 0] == pytest.approx(np.sqrt(6.0))
    assert combined["area_cm2"].tolist() == [[2.0, 4.0]]
    assert "i_sweep/ap_L_cm2" in str(combined["pairing"][0])
    assert "isat/ap_R_cm2" in str(combined["pairing"][0])


def test_geomean_gives_a_crossed_cable_run_the_electrode_each_channel_sat_on():
    """Run 31's cables are crossed, so its channel-to-area map is inverted.

    The two currents are unchanged; what moves is which calibrated area
    normalizes each of them, and therefore each single-face density and the
    pairing string.  The geometric mean divides by the same A_L * A_R either
    way, so its VALUE is invariant -- that invariance is asserted here so a
    later reader does not mistake an unmoved geomean for an unapplied swap.
    """
    combined = _flow_symmetrized_profiles(
        _face("isat", 6.0, run_id="31"), _face("i_sweep", 8.0, run_id="31"), AREAS
    )

    # isat sat on the LEFT electrode (2.0) and i_sweep on the RIGHT one (4.0).
    assert combined["area_cm2"].tolist() == [[2.0, 4.0]]
    assert "isat/ap_L_cm2=2.000000 cm2" in str(combined["pairing"][0])
    assert "i_sweep/ap_R_cm2=4.000000 cm2" in str(combined["pairing"][0])

    nominal = _flow_symmetrized_profiles(
        _face("isat", 6.0), _face("i_sweep", 8.0), AREAS
    )
    assert "isat/ap_R_cm2=4.000000 cm2" in str(nominal["pairing"][0])
    assert combined["profiles"][0, 0, 0] == pytest.approx(nominal["profiles"][0, 0, 0])


def test_geomean_is_symmetric_under_exchanging_which_product_holds_which_face():
    forward = _flow_symmetrized_profiles(
        _face("i_sweep", 6.0), _face("isat", 8.0), AREAS
    )
    swapped = _flow_symmetrized_profiles(
        _face("isat", 8.0), _face("i_sweep", 6.0), AREAS
    )
    assert forward["profiles"][0, 0, 0] == pytest.approx(swapped["profiles"][0, 0, 0])


def test_geomean_cancels_reciprocal_flow_factors():
    # Chung: the faces carry exp(+KM/2) and exp(-KM/2) about a common level.
    boost = np.exp(0.37)
    plain = _flow_symmetrized_profiles(
        _face("i_sweep", 2.0), _face("isat", 4.0), AREAS
    )
    flowing = _flow_symmetrized_profiles(
        _face("i_sweep", 2.0 * boost), _face("isat", 4.0 / boost), AREAS
    )
    assert flowing["profiles"][0, 0, 0] == pytest.approx(plain["profiles"][0, 0, 0])


def test_geomean_relative_error_is_half_the_quadrature_of_the_faces():
    combined = _flow_symmetrized_profiles(
        _face("i_sweep", 2.0, sem=0.2), _face("isat", 4.0, sem=0.8), AREAS
    )
    value = combined["profiles"][0, 0, 0]
    expected = value * 0.5 * np.hypot(0.2 / 2.0, 0.8 / 4.0)
    assert combined["sem"][0, 0, 0] == pytest.approx(expected)


def test_geomean_is_nan_where_a_face_is_not_positive():
    combined = _flow_symmetrized_profiles(
        _face("i_sweep", 2.0), _face("isat", -1.0), AREAS
    )
    assert not np.isfinite(combined["profiles"][0, 0, 0])


def test_geomean_refuses_two_readings_of_the_same_face():
    with pytest.raises(ValueError, match="no second face"):
        _flow_symmetrized_profiles(
            _face("i_sweep", 2.0), _face("i_sweep", 3.0), AREAS
        )


# ---------------------------------------------------------------------------
# The filled T_e product's trust and semi-quantitative records
# ---------------------------------------------------------------------------
def _write_te_records(path, *, omit=(), core_band=(X_MIN_CM, X_MAX_CM)):
    records = {
        "core_mean_te_monotonic_clamped": np.zeros((5, 3), dtype=bool),
        "core_mean_te_monotonic_scale": np.ones((5, 3)),
        "te_trust_radius_cm": np.array([18.415, 18.415, 18.415, 18.415, 10.0]),
        "te_trust_blend_cm": np.array([20.2, 20.2, 20.2, 20.2, 15.0]),
        "te_semi_quantitative_core_count": np.zeros((5, 3), dtype=np.int16),
        "te_semi_quantitative_band_count": np.zeros((5, 3), dtype=np.int16),
        "te_core_window_sem_ev": np.zeros((5, 3)),
        "te_window_dln_core_control": np.array([0.19, 0.24, 0.17, 2.146, 3.049]),
        "te_window_dln_core_control_source": np.array([1, 1, 1, 2, 2], dtype=np.int8),
        "te_row_measured_cells": np.array(
            [[7, 7, 7], [7, 7, 7], [7, 7, 7], [0, 0, 0], [0, 0, 0]], dtype=np.int16
        ),
        "te_row_measured_core_cells": np.array(
            [[5, 5, 5], [5, 5, 5], [5, 5, 5], [0, 0, 0], [0, 0, 0]], dtype=np.int16
        ),
    }
    with h5py.File(path, "w") as hdf:
        group = hdf.create_group("experiment_sets/1")
        for name, value in records.items():
            if name in omit:
                continue
            group.create_dataset(name, data=value)
        group.attrs["te_core_window_sem_x_min_cm"] = core_band[0]
        group.attrs["te_core_window_sem_x_max_cm"] = core_band[1]
        group.attrs["te_trust_model"] = "trust model"
        group.attrs["te_semi_quantitative_rule"] = "semi-quantitative rule"
        group.attrs["te_core_window_sem_definition"] = "window sem"
        group.attrs["te_window_core_control_source_codes"] = "0 none, 1 band, 2 original"
        group.attrs["te_row_provenance_definition"] = "row provenance"
        group.attrs["te_qc_floor_rule"] = "qc floor"
    return path


def test_te_records_are_read_off_the_filled_product(tmp_path):
    path = _write_te_records(tmp_path / "te_filled.hdf5")

    with h5py.File(path, "r") as hdf:
        records = _te_trust_records(hdf["experiment_sets/1"], path, 1)

    assert records["trust_radius_cm"].tolist() == [18.415] * 4 + [10.0]
    assert records["trust_blend_cm"].tolist() == [20.2] * 4 + [15.0]
    assert records["trust_model"] == "trust model"
    assert records["core_control_dln"].tolist() == [0.19, 0.24, 0.17, 2.146, 3.049]
    assert records["core_control_source"].tolist() == [1, 1, 1, 2, 2]
    assert records["row_measured_cells"][:, 0].tolist() == [7, 7, 7, 0, 0]
    assert records["row_measured_core_cells"][:, 0].tolist() == [5, 5, 5, 0, 0]


def test_a_row_with_no_measured_cell_is_flagged_prior_derived(tmp_path):
    """The seven blanked exception ports must not read as measured ports."""
    path = _write_te_records(tmp_path / "te_filled.hdf5")

    with h5py.File(path, "r") as hdf:
        records = _te_trust_records(hdf["experiment_sets/1"], path, 1)
    measured = records["row_measured_cells"] > 0

    assert measured[:, 0].tolist() == [True, True, True, False, False]
    assert not measured[3].any() and not measured[4].any()


@pytest.mark.parametrize("missing", REQUIRED_TE_RECORDS)
def test_the_export_refuses_a_product_missing_a_required_record(tmp_path, missing):
    path = _write_te_records(tmp_path / "te_filled.hdf5", omit=(missing,))

    with h5py.File(path, "r") as hdf:
        with pytest.raises(ValueError, match=f"carries no {missing} record"):
            _te_trust_records(hdf["experiment_sets/1"], path, 1)


def test_the_export_refuses_a_window_uncertainty_from_a_different_core_band(tmp_path):
    path = _write_te_records(tmp_path / "te_filled.hdf5", core_band=(-18.415, 18.415))

    with h5py.File(path, "r") as hdf:
        with pytest.raises(ValueError, match="mixes bands"):
            _te_trust_records(hdf["experiment_sets/1"], path, 1)


def test_the_exported_te_sem_adds_the_window_term_in_quadrature():
    radial = np.array([[0.30, 0.40]])
    window = np.array([[0.40, 0.00]])

    combined = np.hypot(radial, np.nan_to_num(window, nan=0.0))

    assert combined[0, 0] == pytest.approx(0.5)
    assert combined[0, 1] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Both probe faces through the afterglow, and their geometric mean
# ---------------------------------------------------------------------------
def _decay_face(channel, current, sem=1.0, run_id="01", port=11):
    """One face of the x=0 afterglow pair, in ``_isat_decay_stats``' return shape."""
    current = np.atleast_1d(np.asarray(current, dtype=np.float64))
    sem = np.broadcast_to(
        np.atleast_1d(np.asarray(sem, dtype=np.float64)), current.shape
    )
    return {
        "time_ms": np.arange(current.size, dtype=np.float64),
        "mean_a": current[None, :].copy(),
        "sem_a": np.array(sem, dtype=np.float64)[None, :].copy(),
        "port": np.array([port], dtype=np.int16),
        "run_id": np.asarray([run_id]),
        "source_channel": np.asarray([channel]),
    }


def test_decay_geomean_normalizes_each_face_by_its_own_channel_area():
    combined = _isat_decay_geomean(
        _decay_face("i_sweep", 6.0), _decay_face("isat", 8.0), AREAS
    )
    # sqrt((6/2) * (8/4)) = sqrt(6), in A cm^-2 and not in amperes.
    assert combined["geomean_a_per_cm2"][0, 0] == pytest.approx(np.sqrt(6.0))
    assert combined["area_cm2"].tolist() == [[2.0, 4.0]]
    assert "i_sweep/ap_L_cm2" in str(combined["pairing"][0])
    assert "isat/ap_R_cm2" in str(combined["pairing"][0])


def test_decay_geomean_gives_a_crossed_cable_run_the_inverted_channel_areas():
    """The afterglow pair takes the same per-run electrode map as the profiles."""
    combined = _isat_decay_geomean(
        _decay_face("isat", 6.0, run_id="31"),
        _decay_face("i_sweep", 8.0, run_id="31"),
        AREAS,
    )

    assert combined["area_cm2"].tolist() == [[2.0, 4.0]]
    assert "isat/ap_L_cm2=2.000000 cm2" in str(combined["pairing"][0])
    assert "i_sweep/ap_R_cm2=4.000000 cm2" in str(combined["pairing"][0])


def test_decay_geomean_is_symmetric_under_exchanging_the_two_faces():
    forward = _isat_decay_geomean(
        _decay_face("i_sweep", 6.0), _decay_face("isat", 8.0), AREAS
    )
    swapped = _isat_decay_geomean(
        _decay_face("isat", 8.0), _decay_face("i_sweep", 6.0), AREAS
    )
    assert forward["geomean_a_per_cm2"][0, 0] == pytest.approx(
        swapped["geomean_a_per_cm2"][0, 0]
    )


def test_decay_geomean_cancels_reciprocal_flow_factors():
    # Chung: the faces carry exp(+KM/2) and exp(-KM/2) about a common level.
    boost = np.exp(0.37)
    plain = _isat_decay_geomean(
        _decay_face("i_sweep", 2.0), _decay_face("isat", 4.0), AREAS
    )
    flowing = _isat_decay_geomean(
        _decay_face("i_sweep", 2.0 * boost), _decay_face("isat", 4.0 / boost), AREAS
    )
    assert flowing["geomean_a_per_cm2"][0, 0] == pytest.approx(
        plain["geomean_a_per_cm2"][0, 0]
    )


def test_decay_geomean_relative_error_is_half_the_quadrature_of_the_faces():
    combined = _isat_decay_geomean(
        _decay_face("i_sweep", 2.0, sem=0.2),
        _decay_face("isat", 4.0, sem=0.8),
        AREAS,
    )
    value = combined["geomean_a_per_cm2"][0, 0]
    expected = value * 0.5 * np.hypot(0.2 / 2.0, 0.8 / 4.0)
    assert combined["sem_a_per_cm2"][0, 0] == pytest.approx(expected)


def test_decay_geomean_is_nan_where_a_face_is_not_positive_and_clips_nothing():
    """The p50 upstream face decays through zero; that sample is NaN, not clipped."""
    combined = _isat_decay_geomean(
        _decay_face("i_sweep", [2.0, -1.0, 3.0, 0.0]),
        _decay_face("isat", [4.0, 4.0, 4.0, 4.0]),
        AREAS,
    )
    geomean = combined["geomean_a_per_cm2"][0]
    sem = combined["sem_a_per_cm2"][0]

    # J_up = [1, -0.5, 1.5, 0] and J_dn = 1 everywhere, so only 0 and 2 survive.
    assert np.isfinite(geomean).tolist() == [True, False, True, False]
    assert geomean[0] == pytest.approx(1.0)
    assert geomean[2] == pytest.approx(np.sqrt(1.5))
    assert np.isfinite(sem).tolist() == [True, False, True, False]


def test_decay_geomean_refuses_two_readings_of_the_same_face():
    with pytest.raises(ValueError, match="no second face"):
        _isat_decay_geomean(
            _decay_face("i_sweep", 2.0), _decay_face("i_sweep", 3.0), AREAS
        )


def test_decay_geomean_refuses_faces_that_disagree_on_ports_runs_or_time():
    with pytest.raises(ValueError, match="disagree on ports"):
        _isat_decay_geomean(
            _decay_face("i_sweep", 2.0),
            _decay_face("isat", 3.0, port=50),
            AREAS,
        )
    with pytest.raises(ValueError, match="disagree on runs"):
        _isat_decay_geomean(
            _decay_face("i_sweep", 2.0),
            _decay_face("isat", 3.0, run_id="08"),
            AREAS,
        )
    shifted = _decay_face("isat", 3.0)
    shifted["time_ms"] = shifted["time_ms"] + 1.0
    with pytest.raises(ValueError, match="disagree on the time grid"):
        _isat_decay_geomean(_decay_face("i_sweep", 2.0), shifted, AREAS)


# ---------------------------------------------------------------------------
# Interferometer chords
# ---------------------------------------------------------------------------
LECROY_MS = np.array([0.0, 1.0, 2.0])
RIGOL_MS = np.array([-1.0, 0.5, 2.5])


class _StubRun:
    def __init__(self, path):
        self.path = str(path)


class _StubDataset:
    """The two ``LapdDataset`` entry points the interferometer reader uses."""

    def __init__(self, paths_by_run_id):
        self._paths = dict(paths_by_run_id)

    def experiment_set_run_ids(self, experiment_set_id):
        return list(self._paths)

    def run(self, run_id):
        return _StubRun(self._paths[run_id])


def _write_interferometer(
    path,
    shots,
    *,
    calibration=1.0,
    missing=(),
    rigol_time_ms=RIGOL_MS,
):
    """Minimal stand-in for one run's ``diagnostics/interferometer`` group."""
    with h5py.File(path, "w") as hdf:
        group = hdf.create_group("diagnostics/interferometer")
        lecroy = group.create_group("time_array")
        rigol = group.create_group("time_array_p40")
        for port in (20, 29, 40):
            phase_group = group.create_group(f"phase_p{port}")
            phase_group.attrs["calibration factor (m^-3/rad)"] = calibration
            for key, phase in shots[port].items():
                dataset = phase_group.create_dataset(
                    str(key), data=np.asarray(phase, dtype=np.float64)
                )
                if (port, key) in missing:
                    dataset.attrs["rigol_missing"] = True
                grid = rigol if port == 40 else lecroy
                if str(key) not in grid:
                    grid.create_dataset(
                        str(key),
                        data=(rigol_time_ms if port == 40 else LECROY_MS),
                    )
    return path


def _flat_shots(values, n_samples=3):
    return {
        port: {
            index: np.full(n_samples, float(value))
            for index, value in enumerate(values)
        }
        for port in (20, 29, 40)
    }


def test_interferometer_pools_every_shot_of_every_run(tmp_path):
    first = _write_interferometer(tmp_path / "a.hdf5", _flat_shots([1.0, 3.0]))
    second = _write_interferometer(tmp_path / "b.hdf5", _flat_shots([5.0, 7.0]))
    dataset = _StubDataset({"01": first, "02": second})

    stats = _interferometer_decay_stats(dataset, 1)

    scale = 1.0e-6 * PLASMA_DIAMETER_CM
    assert stats["n_shots"].tolist() == [4, 4, 4]
    assert stats["line_density_cm2"][0, 0] == pytest.approx(4.0 * scale)
    # ddof=1 over [1, 3, 5, 7] is sqrt(20/3); the SEM divides by sqrt(4).
    assert stats["sem_cm2"][0, 0] == pytest.approx(
        np.sqrt(20.0 / 3.0) / 2.0 * scale
    )
    assert stats["port"].tolist() == [20, 29, 40]
    assert stats["z_cm"].tolist() == pytest.approx([757.6, 1045.15, 1396.6])
    assert stats["run_id"].shape == (3, 2)
    assert stats["plasma_diameter_cm"] == PLASMA_DIAMETER_CM


def test_interferometer_applies_each_chord_calibration_factor(tmp_path):
    path = _write_interferometer(
        tmp_path / "a.hdf5", _flat_shots([2.0]), calibration=3.0
    )
    dataset = _StubDataset({"01": path})

    stats = _interferometer_decay_stats(dataset, 1)

    assert stats["line_density_cm2"][1, 0] == pytest.approx(
        2.0 * 3.0 * 1.0e-6 * PLASMA_DIAMETER_CM
    )


def test_interferometer_skips_rigol_missing_shots(tmp_path):
    path = _write_interferometer(
        tmp_path / "a.hdf5",
        _flat_shots([1.0, 99.0]),
        missing=((40, 1),),
    )
    dataset = _StubDataset({"01": path})

    stats = _interferometer_decay_stats(dataset, 1)

    assert stats["n_shots"].tolist() == [2, 2, 1]
    assert stats["line_density_cm2"][2, 0] == pytest.approx(
        1.0e-6 * PLASMA_DIAMETER_CM
    )


def test_interferometer_puts_the_rigol_chord_on_the_shared_lecroy_grid(tmp_path):
    ramp = {
        port: {0: (RIGOL_MS if port == 40 else LECROY_MS) * 2.0 + 1.0}
        for port in (20, 29, 40)
    }
    path = _write_interferometer(tmp_path / "a.hdf5", ramp)
    dataset = _StubDataset({"01": path})

    stats = _interferometer_decay_stats(dataset, 1)

    assert np.array_equal(stats["time_ms"], LECROY_MS)
    assert stats["line_density_cm2"].shape == (3, LECROY_MS.size)
    assert stats["sem_cm2"].shape == (3, LECROY_MS.size)
    # The ramp is linear in t on both grids, so the interpolation is exact.
    expected = (LECROY_MS * 2.0 + 1.0) * 1.0e-6 * PLASMA_DIAMETER_CM
    assert stats["line_density_cm2"][2] == pytest.approx(expected)
    assert stats["line_density_cm2"][0] == pytest.approx(expected)


def test_interferometer_refuses_a_chord_that_stops_short_of_the_shared_grid(tmp_path):
    path = _write_interferometer(
        tmp_path / "a.hdf5",
        _flat_shots([1.0]),
        rigol_time_ms=np.array([0.5, 1.0, 1.5]),
    )
    dataset = _StubDataset({"01": path})

    with pytest.raises(ValueError, match="does not cover the shared grid"):
        _interferometer_decay_stats(dataset, 1)


def test_export_refuses_a_non_default_port_ladder_and_names_the_rebuild(tmp_path):
    missing = tmp_path / "not-read.hdf5"

    with pytest.raises(ValueError, match="refusing to export on port_map") as excinfo:
        export_overlay(
            missing,
            missing,
            missing,
            missing,
            missing,
            tmp_path / "out.npz",
            port_map="cad",
        )

    # The refusal has to name the prerequisite, not just say no: the probe-port
    # z grid is copied from products built on the other ladder, so adopting one
    # is a rebuild of those inputs.  It also has to fire before anything is
    # read, which is why none of the paths above exists and none is created.
    message = str(excinfo.value)
    assert "rebuild" in message
    assert "probe-port z grid is copied" in message
    assert not (tmp_path / "out.npz").exists()


# ---------------------------------------------------------------------------
# The exported product, when a regenerated one is on disk
# ---------------------------------------------------------------------------
OVERLAY_NPZ = Path("processed/es1_sim1d_overlay.npz")
NEW_ISAT_KEYS = (
    "isat_decay_dn_mean_a",
    "isat_decay_dn_sem_a",
    "isat_decay_dn_n_shots_used",
    "isat_decay_dn_n_shots_rejected",
    "isat_decay_geomean_a_per_cm2",
    "isat_decay_geomean_sem_a_per_cm2",
    "isat_decay_geomean_area_cm2",
    "isat_decay_face_convention",
)
NEW_INTERF_KEYS = (
    "interf_decay_time_ms",
    "interf_decay_line_density_cm2",
    "interf_decay_sem_cm2",
    "interf_decay_port",
    "interf_decay_z_cm",
    "interf_decay_n_shots",
    "interf_decay_run_ids",
    "interf_decay_convention",
    "interf_decay_chord_caveat",
    "interf_decay_clock_offset",
)


def _overlay_or_skip():
    """The on-disk overlay, skipped past unless it is a both-face vintage.

    A shared checkout can carry an overlay exported before this change; that is
    a stale artifact, not a failure, so the vintage is decided by the product's
    own keys rather than by the file merely existing.
    """
    if not OVERLAY_NPZ.exists():
        pytest.skip("no ES1 overlay on disk")
    overlay = np.load(OVERLAY_NPZ, allow_pickle=True)
    if "isat_decay_geomean_a_per_cm2" not in overlay.files:
        pytest.skip("the ES1 overlay on disk predates the both-face export")
    return overlay


def test_the_exported_overlay_carries_both_faces_and_the_chords():
    overlay = _overlay_or_skip()
    for key in NEW_ISAT_KEYS + NEW_INTERF_KEYS:
        assert key in overlay.files, key

    n_ports = overlay["isat_decay_port"].size
    n_time = overlay["isat_decay_time_ms"].size
    for key in ("isat_decay_dn_mean_a", "isat_decay_dn_sem_a",
                "isat_decay_geomean_a_per_cm2", "isat_decay_geomean_sem_a_per_cm2"):
        assert overlay[key].shape == (n_ports, n_time), key
    assert overlay["isat_decay_dn_n_shots_used"].shape == (n_ports,)
    assert overlay["isat_decay_dn_n_shots_rejected"].shape == (n_ports,)
    assert overlay["isat_decay_geomean_area_cm2"].shape == (n_ports, 2)

    n_chords = overlay["interf_decay_port"].size
    n_interf = overlay["interf_decay_time_ms"].size
    assert overlay["interf_decay_port"].tolist() == [20, 29, 40]
    assert overlay["interf_decay_line_density_cm2"].shape == (n_chords, n_interf)
    assert overlay["interf_decay_sem_cm2"].shape == (n_chords, n_interf)
    assert overlay["interf_decay_z_cm"].shape == (n_chords,)
    assert overlay["interf_decay_n_shots"].shape == (n_chords,)
    assert overlay["interf_decay_run_ids"].shape[0] == n_chords


def test_the_exported_geomean_is_the_two_faces_area_normalized_geometric_mean():
    overlay = _overlay_or_skip()
    areas = overlay["isat_decay_geomean_area_cm2"]
    upstream = overlay["isat_decay_mean_a"] / areas[:, 0][:, None]
    downstream = overlay["isat_decay_dn_mean_a"] / areas[:, 1][:, None]
    product = upstream * downstream
    expected = np.where(
        np.isfinite(product) & (product > 0.0), np.sqrt(np.abs(product)), np.nan
    )

    assert np.array_equal(
        overlay["isat_decay_geomean_a_per_cm2"], expected, equal_nan=True
    )
    # The NaN policy is load-bearing: at least one port decays through zero.
    assert not np.isfinite(overlay["isat_decay_geomean_a_per_cm2"]).all()


def test_the_two_faces_reject_different_shots():
    overlay = _overlay_or_skip()
    upstream = overlay["isat_decay_n_shots_rejected"]
    downstream = overlay["isat_decay_dn_n_shots_rejected"]

    assert upstream.shape == downstream.shape
    assert not np.array_equal(upstream, downstream)
    assert "REJECT" in str(overlay["isat_decay_face_convention"])


# ---------------------------------------------------------------------------
# The raw discharge ensemble (the --raw-discharge-ensemble opt-in)
# ---------------------------------------------------------------------------
RAW_DT_MS = 0.05
RAW_TIME_MS = np.arange(0.0, 25.0, RAW_DT_MS)
RAW_RISE_START_MS = 2.0
RAW_RISE_STOP_MS = 4.0
RAW_FALL_MS = 20.5


def _ramp_shot(plateau_a, time_ms=RAW_TIME_MS):
    """A synthetic discharge shot: zero, a linear rise, a flat plateau, a fall.

    The rise is linear between ``RAW_RISE_START_MS`` and ``RAW_RISE_STOP_MS``
    so a level crossing inside it has a closed-form time, and the plateau is
    flat over the whole scoring window so its mean is exactly ``plateau_a``.
    """
    frac = np.clip(
        (time_ms - RAW_RISE_START_MS) / (RAW_RISE_STOP_MS - RAW_RISE_START_MS),
        0.0,
        1.0,
    )
    return np.where(time_ms >= RAW_FALL_MS, 0.0, plateau_a * frac)


class _DischargeStubRun:
    """The two attributes ``_discharge_stats`` reads off a run."""

    def __init__(self, current, time_ms):
        self._current = np.asarray(current, dtype=np.float64)
        self._time_s = np.asarray(time_ms, dtype=np.float64) / 1000.0

    def discharge_traces(self):
        # The voltage is smoothed and averaged but plays no part in the raw
        # current ensemble under test.
        return self._current, -np.full_like(self._current, 100.0), self._time_s

    def discharge_zero_offset_stats(self):
        return type("_Offsets", (), {"offset_a": 0.0})()


class _DischargeStubDataset:
    """The two lookups ``_discharge_stats`` makes on a dataset."""

    def __init__(self, runs):
        self._runs = runs

    def experiment_set_run_ids(self, experiment_set_id):
        return list(self._runs)

    def run(self, run_id):
        return self._runs[run_id]


def test_the_plateau_helper_is_the_per_shot_mean_over_the_scoring_window():
    window = (
        (RAW_TIME_MS >= RAW_PLATEAU_WINDOW_MS[0])
        & (RAW_TIME_MS <= RAW_PLATEAU_WINDOW_MS[1])
    )
    # In-window samples that average to a known plateau without being
    # constant, so a helper that read one sample instead of the mean fails;
    # out-of-window samples far away, so a helper that ignored the window
    # fails too.
    current = np.full((2, RAW_TIME_MS.size), -9000.0)
    n_in = int(window.sum())
    swing = (np.arange(n_in, dtype=np.float64) - (n_in - 1) / 2.0) * 10.0
    assert np.sum(swing) == 0.0
    current[0, window] = 3000.0 + swing
    current[1, window] = 2500.0 + 2.0 * swing

    plateau = _plateau_current_a(current, RAW_TIME_MS)

    assert plateau.shape == (2,)
    assert plateau[0] == pytest.approx(3000.0)
    assert plateau[1] == pytest.approx(2500.0)
    assert plateau[0] == pytest.approx(np.mean(current[0, window]))


def test_the_scoring_plateau_window_is_the_inherited_beta_plateau():
    # RAW_PLATEAU_WINDOW_MS is not chosen in this repo: it is the drive-plateau
    # window the transport comparison scores over, whose own constant is named
    # BETA_PLATEAU_MS and reads (15.0, 19.5) ms.  Asserted as a literal rather
    # than imported, because the two repositories are not installed together
    # and a silent drift on either side has to show up here as a failure.
    assert RAW_PLATEAU_WINDOW_MS == (15.0, 19.5)


def test_the_common_crossing_level_is_half_the_ensemble_median_plateau():
    runs = {
        "01": np.array([_ramp_shot(3000.0), _ramp_shot(3400.0)]),
        "02": np.array([_ramp_shot(2600.0)]),
    }
    dataset = _DischargeStubDataset({k: _DischargeStubRun(v, RAW_TIME_MS) for k, v in runs.items()})

    stats = _discharge_stats(dataset, 1, raw_ensemble=True)

    plateaus = np.array([3000.0, 3400.0, 2600.0])
    assert stats["raw"]["t_half_level_a"] == pytest.approx(
        0.5 * float(np.median(plateaus))
    )
    # Not half of any single shot's own plateau -- one level for the ensemble.
    assert stats["raw"]["t_half_level_a"] == pytest.approx(1500.0)
    assert stats["raw"]["t_half_level_run_id"].tolist() == ["01", "01", "02"]


def test_a_shot_that_never_reaches_the_level_raises_and_names_that_shot():
    current = np.array([_ramp_shot(3000.0), _ramp_shot(3000.0), _ramp_shot(3000.0)])
    current[1] *= 0.1  # this shot tops out at 300 A, far below the level

    with pytest.raises(ValueError) as excinfo:
        _t_half_level_ms(current, RAW_TIME_MS, 1500.0)

    message = str(excinfo.value)
    assert "[1]" in message
    assert "never reach" in message
    assert "1500" in message


def test_a_spike_after_the_window_does_not_move_the_level_or_the_crossings():
    clean = {
        "01": np.array([_ramp_shot(3000.0), _ramp_shot(2900.0)]),
        "02": np.array([_ramp_shot(3100.0)]),
    }
    spiked = {k: v.copy() for k, v in clean.items()}
    spike_sample = int(np.argmin(np.abs(RAW_TIME_MS - 20.0)))
    assert RAW_PLATEAU_WINDOW_MS[1] < RAW_TIME_MS[spike_sample] < RAW_FALL_MS
    # Run 05's shape: one sample at ~5 kA against a normal 3 kA plateau.
    spiked["01"][0, spike_sample] = 4950.0
    spiked["02"][0, spike_sample] = 4070.0

    clean_stats = _discharge_stats(
        _DischargeStubDataset({k: _DischargeStubRun(v, RAW_TIME_MS) for k, v in clean.items()}),
        1,
        raw_ensemble=True,
    )
    spiked_stats = _discharge_stats(
        _DischargeStubDataset({k: _DischargeStubRun(v, RAW_TIME_MS) for k, v in spiked.items()}),
        1,
        raw_ensemble=True,
    )

    assert spiked_stats["raw"]["t_half_level_a"] == clean_stats["raw"]["t_half_level_a"]
    assert np.array_equal(
        spiked_stats["raw"]["t_half_level_ms"], clean_stats["raw"]["t_half_level_ms"]
    )
    assert spiked_stats["raw"]["t_half_level_sd_ms"] == clean_stats["raw"][
        "t_half_level_sd_ms"
    ]
    # The spike is a real difference between the two ensembles; it just does
    # not reach the amplitude the timing statistic is referred to.
    assert not np.array_equal(
        spiked_stats["raw"]["current_mean_a"], clean_stats["raw"]["current_mean_a"]
    )
