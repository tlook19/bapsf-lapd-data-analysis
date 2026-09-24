"""Chord-implied T_e arithmetic on a synthetic plasma with a closed-form answer.

A flat-top density over the 50 cm scan at a constant filled ``T_e`` makes every
integral elementary: the probe line integral is ``n0 * 0.50 m``, the
density-weighted mean temperature is the constant itself, and the chord is
supplied.  Choosing the chord to be exactly the probe integral makes ``r = 1``,
and choosing it to be ``1 / 0.8`` times the probe integral makes ``r = 0.8``,
so the implied temperature must come back unchanged in the first case and at
``0.64`` times the used temperature in the second.
"""

import numpy as np
import pytest

from crosscheck_chord_implied_te import (
    build_parser,
    collect_rows,
    density_weighted_te_ev,
    edge_over_axis,
    implied_te_ev,
    line_integral_cm2,
    te_pull,
    te_sigma_sys_ev,
    window_rows,
    write_csv,
)


h5py = pytest.importorskip("h5py")

X_CM = np.linspace(-25.0, 25.0, 51)
X_M = X_CM * 1e-2
SCAN_WIDTH_M = 0.50

DENSITY_M3 = 1.0e18
TE_EV = 5.0
SAMPLE_TIMES_MS = np.array([15.5, 16.5, 17.5, 18.5])
TE_CYCLE_TIMES_MS = np.array([14.0, 16.0, 18.0, 20.0])

#: The p29 chord and the p29 probe are the coincident pair, so the synthetic
#: product needs only that one chord.
CHORD_PORT = 29
PROBE_PORT = 29
Z_CM = 1045.15


def _flat_probe_line_integral_cm2() -> float:
    return DENSITY_M3 * SCAN_WIDTH_M / 1e4


def _write_products(tmp_path, ratio: float):
    """Write density, filled-T_e and chord products whose r is ``ratio``."""
    density_path = tmp_path / "density_profiles_isweep.hdf5"
    te_path = tmp_path / "te_filled.hdf5"
    interf_path = tmp_path / "interferometer_experiment_set_stats.npz"

    n_samples = SAMPLE_TIMES_MS.size
    n_cycles = TE_CYCLE_TIMES_MS.size
    with h5py.File(density_path, "w") as hdf:
        hdf.create_dataset("x_cm", data=X_CM)
        group = hdf.create_group("experiment_sets/1")
        group.create_dataset("z_cm", data=np.array([Z_CM]))
        group.create_dataset("inter_sweep_time_s", data=SAMPLE_TIMES_MS * 1e-3)
        group.create_dataset(
            "n_e_m3", data=np.full((1, X_CM.size, n_samples), DENSITY_M3)
        )

    with h5py.File(te_path, "w") as hdf:
        group = hdf.create_group("experiment_sets/1")
        group.create_dataset("x_cm", data=X_CM)
        group.create_dataset("z_cm", data=np.array([Z_CM]))
        group.create_dataset("port", data=np.array([PROBE_PORT], dtype=np.int16))
        group.create_dataset("cycle_time_ms", data=TE_CYCLE_TIMES_MS)
        group.create_dataset(
            "te_filled", data=np.full((1, X_CM.size, n_cycles), TE_EV)
        )
        group.create_dataset(
            "te_semi_quantitative",
            data=np.zeros((1, X_CM.size, n_cycles), dtype=bool),
        )

    chord_cm2 = _flat_probe_line_integral_cm2() / ratio
    np.savez(
        interf_path,
        **{
            f"set1_p{CHORD_PORT}_time_ms": np.array([0.0, 100.0]),
            f"set1_p{CHORD_PORT}_line_integrated_mean_cm2": np.array(
                [chord_cm2, chord_cm2]
            ),
        },
    )
    return density_path, te_path, interf_path


def _window_row(tmp_path, ratio: float) -> dict:
    density_path, te_path, interf_path = _write_products(tmp_path, ratio)
    rows = collect_rows(density_path, te_path, interf_path, (15.0, 19.5))
    aggregates = window_rows(rows)
    assert len(aggregates) == 1
    # One window row plus one row per in-window dead-time sample.
    assert len(rows) == 1 + SAMPLE_TIMES_MS.size
    return aggregates[0]


def test_unit_ratio_returns_the_used_temperature():
    assert implied_te_ev(TE_EV, 1.0) == pytest.approx(TE_EV, rel=1e-12)


def test_ratio_of_four_fifths_scales_the_temperature_by_the_square():
    assert implied_te_ev(TE_EV, 0.8) == pytest.approx(0.64 * TE_EV, rel=1e-12)


def test_a_non_positive_ratio_implies_no_temperature():
    assert np.isnan(implied_te_ev(TE_EV, -0.5))
    assert np.isnan(implied_te_ev(TE_EV, float("nan")))


def test_sigma_and_pull_are_the_declared_band():
    sigma = te_sigma_sys_ev(TE_EV)
    assert sigma == pytest.approx(0.25 * TE_EV + 0.20, rel=1e-12)
    assert te_pull(TE_EV, 0.64 * TE_EV, sigma) == pytest.approx(
        (TE_EV - 0.64 * TE_EV) / sigma, rel=1e-12
    )


def test_flat_profile_line_integral_is_the_closed_form():
    profile = np.full_like(X_M, DENSITY_M3)
    assert line_integral_cm2(profile, X_M) == pytest.approx(
        _flat_probe_line_integral_cm2(), rel=1e-12
    )


def test_density_weighted_mean_of_a_constant_is_that_constant():
    numerator, denominator = density_weighted_te_ev(
        np.full_like(X_M, DENSITY_M3), np.full_like(X_M, TE_EV), X_M
    )
    assert numerator / denominator == pytest.approx(TE_EV, rel=1e-12)


def test_edge_over_axis_reports_both_ends():
    profile = np.full_like(X_M, DENSITY_M3)
    profile[0] = 0.25 * DENSITY_M3
    profile[-1] = 0.50 * DENSITY_M3
    left, right = edge_over_axis(profile, X_CM)
    assert left == pytest.approx(0.25, rel=1e-12)
    assert right == pytest.approx(0.50, rel=1e-12)


def test_synthetic_matched_chord_implies_the_used_temperature(tmp_path):
    row = _window_row(tmp_path, 1.0)
    assert row["ratio"] == pytest.approx(1.0, rel=1e-12)
    assert row["te_used_ev"] == pytest.approx(TE_EV, rel=1e-12)
    assert row["te_implied_ev"] == pytest.approx(TE_EV, rel=1e-12)
    assert row["pull"] == pytest.approx(0.0, abs=1e-12)
    assert row["L_probe_cm2"] == pytest.approx(
        _flat_probe_line_integral_cm2(), rel=1e-12
    )


def test_synthetic_low_ratio_implies_sixty_four_percent_of_it(tmp_path):
    row = _window_row(tmp_path, 0.8)
    assert row["ratio"] == pytest.approx(0.8, rel=1e-12)
    assert row["te_used_ev"] == pytest.approx(TE_EV, rel=1e-12)
    assert row["te_implied_ev"] == pytest.approx(0.64 * TE_EV, rel=1e-12)
    assert row["pull"] == pytest.approx(
        (TE_EV - 0.64 * TE_EV) / te_sigma_sys_ev(TE_EV), rel=1e-12
    )


def test_csv_carries_every_row(tmp_path):
    density_path, te_path, interf_path = _write_products(tmp_path, 0.8)
    rows = collect_rows(density_path, te_path, interf_path, (15.0, 19.5))
    path = write_csv(rows, tmp_path / "chord_implied_te.csv")
    lines = path.read_text().strip().splitlines()
    assert lines[0].startswith("es,chord_port,probe_port")
    assert len(lines) == 1 + len(rows)


def test_parser_defaults_to_the_plateau_window():
    args = build_parser().parse_args([])
    assert tuple(args.window_ms) == (15.0, 19.5)
