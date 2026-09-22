from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.export_es1_sim1d_overlay import (
    DESPIKE_MIN_PEAK_FRACTION,
    FLUX_TUBE_RADIUS_CM,
    ISAT_DECAY_FIT_WINDOW_MS,
    ISAT_DECAY_MATRIX_CONVENTIONS,
    ISAT_DECAY_MATRIX_FACES,
    PLASMA_DIAMETER_CM,
    PORTS,
    RAW_PLATEAU_WINDOW_MS,
    X_MAX_CM,
    X_MIN_CM,
    _check_density_convention_pair,
    _column_edge_cm,
    _decay_efold_ms,
    _decay_noise_floor_a,
    _despike_profile,
    _discharge_stats,
    _flux_tube_profile_stats,
    _flux_tube_series,
    _flux_tube_te_series,
    _flux_tube_te_stats,
    _flow_symmetrized_profiles,
    _flux_tube_weights,
    _interp_onto_time_grid,
    _interferometer_decay_stats,
    _isat_decay_geomean,
    _isat_decay_matrix,
    _measured_coverage_cm,
    _weighted_mean_and_sem,
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


# ---------------------------------------------------------------------------
# The flux-tube T_e comparand: density-weighted, plain, and its coverage
# ---------------------------------------------------------------------------
#: A density profile that is FLAT across the whole integration disc and tapers
#: outside it, so the scalar background is well defined and the despike gate
#: sees no step.  Same shape as the flat-column density test above.
_TAPER = {21: 2.0, 22: 1.0, 23: 0.5, 24: 0.25, 25: 0.0}


def _flat_density(level=3.0):
    return np.array(
        [_TAPER.get(int(abs(position)), level) for position in X_CM],
        dtype=np.float64,
    )


def _peaked(width_cm, pedestal=0.0, amplitude=1.0):
    """A smooth Gaussian the despike gate leaves alone."""
    return pedestal + amplitude * np.exp(-((X_CM / width_cm) ** 2))


def _hollow_density(ring_cm=12.0, width_cm=6.0):
    """A HOLLOW column: an annular density peak with a depressed axis.

    Still falls away at the scan edge, so the ledger's scalar background is
    small and the profile survives the clip; what makes it hollow is that it
    is lowest exactly where a peaked T_e is hottest.
    """
    return np.exp(-(((np.abs(X_CM) - ring_cm) / width_cm) ** 2))


def _core_band_mean(profile):
    band = (X_CM >= X_MIN_CM) & (X_CM <= X_MAX_CM)
    return float(np.mean(np.asarray(profile)[band]))


def test_flat_te_on_flat_density_gives_the_core_value_under_both_weightings():
    """(i) Nothing to weight and nothing to average: all three agree."""
    te = np.full(X_CM.size, 4.25)
    stats = _flux_tube_te_stats(te, _flat_density(), X_CM)

    assert stats["n_despiked"] == 0
    assert stats["ftavg"] == pytest.approx(4.25, rel=1e-12)
    assert stats["plain"] == pytest.approx(4.25, rel=1e-12)
    assert stats["plain"] == pytest.approx(_core_band_mean(te), rel=1e-12)
    # A flat profile has no radial scatter, so the scatter SEM is zero.
    assert stats["ftavg_sem"] == pytest.approx(0.0, abs=1e-12)


def test_a_peaked_te_on_a_flat_density_weights_to_its_own_plain_area_mean():
    """(ii) A flat weight is no weight; both fall below the core-band mean."""
    te = _peaked(8.0, pedestal=1.0, amplitude=4.0)
    stats = _flux_tube_te_stats(te, _flat_density(), X_CM)

    assert stats["n_despiked"] == 0
    assert stats["ftavg"] == pytest.approx(stats["plain"], rel=1e-12)
    # The area average reaches radii the core-band line cut never sees, and
    # T_e is lower out there.
    assert stats["plain"] < _core_band_mean(te)


def test_a_peaked_density_pulls_the_te_average_above_the_plain_one():
    """(iii) The weight is largest where T_e is hottest."""
    te = _peaked(8.0, pedestal=1.0, amplitude=4.0)
    density = _peaked(9.0, pedestal=0.0, amplitude=1.0)
    stats = _flux_tube_te_stats(te, density, X_CM)

    assert stats["ftavg"] > stats["plain"]
    assert stats["plain"] < _core_band_mean(te)


def _quadrature_nodes(density_profile, te_profile, x_cm):
    """The quadrature nodes, weights and values, from the single-source parts.

    Assembles what the average is taken over -- despiked and
    background-subtracted density, despiked T_e, the centroid fold and
    ``_flux_tube_weights`` -- WITHOUT taking the average, so a test can form
    the two quadrature sums itself and check the reduction against them.
    Returns ``None`` where the reduction is not defined.
    """
    density, _ = _despike_profile(density_profile)
    density = _subtract_background(density)
    te, _ = _despike_profile(te_profile)
    finite = np.isfinite(density) & np.isfinite(te)
    if np.count_nonzero(finite) < 5:
        return None
    values = np.clip(density[finite], 0.0, None)
    total = float(np.sum(values))
    if total <= 0.0:
        return None
    centroid = float(np.sum(values * x_cm[finite]) / total)
    try:
        weights = _flux_tube_weights(
            np.abs(x_cm[finite] - centroid), FLUX_TUBE_RADIUS_CM
        )
    except ValueError:
        return None
    return weights, values, te[finite]


def _assert_is_the_two_quadrature_sums(stats, nodes):
    """``ftavg`` is sum w n T / sum w n and ``plain`` is sum w T / sum w.

    Where no weight-carrying node holds any density -- ``sum w n == 0``, which
    the placed products do reach at a noise-floor row -- the weighted average
    is not defined and must be NaN while the plain one stays finite.
    """
    weights, density, te = nodes
    assert stats["plain"] == pytest.approx(
        float(weights @ te) / float(weights.sum()), rel=1e-12
    )
    weighted = weights * density
    if float(weighted.sum()) <= 0.0:
        assert np.isnan(stats["ftavg"])
        return
    assert stats["ftavg"] == pytest.approx(
        float(weighted @ te) / float(weighted.sum()), rel=1e-12
    )


def _assert_the_gap_is_the_weighted_covariance(stats, nodes):
    """``ftavg - plain == cov_w(n, T_e) / <n>_w``, the sign-carrying identity."""
    weights, density, te = nodes
    normalized = weights / weights.sum()
    mean_n = float(normalized @ density)
    if mean_n <= 0.0:
        assert np.isnan(stats["ftavg"])
        return
    covariance = float(normalized @ (density * te)) - mean_n * float(normalized @ te)
    assert stats["ftavg"] - stats["plain"] == pytest.approx(
        covariance / mean_n, rel=1e-9, abs=1e-12
    )


def test_the_weighted_row_is_the_density_weighted_quadrature_on_the_same_nodes():
    """The property the two rows actually have: one quadrature, two weightings."""
    te = _peaked(8.0, pedestal=1.0, amplitude=4.0)
    density = _peaked(9.0, pedestal=0.0, amplitude=1.0)

    stats = _flux_tube_te_stats(te, density, X_CM)
    nodes = _quadrature_nodes(density, te, X_CM)

    _assert_is_the_two_quadrature_sums(stats, nodes)
    _assert_the_gap_is_the_weighted_covariance(stats, nodes)


def test_the_two_rows_differ_by_the_weighted_covariance_of_density_and_te():
    """Which row is larger is the SIGN of that covariance, and nothing else."""
    te = _peaked(8.0, pedestal=1.0, amplitude=4.0)
    for density in (
        _peaked(9.0, amplitude=1.0),   # peaked with T_e: positive covariance
        _flat_density(1.0),            # flat across the disc: zero
        _hollow_density(),             # ring: negative
    ):
        stats = _flux_tube_te_stats(te, density, X_CM)
        _assert_the_gap_is_the_weighted_covariance(
            stats, _quadrature_nodes(density, te, X_CM)
        )


def test_a_hollow_density_puts_the_weighted_average_below_the_plain_one():
    """The ordering is NOT universal: anti-correlated n and T_e invert it.

    This is why no per-sample ``ftavg >= plain`` assertion may be made against
    a real product.  A hollow density weights the cold outer radii hardest.
    """
    te = _peaked(8.0, pedestal=1.0, amplitude=4.0)

    stats = _flux_tube_te_stats(te, _hollow_density(), X_CM)

    assert stats["ftavg"] < stats["plain"]


def test_the_te_weight_denominator_is_the_exported_density_flux_tube_average():
    """The two rows must be one quantity: same nodes, same weights."""
    te = _peaked(8.0, pedestal=1.0, amplitude=4.0)
    density = _peaked(9.0, pedestal=0.0, amplitude=1.0)

    stats = _flux_tube_te_stats(te, density, X_CM)
    density_only = _flux_tube_profile_stats(density, X_CM)

    assert stats["weight_density"] == pytest.approx(
        density_only["ftavg"], rel=1e-12
    )
    assert stats["centroid"] == pytest.approx(density_only["centroid"], rel=1e-12)


def test_the_weighted_scatter_sem_reduces_to_the_core_band_convention():
    """Equal weights must give back std(ddof=1)/sqrt(N), the core-band form."""
    values = np.array([1.0, 2.0, 4.0, 8.0, 9.0])
    mean, sem = _weighted_mean_and_sem(values, np.full(values.size, 0.37))

    assert mean == pytest.approx(values.mean())
    assert sem == pytest.approx(values.std(ddof=1) / np.sqrt(values.size))


def test_a_negative_weight_is_carried_and_the_mean_may_leave_its_own_range():
    """The signed density weight is not a convex combination, and is not clipped.

    ``ftavg`` is the ratio of two quadrature sums, ``sum w n T / sum w n``.
    On the comparand chain ``n`` is the measured density WITH ITS SIGN, so a
    cell on noise about zero enters with a negative weight and the ratio stops
    being an average between its own nodes.  It is reported as computed: a
    clip back into ``[min, max]`` is the retired sign test under another name.
    """
    values = np.array([1.0, 2.0, 3.0])
    weights = np.array([-0.2, 0.1, 0.2])

    mean, _ = _weighted_mean_and_sem(values, weights)

    assert mean == pytest.approx(
        float(weights @ values) / float(weights.sum()), rel=1e-12
    )
    assert mean > values.max()  # outside the range of its own nodes


def test_a_non_positive_column_reports_an_axial_area_mean_and_no_temperature():
    """The ES4 p50 state: a whole column whose intensity sums non-positive.

    Such a column has NO CENTROID -- an intensity-weighted mean position is
    undefined for a non-positive intensity -- but it is still a MEASUREMENT of
    a port with no plasma left in it, and NaN would say otherwise.  The row is
    reported on the one fold centre that needs no intensity, the geometric
    axis x = 0, and carries the SIGNED AREA MEAN in the row's own units.  The
    T_e rows, ratios over that same non-positive weight, stay NaN, and the
    weight-density field carries the denominator that refused.
    """
    # Noise about zero with a net negative sum, and no step the despike gate
    # would repair: a slow cosine ripple sitting below zero.
    column = -1.0 + 0.5 * np.cos(X_CM * (np.pi / 12.5))
    assert np.sum(column) < 0.0  # the premise, measured not assumed

    density = _flux_tube_profile_stats(column, X_CM, subtract_background=False)

    axial_weights = _flux_tube_weights(np.abs(X_CM), FLUX_TUBE_RADIUS_CM)
    expected = float(axial_weights @ column) / float(axial_weights.sum())
    assert density["ftavg"] == pytest.approx(expected, rel=1e-12)
    assert density["ftavg"] < 0.0
    # An AREA MEAN, in the row's own units -- it sits between the extremes of
    # the profile it averages, which a sum over 51 cells could not.
    assert column.min() <= density["ftavg"] <= column.max()
    assert abs(density["ftavg"]) < abs(float(np.sum(column)))
    # No centroid exists; the extent the weights were formed over is reported.
    assert np.isnan(density["centroid"])
    assert density["edge"] == pytest.approx(FLUX_TUBE_RADIUS_CM)
    assert np.isnan(density["ftavg_sem"])
    assert np.isnan(density["scatter_sem"])

    # The whole-column convention folds about the same axis, out to the scan
    # limit, and does NOT take the tube-area renormalization there -- which is
    # what makes pi R^2 times the row a consistent inventory.
    column_stats = _flux_tube_profile_stats(
        column,
        X_CM,
        radius_cm=None,
        normalize_radius_cm=FLUX_TUBE_RADIUS_CM,
        subtract_background=False,
    )
    scan_limit = float(np.max(np.abs(X_CM)))
    scan_weights = _flux_tube_weights(np.abs(X_CM), scan_limit)
    assert column_stats["edge"] == pytest.approx(scan_limit)
    assert column_stats["ftavg"] == pytest.approx(
        float(scan_weights @ column) / float(scan_weights.sum()), rel=1e-12
    )

    te = _flux_tube_te_stats(
        _peaked(8.0, pedestal=1.0, amplitude=4.0),
        column,
        X_CM,
        subtract_background=False,
    )

    assert np.isnan(te["ftavg"])
    assert np.isnan(te["plain"])
    assert np.isnan(te["ftavg_sem"])
    assert np.isnan(te["plain_sem"])
    # The denominator is exported beside the refusal, and is the same number
    # the density row carries.
    assert te["weight_density"] == pytest.approx(density["ftavg"], rel=1e-12)


def test_a_column_with_plasma_is_untouched_by_the_non_positive_rule():
    """The rule is inert wherever the column sums positive, which is everywhere else."""
    column = _peaked(9.0, pedestal=0.0, amplitude=1.0)
    stats = _flux_tube_profile_stats(column, X_CM, subtract_background=False)

    assert np.isfinite(stats["ftavg"])
    assert np.isfinite(stats["centroid"])
    assert np.isfinite(stats["edge"])
    # The CENTROID fold is the one that was used, not the axial fallback: for
    # an off-axis column the two give different numbers, and this is the
    # centroid one.
    offset = np.exp(-(((X_CM - 6.0) / 9.0) ** 2))
    off_stats = _flux_tube_profile_stats(offset, X_CM, subtract_background=False)
    assert off_stats["centroid"] == pytest.approx(6.0, abs=0.5)
    axial_weights = _flux_tube_weights(np.abs(X_CM), FLUX_TUBE_RADIUS_CM)
    axial_mean = float(axial_weights @ offset) / float(axial_weights.sum())
    assert off_stats["ftavg"] != pytest.approx(axial_mean, rel=1e-3)


def test_the_legacy_subtracted_chain_keeps_refusing_a_column_it_erased():
    """The signed-total rule does not reach the clipped legacy rows.

    ``subtract_background=True`` clips at zero, so its total can only ever
    reach the boundary, and a zero there means the retired subtraction erased
    the profile rather than that the column measured negative.  Those rows
    exist to reproduce the retired METHOD, so they keep returning NaN --
    see ``test_the_retired_subtraction_erases_a_flat_profile_entirely``.
    """
    column = -1.0 + 0.5 * np.cos(X_CM * (np.pi / 12.5))
    stats = _flux_tube_profile_stats(column, X_CM, subtract_background=True)

    assert np.isnan(stats["ftavg"])


def test_a_negative_weighted_variance_is_refused_rather_than_floored():
    """A negative scatter variance is undefined, and must not read as zero.

    Zero would claim the retained cells agree exactly, which is the opposite
    of what a negative variance says, so the SEM comes back NaN while the
    mean -- which is still the ratio that was wanted -- stays finite.
    """
    values = np.array([0.0, 1.0, 10.0])
    weights = np.array([1.0, -0.55, 0.2])

    normalized = weights / weights.sum()
    mean = float(normalized @ values)
    variance = float(normalized @ (values - mean) ** 2) / (
        1.0 - float(normalized @ normalized)
    )
    assert variance < 0.0  # the premise of this test, measured not assumed

    reported_mean, sem = _weighted_mean_and_sem(values, weights)

    assert reported_mean == pytest.approx(mean, rel=1e-12)
    assert np.isnan(sem)


def test_non_negative_weights_can_never_reach_either_behaviour():
    """The signed-weight handling is inert for every unsigned caller.

    The Isat families, the clipped legacy rows and the unweighted ``plain``
    average all pass non-negative weights, for which the mean is a convex
    combination and ``s^2 >= 0`` identically.  Measured over a spread of
    random non-negative weightings rather than asserted.
    """
    rng = np.random.default_rng(20260922)
    for _ in range(2000):
        values = rng.normal(size=6) * 5.0
        weights = rng.random(6)
        mean, sem = _weighted_mean_and_sem(values, weights)
        assert values.min() - 1e-12 <= mean <= values.max() + 1e-12
        assert np.isfinite(sem) and sem >= 0.0


def test_the_te_average_carries_the_semi_quantitative_weight_it_integrates():
    te = np.full(X_CM.size, 3.0)
    marks = np.abs(X_CM) > 12.0
    stats = _flux_tube_te_stats(
        te, _flat_density(), X_CM, semi_quantitative=marks
    )

    inside = int(np.count_nonzero(marks & (np.abs(X_CM) <= FLUX_TUBE_RADIUS_CM)))
    # Every marked cell inside the tube, plus the ONE sample just outside it
    # that carries the quadrature's closing weight at r = R.
    assert stats["semi_quant_count"] == inside + 1
    # 2 r dr weighting: the marked annulus 12 to 18.415 cm is most of the disc.
    assert 0.4 < stats["semi_quant_weight"] < 0.9


def test_a_density_profile_that_stops_short_of_the_radius_gives_no_te_row():
    short_x = np.linspace(-9.0, 9.0, 19)
    stats = _flux_tube_te_stats(
        np.full(short_x.size, 3.0), np.full(short_x.size, 1.0), short_x
    )

    assert np.isnan(stats["ftavg"]) and np.isnan(stats["plain"])


def test_the_te_series_refuses_grids_that_are_not_the_same_shape():
    with pytest.raises(ValueError, match="same"):
        _flux_tube_te_series(
            np.ones((2, X_CM.size, 3)), np.ones((2, X_CM.size, 4)), X_CM
        )


def test_coverage_is_the_outermost_measured_radius_capped_at_the_trust_radius():
    """(iv) A row measured only to 10 cm must say so, per port and sample."""
    n_t = 2
    measured = np.full((3, X_CM.size, n_t), np.nan)
    measured[0, np.abs(X_CM) <= 10.0, :] = 3.0   # truncated at 10 cm
    measured[1, np.abs(X_CM) <= 22.0, :] = 3.0   # measured past the aperture
    measured[2, np.abs(X_CM) <= 22.0, :] = 3.0   # measured, but untrusted port
    trust = np.array([18.415, 18.415, 10.0])
    rows = np.array([[7, 7], [7, 7], [7, 7]], dtype=np.int16)

    coverage = _measured_coverage_cm(measured, X_CM, trust, rows)

    assert coverage[0].tolist() == [10.0, 10.0]
    assert coverage[1].tolist() == [18.415, 18.415]
    assert coverage[2].tolist() == [10.0, 10.0]
    prior = ~(coverage >= FLUX_TUBE_RADIUS_CM)
    assert prior[0].tolist() == [True, True]
    assert prior[1].tolist() == [False, False]
    assert prior[2].tolist() == [True, True]


def test_a_prior_derived_row_has_no_coverage_and_is_flagged():
    measured = np.full((1, X_CM.size, 2), np.nan)
    measured[0, np.abs(X_CM) <= 22.0, 0] = 3.0
    trust = np.array([18.415])
    # The product itself calls the row prior-derived at both samples.
    rows = np.array([[0, 0]], dtype=np.int16)

    coverage = _measured_coverage_cm(measured, X_CM, trust, rows)

    assert np.isnan(coverage).all()
    assert (~(coverage >= FLUX_TUBE_RADIUS_CM)).all()


def test_the_density_weight_is_moved_onto_the_te_clock_without_inventing_cells():
    source_time = np.array([0.375, 0.875, 1.375, 1.875])
    target_time = np.array([0.0, 0.5, 1.0, 1.5])
    profiles = np.array([[[1.0, 2.0, np.nan, 4.0], [1.0, 1.0, 1.0, 1.0]]])

    moved = _interp_onto_time_grid(profiles, source_time, target_time)

    # Before the first source sample np.interp holds its finite end value.
    assert moved[0, 0, 0] == pytest.approx(1.0)
    assert moved[0, 0, 1] == pytest.approx(1.25)
    # Both target samples bracketed by the non-finite source sample are gone,
    # rather than filled in from the finite samples further away.
    assert np.isnan(moved[0, 0, 2]) and np.isnan(moved[0, 0, 3])
    assert np.allclose(moved[0, 1], 1.0)


def test_a_cell_with_one_finite_sample_does_not_reach_the_te_clock():
    source_time = np.array([0.0, 1.0, 2.0])
    profiles = np.array([[[np.nan, 5.0, np.nan]]])

    moved = _interp_onto_time_grid(profiles, source_time, np.array([0.5, 1.5]))

    assert np.isnan(moved).all()


# ---------------------------------------------------------------------------
# The WHOLE-COLUMN comparand: inventory out to the column edge, over the tube
# ---------------------------------------------------------------------------
def _column_density_stats(profile, x_cm=None, subtract_background=False):
    """The density row's column reduction, spelled out once for the tests.

    ``subtract_background`` defaults to FALSE, which is the comparand chain:
    the probe's own end-of-shot zeroing is the baseline, so the exporter's
    density rows subtract nothing further.
    """
    return _flux_tube_profile_stats(
        profile,
        X_CM if x_cm is None else x_cm,
        radius_cm=None,
        normalize_radius_cm=FLUX_TUBE_RADIUS_CM,
        subtract_background=subtract_background,
    )


#: A scan whose own extent IS the flux-tube radius, so the column integral and
#: the flux-tube integral have the same outer limit and nothing separates the
#: three conventions but the profile itself.
X_TUBE_CM = np.linspace(-FLUX_TUBE_RADIUS_CM, FLUX_TUBE_RADIUS_CM, 51)


def test_a_flat_profile_reads_the_same_under_all_three_conventions():
    """column == ftavg == core, exactly, with no background subtracted.

    This identity was unreachable while the effective-width ledger's baseline
    was still being removed: a profile flat to the scan edge had its own level
    as that baseline, so the subtraction took it to zero and the row came back
    NaN.  With the subtraction retired it is the plainest check there is --
    a uniform column is one number under a line cut, an area mean over the
    tube, and the tube-normalized inventory to the scan edge alike.
    """
    level = 3.0
    flat = np.full(X_TUBE_CM.size, level)

    column = _column_density_stats(flat, X_TUBE_CM)
    flux_tube = _flux_tube_profile_stats(
        flat, X_TUBE_CM, subtract_background=False
    )

    assert flux_tube["n_despiked"] == 0
    assert column["edge"] == pytest.approx(FLUX_TUBE_RADIUS_CM, rel=1e-12)
    assert flux_tube["ftavg"] == pytest.approx(level, rel=1e-12)
    assert column["ftavg"] == pytest.approx(level, rel=1e-12)
    assert flux_tube["core"] == pytest.approx(level, rel=1e-12)


def test_the_retired_subtraction_erases_a_flat_profile_entirely():
    """Why the identity above could not be written before the ruling."""
    flat = np.full(X_TUBE_CM.size, 3.0)

    subtracted = _flux_tube_profile_stats(
        flat, X_TUBE_CM, subtract_background=True
    )

    assert np.isnan(subtracted["ftavg"])
    assert subtracted["core"] == pytest.approx(0.0, abs=1e-12)


def test_a_flat_column_wider_than_the_tube_carries_more_than_the_tube():
    """The normalization, isolated: same level, more area, more inventory."""
    level = 3.0
    flat = np.full(X_CM.size, level)

    column = _column_density_stats(flat)
    flux_tube = _flux_tube_profile_stats(flat, X_CM, subtract_background=False)
    edge = column["edge"]

    assert edge == pytest.approx(25.0, rel=1e-12)
    assert flux_tube["ftavg"] == pytest.approx(level, rel=1e-12)
    assert column["ftavg"] == pytest.approx(
        level * (edge / FLUX_TUBE_RADIUS_CM) ** 2, rel=1e-12
    )


def _top_hat(level=3.0, flat_to_cm=16.0):
    """A column that stops INSIDE the tube, with no step the despike gate bites.

    Flat at ``level`` out to ``flat_to_cm``, one half-level cell, then zero.
    The half-level cell matters: a bare step puts the outermost flat cell's
    neighbour median at half its own value, which is past DESPIKE_TOLERANCE and
    would be "repaired" away.  Zero at both cells bracketing
    FLUX_TUBE_RADIUS_CM is what makes the two conventions coincide -- the
    quadrature interpolates the profile at r = R between them.
    """
    radius = np.abs(X_CM)
    return np.where(
        radius <= flat_to_cm,
        level,
        np.where(radius <= flat_to_cm + 1.0, level / 2.0, 0.0),
    )


def test_a_column_that_stops_inside_the_tube_reads_the_flux_tube_row():
    """Nothing outside the tube: the two area conventions are one number."""
    profile = _top_hat()

    column = _column_density_stats(profile)
    flux_tube = _flux_tube_profile_stats(
        profile, X_CM, subtract_background=False
    )

    assert flux_tube["n_despiked"] == 0
    assert column["edge"] == pytest.approx(np.max(np.abs(X_CM)), rel=1e-12)
    assert column["ftavg"] == pytest.approx(flux_tube["ftavg"], rel=1e-12)
    # The share outside the tube is zero, so the exported ratio is one.
    assert column["ftavg"] / flux_tube["ftavg"] == pytest.approx(1.0, rel=1e-12)
    # The core-band row is the flat level and both area rows sit below it,
    # because the disc reaches radii the line cut never does.
    assert flux_tube["core"] == pytest.approx(3.0, rel=1e-12)
    assert column["ftavg"] < flux_tube["core"]


def test_a_flat_te_reads_the_same_under_all_three_conventions():
    """T_e is where flat DOES give column == ftavg == core, for any weight.

    Both T_e rows are ratios over one node set, so a constant profile comes
    back unchanged whatever the weight, whatever the extent and whatever the
    normalization -- which is the property the column row has to have before
    its value can be read as a temperature at all.
    """
    te = np.full(X_CM.size, 4.25)
    density = _peaked(9.0, pedestal=0.0, amplitude=1.0)

    column = _flux_tube_te_stats(
        te,
        density,
        X_CM,
        radius_cm=None,
        normalize_radius_cm=FLUX_TUBE_RADIUS_CM,
        subtract_background=False,
    )
    flux_tube = _flux_tube_te_stats(te, density, X_CM, subtract_background=False)

    assert column["ftavg"] == pytest.approx(4.25, rel=1e-12)
    assert column["plain"] == pytest.approx(4.25, rel=1e-12)
    assert column["ftavg"] == pytest.approx(flux_tube["ftavg"], rel=1e-12)
    assert column["plain"] == pytest.approx(flux_tube["plain"], rel=1e-12)
    assert column["ftavg"] == pytest.approx(_core_band_mean(te), rel=1e-12)
    # The extents behind the two rows are NOT the same, which is what makes
    # the agreement a property of the profile rather than of the machinery.
    assert column["edge"] > flux_tube["edge"] == FLUX_TUBE_RADIUS_CM


def _gaussian_column_over_ftavg(width_cm, edge_cm=25.0):
    """The analytic ``column_over_ftavg_ratio`` of a centred Gaussian.

    Nothing is subtracted from the comparand chain, so for
    ``f(r) = exp(-(r/w)^2)`` the closed form is the bare pair of integrals,
    ``int_0^a 2 r f dr = w^2 (1 - exp(-a^2/w^2))``:

        ratio = (1 - e^{-edge^2/w^2}) / (1 - e^{-R^2/w^2})

    and ``1 - 1/ratio`` is the share of the column's inventory lying outside
    the tube.
    """
    return (1.0 - np.exp(-((edge_cm / width_cm) ** 2))) / (
        1.0 - np.exp(-((FLUX_TUBE_RADIUS_CM / width_cm) ** 2))
    )


def test_a_gaussian_column_gives_the_analytic_share_outside_the_tube():
    """A profile with a KNOWN fraction outside the tube, end to end.

    The tolerance is the 1 cm trapezoid's own discretization error against the
    closed form; nothing here is fitted to the implementation.
    """
    width_cm = 15.0
    profile = np.exp(-((X_CM / width_cm) ** 2))

    column = _column_density_stats(profile)
    flux_tube = _flux_tube_profile_stats(
        profile, X_CM, subtract_background=False
    )
    expected = _gaussian_column_over_ftavg(width_cm)

    assert flux_tube["n_despiked"] == 0
    assert column["ftavg"] / flux_tube["ftavg"] == pytest.approx(
        expected, rel=1e-3
    )
    # The fraction this width puts outside the tube, stated rather than implied.
    assert 1.0 - 1.0 / expected == pytest.approx(0.1699, abs=5e-4)


def test_the_column_te_row_is_the_two_quadrature_sums_over_the_column_nodes():
    """The identity the T_e column row is defined by, recomputed by hand."""
    density = _peaked(9.0, pedestal=0.0, amplitude=1.0)
    te = _peaked(7.0, pedestal=1.0, amplitude=3.0)

    stats = _flux_tube_te_stats(
        te,
        density,
        X_CM,
        radius_cm=None,
        normalize_radius_cm=FLUX_TUBE_RADIUS_CM,
        subtract_background=False,
    )

    # The comparand chain: despiked, nothing subtracted, nothing clipped.
    prepared = _despike_profile(density)[0]
    finite = np.isfinite(prepared) & np.isfinite(te)
    values = prepared[finite]
    positions = X_CM[finite]
    centroid = float(np.sum(values * positions) / np.sum(values))
    folded = np.abs(positions - centroid)
    edge = _column_edge_cm(folded)
    weights = _flux_tube_weights(folded, edge)

    assert stats["edge"] == pytest.approx(edge, rel=1e-12)
    assert stats["ftavg"] == pytest.approx(
        float((weights * values) @ te[finite] / np.sum(weights * values)),
        rel=1e-12,
    )
    assert stats["plain"] == pytest.approx(
        float(weights @ te[finite] / np.sum(weights)), rel=1e-12
    )
    # The weight denominator is the column DENSITY row, tube-area normalized.
    assert stats["weight_density"] == pytest.approx(
        _column_density_stats(density)["ftavg"], rel=1e-12
    )


def test_a_ten_centimetre_truncated_te_row_flags_the_prior_it_integrates():
    """The coverage rule, at the trust radius every ES3 and p50 row keeps.

    The coverage radius is the flux-tube one unchanged; what the column
    convention changes is what it is compared against, and past the tube the
    comparison is against an edge no port's T_e reaches.
    """
    measured = np.full((2, X_CM.size, 1), np.nan)
    measured[0, np.abs(X_CM) <= 22.0, :] = 3.0  # measured past the aperture
    measured[1, np.abs(X_CM) <= 22.0, :] = 3.0
    trust = np.array([FLUX_TUBE_RADIUS_CM, 10.0])  # an adopting port, and p50
    rows = np.array([[7], [7]], dtype=np.int16)

    coverage = _measured_coverage_cm(measured, X_CM, trust, rows)
    edge = _column_density_stats(_peaked(9.0, amplitude=1.0))["edge"]

    assert coverage[:, 0].tolist() == [FLUX_TUBE_RADIUS_CM, 10.0]
    # The flux-tube flag separates the two ports; the column flag cannot,
    # because the column runs past both trust radii at every port.
    assert (~(coverage >= FLUX_TUBE_RADIUS_CM))[:, 0].tolist() == [False, True]
    assert (~(coverage >= edge))[:, 0].tolist() == [True, True]


def test_the_column_edge_is_the_outermost_retained_radius():
    """The edge is the scan's own reach, not a declared column radius."""
    profile = _peaked(9.0, pedestal=0.0, amplitude=1.0)
    short_x = np.linspace(-20.0, 20.0, 41)
    short = np.exp(-((short_x / 9.0) ** 2))

    assert _column_edge_cm(np.array([1.0, 7.5, 3.0])) == 7.5
    assert _column_density_stats(profile)["edge"] == pytest.approx(25.0)
    assert _column_density_stats(short, short_x)["edge"] == pytest.approx(20.0)


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
    # The density grid holds SIGNED measurements or NaN, and a mean of
    # measured cells does not land on exactly zero, so a zero core-band mean
    # means the grid has begun carrying zero-filled cells -- and the
    # flux-tube row's error is transferred from that mean.
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


def test_a_negative_core_band_mean_is_a_measurement_and_is_not_refused():
    """A decayed column reads negative in the core band, and that is data.

    The density product carries the sign of the measured current, so a port
    whose column has gone into the noise averages to a negative core-band
    mean.  The guard judges EXACT ZERO and non-finite, which are input
    defects; a negative mean is a measurement and must pass.
    """
    _check_density_convention_pair(*_convention_pair([4.0, -0.3, 6.0], [3.0, 5.0, 5.5]))


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
def _window_mean(values, time_ms, window=RAW_PLATEAU_WINDOW_MS):
    """Per-port mean over the scoring plateau window, NaN-safe and warning-free."""
    band = np.asarray(values, dtype=np.float64)[
        :, (time_ms >= window[0]) & (time_ms <= window[1])
    ]
    finite = np.isfinite(band)
    counts = finite.sum(axis=1)
    out = np.full(band.shape[0], np.nan)
    covered = counts > 0
    out[covered] = (
        np.where(finite, band, 0.0)[covered].sum(axis=1) / counts[covered]
    )
    return out


def _plateau_ordering_margins_ev(core, ftavg, plain, time_ms, density_core=None):
    """Assert plain < weighted < core ON THE PLATEAU-WINDOW MEANS; return margins.

    This is a WINDOW-MEAN property and is asserted as one.  The per-sample
    ordering is not a property of these rows at all: the gap between the
    weighted and the plain row is cov_w(n, T_e) / <n>, which changes sign
    wherever the density and T_e anti-correlate across the disc.

    ``weighted < core`` holds at every port that carries a row and is asserted
    unconditionally.  ``plain < weighted`` is the statement that the density
    and T_e are positively correlated across the disc, and it is NOT universal
    either: at a port whose column has decayed into the noise there is no
    correlation left to weight by.  Pass ``density_core`` -- the same port rows
    of the core-band DENSITY -- to have the claim made where it is a claim, at
    every port whose plateau-mean core density is positive, and MEASURED
    elsewhere: a violation at a port whose core density averages non-positive
    is reported as the measured fact it is, and a violation anywhere else
    fails.  Without it the ordering is asserted at every port, which is what
    the synthetic callers want.
    """
    means = [_window_mean(v, time_ms) for v in (core, ftavg, plain)]
    rows = np.isfinite(means[0]) & np.isfinite(means[1]) & np.isfinite(means[2])
    assert rows.any(), "no port carries a plateau-window mean"
    core_mean, ftavg_mean, plain_mean = (m[rows] for m in means)

    lifted = plain_mean < ftavg_mean
    if density_core is None:
        carries_plasma = np.ones_like(lifted, dtype=bool)
    else:
        carries_plasma = _window_mean(density_core, time_ms)[rows] > 0.0
    assert np.all(lifted[carries_plasma]), (
        "the density weighting must lift the plateau-mean T_e above the plain "
        "area mean wherever the column carries plasma: "
        f"{plain_mean[carries_plasma]} vs {ftavg_mean[carries_plasma]}"
    )
    assert np.all(ftavg_mean < core_mean), (
        "the area average reaches radii the core-band line cut never sees and "
        f"must sit below it: {ftavg_mean} vs {core_mean}"
    )
    return list(zip(ftavg_mean - plain_mean, core_mean - ftavg_mean))


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
        "te_masked": np.where(
            np.abs(X_CM)[None, :, None] <= 10.0,
            3.0,
            np.nan,
        ).repeat(5, axis=0).repeat(3, axis=2),
        "te_semi_quantitative": np.zeros((5, X_CM.size, 3), dtype=bool),
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
    # The two per-cell grids the flux-tube T_e rows stand on come out too.
    assert records["measured_te"].shape == (5, X_CM.size, 3)
    assert records["semi_quantitative"].shape == (5, X_CM.size, 3)
    assert np.isfinite(records["measured_te"][0, :, 0]).sum() == int(
        np.count_nonzero(np.abs(X_CM) <= 10.0)
    )


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


def test_the_exported_overlay_carries_the_flux_tube_te_comparand():
    """The placed product, when it is a flux-tube-T_e vintage."""
    overlay = _overlay_or_skip()
    if "te_ftavg_ev" not in overlay.files:
        pytest.skip("the ES1 overlay on disk predates the flux-tube T_e rows")

    shape = overlay["te_mean_ev"].shape
    for key in (
        "te_ftavg_ev",
        "te_ftavg_sem_ev",
        "te_ftavg_radial_sem_ev",
        "te_ftavg_plain_ev",
        "te_ftavg_plain_sem_ev",
        "te_ftavg_weight_density_cm3",
        "te_ftavg_centroid_cm",
        "ftavg_coverage_cm",
        "ftavg_prior_beyond_coverage",
    ):
        assert overlay[key].shape == shape, key
    assert np.array_equal(overlay["te_ftavg_time_ms"], overlay["te_time_ms"])

    # The ordering of the three rows is a PLATEAU-WINDOW-MEAN property and is
    # asserted as one.  Per SAMPLE it does not hold and must not be asserted:
    # te_ftavg_ev - te_ftavg_plain_ev is cov_w(n, T_e) / <n>, which changes
    # sign wherever n and T_e anti-correlate across the disc, and a flat or
    # inverted profile across the core-band edge can put a sample above its
    # own core-band row.  See the two synthetic tests above.
    assert _plateau_ordering_margins_ev(
        overlay["te_mean_ev"],
        overlay["te_ftavg_ev"],
        overlay["te_ftavg_plain_ev"],
        overlay["te_time_ms"],
    )

    # Every p50 row integrates over a disc the measurement does not fill.
    p50 = int(np.flatnonzero(overlay["port"] == 50)[0])
    assert overlay["ftavg_prior_beyond_coverage"][p50].all()


def test_the_exported_overlay_carries_the_whole_column_comparand():
    """The placed product, when it is a whole-column vintage."""
    overlay = _overlay_or_skip()
    if "density_column_cm3" not in overlay.files:
        pytest.skip("the ES1 overlay on disk predates the whole-column rows")

    density_shape = overlay["density_mean_cm3"].shape
    for key in (
        "density_column_cm3",
        "density_column_sem_cm3",
        "density_column_radial_sem_cm3",
        "column_inventory_per_cm",
        "column_edge_cm",
        "column_over_ftavg_ratio",
    ):
        assert overlay[key].shape == density_shape, key
    te_shape = overlay["te_mean_ev"].shape
    for key in (
        "te_column_ev",
        "te_column_sem_ev",
        "te_column_radial_sem_ev",
        "te_column_plain_ev",
        "te_column_plain_sem_ev",
        "te_column_plain_radial_sem_ev",
        "te_column_edge_cm",
        "te_column_prior_weight",
        "te_column_pure_prior_weight",
        "te_column_plain_prior_weight",
        "te_column_plain_pure_prior_weight",
        "column_coverage_cm",
        "column_prior_beyond_coverage",
    ):
        assert overlay[key].shape == te_shape, key

    # The inventory and the density row are one number under two units.
    assert np.allclose(
        overlay["column_inventory_per_cm"],
        np.pi * FLUX_TUBE_RADIUS_CM**2 * overlay["density_column_cm3"],
        equal_nan=True,
    )
    # The column integral runs past the tube at every port and sample, so no
    # row may claim its T_e is measured all the way out.
    finite_edge = np.isfinite(overlay["te_column_edge_cm"])
    assert finite_edge.any()
    assert np.all(overlay["te_column_edge_cm"][finite_edge] > FLUX_TUBE_RADIUS_CM)
    assert overlay["column_prior_beyond_coverage"].all()
    # The coverage radius itself is the flux-tube one; only the comparison moved.
    assert np.array_equal(
        overlay["column_coverage_cm"], overlay["ftavg_coverage_cm"], equal_nan=True
    )
    # All three conventions are named for a consumer that has to pick one.
    comparand_map = str(overlay["ftavg_comparand_map"])
    for name in ("density_mean_cm3", "density_ftavg_cm3", "density_column_cm3"):
        assert name in comparand_map, name


#: The ES1 and ES3 p11 shares of the column T_e's density-weighted quadrature
#: weight lying beyond that port's trust radius, over the scoring plateau.
#: ES1 p11 is an 18.415 cm aperture port and ES3 p11 keeps the historical
#: 10 cm, which is the whole spread of the trust model in two numbers.
PLATEAU_PRIOR_WEIGHT_P11 = {1: 0.236, 3: 0.735}

#: The plateau window every scored row is taken over, in ms, as the share
#: fields below are judged on.  Named here so the measured statement about
#: where the shares stay inside [0, 1] says which samples it is about.
PRIOR_WEIGHT_PLATEAU_MS = (15.0, 19.5)


def _plateau_mask(time_ms):
    return (time_ms >= RAW_PLATEAU_WINDOW_MS[0]) & (
        time_ms <= RAW_PLATEAU_WINDOW_MS[1]
    )


@pytest.mark.parametrize("experiment_set", sorted(PLATEAU_PRIOR_WEIGHT_P11))
def test_the_column_te_prior_weight_is_the_measured_share(experiment_set):
    """How much of each column T_e is the SOL prior, pinned on the product.

    The share is ``sum w n`` beyond the trust radius over ``sum w n`` over the
    whole column, and BOTH SUMS ARE SIGNED: since the density grid stopped
    deleting its negative cells, a sample whose outer cells are noise about
    zero can carry a negative numerator or a numerator larger than its own
    denominator.  So [0, 1] is not a property of this field and is not
    asserted; it is MEASURED to hold over the plateau window every scored row
    is taken from, and to fail only at early samples, which is the statement
    that is actually true of the product.
    """
    path = Path(f"processed/es{experiment_set}_sim1d_overlay.npz")
    if not path.exists():
        pytest.skip(f"no ES{experiment_set} overlay on disk")
    overlay = np.load(path, allow_pickle=True)
    if "te_column_prior_weight" not in overlay.files:
        pytest.skip("the overlay on disk predates the column prior weight")

    share = overlay["te_column_prior_weight"]
    pure = overlay["te_column_pure_prior_weight"]
    finite = np.isfinite(share)
    assert finite.any()

    time_ms = overlay["te_time_ms"]
    scored = (time_ms >= PRIOR_WEIGHT_PLATEAU_MS[0]) & (
        time_ms <= PRIOR_WEIGHT_PLATEAU_MS[1]
    )
    inside = finite & scored[None, :]
    assert inside.any()
    assert np.all((share[inside] >= 0.0) & (share[inside] <= 1.0)), (
        "a prior-weight share left [0, 1] inside the scored plateau window, "
        "where the column still carries plasma and the denominator is not "
        f"noise: {share[inside][(share[inside] < 0.0) | (share[inside] > 1.0)]}"
    )
    outside = finite & ~scored[None, :]
    excursions = outside & ((share < 0.0) | (share > 1.0))
    if excursions.any():
        # Measured, not asserted away: every one of them is an early sample.
        assert np.all(time_ms[np.nonzero(excursions)[1]] < 10.0)
    # Past the blend radius is a subset of past the trust radius -- an
    # inclusion of node SETS, so it survives the sign only where the shares
    # themselves do; judged on the scored window for the same reason.
    assert np.all(pure[inside] <= share[inside] + 1e-12)

    p11 = int(np.flatnonzero(overlay["port"] == 11)[0])
    window = _plateau_mask(overlay["te_time_ms"])
    assert np.nanmean(share[p11, window]) == pytest.approx(
        PLATEAU_PRIOR_WEIGHT_P11[experiment_set], abs=0.005
    )


#: Every area-averaged comparand and the legacy row that carries its retired,
#: background-subtracted value.
SUBTRACTED_LEGACY_PAIRS = (
    ("density_ftavg_subtracted_cm3", "density_ftavg_cm3"),
    ("density_column_subtracted_cm3", "density_column_cm3"),
    ("isat_ftavg_upstream_subtracted_a", "isat_ftavg_upstream_a"),
    ("isat_ftavg_subtracted_a", "isat_ftavg_a"),
    ("isat_ftavg_geomean_subtracted_a_per_cm2", "isat_ftavg_geomean_a_per_cm2"),
)


def test_the_legacy_subtracted_rows_read_low_and_are_not_the_comparand():
    """The retired convention is kept, named, and reads low on balance.

    It reads low, not uniformly low, and the difference is NOT pointwise
    one-signed.  The retired path did TWO things -- removed an edge-median
    baseline and clipped the result at zero -- and the Isat line scans carry
    genuinely negative cells in the far skirt (349 of 10,200 finite despiked
    cells on the ES1 upstream face), which the clip used to lift to zero.  At
    a sample whose baseline is itself zero, that lift is all there is, so the
    legacy row can sit a little ABOVE its comparand; the re-derived centroid
    moves the quadrature nodes by a hair on top of it.  What holds is the
    MEDIAN, and the monotone statement belongs on a profile with a strictly
    positive pedestal and no negative cells -- the test below.
    """
    overlay = _overlay_or_skip()
    if "density_ftavg_subtracted_cm3" not in overlay.files:
        pytest.skip("the ES1 overlay on disk predates the background ruling")

    for legacy, comparand in SUBTRACTED_LEGACY_PAIRS:
        assert legacy in overlay.files, legacy
        both = np.isfinite(overlay[legacy]) & np.isfinite(overlay[comparand])
        assert both.any(), legacy
        relative = (
            overlay[legacy][both] - overlay[comparand][both]
        ) / overlay[comparand][both]
        # The retired convention took signal away from the typical sample.
        assert np.median(relative) < 0.0, legacy
        assert legacy in str(overlay["subtracted_legacy_definition"]), legacy

    ruling = str(overlay["ftavg_background"])
    assert "NO BACKGROUND IS SUBTRACTED" in ruling
    assert "END OF THE SHOT" in ruling
    assert "NOTHING IN THIS PRODUCT STILL SUBTRACTS." in ruling


def test_a_positive_pedestal_is_what_the_retired_convention_removed():
    """The monotone statement, on a profile where only the baseline differs.

    The ledger's baseline is the median of the outermost three samples, which
    on a monotone skirt is the middle one, |x| = 24 cm -- so what it removes
    is the pedestal PLUS whatever the Gaussian still has out there, and the
    weights summing to one for a uniform profile make the area average drop by
    exactly that much.
    """
    pedestal = 0.4
    profile = pedestal + np.exp(-((X_CM / 9.0) ** 2))
    removed = pedestal + np.exp(-((24.0 / 9.0) ** 2))

    comparand = _flux_tube_profile_stats(
        profile, X_CM, subtract_background=False
    )
    legacy = _flux_tube_profile_stats(profile, X_CM, subtract_background=True)

    assert profile.min() > 0.0  # nothing for the retired clip to lift
    assert comparand["ftavg"] - legacy["ftavg"] == pytest.approx(
        removed, rel=1e-9
    )


TE_FILLED_HDF5 = Path("processed/te_filled.hdf5")
DENSITY_PROFILES_HDF5 = Path("processed/density_profiles_isweep.hdf5")


def test_the_flux_tube_te_rows_rebuilt_from_the_placed_profile_products():
    """The rows REGENERATED from the real inputs, at every experiment set.

    The placed overlay npz can be any vintage, so the test above skips on an
    old one; this one does not depend on it.  It rebuilds the flux-tube T_e
    rows from the two profile products the exporter reads, with the exporter's
    own functions, and checks the two things that must hold of them:

    * each row IS the two quadrature sums over its nodes, and the gap between
      them IS the weighted covariance of density and T_e -- exactly, at every
      port and every sample;
    * the PLATEAU-WINDOW MEANS are ordered plain < weighted < core at every
      port THAT CARRIES PLASMA.  Per sample they are not, and neither is the
      lift universal: since the density grid stopped deleting its negative
      cells, ES4 p50 -- whose column has decayed into the noise and whose
      core-band density averages NEGATIVE over the plateau -- reads
      plain 0.648 eV against weighted 0.584 eV on this legacy chain, the
      density weighting pulling the average down instead of lifting it.  The
      core-band density is passed so the claim is made where it is a claim.
    """
    if not (TE_FILLED_HDF5.exists() and DENSITY_PROFILES_HDF5.exists()):
        pytest.skip("the placed T_e / density profile products are not on disk")

    checked_sets = 0
    with h5py.File(TE_FILLED_HDF5, "r") as te_hdf, h5py.File(
        DENSITY_PROFILES_HDF5, "r"
    ) as density_hdf:
        x_cm = density_hdf["x_cm"][()]
        shared = sorted(
            set(te_hdf["experiment_sets"]) & set(density_hdf["experiment_sets"])
        )
        for set_id in shared:
            te_group = te_hdf[f"experiment_sets/{set_id}"]
            density_group = density_hdf[f"experiment_sets/{set_id}"]
            if not np.allclose(te_group["x_cm"][()], x_cm):
                pytest.skip(f"ES{set_id} radial grids differ between products")
            te_profiles = te_group["te_filled"][()]
            te_time_ms = te_group["cycle_time_ms"][()]
            weight = _interp_onto_time_grid(
                density_group["n_e_m3"][()],
                density_group["inter_sweep_time_s"][()] * 1000.0,
                te_time_ms,
            )
            rows = _flux_tube_te_series(te_profiles, weight, x_cm)

            n_z, _, n_t = te_profiles.shape
            for zi in range(n_z):
                for ti in range(n_t):
                    nodes = _quadrature_nodes(
                        weight[zi, :, ti], te_profiles[zi, :, ti], x_cm
                    )
                    stats = {
                        "ftavg": rows["ftavg"][zi, ti],
                        "plain": rows["plain"][zi, ti],
                    }
                    if nodes is None:
                        assert np.isnan(stats["ftavg"])
                        assert np.isnan(stats["plain"])
                        continue
                    _assert_is_the_two_quadrature_sums(stats, nodes)
                    _assert_the_gap_is_the_weighted_covariance(stats, nodes)

            core_mean, _, _, _ = _nan_core_stats(te_profiles, x_cm, X_MIN_CM, X_MAX_CM)
            density_core, _, _, _ = _nan_core_stats(weight, x_cm, X_MIN_CM, X_MAX_CM)
            _plateau_ordering_margins_ev(
                core_mean,
                rows["ftavg"],
                rows["plain"],
                te_time_ms,
                density_core=density_core,
            )
            checked_sets += 1

    assert checked_sets >= 1


def test_the_density_product_keeps_the_cells_the_sign_test_used_to_delete():
    """The density grid is SIGNED, and says which cells the sign test took.

    A cell whose measured current is negative is noise about zero, not a
    failure, so it is stored with its sign; NaN is reserved for a cell with no
    usable measurement.  ``n_e_sign_masked`` is the traceability companion --
    exactly the finite non-positive cells, the set the retired sign test would
    have written as NaN -- and nothing masks anything by it.
    """
    if not DENSITY_PROFILES_HDF5.exists():
        pytest.skip("the placed density profile product is not on disk")

    checked = 0
    with h5py.File(DENSITY_PROFILES_HDF5, "r") as density_hdf:
        assert "SIGNED" in str(density_hdf.attrs["n_e_sign_convention"])
        for set_id in sorted(density_hdf["experiment_sets"]):
            group = density_hdf[f"experiment_sets/{set_id}"]
            density = group["n_e_m3"][()]
            sign_masked = group["n_e_sign_masked"][()].astype(bool)

            assert sign_masked.shape == density.shape
            assert np.array_equal(
                sign_masked, np.isfinite(density) & (density <= 0.0)
            )
            # The cells the sign test used to delete are present and finite.
            assert np.all(np.isfinite(density[sign_masked]))
            checked += 1

    assert checked >= 1
    # At least one set must actually carry them, or this asserts nothing.
    assert sign_masked.any()


def _comparand_nodes(density_profile, te_profile, x_cm, radius_cm=FLUX_TUBE_RADIUS_CM):
    """``_quadrature_nodes`` on THE COMPARAND CHAIN: no subtraction, no clip.

    This is the signed path -- what every density and T_e row of record is
    reduced on -- so ``values`` here can be negative and the weights ``w n``
    can change sign.  Returns ``None`` where no row is formed.
    """
    density, _ = _despike_profile(density_profile)
    te, _ = _despike_profile(te_profile)
    finite = np.isfinite(density) & np.isfinite(te)
    if np.count_nonzero(finite) < 5:
        return None
    values = density[finite]
    total = float(np.sum(values))
    if total <= 0.0:
        return None
    centroid = float(np.sum(values * x_cm[finite]) / total)
    folded = np.abs(x_cm[finite] - centroid)
    limit = _column_edge_cm(folded) if radius_cm is None else float(radius_cm)
    try:
        weights = _flux_tube_weights(folded, limit)
    except ValueError:
        return None
    return weights, values, te[finite]


def test_the_signed_weight_breaks_the_convex_combination_and_the_break_is_measured():
    """What the comparand T_e rows ARE, and what they have stopped being.

    With the density sign test retired the weight ``w_i n_i`` can be negative,
    so ``te_ftavg_ev`` is no longer bounded by the T_e values at its own
    nodes.  Nothing here asserts a bound, because the bound is not a property
    of these rows any more.  What IS asserted, at every port and every sample
    of the real products:

    * the row is still exactly the ratio of its two quadrature sums, and the
      gap to the plain row is still exactly the weighted covariance;
    * every sample that leaves ``[min T, max T]`` has at least one
      negative-weight node -- the excursion is the signed weight and nothing
      else, never a defect of the quadrature;
    * under the FLUX-TUBE convention no sample inside the 10-19 ms plateau
      window leaves its bound, which is the window every scored row is taken
      over.  The whole-column convention is not asserted to that: it
      integrates to the scan edge, where the noise cells are, and it does
      leave the bound at early samples.
    """
    if not (TE_FILLED_HDF5.exists() and DENSITY_PROFILES_HDF5.exists()):
        pytest.skip("the placed T_e / density profile products are not on disk")

    excursions = 0
    plateau_excursions = 0
    negative_weight_samples = 0
    checked_sets = 0
    with h5py.File(TE_FILLED_HDF5, "r") as te_hdf, h5py.File(
        DENSITY_PROFILES_HDF5, "r"
    ) as density_hdf:
        x_cm = density_hdf["x_cm"][()]
        shared = sorted(
            set(te_hdf["experiment_sets"]) & set(density_hdf["experiment_sets"])
        )
        for set_id in shared:
            te_group = te_hdf[f"experiment_sets/{set_id}"]
            density_group = density_hdf[f"experiment_sets/{set_id}"]
            if not np.allclose(te_group["x_cm"][()], x_cm):
                pytest.skip(f"ES{set_id} radial grids differ between products")
            te_profiles = te_group["te_filled"][()]
            te_time_ms = te_group["cycle_time_ms"][()]
            weight = _interp_onto_time_grid(
                density_group["n_e_m3"][()],
                density_group["inter_sweep_time_s"][()] * 1000.0,
                te_time_ms,
            )
            rows = _flux_tube_te_series(
                te_profiles, weight, x_cm, subtract_background=False
            )

            n_z, _, n_t = te_profiles.shape
            for zi in range(n_z):
                for ti in range(n_t):
                    nodes = _comparand_nodes(
                        weight[zi, :, ti], te_profiles[zi, :, ti], x_cm
                    )
                    stats = {
                        "ftavg": rows["ftavg"][zi, ti],
                        "plain": rows["plain"][zi, ti],
                    }
                    if nodes is None:
                        assert np.isnan(stats["ftavg"])
                        assert np.isnan(stats["plain"])
                        continue
                    _assert_is_the_two_quadrature_sums(stats, nodes)
                    _assert_the_gap_is_the_weighted_covariance(stats, nodes)

                    weights, density, te_values = nodes
                    weighted = weights * density
                    if np.any(weighted < 0.0):
                        negative_weight_samples += 1
                    if not np.isfinite(stats["ftavg"]):
                        continue
                    carrying = weights > 0.0
                    low = float(np.min(te_values[carrying]))
                    high = float(np.max(te_values[carrying]))
                    if low <= stats["ftavg"] <= high:
                        continue
                    excursions += 1
                    assert np.any(weighted < 0.0), (
                        f"ES{set_id} port index {zi} sample {ti}: the weighted "
                        "T_e left [min, max] of its own nodes with every "
                        "quadrature weight non-negative, which cannot happen"
                    )
                    in_plateau = 10.0 <= float(te_time_ms[ti]) <= 19.0
                    plateau_excursions += int(in_plateau)
            checked_sets += 1

    assert checked_sets >= 1
    # The premise: the signed weight really is present in these products.
    assert negative_weight_samples > 0
    assert excursions > 0
    # The scored window is clean of it under the flux-tube convention.
    assert plateau_excursions == 0


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


# ---------------------------------------------------------------------------
# The afterglow e-fold matrix: face x radial convention x port
# ---------------------------------------------------------------------------
DECAY_DT_MS = 0.1
DECAY_SPAN_MS = 10.0


def _decay_time_grid():
    """A trace grid long enough to carry both the fit window and its tail."""
    start = float(ISAT_DECAY_FIT_WINDOW_MS[0])
    n = int(round(DECAY_SPAN_MS / DECAY_DT_MS)) + 1
    return start + DECAY_DT_MS * np.arange(n, dtype=np.float64)


def _decay_profiles(tau_cm, t_ms, x_cm=X_CM, peak=1.0):
    """One port's synthetic afterglow line scan, shaped ``(port, x, time)``.

    A fixed Gaussian radial shape decaying with a per-radius e-fold time
    ``tau_cm(|x|)``, so a convention that weights the outer column more heavily
    sees a different effective decay from one that reads the axis alone.
    """
    shape = peak * np.exp(-((x_cm / 8.0) ** 2))
    tau = np.asarray(tau_cm(np.abs(x_cm)), dtype=np.float64)
    elapsed = t_ms - t_ms[0]
    return (
        shape[None, :, None]
        * np.exp(-elapsed[None, None, :] / tau[None, :, None])
    )


def _decay_matrix_cells(profiles, sem_value=1.0e-4, x_cm=X_CM):
    """Reduce one line scan into the nine (face, convention) matrix cells.

    The same wiring ``export_overlay`` uses: the x=0 column for ``x0`` and the
    two radial quadratures for ``ftavg`` / ``column``.  All three faces are
    handed the same traces here, so any difference between faces in a result
    would be the fit reading its own axis labels.
    """
    sem = np.full_like(profiles, sem_value)
    ftavg = _flux_tube_series(profiles, x_cm, sem, subtract_background=False)
    column = _flux_tube_series(
        profiles,
        x_cm,
        sem,
        radius_cm=None,
        normalize_radius_cm=FLUX_TUBE_RADIUS_CM,
        subtract_background=False,
    )
    axis = int(np.argmin(np.abs(x_cm)))
    per_convention = {
        "x0": (profiles[:, axis, :], sem[:, axis, :]),
        "ftavg": (ftavg["ftavg"], ftavg["ftavg_sem"]),
        "column": (column["ftavg"], column["ftavg_sem"]),
    }
    return {
        (face, convention): per_convention[convention]
        for face in ISAT_DECAY_MATRIX_FACES
        for convention in ISAT_DECAY_MATRIX_CONVENTIONS
    }


def _fit_matrix(cells, t_ms, ports=(11,)):
    ports = np.asarray(ports, dtype=np.int16)
    clear = {face: np.zeros(ports.size, dtype=bool) for face in ISAT_DECAY_MATRIX_FACES}
    reasons = {face: np.asarray([""] * ports.size) for face in ISAT_DECAY_MATRIX_FACES}
    return _isat_decay_matrix(t_ms, cells, ports, clear, reasons)


def test_the_matrix_x0_slice_is_the_existing_x0_decay_fit_byte_for_byte():
    """The proof that the nine cells share ONE fit.

    The ``x0`` convention is the ``isat_decay_*`` family the transport
    comparison's stage (iii) already fits, so fitting those traces
    independently -- outside the matrix, through the same two helpers -- must
    reproduce the ``[:, 0, :]`` slice exactly, not merely closely.  Each face
    is given a DIFFERENT x=0 trace so the check cannot pass by the three
    happening to coincide.
    """
    t_ms = _decay_time_grid()
    elapsed = t_ms - t_ms[0]
    x0_traces = {
        "upstream": 0.030 * np.exp(-elapsed / 0.62),
        "downstream": 0.012 * np.exp(-elapsed / 1.15),
        "geomean": 0.190 * np.exp(-elapsed / 0.83),
    }
    sem = np.full(t_ms.size, 1.0e-5)
    profiles = _decay_profiles(lambda r: 0.8 - 0.01 * r, t_ms)
    cells = _decay_matrix_cells(profiles)
    for face, trace in x0_traces.items():
        cells[(face, "x0")] = (trace[None, :], sem[None, :])

    matrix = _fit_matrix(cells, t_ms)

    assert list(matrix["convention"]) == ["x0", "ftavg", "column"]
    window = (t_ms >= ISAT_DECAY_FIT_WINDOW_MS[0]) & (
        t_ms <= ISAT_DECAY_FIT_WINDOW_MS[1]
    )
    tail = t_ms >= t_ms.max() - 5.0
    independent = np.array(
        [
            [
                _decay_efold_ms(
                    t_ms[window],
                    x0_traces[face][window],
                    _decay_noise_floor_a(x0_traces[face], tail),
                )
            ]
            for face in ISAT_DECAY_MATRIX_FACES
        ],
        dtype=np.float64,
    )

    assert np.all(np.isfinite(independent))
    assert matrix["tau_ms"][:, 0, :].tobytes() == independent.tobytes()


def test_a_self_similar_decay_reads_the_same_tau_in_all_nine_cells():
    """One e-fold time at every radius must come back as one number.

    If the column decays at a single rate, no radial average can change that
    rate, and neither can the choice of face: the nine cells differ only in
    which linear combination of the same exponentials they fit.
    """
    t_ms = _decay_time_grid()
    tau_ms = 0.80
    uniform = _decay_profiles(lambda r: np.full_like(r, tau_ms), t_ms)
    cells = _decay_matrix_cells(uniform)

    matrix = _fit_matrix(cells, t_ms)

    # The noise floor must not have removed a sample; otherwise the cells
    # would be fitted over different sample sets and the equality would be
    # about the mask rather than about the decay.
    assert np.all(matrix["n_fit"] == matrix["n_window"])
    tau = matrix["tau_ms"]
    assert np.all(np.isfinite(tau))
    assert tau == pytest.approx(np.full_like(tau, tau_ms), rel=1e-9)


def test_an_edge_that_cools_faster_shortens_the_column_tau_below_the_axis_tau():
    """The SIGN the matrix exists to read.

    With the e-fold time falling toward the edge, an average that counts the
    outer column decays faster than the axis does.  The whole-column
    convention counts the most of it, the flux tube less, and the x=0 point
    none -- so the three must come out ordered, and a column tau SHORTER than
    the x0 tau is the statement that the edge cools faster.
    """
    t_ms = _decay_time_grid()
    axis_tau_ms = 0.80
    cells = _decay_matrix_cells(
        _decay_profiles(lambda r: axis_tau_ms / (1.0 + 0.05 * r), t_ms)
    )

    matrix = _fit_matrix(cells, t_ms)

    assert np.all(matrix["n_fit"] == matrix["n_window"])
    tau = matrix["tau_ms"]
    assert np.all(np.isfinite(tau))
    for face in range(len(ISAT_DECAY_MATRIX_FACES)):
        x0_tau, ftavg_tau, column_tau = tau[face, :, 0]
        assert x0_tau == pytest.approx(axis_tau_ms, rel=1e-9)
        assert column_tau < ftavg_tau < x0_tau


def test_a_face_excluded_on_one_convention_is_excluded_on_all_three():
    """An exclusion belongs to the run's channel, not to a radial average."""
    t_ms = _decay_time_grid()
    cells = _decay_matrix_cells(_decay_profiles(lambda r: np.full_like(r, 0.8), t_ms))
    ports = np.array([11], dtype=np.int16)
    excluded = {face: np.zeros(1, dtype=bool) for face in ISAT_DECAY_MATRIX_FACES}
    reasons = {face: np.asarray([""]) for face in ISAT_DECAY_MATRIX_FACES}
    excluded["downstream"] = np.ones(1, dtype=bool)
    reasons["downstream"] = np.asarray(["probe-local current"])

    matrix = _isat_decay_matrix(t_ms, cells, ports, excluded, reasons)

    face_index = ISAT_DECAY_MATRIX_FACES.index("downstream")
    assert matrix["excluded"].shape == (3, 1)
    assert bool(matrix["excluded"][face_index, 0])
    assert not matrix["excluded"][[0, 2], 0].any()
    assert str(matrix["excluded_reason"][face_index, 0]) == "probe-local current"


def test_the_matrix_refuses_a_cell_that_is_not_on_the_exported_grid():
    t_ms = _decay_time_grid()
    cells = _decay_matrix_cells(_decay_profiles(lambda r: np.full_like(r, 0.8), t_ms))
    short = cells[("upstream", "x0")][0][:, :-1]
    cells[("upstream", "x0")] = (short, short)

    with pytest.raises(ValueError, match="decay matrix cell"):
        _fit_matrix(cells, t_ms)
