import json

import h5py
import numpy as np
import pytest

from scripts.fit_te_spatial import (
    CONTROL_SOURCE_BAND,
    CONTROL_SOURCE_LEGACY,
    CONTROL_SOURCE_NONE,
    SEMI_QUANT_CORE_CONTROL,
    SEMI_QUANT_SUB_EV,
    SEMI_QUANT_WINDOW,
    SEMI_QUANTITATIVE_DLN,
    TRUST_MODEL_BY_SET_PORT,
    X_CORE_CM,
    X_EDGE_CM,
    X_TRUST_APERTURE_CM,
    X_TRUST_BLEND_CM,
    _core_window_sem,
    _enforce_core_mean_monotonic_z,
    _mask_untrusted_te_ports,
    _monotonic_z_bound,
    _qc_surviving_cells,
    _robust_current_spike_mask,
    _semi_quantitative_marks,
    _trust_model_for_ports,
    _window_spread_grids,
    QC_FLOOR_MIN_N_OK,
    _legacy_dln,
    fill_te_cycle,
    load_window_band_spreads,
    measured_weight_allowed,
    merge_legacy_x0_controls,
    sparse_z_rows,
)

ADOPTED_SET_PORTS = (
    ("1", 11), ("1", 21), ("1", 29), ("1", 41),
    ("2", 11), ("2", 21), ("2", 29), ("2", 41),
)


def test_current_spike_mask_preserves_only_core_spikes():
    x_cm = np.array([-20.0, -5.0, 0.0, 5.0, 20.0])
    current = np.full((5, 2), 10.0)
    current[2, 1] = 30.0
    current[4, 1] = 100.0

    mask = _robust_current_spike_mask(current, x_cm, core_x_cm=10.0, sigma=3.0, ratio=1.25)

    assert mask[2, 1]
    assert not mask[4, 1]
    assert mask.sum() == 1


def test_monotonic_z_bound_keeps_corroborated_core_hotspot_only():
    te_grid = np.array(
        [
            [[2.0], [2.0], [2.0]],
            [[4.0], [4.0], [1.0]],
        ]
    )
    preserve = np.zeros(te_grid.shape, dtype=bool)
    preserve[1, 0, 0] = True

    bounded, n_masked, n_preserved = _monotonic_z_bound(
        te_grid,
        np.array([100.0, 200.0]),
        padding=0.25,
        preserve_mask=preserve,
    )

    assert bounded[1, 0, 0] == pytest.approx(4.0)
    assert np.isnan(bounded[1, 1, 0])
    assert bounded[1, 2, 0] == pytest.approx(1.0)
    assert n_masked == 1
    assert n_preserved == 1


def test_monotonic_z_bound_rejects_mismatched_preserve_mask_shape():
    te_grid = np.ones((2, 3, 1))

    with pytest.raises(ValueError, match="preserve_mask shape"):
        _monotonic_z_bound(
            te_grid,
            np.array([100.0, 200.0]),
            preserve_mask=np.zeros((2, 3), dtype=bool),
        )


def test_later_set_untrusted_ports_are_blanked():
    te = np.ones((5, 2, 1))
    ports = np.array([11, 21, 29, 41, 50])

    masked, excluded = _mask_untrusted_te_ports(te, ports, "3")

    assert excluded == (21, 41, 50)
    assert np.all(np.isfinite(masked[[0, 2]]))
    assert np.all(np.isnan(masked[[1, 3, 4]]))


def test_core_mean_monotonic_enforcement_preserves_radial_shape_about_floor():
    te = np.array(
        [
            [[2.0], [4.0], [2.0]],
            [[3.0], [6.0], [3.0]],
            [[1.0], [2.0], [1.0]],
        ]
    )
    x_cm = np.array([-5.0, 0.0, 5.0])

    bounded, adjusted, max_reduction, clamped, scales = _enforce_core_mean_monotonic_z(
        te,
        x_cm,
        te_floor=0.1,
    )

    means = bounded[:, :, 0].mean(axis=1)
    assert np.all(np.diff(means) <= 1e-12)
    assert adjusted == 1
    assert max_reduction > 0.0
    assert clamped.tolist() == [[False], [True], [False]]
    assert scales[1, 0] < 1.0
    assert scales[0, 0] == pytest.approx(1.0)
    original_ratio = (te[1, 1, 0] - 0.1) / (te[1, 0, 0] - 0.1)
    bounded_ratio = (bounded[1, 1, 0] - 0.1) / (bounded[1, 0, 0] - 0.1)
    assert bounded_ratio == pytest.approx(original_ratio)


def test_core_mean_clamp_skips_inversions_absent_from_the_measurement():
    """A row whose measurement is monotonic is not rewritten by the clamp."""
    te = np.array(
        [
            [[2.0], [4.0], [2.0]],
            [[3.0], [6.0], [3.0]],
        ]
    )
    x_cm = np.array([-5.0, 0.0, 5.0])
    # The measurement puts row 1 BELOW row 0; the inversion exists only in the
    # filled surface, which is exactly the fill artefact the gate is there for.
    measured = np.array(
        [
            [[5.0], [7.0], [5.0]],
            [[1.0], [2.0], [1.0]],
        ]
    )

    bounded, adjusted, max_reduction, clamped, scales = _enforce_core_mean_monotonic_z(
        te,
        x_cm,
        te_floor=0.1,
        measured_grid=measured,
    )

    assert adjusted == 0
    assert max_reduction == 0.0
    assert not clamped.any()
    assert np.all(scales == 1.0)
    assert bounded == pytest.approx(te)


def test_core_mean_clamp_still_binds_on_a_measured_inversion():
    """The prior keeps acting where the measurement itself runs the wrong way."""
    te = np.array(
        [
            [[2.0], [4.0], [2.0]],
            [[3.0], [6.0], [3.0]],
        ]
    )
    x_cm = np.array([-5.0, 0.0, 5.0])
    measured = te.copy()

    bounded, adjusted, _, clamped, _ = _enforce_core_mean_monotonic_z(
        te,
        x_cm,
        te_floor=0.1,
        measured_grid=measured,
    )

    assert adjusted == 1
    assert clamped.tolist() == [[False], [True]]
    assert bounded[1, :, 0].mean() == pytest.approx(te[0, :, 0].mean())


def test_core_mean_clamp_applies_to_rows_with_no_measured_core_cells():
    """A reconstructed row has no measurement to judge it, so the prior stands."""
    te = np.array(
        [
            [[2.0], [4.0], [2.0]],
            [[3.0], [6.0], [3.0]],
        ]
    )
    x_cm = np.array([-5.0, 0.0, 5.0])
    measured = np.array(
        [
            [[2.0], [4.0], [2.0]],
            [[np.nan], [np.nan], [np.nan]],
        ]
    )

    _, adjusted, _, clamped, _ = _enforce_core_mean_monotonic_z(
        te,
        x_cm,
        te_floor=0.1,
        measured_grid=measured,
    )

    assert adjusted == 1
    assert clamped[1, 0]


def test_core_mean_clamp_rejects_a_mismatched_measured_grid():
    te = np.ones((2, 3, 1))

    with pytest.raises(ValueError, match="measured grid shape"):
        _enforce_core_mean_monotonic_z(
            te,
            np.array([-5.0, 0.0, 5.0]),
            measured_grid=np.ones((2, 3)),
        )


def test_fill_keeps_measured_core_cells_and_leaves_the_edge_prior_alone():
    """The sentinels may shape the gaps; they may not move a measured core cell.

    Outside the core the scrape-off-layer prior is intended, so the fitted
    surface must still be what comes out there.
    """
    x_cm = np.linspace(-25.0, 25.0, 51)
    z_cm = np.array([470.1, 789.5, 1045.2])
    rng = np.random.default_rng(0)
    te_2d = 8.0 - 0.002 * (z_cm[:, None] - z_cm[0]) - 0.004 * x_cm[None, :] ** 2
    te_2d = te_2d + rng.normal(scale=0.05, size=te_2d.shape)
    te_2d[1, 24] = np.nan
    te_2d[2, 37] = np.nan
    measured = np.isfinite(te_2d)
    core = np.broadcast_to(np.abs(x_cm) <= 10.0, te_2d.shape)
    beyond_edge = np.broadcast_to(np.abs(x_cm) >= 15.0, te_2d.shape)

    filled = fill_te_cycle(te_2d, x_cm, z_cm)
    unpreserved = fill_te_cycle(te_2d, x_cm, z_cm, preserve_measured=False)

    assert np.all(np.isfinite(filled))
    assert filled[measured & core] == pytest.approx(te_2d[measured & core])
    assert filled[~measured] == pytest.approx(unpreserved[~measured])
    assert filled[beyond_edge] == pytest.approx(unpreserved[beyond_edge])
    # The row nearest the cathode end plate is the one the sentinels drag down.
    core_row = measured[0] & core[0]
    assert unpreserved[0][core_row].mean() < te_2d[0][core_row].mean()


def test_fill_blends_across_the_transition_zone_without_a_step():
    """Trust in the measurement falls off on the ramp the smoothing already uses."""
    x_cm = np.linspace(-25.0, 25.0, 51)
    z_cm = np.array([470.1, 789.5, 1045.2])
    te_2d = np.broadcast_to(
        8.0 - 0.004 * x_cm ** 2, (z_cm.size, x_cm.size)
    ).copy()
    te_2d[:, np.abs(x_cm) > 12.0] += 3.0  # a hot shoulder only the prior should damp

    filled = fill_te_cycle(te_2d, x_cm, z_cm)
    unpreserved = fill_te_cycle(te_2d, x_cm, z_cm, preserve_measured=False)

    weight = (filled - unpreserved) / np.where(
        te_2d - unpreserved == 0.0, np.nan, te_2d - unpreserved
    )
    inner = np.abs(x_cm) <= 10.0
    outer = np.abs(x_cm) >= 15.0
    middle = (np.abs(x_cm) > 10.0) & (np.abs(x_cm) < 15.0)
    assert np.nanmin(weight[:, inner]) == pytest.approx(1.0)
    assert np.nanmax(weight[:, outer]) == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.nan_to_num(weight[:, middle]) < 1.0)
    assert np.all(np.nan_to_num(weight[:, middle]) >= 0.0)
    # The blend weight is monotone in |x|, so the profile picks up no step.
    order = np.argsort(np.abs(x_cm))
    ramp = np.nan_to_num(weight[0][order], nan=1.0)
    assert np.all(np.diff(ramp) <= 1e-12)


# ---------------------------------------------------------------------------
# The QC gate
# ---------------------------------------------------------------------------
def test_qc_gate_keeps_a_cell_whose_surviving_cycles_are_not_outnumbered():
    n_ok = np.array([[5, 3, 0, 0]])
    n_bad = np.array([[1, 3, 4, 0]])

    surviving = _qc_surviving_cells(n_ok, n_bad)

    # A clear majority survives, a tie survives, and a cell whose cycles all
    # only warned (n_ok = n_bad = 0) survives; only the outnumbered cell goes.
    assert surviving.tolist() == [[True, True, False, True]]


def test_qc_gate_is_the_rule_the_loader_applies_at_every_radius():
    """The gate is one function of the severity counts and knows nothing of x."""
    n_ok = np.array([4, 4])
    n_bad = np.array([9, 9])

    assert not _qc_surviving_cells(n_ok, n_bad).any()


# ---------------------------------------------------------------------------
# The per-port trust model
# ---------------------------------------------------------------------------
def test_adopted_set_ports_take_the_aperture_trust_radius():
    ports = np.array([11, 21, 29, 41, 50])

    radius, blend = _trust_model_for_ports("1", ports)

    assert radius[:4].tolist() == [X_TRUST_APERTURE_CM] * 4
    assert blend[:4].tolist() == [X_TRUST_BLEND_CM] * 4
    assert radius[4] == X_CORE_CM
    assert blend[4] == X_EDGE_CM


def test_the_adoption_table_holds_exactly_the_eight_ruled_set_ports():
    assert set(TRUST_MODEL_BY_SET_PORT) == set(ADOPTED_SET_PORTS)
    assert all(
        value == (X_TRUST_APERTURE_CM, X_TRUST_BLEND_CM)
        for value in TRUST_MODEL_BY_SET_PORT.values()
    )


@pytest.mark.parametrize("es_id", ["3", "4"])
def test_unadopted_experiment_sets_keep_the_historical_core(es_id):
    ports = np.array([11, 21, 29, 41, 50])

    radius, blend = _trust_model_for_ports(es_id, ports)

    assert np.all(radius == X_CORE_CM)
    assert np.all(blend == X_EDGE_CM)


def test_a_blend_radius_inside_the_trusted_radius_is_refused():
    with pytest.raises(ValueError, match="must be outside the trusted radius"):
        _trust_model_for_ports(
            "1", np.array([11]), trust_model={("1", 11): (20.0, 12.0)}
        )


def test_trust_radii_that_do_not_match_the_rows_are_refused():
    x_cm = np.linspace(-25.0, 25.0, 51)
    z_cm = np.array([470.1, 789.5, 1045.2])
    te_2d = np.broadcast_to(8.0 - 0.004 * x_cm**2, (z_cm.size, x_cm.size)).copy()

    with pytest.raises(ValueError, match="do not broadcast"):
        fill_te_cycle(
            te_2d,
            x_cm,
            z_cm,
            trust_radius_cm=np.array([10.0, 10.0]),
            trust_blend_cm=np.array([15.0, 15.0]),
        )


def _blend_weight(te_2d, x_cm, z_cm, **kwargs):
    """Return how much of the measurement survives into the fill, per cell."""
    filled = fill_te_cycle(te_2d, x_cm, z_cm, **kwargs)
    surface = fill_te_cycle(te_2d, x_cm, z_cm, preserve_measured=False, **kwargs)
    denominator = te_2d - surface
    return (filled - surface) / np.where(denominator == 0.0, np.nan, denominator)


def test_the_aperture_blend_runs_from_the_aperture_to_the_column_edge():
    """Trust is total at 18.415 cm, gone at 20.2 cm, and monotone between."""
    x_cm = np.array(
        [0.0, 5.0, 10.0, 14.0, 18.0, X_TRUST_APERTURE_CM, 19.0, 19.5,
         X_TRUST_BLEND_CM, 21.0, 24.0]
    )
    x_cm = np.concatenate([-x_cm[::-1][:-1], x_cm])
    z_cm = np.array([470.1, 789.5, 1045.2])
    te_2d = np.broadcast_to(8.0 - 0.004 * x_cm**2, (z_cm.size, x_cm.size)).copy()
    te_2d[:, np.abs(x_cm) > 12.0] += 3.0

    weight = _blend_weight(
        te_2d,
        x_cm,
        z_cm,
        trust_radius_cm=X_TRUST_APERTURE_CM,
        trust_blend_cm=X_TRUST_BLEND_CM,
    )

    inside = np.abs(x_cm) <= X_TRUST_APERTURE_CM
    outside = np.abs(x_cm) >= X_TRUST_BLEND_CM
    between = ~inside & ~outside
    assert np.nanmin(weight[:, inside]) == pytest.approx(1.0)
    assert np.nanmax(weight[:, outside]) == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.nan_to_num(weight[:, between]) < 1.0)
    assert np.all(np.nan_to_num(weight[:, between]) > 0.0)
    # No step anywhere: the weight falls monotonically with |x|.
    order = np.argsort(np.abs(x_cm))
    ramp = np.nan_to_num(weight[0][order], nan=1.0)
    assert np.all(np.diff(ramp) <= 1e-12)


def test_one_row_adopting_the_aperture_does_not_move_the_others():
    """The trust radius is per row and never enters the RBF, so p50 cannot move."""
    x_cm = np.linspace(-25.0, 25.0, 51)
    z_cm = np.array([470.1, 789.5, 1045.2])
    rng = np.random.default_rng(3)
    te_2d = 8.0 - 0.002 * (z_cm[:, None] - z_cm[0]) - 0.004 * x_cm[None, :] ** 2
    te_2d = te_2d + rng.normal(scale=0.05, size=te_2d.shape)

    historical = fill_te_cycle(te_2d, x_cm, z_cm)
    mixed = fill_te_cycle(
        te_2d,
        x_cm,
        z_cm,
        trust_radius_cm=np.array([X_TRUST_APERTURE_CM, X_CORE_CM, X_CORE_CM]),
        trust_blend_cm=np.array([X_TRUST_BLEND_CM, X_EDGE_CM, X_EDGE_CM]),
    )

    assert np.array_equal(mixed[1:].view(np.uint8), historical[1:].view(np.uint8))
    band = (np.abs(x_cm) > X_CORE_CM) & (np.abs(x_cm) <= X_TRUST_APERTURE_CM)
    assert np.all(mixed[0][band] == pytest.approx(te_2d[0][band]))


def test_the_default_trust_radii_reproduce_the_historical_fill_exactly():
    x_cm = np.linspace(-25.0, 25.0, 51)
    z_cm = np.array([470.1, 789.5, 1045.2])
    rng = np.random.default_rng(7)
    te_2d = 6.0 - 0.003 * x_cm[None, :] ** 2 + rng.normal(
        scale=0.05, size=(z_cm.size, x_cm.size)
    )

    implicit = fill_te_cycle(te_2d, x_cm, z_cm)
    explicit = fill_te_cycle(
        te_2d, x_cm, z_cm, trust_radius_cm=X_CORE_CM, trust_blend_cm=X_EDGE_CM
    )

    assert np.array_equal(implicit.view(np.uint8), explicit.view(np.uint8))


# ---------------------------------------------------------------------------
# Semi-quantitative marking
# ---------------------------------------------------------------------------
def _write_band_csv(path, rows):
    header = (
        "set_id,port,run_id,x_cm,is_x0_control,in_band,dln_te_window,"
        "te_default_med_ev,n_sweeps,at_or_above_criterion\n"
    )
    path.write_text(header + "".join(rows))
    return path


def test_band_spreads_round_trip_from_the_tracked_csv(tmp_path):
    path = _write_band_csv(
        tmp_path / "band.csv",
        [
            "2,50,28,-11,0,1,1.88,1.70,400,1\n",
            "2,50,28,0,1,0,0.603,2.16,400,1\n",
            "1,11,01,11,0,1,0.12,7.0,400,0\n",
        ],
    )

    spreads = load_window_band_spreads(path)

    assert set(spreads) == {("2", 50), ("1", 11)}
    assert spreads[("2", 50)]["x0_dln"] == pytest.approx(0.603)
    assert spreads[("1", 11)]["x0_dln"] != spreads[("1", 11)]["x0_dln"]  # NaN
    assert spreads[("2", 50)]["x_cm"].tolist() == [-11.0, 0.0]


def test_window_spreads_land_on_the_te_grid_cells_they_were_measured_at(tmp_path):
    path = _write_band_csv(
        tmp_path / "band.csv",
        ["1,11,01,11,0,1,0.75,7.0,400,1\n", "1,11,01,0,1,0,0.19,7.0,400,0\n"],
    )
    x_cm = np.array([-11.0, 0.0, 11.0])

    dln, control, source = _window_spread_grids(
        x_cm, np.array([11, 50]), "1", load_window_band_spreads(path)
    )

    assert np.isnan(dln[0, 0])          # x = -11 was not in the CSV
    assert dln[0, 1] == pytest.approx(0.19)
    assert dln[0, 2] == pytest.approx(0.75)
    assert np.all(np.isnan(dln[1]))
    assert control[0] == pytest.approx(0.19)
    assert np.isnan(control[1])
    assert source.tolist() == [CONTROL_SOURCE_BAND, CONTROL_SOURCE_NONE]


def test_a_window_spread_off_the_te_grid_is_refused(tmp_path):
    path = _write_band_csv(
        tmp_path / "band.csv", ["1,11,01,11.5,0,1,0.75,7.0,400,1\n"]
    )
    x_cm = np.array([-11.0, 0.0, 11.0])

    with pytest.raises(ValueError, match="is not on the T_e x grid"):
        _window_spread_grids(
            x_cm, np.array([11]), "1", load_window_band_spreads(path)
        )


def test_semi_quantitative_marks_name_the_reason_and_keep_the_value():
    x_cm = np.array([-11.0, 0.0, 11.0])
    te = np.array([[[5.0], [6.0], [0.4]]])
    dln = np.array([[np.nan, 0.19, 0.75]])
    control = np.array([0.19])

    reason, inflating = _semi_quantitative_marks(
        te, x_cm, dln, control, core_x_cm=10.0
    )

    assert reason[0, 0, 0] == 0
    assert reason[0, 1, 0] == 0
    assert reason[0, 2, 0] == SEMI_QUANT_SUB_EV | SEMI_QUANT_WINDOW
    assert np.isnan(inflating[0, 0])
    assert np.isnan(inflating[0, 1])
    assert inflating[0, 2] == pytest.approx(0.75)


def test_a_failing_x0_control_marks_that_ports_whole_core():
    """The ES2 p50 rider, as uniform machinery rather than a port special case."""
    x_cm = np.array([-11.0, -5.0, 0.0, 5.0, 11.0])
    te = np.full((2, 5, 1), 3.0)
    dln = np.full((2, 5), np.nan)
    dln[:, 2] = [0.603, 0.19]          # p50 x=0 control fails, p41's passes
    control = np.array([0.603, 0.19])

    reason, inflating = _semi_quantitative_marks(
        te, x_cm, dln, control, core_x_cm=10.0
    )

    core = np.abs(x_cm) <= 10.0
    assert np.all(reason[0, core, 0] & SEMI_QUANT_CORE_CONTROL)
    assert not np.any(reason[0, ~core, 0])          # only the core it speaks for
    assert not np.any(reason[1])                    # the passing port is untouched
    assert np.all(inflating[0, core] == pytest.approx(0.603))
    assert np.all(np.isnan(inflating[1]))


def test_a_sub_ev_cell_is_marked_but_carries_no_window_inflation():
    x_cm = np.array([0.0])
    te = np.array([[[0.5, 2.0]]])

    reason, inflating = _semi_quantitative_marks(
        te, x_cm, np.array([[np.nan]]), np.array([np.nan])
    )

    assert reason[0, 0, 0] == SEMI_QUANT_SUB_EV
    assert reason[0, 0, 1] == 0
    assert np.isnan(inflating[0, 0])


def test_the_criterion_is_inclusive_at_its_threshold():
    x_cm = np.array([0.0])
    te = np.full((1, 1, 1), 5.0)
    dln = np.array([[SEMI_QUANTITATIVE_DLN]])

    reason, _ = _semi_quantitative_marks(te, x_cm, dln, np.array([np.nan]))

    assert reason[0, 0, 0] & SEMI_QUANT_WINDOW


def test_the_window_sem_is_the_correlated_core_average_of_the_half_spreads():
    x_cm = np.array([-5.0, 0.0, 5.0, 20.0])
    te = np.array([[[4.0], [2.0], [4.0], [9.0]]])
    inflating = np.array([[0.8, np.nan, 0.8, 3.0]])

    sem = _core_window_sem(te, x_cm, inflating, x_min=-10.0, x_max=10.0)

    # Two of the three core cells are marked; each contributes T sinh(d/2),
    # they add linearly because the window choice is shared, and the sum is
    # divided by the number of core cells the mean is taken over.
    expected = 2.0 * 4.0 * np.sinh(0.4) / 3.0
    assert sem[0, 0] == pytest.approx(expected)


def test_the_window_sem_vanishes_where_nothing_is_marked():
    x_cm = np.array([-5.0, 0.0, 5.0])
    te = np.full((2, 3, 4), 3.0)

    sem = _core_window_sem(
        te, x_cm, np.full((2, 3), np.nan), x_min=-10.0, x_max=10.0
    )

    assert np.all(sem == 0.0)


# ---------------------------------------------------------------------------
# The union of the two window-refit products, and its metric-identity guard
# ---------------------------------------------------------------------------
P_LOW = (3.0, 8.0, 15.0, 25.0, 35.0)
F_HIGH = (0.05, 0.10, 0.15, 0.30, 0.50)


def _grid_with_spread(dln, *, level=4.0):
    """Return a 5 x 5 window grid whose ln(max/min) is exactly *dln*."""
    grid = np.full((5, 5), level)
    grid[0, 0] = level * np.exp(dln)
    return grid


def _write_legacy(path, ports, *, p_low=P_LOW, f_high=F_HIGH,
                  plateau=(10.0, 19.5), recorded=None):
    with h5py.File(path, "w") as hdf:
        hdf.attrs["p_low"] = np.asarray(p_low, dtype=np.float64)
        hdf.attrs["f_high"] = np.asarray(f_high, dtype=np.float64)
        hdf.attrs["plateau_ms"] = np.asarray(plateau, dtype=np.float64)
        for (set_id, port), dln in ports.items():
            grid = _grid_with_spread(dln)
            group = hdf.create_group(f"set{set_id}/port{port}")
            group.attrs["dln_te_window"] = (
                dln if recorded is None else recorded.get((set_id, port), dln)
            )
            group.create_dataset("te_window_ev", data=grid)
    return path


def _write_metadata(path, *, p_low=P_LOW, f_high=F_HIGH, plateau=(10.0, 19.5)):
    path.write_text(json.dumps({
        "window_p_low_percent": list(p_low),
        "window_f_high_fraction": list(f_high),
        "plateau_ms": list(plateau),
    }))
    return path


def _band_spreads(ports):
    return {
        key: {"x_cm": np.empty(0), "dln": np.empty(0), "x0_dln": dln}
        for key, dln in ports.items()
    }


def test_the_union_adds_a_port_the_band_pass_never_covered(tmp_path):
    """ES3 has no band pass, so its x = 0 control can only come from the old product."""
    legacy = _write_legacy(
        tmp_path / "legacy.hdf5", {("1", 11): 0.19, ("3", 41): 2.146}
    )
    metadata = _write_metadata(tmp_path / "meta.json")

    merged = merge_legacy_x0_controls(
        _band_spreads({("1", 11): 0.19}), legacy, metadata
    )

    assert set(merged) == {("1", 11), ("3", 41)}
    assert merged[("3", 41)]["x0_dln"] == pytest.approx(2.146)
    assert merged[("3", 41)]["x0_source"] == CONTROL_SOURCE_LEGACY
    assert merged[("3", 41)]["x_cm"].size == 0      # no per-cell band coverage


def test_a_shared_port_keeps_the_band_products_own_value(tmp_path):
    legacy = _write_legacy(tmp_path / "legacy.hdf5", {("1", 11): 0.19})
    metadata = _write_metadata(tmp_path / "meta.json")
    band = _band_spreads({("1", 11): 0.19})
    band[("1", 11)]["x0_dln"] = 0.19 * (1.0 + 1e-9)   # same value, other platform

    merged = merge_legacy_x0_controls(band, legacy, metadata)

    assert merged[("1", 11)]["x0_dln"] == pytest.approx(0.19 * (1.0 + 1e-9))
    assert merged[("1", 11)]["x0_source"] == CONTROL_SOURCE_BAND


def test_the_guard_refuses_a_different_window_family(tmp_path):
    legacy = _write_legacy(
        tmp_path / "legacy.hdf5", {("1", 11): 0.19}, p_low=(3.0, 8.0, 15.0, 25.0, 40.0)
    )
    metadata = _write_metadata(tmp_path / "meta.json")

    with pytest.raises(ValueError, match="not the same window sensitivity"):
        merge_legacy_x0_controls(_band_spreads({("1", 11): 0.19}), legacy, metadata)


def test_the_guard_refuses_a_different_plateau(tmp_path):
    legacy = _write_legacy(
        tmp_path / "legacy.hdf5", {("1", 11): 0.19}, plateau=(8.0, 19.5)
    )
    metadata = _write_metadata(tmp_path / "meta.json")

    with pytest.raises(ValueError, match="not the same window sensitivity"):
        merge_legacy_x0_controls(_band_spreads({("1", 11): 0.19}), legacy, metadata)


def test_the_guard_recomputes_the_metric_and_refuses_a_relabelled_one(tmp_path):
    """A product whose recorded number is not ln(max/min) of its own grid is refused."""
    legacy = _write_legacy(
        tmp_path / "legacy.hdf5",
        {("1", 11): 0.19},
        recorded={("1", 11): 0.42},
    )
    metadata = _write_metadata(tmp_path / "meta.json")

    with pytest.raises(ValueError, match="not\\s+ln\\(max/min\\) over that grid"):
        merge_legacy_x0_controls(_band_spreads({("1", 11): 0.19}), legacy, metadata)


def test_the_guard_refuses_two_products_that_disagree_on_a_shared_port(tmp_path):
    legacy = _write_legacy(tmp_path / "legacy.hdf5", {("1", 11): 0.19})
    metadata = _write_metadata(tmp_path / "meta.json")

    with pytest.raises(ValueError, match="not measuring the same quantity"):
        merge_legacy_x0_controls(_band_spreads({("1", 11): 0.55}), legacy, metadata)


def test_the_guard_refuses_a_product_it_cannot_check_against_anything(tmp_path):
    legacy = _write_legacy(tmp_path / "legacy.hdf5", {("3", 41): 2.146})
    metadata = _write_metadata(tmp_path / "meta.json")

    with pytest.raises(ValueError, match="shares no set-port"):
        merge_legacy_x0_controls(_band_spreads({("1", 11): 0.19}), legacy, metadata)


def test_the_guard_refuses_a_grid_the_declared_family_does_not_fit(tmp_path):
    legacy = tmp_path / "legacy.hdf5"
    _write_legacy(legacy, {("1", 11): 0.19})
    with h5py.File(legacy, "r+") as hdf:
        del hdf["set1/port11/te_window_ev"]
        hdf["set1/port11"].create_dataset("te_window_ev", data=np.full((4, 5), 4.0))
    metadata = _write_metadata(tmp_path / "meta.json")

    with pytest.raises(ValueError, match="window family implies"):
        merge_legacy_x0_controls(_band_spreads({("1", 11): 0.19}), legacy, metadata)


def test_the_shared_metric_is_the_log_of_the_grid_extremes():
    assert _legacy_dln(_grid_with_spread(0.75)) == pytest.approx(0.75)
    assert np.isnan(_legacy_dln(np.full((5, 5), np.nan)))


def test_a_legacy_only_control_marks_that_ports_core(tmp_path):
    """The ES3 p41/p50 rider: the old product's control marks the core, sem only."""
    legacy = _write_legacy(
        tmp_path / "legacy.hdf5", {("1", 11): 0.19, ("3", 41): 2.146}
    )
    metadata = _write_metadata(tmp_path / "meta.json")
    merged = merge_legacy_x0_controls(
        _band_spreads({("1", 11): 0.19}), legacy, metadata
    )
    x_cm = np.array([-11.0, 0.0, 11.0])
    dln, control, source = _window_spread_grids(x_cm, np.array([41]), "3", merged)
    te = np.full((1, 3, 1), 2.0)

    reason, inflating = _semi_quantitative_marks(
        te, x_cm, dln, control, core_x_cm=10.0
    )

    assert source.tolist() == [CONTROL_SOURCE_LEGACY]
    assert reason[0, 1, 0] & SEMI_QUANT_CORE_CONTROL      # x = 0 is in the core
    assert not reason[0, 0, 0] and not reason[0, 2, 0]    # |x| = 11 is not
    assert inflating[0, 1] == pytest.approx(2.146)


# ---------------------------------------------------------------------------
# The radius-conditional QC floor
# ---------------------------------------------------------------------------
def test_the_floor_only_asks_about_evidence_beyond_the_aperture():
    x_cm = np.array([-20.0, -18.0, 0.0, 18.0, 20.0])
    n_ok = np.full((1, 5, 1), 1, dtype=np.int32)

    allowed = measured_weight_allowed(n_ok, x_cm)

    # Inside 18.415 cm a single surviving cycle is still allowed; outside it is
    # not, and the boundary sits between the 18 and 20 cm samples.
    assert allowed[0, :, 0].tolist() == [False, True, True, True, False]


def test_the_floor_admits_a_far_cell_with_enough_surviving_cycles():
    x_cm = np.array([20.0])
    below = measured_weight_allowed(
        np.full((1, 1, 1), QC_FLOOR_MIN_N_OK - 1, dtype=np.int32), x_cm
    )
    at = measured_weight_allowed(
        np.full((1, 1, 1), QC_FLOOR_MIN_N_OK, dtype=np.int32), x_cm
    )

    assert not below[0, 0, 0]
    assert at[0, 0, 0]


def _floor_case(trust_radius, trust_blend):
    x_cm = np.linspace(-25.0, 25.0, 51)
    z_cm = np.array([470.1, 789.5, 1045.2])
    rng = np.random.default_rng(11)
    te_2d = 8.0 - 0.004 * x_cm[None, :] ** 2 + rng.normal(
        scale=0.05, size=(z_cm.size, x_cm.size)
    )
    gate = np.ones(te_2d.shape, dtype=bool)
    gate[:, np.abs(x_cm) > X_TRUST_APERTURE_CM] = False   # everything far is gated
    kwargs = dict(trust_radius_cm=trust_radius, trust_blend_cm=trust_blend)
    ungated = fill_te_cycle(te_2d, x_cm, z_cm, **kwargs)
    gated = fill_te_cycle(te_2d, x_cm, z_cm, measured_weight_mask=gate, **kwargs)
    return x_cm, ungated, gated


def test_the_floor_has_no_effect_where_the_measurement_has_no_weight_out_there():
    """A port on the historical 10 cm model cannot be moved by the floor."""
    _, ungated, gated = _floor_case(X_CORE_CM, X_EDGE_CM)

    assert np.array_equal(gated.view(np.uint8), ungated.view(np.uint8))


def test_the_floor_acts_only_in_the_blend_at_a_port_that_adopted_the_aperture():
    x_cm, ungated, gated = _floor_case(X_TRUST_APERTURE_CM, X_TRUST_BLEND_CM)

    moved = np.flatnonzero((gated != ungated).any(axis=0))
    assert moved.size > 0
    assert np.all(np.abs(x_cm[moved]) > X_TRUST_APERTURE_CM)
    assert np.all(np.abs(x_cm[moved]) <= X_TRUST_BLEND_CM)


def test_a_measured_weight_mask_of_the_wrong_shape_is_refused():
    x_cm = np.linspace(-25.0, 25.0, 51)
    z_cm = np.array([470.1, 789.5, 1045.2])
    te_2d = np.broadcast_to(8.0 - 0.004 * x_cm**2, (z_cm.size, x_cm.size)).copy()

    with pytest.raises(ValueError, match="measured weight mask shape"):
        fill_te_cycle(
            te_2d, x_cm, z_cm, measured_weight_mask=np.ones((2, 51), dtype=bool)
        )


def test_sparse_rows_are_the_ones_below_the_coverage_threshold():
    grid = np.full((3, 4, 5), np.nan)
    grid[0] = 1.0                      # fully covered
    grid[1, 0, 0] = 1.0                # 1 of 20 cells
    sparse = sparse_z_rows(grid, 0.10)

    assert sparse.tolist() == [False, True, True]
