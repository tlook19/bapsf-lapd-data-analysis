import numpy as np
import pytest

from scripts.fit_te_spatial import (
    _enforce_core_mean_monotonic_z,
    _mask_untrusted_te_ports,
    _monotonic_z_bound,
    _robust_current_spike_mask,
    fill_te_cycle,
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


def test_fill_keeps_measured_cells_and_fills_only_the_gaps():
    """The sentinels may shape the gaps; they may not move a measured cell."""
    x_cm = np.linspace(-25.0, 25.0, 11)
    z_cm = np.array([470.1, 789.5, 1045.2])
    rng = np.random.default_rng(0)
    te_2d = 8.0 - 0.002 * (z_cm[:, None] - z_cm[0]) - 0.004 * x_cm[None, :] ** 2
    te_2d = te_2d + rng.normal(scale=0.05, size=te_2d.shape)
    te_2d[1, 4] = np.nan
    te_2d[2, 7] = np.nan
    measured = np.isfinite(te_2d)

    filled = fill_te_cycle(te_2d, x_cm, z_cm)
    unpreserved = fill_te_cycle(te_2d, x_cm, z_cm, preserve_measured=False)

    assert np.all(np.isfinite(filled))
    assert filled[measured] == pytest.approx(te_2d[measured])
    assert filled[~measured] == pytest.approx(unpreserved[~measured])
    # The row nearest the cathode end plate is the one the sentinels drag down.
    assert unpreserved[0][measured[0]].mean() < te_2d[0][measured[0]].mean()
