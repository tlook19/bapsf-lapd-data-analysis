import numpy as np
import pytest

from scripts.fit_te_spatial import (
    _enforce_core_mean_monotonic_z,
    _mask_untrusted_te_ports,
    _monotonic_z_bound,
    _robust_current_spike_mask,
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

    bounded, adjusted, max_reduction = _enforce_core_mean_monotonic_z(
        te,
        x_cm,
        te_floor=0.1,
    )

    means = bounded[:, :, 0].mean(axis=1)
    assert np.all(np.diff(means) <= 1e-12)
    assert adjusted == 1
    assert max_reduction > 0.0
    original_ratio = (te[1, 1, 0] - 0.1) / (te[1, 0, 0] - 0.1)
    bounded_ratio = (bounded[1, 1, 0] - 0.1) / (bounded[1, 0, 0] - 0.1)
    assert bounded_ratio == pytest.approx(original_ratio)
