import numpy as np
import pytest

from scripts.plot_isat_profiles import (
    _calibrate_probe_a_density_factor,
    _high_shot_outlier_mask,
)
from scripts.plot_density_profiles import _probe_a_area_relative_uncertainty
from scripts.plot_core_density_temperature_timeseries import CoreStats, _plot_uncertainty


def test_density_bracket_uses_increasing_endpoint_constraints():
    factor, metadata = _calibrate_probe_a_density_factor(
        {
            "01": 4.5,
            "02": 9.0,
            "06": 12.0,
            "08": 5.0,
        }
    )

    assert metadata["factor_lower_bound"] == pytest.approx(2.4)
    assert metadata["factor_upper_bound"] == pytest.approx(2.0)
    assert metadata["exact_bracket"] == 0.0
    assert factor == pytest.approx(np.sqrt(2.4 * 2.0))


def test_density_bracket_exact_interval():
    factor, metadata = _calibrate_probe_a_density_factor(
        {
            "01": 5.0,
            "02": 10.5,
            "06": 10.0,
            "08": 5.0,
        }
    )

    assert metadata["factor_lower_bound"] == pytest.approx(2.0)
    assert metadata["factor_upper_bound"] == pytest.approx(2.1)
    assert metadata["exact_bracket"] == 1.0
    assert 2.0 <= factor <= 2.1


def test_high_current_outlier_is_excluded():
    shot_means = np.full((1, 20, 1), 1.0)
    shot_means[0, -1, 0] = 10.0

    mask = _high_shot_outlier_mask(
        shot_means,
        sigma=3.0,
        ratio=1.5,
        min_shots_used=10,
    )

    assert mask.sum() == 1
    assert mask[0, -1, 0]


def test_probe_a_area_uncertainty_combines_reference_and_factor_terms():
    relative = _probe_a_area_relative_uncertainty(
        area_m2=5.0,
        area_std_m2=0.15,
        factor_relative_uncertainty=0.004,
    )

    assert relative == pytest.approx(np.hypot(0.03, 0.004))


def test_sem_combines_probe_a_calibration_uncertainty_only_in_quadrature():
    stats = CoreStats(
        time_ms=np.array([1.0]),
        z_cm=np.array([100.0]),
        mean=np.array([[10.0]]),
        std=np.array([[3.0]]),
        sem=np.array([[1.0]]),
        count=np.array([[9]]),
        calibration_uncertainty=np.array([[2.0]]),
    )

    np.testing.assert_allclose(_plot_uncertainty(stats, "sem"), [[np.sqrt(5.0)]])
    np.testing.assert_allclose(_plot_uncertainty(stats, "std"), [[3.0]])
