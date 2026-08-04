from pathlib import Path

import numpy as np
import pytest

from bapsf_lapd.density import (
    apply_probe_a_area_factor,
    load_probe_a_area_calibration,
)


CALIBRATION = Path("config/may2026_probe_a_area_calibration.toml")


def test_canonical_probe_a_factor_and_uncertainty():
    calibration = load_probe_a_area_calibration(CALIBRATION)

    assert calibration.factor == pytest.approx(2.03030067440105)
    assert calibration.relative_uncertainty == pytest.approx(0.003421018746225931)
    assert calibration.applies_to_both_faces


def test_probe_a_factor_only_scales_probe_a_current():
    calibration = load_probe_a_area_calibration(CALIBRATION)
    current = np.array([1.0, 2.0])

    assert np.allclose(
        apply_probe_a_area_factor(current, "A", calibration),
        current * calibration.factor,
    )
    assert np.array_equal(
        apply_probe_a_area_factor(current, "B", calibration),
        current,
    )
