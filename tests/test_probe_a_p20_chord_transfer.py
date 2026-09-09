"""Probe-A chord-transfer factor arithmetic against a closed-form answer.

A flat-top ion-saturation profile at a constant sound speed makes the
calibration integral elementary:

  P = trapz(I0 / [e * C_s * exp(-1/2)], x) = W * I0 / (e * C_s * exp(-1/2))

over a scan of width ``W``.  With a chord line density ``L`` and a declared
transfer ratio ``r`` the chord-implied area is ``P / (r * L)``, so the numbers
below are chosen to make the implied area, and the factor built from it,
exactly representable rather than merely close.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from probe_a_p20_chord_transfer import (
    FORBIDDEN_OUTPUT_DIR,
    build_parser,
    checked_output_path,
    chord_transfer_area_m2,
    probe_a_factor,
)


ELEMENTARY_CHARGE_C = 1.602176634e-19
HALF_BOHM = math.exp(-0.5)

# A flat-top profile over a 0.50 m scan at a constant sound speed.
SCAN_WIDTH_M = 0.50
X_M = np.linspace(-0.25, 0.25, 51)
ISAT_A = 0.02
CS_M_S = 1.5e4
CHORD_LINE_INTEGRATED_M2 = 4.0e18
TRANSFER_RATIO = 1.25


def _closed_form_area_m2(transfer_ratio: float) -> float:
    probe_line_integrated_m2 = (
        SCAN_WIDTH_M * ISAT_A / (ELEMENTARY_CHARGE_C * CS_M_S * HALF_BOHM)
    )
    return probe_line_integrated_m2 / (transfer_ratio * CHORD_LINE_INTEGRATED_M2)


def _flat_profile() -> tuple[np.ndarray, np.ndarray]:
    return np.full_like(X_M, ISAT_A), np.full_like(X_M, CS_M_S)


def test_chord_transfer_area_matches_the_closed_form():
    isat, cs = _flat_profile()
    area = chord_transfer_area_m2(
        isat, cs, X_M, CHORD_LINE_INTEGRATED_M2, TRANSFER_RATIO
    )
    assert area == pytest.approx(_closed_form_area_m2(TRANSFER_RATIO), rel=1e-12)


def test_factor_is_the_reference_area_over_the_implied_area():
    isat, cs = _flat_profile()
    area = chord_transfer_area_m2(
        isat, cs, X_M, CHORD_LINE_INTEGRATED_M2, TRANSFER_RATIO
    )
    # A reference area of exactly twice the implied area must give a factor of 2.
    reference_area_m2 = 2.0 * _closed_form_area_m2(TRANSFER_RATIO)
    assert probe_a_factor(reference_area_m2, area) == pytest.approx(2.0, rel=1e-12)


def test_factor_is_exactly_proportional_to_the_transfer_ratio():
    isat, cs = _flat_profile()
    reference_area_m2 = 3.0e-06
    factors = {}
    for ratio in (0.80, 1.00, 1.25):
        area = chord_transfer_area_m2(isat, cs, X_M, CHORD_LINE_INTEGRATED_M2, ratio)
        factors[ratio] = probe_a_factor(reference_area_m2, area)
    # d ln f / d ln r = 1: the factor scales one-for-one with the model ratio.
    assert factors[1.25] / factors[1.00] == pytest.approx(1.25, rel=1e-12)
    assert factors[0.80] / factors[1.00] == pytest.approx(0.80, rel=1e-12)


def test_a_non_positive_transfer_ratio_is_refused():
    isat, cs = _flat_profile()
    with pytest.raises(ValueError):
        chord_transfer_area_m2(isat, cs, X_M, CHORD_LINE_INTEGRATED_M2, 0.0)


def test_an_unusable_profile_gives_a_nan_factor():
    isat, cs = _flat_profile()
    area = chord_transfer_area_m2(
        np.zeros_like(isat), cs, X_M, CHORD_LINE_INTEGRATED_M2, TRANSFER_RATIO
    )
    assert math.isnan(area)
    assert math.isnan(probe_a_factor(3.0e-06, area))


def test_output_defaults_outside_the_product_directory():
    # A one-shot instrument's default output must not land in processed/,
    # which would put it in the scoring chain unasked.
    args = build_parser().parse_args([])
    assert FORBIDDEN_OUTPUT_DIR not in args.output.parts
    assert checked_output_path(args.output) == args.output.resolve()


def test_output_inside_the_product_directory_is_refused(tmp_path):
    allowed = tmp_path / "probe_a_p20_chord_transfer.csv"
    assert checked_output_path(allowed) == allowed.resolve()

    forbidden = tmp_path / FORBIDDEN_OUTPUT_DIR / "probe_a_p20_chord_transfer.csv"
    with pytest.raises(ValueError, match=FORBIDDEN_OUTPUT_DIR):
        checked_output_path(forbidden)

    (tmp_path / FORBIDDEN_OUTPUT_DIR).mkdir()
    with pytest.raises(ValueError, match=FORBIDDEN_OUTPUT_DIR):
        checked_output_path(tmp_path / FORBIDDEN_OUTPUT_DIR / ".." / FORBIDDEN_OUTPUT_DIR / "f.csv")
