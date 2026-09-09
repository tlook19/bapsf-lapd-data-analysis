"""Conventions and gate arithmetic of the rot-180 ISAT upstream-row instrument.

Every test here is closed-form or constructed: synthetic profiles whose line
integral and density can be written down, a bin the constructed inputs are
placed inside and outside of by hand, and the two area-key conventions compared
against each other rather than against a run.  Nothing reads a data file, so
these run in a checkout with no products placed.

One test carries a disclosure rather than an assertion of correctness: the area
key this instrument uses for the ISAT face and the key
``density_area_key_for_deadtime_source`` returns for it disagree at ports 29 and
41, and the test pins that disagreement so a later change to either side is
visible instead of silent.
"""

import numpy as np
import pytest

from bapsf_lapd.config import ChannelKind
from bapsf_lapd.corrections import density_area_key_for_deadtime_source
from bapsf_lapd.density import electron_density_m3, ion_sound_speed_m_s

from scripts.es4_upstream_rows_rot180_isat import (
    CORE_CM,
    DEFICIT_BIN,
    FORBIDDEN_OUTPUT_DIR,
    M_I_AMU,
    PLATEAU_MS,
    R_UP_REFERENCE,
    R_UP_REFERENCE_X_CM,
    area_key_for_electrode,
    checked_output_path,
    deficit_gate,
    line_integral_m2,
    plateau_window_mask,
    shape_gate,
    sqrt_te_rescale,
)

# The 20-window, 1 ms inter-sweep dead-time grid both ES4 products carry.
DEADTIME_T_MS = np.arange(0.75, 20.0, 1.0)


def test_area_key_follows_the_collecting_electrode():
    """ISAT collects on the right face and I_SWEEP on the left, always."""
    assert area_key_for_electrode(ChannelKind.ISAT) == "ap_R_cm2"
    assert area_key_for_electrode(ChannelKind.I_SWEEP) == "ap_L_cm2"

    with pytest.raises(ValueError, match="no calibrated face area"):
        area_key_for_electrode(ChannelKind.V_SWEEP)


def test_the_disclosed_divergence_from_the_deadtime_source_helper_is_pinned():
    """The helper agrees at port 11 and disagrees at the ports built here.

    ``density_area_key_for_deadtime_source`` special-cases port 11 with an ISAT
    source and answers ``ap_L_cm2`` everywhere else, so it names the I_SWEEP
    face's area for the ISAT face at ports 21/29/41.  This instrument uses the
    collecting electrode's own area instead.  Pinned, not corrected: changing
    the helper is a separate, registered step.
    """
    assert (
        density_area_key_for_deadtime_source(11, ChannelKind.ISAT)
        == area_key_for_electrode(ChannelKind.ISAT)
        == "ap_R_cm2"
    )
    for port in (21, 29, 41):
        assert density_area_key_for_deadtime_source(port, ChannelKind.ISAT) == "ap_L_cm2"
        assert area_key_for_electrode(ChannelKind.ISAT) == "ap_R_cm2"

    # The I_SWEEP face is the one case the two conventions always agree on.
    for port in (21, 29, 41):
        assert (
            density_area_key_for_deadtime_source(port, ChannelKind.I_SWEEP)
            == area_key_for_electrode(ChannelKind.I_SWEEP)
        )


def test_plateau_window_selects_the_four_scoring_windows():
    """15.0-19.5 ms picks the 15.75-18.75 ms dead-time windows and no other."""
    mask = plateau_window_mask(DEADTIME_T_MS, PLATEAU_MS)

    assert mask.sum() == 4
    assert np.allclose(DEADTIME_T_MS[mask], [15.75, 16.75, 17.75, 18.75])
    # The window is inclusive at both edges and empty off the grid.
    assert plateau_window_mask(np.array([15.0, 19.5]), PLATEAU_MS).all()
    assert not plateau_window_mask(DEADTIME_T_MS, (30.0, 40.0)).any()


def test_density_row_scales_as_one_over_sqrt_te():
    """n_e goes as 1/sqrt(T_e) at fixed current, and the rescale factor agrees."""
    isat_a = np.array([1.0e-3, 2.0e-3, 4.0e-3])
    area_m2 = 5.0e-6
    te_old, te_new = 2.0, 8.0  # a factor of four, so the ratio is exactly 1/2

    n_old = electron_density_m3(isat_a, area_m2, ion_sound_speed_m_s(te_old, M_I_AMU))
    n_new = electron_density_m3(isat_a, area_m2, ion_sound_speed_m_s(te_new, M_I_AMU))

    assert np.allclose(n_new / n_old, 0.5)
    assert sqrt_te_rescale(te_old, te_new) == pytest.approx(0.5)
    # The factor is exactly what carries an already-built row onto the new T_e.
    assert np.allclose(n_old * sqrt_te_rescale(te_old, te_new), n_new)

    for bad in ((0.0, 1.0), (1.0, 0.0), (-1.0, 1.0)):
        with pytest.raises(ValueError, match="T_e must be positive"):
            sqrt_te_rescale(*bad)


def test_line_integral_matches_a_closed_form_profile():
    """A rectangle integrates to its area, and cells are admitted by the
    finite-and-positive rule the repo's own line integral uses."""
    x_cm = np.arange(-25.0, 26.0, 1.0)
    x_m = x_cm / 100.0

    # A flat 1e18 m^-3 slab over |x| <= 10 cm, strictly positive on every cell
    # it occupies: area = 0.20 m * 1e18 = 2e17 m^-2.
    slab = np.where(np.abs(x_cm) <= 10.0, 1.0e18, np.nan)
    assert line_integral_m2(slab, x_m) == pytest.approx(2.0e17, rel=1e-12)

    # The admission rule is finite AND POSITIVE, so a profile that reaches zero
    # at its feet loses those two endpoint trapezoids rather than counting them:
    # a triangle of peak 1e18 and half-width 10 cm has area 1e17, and dropping
    # x = +-10 removes 2 * (0.01 m * 0.1e18 / 2) = 1e15.
    triangle = np.clip(1.0e18 * (1.0 - np.abs(x_cm) / 10.0), 0.0, None)
    assert line_integral_m2(triangle, x_m) == pytest.approx(9.9e16, rel=1e-12)

    # Masking the already-excluded cells to nan changes nothing.
    holed = triangle.copy()
    holed[np.abs(x_cm) >= 10.0] = np.nan
    assert line_integral_m2(holed, x_m) == pytest.approx(9.9e16, rel=1e-12)

    # Too few usable cells is not a number, and never a spurious zero.
    assert np.isnan(line_integral_m2(np.array([np.nan, -1.0, 0.0]), np.arange(3.0)))


def test_deficit_gate_arithmetic_on_a_constructed_pass_and_fail():
    """The chord-over-probe deficit lands in, outside, or refuses its bin."""
    probe = 2.0e18
    inside = probe * 0.5 * (DEFICIT_BIN[0] + DEFICIT_BIN[1])
    verdict, value = deficit_gate(inside, probe)
    assert verdict == "PASS"
    assert value == pytest.approx(0.5 * (DEFICIT_BIN[0] + DEFICIT_BIN[1]))

    # Both edges are inside the bin; just past either edge is not.
    assert deficit_gate(probe * DEFICIT_BIN[0], probe)[0] == "PASS"
    assert deficit_gate(probe * DEFICIT_BIN[1], probe)[0] == "PASS"
    for outside in (DEFICIT_BIN[0] - 0.01, DEFICIT_BIN[1] + 0.01, 5.0):
        verdict, value = deficit_gate(probe * outside, probe)
        assert verdict == "outside the bins"
        assert value == pytest.approx(outside)

    # A gate that cannot be formed refuses; it never reports a nan as a verdict.
    for chord, probe_value in ((np.nan, probe), (1.0e18, 0.0), (-1.0, probe), (1.0e18, np.nan)):
        verdict, value = deficit_gate(chord, probe_value)
        assert verdict == "REFUSED"
        assert np.isnan(value)


def test_shape_gate_arithmetic_on_a_constructed_pass_and_fail():
    """Per-cell agreement with R_up is judged against the pair's spread."""
    # Every value here is exactly representable in binary, so the on-the-edge
    # cell is a real edge test and not a floating-point coin toss.
    r_up = np.array([1.0, 1.5, 2.0, 1.25, 0.75])
    spread = np.full(5, 0.25)

    # Every cell inside its own spread, including exactly on the edge.
    passing = r_up + np.array([0.125, -0.125, 0.0, 0.25, -0.25])
    verdict, in_spreads, abs_dev, n_cells = shape_gate(passing, r_up, spread)
    assert verdict == "PASS"
    assert n_cells == 5
    assert in_spreads == pytest.approx(1.0)
    assert abs_dev == pytest.approx(0.25)

    # One cell past its spread fails the whole profile, and the reported
    # deviation is that cell's.
    failing = r_up.copy()
    failing[2] += 0.625
    verdict, in_spreads, abs_dev, n_cells = shape_gate(failing, r_up, spread)
    assert verdict == "outside the bins"
    assert n_cells == 5
    assert abs_dev == pytest.approx(0.625)
    assert in_spreads == pytest.approx(2.5)

    # A cell with no usable ratio, reference or spread is not counted.
    holed = failing.copy()
    holed[2] = np.nan
    verdict, _, _, n_cells = shape_gate(holed, r_up, spread)
    assert verdict == "PASS"
    assert n_cells == 4

    # No admitted cell refuses rather than reporting a number.
    verdict, in_spreads, abs_dev, n_cells = shape_gate(
        np.full(5, np.nan), r_up, spread
    )
    assert verdict == "REFUSED"
    assert n_cells == 0
    assert np.isnan(in_spreads) and np.isnan(abs_dev)


def test_the_banked_r_up_reference_covers_every_core_cell():
    """The banked profile spans the core band with a finite value in each cell."""
    core = np.abs(R_UP_REFERENCE_X_CM) <= CORE_CM
    assert core.sum() == 21

    for port, reference in R_UP_REFERENCE.items():
        assert reference.shape == R_UP_REFERENCE_X_CM.shape, port
        assert np.isfinite(reference).all(), port
        assert (reference[core] > 0).all(), port


def test_output_inside_the_product_directory_is_refused(tmp_path):
    allowed = tmp_path / "rows.csv"
    assert checked_output_path(allowed) == allowed.resolve()

    forbidden = tmp_path / FORBIDDEN_OUTPUT_DIR / "rows.csv"
    with pytest.raises(ValueError, match=FORBIDDEN_OUTPUT_DIR):
        checked_output_path(forbidden)

    # A path that only reaches the product directory after resolution is caught too.
    (tmp_path / FORBIDDEN_OUTPUT_DIR).mkdir()
    with pytest.raises(ValueError, match=FORBIDDEN_OUTPUT_DIR):
        checked_output_path(
            tmp_path / FORBIDDEN_OUTPUT_DIR / ".." / FORBIDDEN_OUTPUT_DIR / "rows.csv"
        )
