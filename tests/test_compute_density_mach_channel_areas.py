"""process_run must key each channel's density by the electrode it reached.

``scripts/compute_density_mach.py`` used to hard-code ISAT -> ``ap_R_m2`` and
I_SWEEP -> ``ap_L_m2`` regardless of wiring, even though the Mach-direction
formula next to it already special-cased ``ELECTRICAL_SWAP_RUN_IDS``.  On a
run in that set the ISAT channel actually reads the LEFT electrode and
I_SWEEP the RIGHT one (the connector was crossed), so the two density
products must swap areas too -- through
``density_area_key_for_deadtime_source``, the single owner of the
channel -> electrode map.  A nominal run must see no change.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from bapsf_lapd import ChannelKind, ELECTRICAL_SWAP_RUN_IDS
from bapsf_lapd.config import AcquisitionConfig, ProbeConfig, SweepConfig
from bapsf_lapd.density import electron_density_m3, ion_sound_speed_m_s

from scripts.compute_density_mach import MACH_K, M_I_AMU, X_CM, process_run

N_POS = len(X_CM)

ISAT_A = 2.0e-3
ISWEEP_RAW_A = -3.0e-3  # negated inside process_run to a positive 3.0e-3 A
TE_EV = 5.0

CALIB_M2 = {
    "ap_L_m2": 1.3e-5,
    "ap_R_m2": 2.1e-5,
    "ap_L_cm2": 1.3e-5 * 1e4,
    "ap_R_cm2": 2.1e-5 * 1e4,
    "estimated": False,
}


class _FakeRun:
    """Just enough of ``LapdRun`` for ``process_run``: constant dead-time currents."""

    def __init__(self, run_id: str, isat_a: float, isweep_raw_a: float, rotation_deg: float = 0.0):
        self.config = SimpleNamespace(
            run_id=run_id,
            sweep=SweepConfig(
                ramp_voltage=0.0, tau_ramp_s=1e-4, tau_cycle_s=2e-4, n_cycles=1, t0_s=0.0
            ),
            acquisition=AcquisitionConfig(n_positions=N_POS, n_shots_per_position=1),
            probe=ProbeConfig(id=0, rotation_deg=rotation_deg, port=11, z_cm=0.0),
        )
        self._isat_a = isat_a
        self._isweep_raw_a = isweep_raw_a

    def langmuir_traces(self, channel, sample_slice, zero_offset_v=None):
        del sample_slice, zero_offset_v
        value = self._isat_a if channel == ChannelKind.ISAT else self._isweep_raw_a
        return np.full((N_POS, 1, 4), value)

    def default_zero_offset_v(self, channel):
        del channel
        return 0.0


def _te_grid():
    return np.full((N_POS, 1), TE_EV)


def _cs():
    return ion_sound_speed_m_s(np.full(N_POS, TE_EV), M_I_AMU)


def _run_process(run_id: str, rotation_deg: float = 0.0):
    run = _FakeRun(run_id, ISAT_A, ISWEEP_RAW_A, rotation_deg=rotation_deg)
    return process_run(
        run,
        calib_m2=CALIB_M2,
        probe_id="B",
        te_grid=_te_grid(),
        probe_a_calibration=None,
    )


def test_nominal_run_areas_unchanged():
    """A non-swapped run still divides ISAT by ap_R and I_SWEEP by ap_L."""
    assert "41" not in ELECTRICAL_SWAP_RUN_IDS
    results = _run_process("41", rotation_deg=180.0)

    cs = _cs()
    expected_n_e_R = electron_density_m3(ISAT_A, CALIB_M2["ap_R_m2"], cs)
    expected_n_e_L = electron_density_m3(-ISWEEP_RAW_A, CALIB_M2["ap_L_m2"], cs)

    np.testing.assert_allclose(results["n_e_R_m3"][:, 0], expected_n_e_R)
    np.testing.assert_allclose(results["n_e_L_m3"][:, 0], expected_n_e_L)


def test_run_31_isat_channel_is_divided_by_ap_l():
    """Run 31's ISAT channel read the left electrode -- it must use ap_L, not ap_R."""
    assert "31" in ELECTRICAL_SWAP_RUN_IDS
    results = _run_process("31")

    cs = _cs()
    expected_n_e_R = electron_density_m3(ISAT_A, CALIB_M2["ap_L_m2"], cs)
    expected_n_e_L = electron_density_m3(-ISWEEP_RAW_A, CALIB_M2["ap_R_m2"], cs)

    np.testing.assert_allclose(results["n_e_R_m3"][:, 0], expected_n_e_R)
    np.testing.assert_allclose(results["n_e_L_m3"][:, 0], expected_n_e_L)


def test_mach_shift_on_run_31_is_the_area_ratio_identity():
    """Fixing the areas shifts run 31's Mach by ``2 * ln(A_R / A_L) / K``.

    Before this fix the density areas were hard-coded (ISAT -> ap_R,
    I_SWEEP -> ap_L) even though the Mach-direction formula was already
    swap-aware.  That fed swap-aware Mach (``ln(n_e_R / n_e_L) / K``) the
    WRONG areas; the corrected density feeds it the exchanged ones.  The
    predicted shift is a pure area-ratio effect, independent of the
    currents or T_e used to build the two densities.
    """
    fixed = _run_process("31")

    ratio = CALIB_M2["ap_R_m2"] / CALIB_M2["ap_L_m2"]
    expected_shift = 2.0 * np.log(ratio) / MACH_K

    cs = _cs()
    buggy_n_e_R = electron_density_m3(ISAT_A, CALIB_M2["ap_R_m2"], cs)
    buggy_n_e_L = electron_density_m3(-ISWEEP_RAW_A, CALIB_M2["ap_L_m2"], cs)
    buggy_mach = np.log(buggy_n_e_R / buggy_n_e_L) / MACH_K

    np.testing.assert_allclose(fixed["mach"][:, 0] - buggy_mach, expected_shift)
