"""Pure physics functions for plasma density and Mach number analysis.

Mach probe convention
---------------------
The probe has two planar faces:

* **Left face** (toward cathode / upstream at rot=0) — Isweep channel,
  collection area A_p_L.
* **Right face** (toward anode / downstream at rot=0) — Isat channel,
  collection area A_p_R.

Isweep and Isat have **opposite polarity**.  Ion saturation current from the
Isweep face must be negated before use: ``isat_L = −I_sweep``.

The Mach number uses K in the denominator (Chung convention)::

    M = ln(I_upstream / I_downstream) / MACH_K

where currents are area-normalised (I / A_p).

Shadow offset
-------------
``MACH_SHADOW_OFFSET`` is the apparent Mach number at M_true = 0 caused by
the probe body geometrically shadowing the downstream face along B.  It is
**not** applied to the recorded M_measured; store it for post-hoc correction::

    M_true = M_measured − sign(M_measured) × MACH_SHADOW_OFFSET

Geometry: 2 × (1 × 4 mm²) planes per face, body cross-section ~12 mm²,
area ratio A_body / A_face ≈ 3/2.  At B = 1.4 kG with He-4 and
T_i ~ 1–2 eV, ρ_i ~ 1.5–2 mm ≈ body half-width (2 mm), so the strongly-
magnetised limit used here somewhat overestimates the correction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import tomllib

import numpy as np

from bapsf_lapd.config import AcquisitionConfig, SweepConfig

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_E_C = 1.602176634e-19        # elementary charge, C
_EV_TO_J = _E_C               # 1 eV in joules
_AMU_TO_KG = 1.66053906660e-27  # 1 amu in kg
_HALF_BOHM = math.exp(-0.5)   # exp(-1/2), Bohm-limit prefactor in Isat formula

# ---------------------------------------------------------------------------
# Mach probe calibration constants
# ---------------------------------------------------------------------------

#: Mach probe calibration constant (Chung et al.).
#: Formula: ``M = ln(I_upstream / I_downstream) / MACH_K``
MACH_K: float = 1.66

#: Shadow-correction offset (strongly-magnetised area-ratio limit).
#: ``ln(A_body / A_face) / MACH_K = ln(3/2) / 1.66 ≈ 0.244``
#: **Not applied** to recorded M_measured; see module docstring.
MACH_SHADOW_OFFSET: float = math.log(1.5) / MACH_K   # ≈ 0.244


@dataclass(frozen=True)
class ProbeAAreaCalibration:
    """Canonical empirical Probe A current-to-area normalization."""

    factor: float
    lower_bound: float
    upper_bound: float
    applies_to_both_faces: bool

    @property
    def uncertainty(self) -> float:
        """Half-width of the density-bracket interval."""
        return abs(self.upper_bound - self.lower_bound) / 2.0

    @property
    def relative_uncertainty(self) -> float:
        return self.uncertainty / self.factor


def load_probe_a_area_calibration(path: str | Path) -> ProbeAAreaCalibration:
    """Load and validate the canonical p11/p50 empirical area factor."""
    with open(path, "rb") as stream:
        data = tomllib.load(stream)["probe_A"]
    calibration = ProbeAAreaCalibration(
        factor=float(data["factor"]),
        lower_bound=float(data["lower_bound"]),
        upper_bound=float(data["upper_bound"]),
        applies_to_both_faces=bool(data.get("applies_to_both_faces", False)),
    )
    if (
        calibration.factor <= 0.0
        or calibration.lower_bound <= 0.0
        or calibration.upper_bound <= 0.0
        or calibration.lower_bound > calibration.upper_bound
    ):
        raise ValueError(f"Invalid Probe A area calibration in {path}")
    if not calibration.applies_to_both_faces:
        raise ValueError(
            "The May 2026 Probe A calibration must explicitly apply to both faces"
        )
    return calibration


def apply_probe_a_area_factor(
    current_a: np.ndarray,
    probe_id: str,
    calibration: ProbeAAreaCalibration,
) -> np.ndarray:
    """Return current normalized to the canonical effective Probe A area."""
    scale = calibration.factor if probe_id == "A" else 1.0
    return np.asarray(current_a) * scale


def ion_sound_speed_m_s(te_ev: np.ndarray, m_i_amu: float) -> np.ndarray:
    """Ion sound speed C_s = sqrt(k_B T_e / m_i) in m/s.

    te_ev: electron temperature in eV; NaN entries propagate.
    m_i_amu: ion mass in atomic mass units.
    """
    te_j = np.asarray(te_ev, dtype=np.float64) * _EV_TO_J
    m_i_kg = m_i_amu * _AMU_TO_KG
    return np.sqrt(te_j / m_i_kg)


def electron_density_m3(
    isat_a: np.ndarray,
    probe_area_m2: float,
    cs_m_s: np.ndarray,
) -> np.ndarray:
    """Electron density n_e in m^-3 from the ion saturation current.

    Inverts: I_sat = exp(-1/2) * A_p * e * n_e * C_s
    => n_e = I_sat / (exp(-1/2) * A_p * e * C_s)

    isat_a: ion saturation current in A (positive); NaN/negative → NaN density.
    probe_area_m2: probe collection area in m^2.
    cs_m_s: ion sound speed in m/s; must be broadcastable with isat_a.
    """
    return np.asarray(isat_a) / (_HALF_BOHM * probe_area_m2 * _E_C * np.asarray(cs_m_s))


def inter_sweep_sample_slices(
    sweep: SweepConfig,
    acq: AcquisitionConfig,
    *,
    clip_s: float = 0.0,
) -> list[slice]:
    """Sample slices for the dead-time window after each voltage ramp.

    Slice k spans from the end of ramp k to the start of ramp k+1 (or the
    equivalent endpoint for the last cycle), with optional edge trimming.

    This is the complement of LapdRun.sweep_ramp_sample_slices(): those
    slices cover the voltage ramp; these cover the idle period between ramps
    when both probe faces are biased to collect ion saturation current.
    """
    if clip_s < 0:
        raise ValueError("clip_s must be non-negative")
    dt = acq.sample_dt_s
    clip_samples = int(round(clip_s / dt))
    slices = []
    for k in range(sweep.n_cycles):
        dead_start_s = sweep.t0_s + k * sweep.tau_cycle_s + sweep.tau_ramp_s
        dead_stop_s = sweep.t0_s + (k + 1) * sweep.tau_cycle_s
        start = int(round(dead_start_s / dt)) + clip_samples
        stop = int(round(dead_stop_s / dt)) - clip_samples
        if stop <= start:
            raise ValueError(
                f"inter_sweep_sample_slices: dead-time slice {k} is empty "
                f"({start}:{stop}). The clip window may be too large for the "
                f"dead period ({dead_stop_s - dead_start_s:.1e} s)."
            )
        slices.append(slice(start, stop))
    return slices


def calibrate_probe_area_m2(
    isat_profile_a: np.ndarray,
    cs_profile_m_s: np.ndarray,
    x_m: np.ndarray,
    interf_line_integrated_m2: float,
) -> float:
    """Calibrate probe face area from interferometer line-integrated density.

    Solves A_p so that ∫ n_e(x) dx = interferometer value, where
    n_e = I_sat / (exp(-0.5) * A_p * e * C_s):

      A_p = trapz(I_sat / (e * C_s * exp(-0.5)), x) / interf_line_integrated

    isat_profile_a: (n_positions,) ion saturation current in A (positive).
    cs_profile_m_s: (n_positions,) ion sound speed in m/s.
    x_m: (n_positions,) probe scan positions in metres.
    interf_line_integrated_m2: interferometer ∫ n_e dl in m^-2.

    NaN positions (bad T_e / bad isat) are excluded from the trapz.
    Returns A_p in m^2, or NaN if calibration is not possible.
    """
    isat = np.asarray(isat_profile_a, dtype=np.float64)
    cs = np.asarray(cs_profile_m_s, dtype=np.float64)
    x = np.asarray(x_m, dtype=np.float64)

    integrand = isat / (_E_C * cs * _HALF_BOHM)
    valid = np.isfinite(integrand) & (cs > 0) & (isat > 0)
    if valid.sum() < 2:
        return np.nan

    probe_line_integrated = np.trapezoid(integrand[valid], x[valid])
    if probe_line_integrated <= 0 or not np.isfinite(interf_line_integrated_m2) or interf_line_integrated_m2 <= 0:
        return np.nan
    return float(probe_line_integrated / interf_line_integrated_m2)


def density_fwhm_cm(
    n_e_profile: np.ndarray,
    x_cm: np.ndarray,
) -> float:
    """FWHM of a radial density profile in cm by linear interpolation at half-max.

    Finds the two crossings of n_e = n_max / 2 on either side of the peak.
    Returns NaN if fewer than three finite points, peak is at an edge, or a
    crossing is not found on one side.
    """
    n_e = np.asarray(n_e_profile, dtype=np.float64)
    x = np.asarray(x_cm, dtype=np.float64)

    valid = np.isfinite(n_e)
    if valid.sum() < 3:
        return np.nan

    peak_idx = int(np.nanargmax(n_e))
    if peak_idx == 0 or peak_idx == len(n_e) - 1:
        return np.nan

    half_max = n_e[peak_idx] / 2.0

    def _find_crossing(indices: range) -> float:
        prev_i = peak_idx
        for i in indices:
            if not valid[i]:
                continue
            if n_e[i] <= half_max:
                # Interpolate between i and the last valid point above half_max.
                hi, lo = prev_i, i
                if n_e[hi] <= half_max or n_e[lo] > half_max:
                    return np.nan
                frac = (half_max - n_e[lo]) / (n_e[hi] - n_e[lo])
                return float(x[lo] + frac * (x[hi] - x[lo]))
            prev_i = i
        return np.nan

    x_left = _find_crossing(range(peak_idx - 1, -1, -1))
    x_right = _find_crossing(range(peak_idx + 1, len(n_e)))

    if not np.isfinite(x_left) or not np.isfinite(x_right):
        return np.nan
    return float(x_right - x_left)
