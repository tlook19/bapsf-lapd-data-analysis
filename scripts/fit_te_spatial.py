"""Fit T_e spatial profiles with machine-boundary conditions.

Loads the strict-masked T_e(x, z, cycle) grid from langmuir_sweeps.hdf5,
adds boundary sentinel points (T_e = TE_BOUNDARY_EV) at the machine walls
in x and at the cathode/anode ends in z, then fills all NaN cells using a
2-D thin-plate-spline RBF interpolant.

The resulting te_filled array is intended for probe-area calibration and
density analysis where a continuous T_e map is required.

Inside the trusted radius the fill fills gaps: a cell that carries a finite
measurement keeps its measured value, and the surface is used only where there
is no measurement.  The sentinels are enforced near-exactly while data points
carry a finite smoothing, so a fitted surface is otherwise free to sit below a
measured row in order to reach the imposed end-plate temperature, and it does
so by an amount that grows towards the ends -- a z-dependent bias on rows that
have data.  Beyond the trusted radius the scrape-off-layer prior is intended and
is left in place; the two are blended across a ramp.  See ``fill_te_cycle``.

How far that trusted radius reaches is a per-port, measured question, and the
answer is read off ``TRUST_MODEL_BY_SET_PORT``.  At the eight set-ports whose
band-wide fit-window sensitivity was measured and passed
(``scripts/refit_window_band.py``) the measurement is authoritative out to the
cathode frame aperture; everywhere else the historical 10 cm core stands.  See
``_trust_model_for_ports``.

A cell is additionally marked SEMI-QUANTITATIVE where the measurement is at the
diagnostic's limit -- sub-eV ``T_e``, or a fit-window spread at or above
``SEMI_QUANTITATIVE_DLN`` in the band product.  A marked cell KEEPS its measured
value; the marking is an uncertainty statement, not a mask.  See
``_semi_quantitative_marks``.

The core-mean monotonic-z clamp that runs after the fill is likewise judged on
the measurement: it rewrites a measured row only where the MEASURED core means
are themselves non-monotonic, and applies unconditionally only to rows that
have no measured core cells and are therefore reconstructed.  See
``_enforce_core_mean_monotonic_z``.

Boundary conditions
-------------------
  |x| = X_WALL_CM   →  T_e = TE_BOUNDARY_EV  (drift-tube wall)
  z  = Z_CATHODE_CM →  T_e = TE_BOUNDARY_EV  (cathode end plate)
  z  = Z_ANODE_CM   →  T_e = TE_BOUNDARY_EV  (anode end plate)

Coordinates are normalised to [0, 1] before the RBF fit so the
~25-fold difference in x vs z span does not make the kernel anisotropic.

Outputs
-------
processed/te_filled.hdf5
  /experiment_sets/{es_id}/
    attrs : label, v_bank_v
    x_cm            (n_x,)
    z_cm            (n_z,)
    cycle_time_ms   (n_cycles,)
    te_masked       (n_z, n_x, n_cycles)   strict-masked data (NaN where hidden)
    te_filled       (n_z, n_x, n_cycles)   measured cells, RBF fill elsewhere
    core_mean_te_monotonic_clamped (n_z, n_cycles)  where the clamp acted
    core_mean_te_monotonic_scale   (n_z, n_cycles)  factor it applied (1 = none)
    te_trust_radius_cm  (n_z,)   measurement-authoritative |x| for that port
    te_trust_blend_cm   (n_z,)   |x| beyond which the prior is used alone
    te_window_dln       (n_z, n_x)  measured fit-window spread, NaN where none
    te_window_dln_core_control (n_z,)  the port's x = 0 window spread
    te_window_dln_core_control_source (n_z,)  0 none, 1 band, 2 original x=0
    te_window_dln_inflating    (n_z, n_x)  the spread that marked a cell
    te_semi_quantitative       (n_z, n_x, n_cycles)  bool
    te_semi_quantitative_reason(n_z, n_x, n_cycles)  bit 1 sub-eV, 2 window,
                                                     4 core control
    te_semi_quantitative_core_count (n_z, n_cycles)  marked cells in the core
    te_semi_quantitative_band_count (n_z, n_cycles)  marked cells in the band
    te_core_window_sem_ev           (n_z, n_cycles)  window term for the SEM

figures/te_filled_expset{es_id}.png
  Masked vs filled T_e at four representative cycles.

Usage
-----
  MPLCONFIGDIR=.matplotlib ./.venv/bin/python scripts/fit_te_spatial.py
  MPLCONFIGDIR=.matplotlib ./.venv/bin/python scripts/fit_te_spatial.py \\
      --input processed/langmuir_sweeps.hdf5 --output processed/te_filled.hdf5 \\
      --x-wall 35 --z-cathode 100 --z-anode 1900 --te-boundary 0.1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.interpolate import RBFInterpolator

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
HDF5_INPUT  = Path("processed/langmuir_sweeps.hdf5")
HDF5_OUTPUT = Path("processed/te_filled.hdf5")
ISWEEP_DEADTIME_INPUT = Path("processed/isweep_deadtime_profiles.hdf5")
#: Per-cell fit-window spreads across the measurement band, written by
#: ``scripts/refit_window_band.py``.  Tracked, because the trust model below
#: conditions on it and a fresh checkout has to be able to reproduce the fill.
WINDOW_BAND_INPUT = Path("processed/window_refit_band_summary.csv")
#: The band product's protocol record, which carries the window family and the
#: plateau the metric-identity guard checks the older product against.
WINDOW_BAND_METADATA = Path("processed/window_refit_band_metadata.json")
#: The ORIGINAL x = 0 window-refit product (2026-07-22).  It measures the same
#: quantity over the same window family as the band product, but covers
#: experiment sets 1-3 rather than 1-2, so it is the only evidence there is
#: about the x = 0 window control at ES3.  Read for the core-control marking
#: only, and only once the metric-identity guard has passed; gitignored like
#: the other large derived products and regenerated by refit_sweep_windows.py.
LEGACY_WINDOW_REFITS = Path("processed/sweep_window_refits.hdf5")
#: Relative tolerance the two products' x = 0 controls must agree to for the
#: metric-identity guard to accept them as the same quantity.  Exact equality
#: is not available: the older product was computed on macOS/arm64 before the
#: 2026-08-16 desktop migration and the band product on linux-64, and the two
#: ISAs contract floating-point expressions differently.  The measured
#: disagreement over the ten overlapping set-ports is 1.4e-8; 1e-6 is two
#: orders looser than that and many orders tighter than any change of
#: definition could hide under.
WINDOW_METRIC_RTOL = 1.0e-6
OUTPUT_DIR  = Path("figures")

PORT_SPACING_CM = 31.95   # from config.py
PORT_2_Z_CM     = 182.5   # from config.py

TE_BOUNDARY_EV  = 0.1     # T_e at machine boundaries
X_WALL_CM       = 35.0    # x boundary (outside ±25 cm scan)
Z_CATHODE_CM    = PORT_2_Z_CM - 2 * PORT_SPACING_CM   # ≈ 119 cm (before port 1)
Z_ANODE_CM      = PORT_2_Z_CM + 64 * PORT_SPACING_CM  # ≈ 2227 cm (past last port)

# Per-zone RBF smoothing: lower → surface fits that point more exactly.
# Core points (|x| ≤ X_CORE_CM) are trusted most; edge points (|x| ≥ X_EDGE_CM)
# are allowed to deviate more because the signal is noisier there.
# Values between X_CORE_CM and X_EDGE_CM get linearly interpolated smoothing.
X_CORE_CM          = 10.0   # inner boundary of transition zone (cm)
X_EDGE_CM          = 15.0   # outer boundary of transition zone (cm)
SMOOTHING_CORE     = 0.05   # tight fit in plasma core
SMOOTHING_EDGE     = 2.0    # loose fit at plasma edge / scrape-off layer
SMOOTHING_SENTINEL = 1e-4   # near-exact enforcement of wall boundary conditions

# Edge prior: measurements near the scan edge are fluctuation-prone and can
# create unrealistically hot shoulders in the filled profile.  Add low-Te anchor
# points in the SOL so the RBF is pinned in the core but pulled down near edges.
EDGE_ANCHOR_CM        = 22.0
TE_EDGE_ANCHOR_EV     = 0.5
SMOOTHING_EDGE_ANCHOR = 0.02
N_EDGE_ANCHOR_Z       = 40

# Z-row coverage threshold: any z-position whose overall fraction of finite cells
# (summed across all x-positions and cycles) falls below this value is excluded
# from the RBF data entirely.  The RBF then smoothly extrapolates from the last
# reliable z-row toward the anode boundary.  ES4's last port has only ~0.3 %
# coverage (3 cells out of 1020); without this filter those outlier cells corrupt
# the fit.  10 % keeps ES4's penultimate port (32 %) while dropping the last one.
MIN_Z_COVERAGE  = 0.10

# Later low-bank sets reach the lower T_e resolution of the swept analysis at
# downstream ports. Keep only the axial rows previously judged reliable; the
# remaining rows are reconstructed from these anchors and the end boundaries.
TRUSTED_TE_PORTS_BY_ES = {
    "3": (11, 29),
    "4": (11,),
}

# Monotonic-z upper bound.  T_e is expected to decrease monotonically with
# axial distance z (away from the cathode).  Any downstream cell whose Te
# exceeds the minimum finite upstream Te by more than this fractional padding
# is masked before the RBF fill.  25 % gives room for shot-to-shot
# fluctuations while still capping grossly inflated downstream estimates.
MONO_Z_PADDING  = 0.25

# Core hot spots can be real when a high T_e cell coincides with an upstream
# -I_SWEEP dead-time current spike.  Preserve those core cells from the
# monotonic-z outlier mask; edge spikes remain masked because they are usually
# edge instability/fluctuation artifacts.
CORE_HOTSPOT_CURRENT_SIGMA = 3.0
CORE_HOTSPOT_CURRENT_RATIO = 1.25

# ---------------------------------------------------------------------------
# Trust model: how far out the measurement is authoritative, per set-port
# ---------------------------------------------------------------------------
#: Outer radius of the measurement-authoritative band at an adopting port: half
#: the caliper-measured 14.5 in cathode frame aperture (36.830 cm diameter,
#: 2026-08-17).  Field lines inside the aperture-mapped radius connect to the
#: source, so this is the radius across which T_e has a reason to be flat, and
#: it is the same radius the flux-tube average in the overlay export uses.
X_TRUST_APERTURE_CM = 18.415
#: Outer end of the blend at an adopting port.  The measured column edge is
#: bracketed at 18.9--20.2 cm; taking the far end as the point where the prior
#: stands alone is a stated CONVENTION, not a measurement.
X_TRUST_BLEND_CM = 20.2

#: The measurement-authoritative radius and the outer end of the blend, keyed by
#: (experiment set id, port).  A set-port that is not listed keeps the
#: historical (X_CORE_CM, X_EDGE_CM) model, so the table is a declared adoption
#: list rather than a set of scattered conditionals and an unlisted port cannot
#: change behaviour by accident.
#:
#: The eight listed set-ports are the ones whose in-band per-cell fit-window
#: spread was measured across the band and passed the pre-declared criterion
#: (median dln_te_window < SEMI_QUANTITATIVE_DLN; medians 0.107--0.288).  Both
#: p50 rows FAILED it (1.056 and 2.688) and keep the 10 cm model; experiment
#: sets 3 and 4 were never measured across the band and keep it as well.
TRUST_MODEL_BY_SET_PORT = {
    ("1", 11): (X_TRUST_APERTURE_CM, X_TRUST_BLEND_CM),
    ("1", 21): (X_TRUST_APERTURE_CM, X_TRUST_BLEND_CM),
    ("1", 29): (X_TRUST_APERTURE_CM, X_TRUST_BLEND_CM),
    ("1", 41): (X_TRUST_APERTURE_CM, X_TRUST_BLEND_CM),
    ("2", 11): (X_TRUST_APERTURE_CM, X_TRUST_BLEND_CM),
    ("2", 21): (X_TRUST_APERTURE_CM, X_TRUST_BLEND_CM),
    ("2", 29): (X_TRUST_APERTURE_CM, X_TRUST_BLEND_CM),
    ("2", 41): (X_TRUST_APERTURE_CM, X_TRUST_BLEND_CM),
}

# ---------------------------------------------------------------------------
# Semi-quantitative marking
# ---------------------------------------------------------------------------
#: A filled T_e below this is at the swept diagnostic's low-temperature limit
#: and is semi-quantitative by the standing convention.
SEMI_QUANTITATIVE_TE_EV = 1.0
#: A measured fit-window spread ln(Te_max / Te_min) at or above this makes the
#: cell semi-quantitative.  It is the same threshold as the band product's
#: per-port pass criterion, applied cell by cell.
SEMI_QUANTITATIVE_DLN = 0.50

#: Bit flags recorded in ``te_semi_quantitative_reason``.
SEMI_QUANT_SUB_EV = 1
SEMI_QUANT_WINDOW = 2
SEMI_QUANT_CORE_CONTROL = 4

#: What the trust model is, stated once so the product carries it verbatim.
TRUST_MODEL_STATEMENT = (
    "The filled T_e reports the MEASUREMENT out to te_trust_radius_cm, blends "
    "measurement into the scrape-off-layer prior linearly from there to "
    "te_trust_blend_cm, and reports the prior alone beyond it; a cell with no "
    "surviving measurement takes the fitted surface at every radius.  The "
    f"radius is {X_TRUST_APERTURE_CM:g} cm -- half the caliper-measured 14.5 in "
    f"cathode frame aperture -- with the blend closing at {X_TRUST_BLEND_CM:g} "
    "cm at the set-ports listed in te_trust_adopted_ports, and the historical "
    "10 cm core elsewhere.  The adoption list is per PORT, not per radius and "
    "not inherited across a set: it is the eight set-ports whose per-cell "
    "fit-window spread was measured across the band and passed the "
    "pre-declared criterion.  The outer blend radius is the far end of the "
    "18.9-20.2 cm measured column-edge bracket and is a stated CONVENTION.  "
    "The per-point RBF smoothing zone is unchanged by the adoption, so the "
    "fitted surface is the same one an unadopted port would have seen."
)

#: What the semi-quantitative class is, stated once for the same reason.
SEMI_QUANTITATIVE_STATEMENT = (
    "A cell is semi-quantitative when its filled T_e is below "
    f"{SEMI_QUANTITATIVE_TE_EV:g} eV (the swept diagnostic's low-temperature "
    "limit), OR its own measured fit-window spread ln(Te_max/Te_min) is at or "
    f"above {SEMI_QUANTITATIVE_DLN:g}, OR its port's x = 0 window control is, "
    "in which case that port's whole core is marked.  The control is read from "
    "the UNION of the two window-refit products -- the band product and the "
    "original x = 0 product, which between them cover experiment sets 1-3 -- "
    "so a port is marked wherever its control fails and not according to which "
    "product happened to measure it; the two are proved to be the same "
    "quantity before the union is taken, and the source is recorded per port "
    "in te_window_dln_core_control_source.  te_semi_quantitative_"
    "reason records which, as bits 1, 2 and 4.  A marked cell KEEPS its "
    "measured value: the class is an uncertainty statement and NOT a mask -- "
    "the defect being corrected was a prior overriding measurements.  One rule "
    "everywhere, no port special cases and no second trust radius."
)

#: What the window term added to the T_e uncertainty is.
CORE_WINDOW_SEM_STATEMENT = (
    "te_core_window_sem_ev is the core-band T_e uncertainty contributed by the "
    "fit-window convention, in eV.  A cell whose measured spread is d = "
    "ln(Te_max/Te_min) has half-spread T * sinh(d/2) about the window family's "
    "geometric centre; the window convention is one choice applied to every "
    "cell at once, so those half-spreads are fully correlated and are averaged "
    "(not added in quadrature) over the core band, with an unmarked cell "
    "contributing zero.  A cell marked only for being sub-eV contributes "
    "nothing here: it has no measured window spread, and its rider is the "
    "standing sub-eV one.  Combine with the radial SEM in quadrature -- the "
    "two are independent."
)

CMAP            = "plasma"


# ---------------------------------------------------------------------------
# Data loading and the QC gate
# ---------------------------------------------------------------------------
#: The QC rule, stated once so the product can carry it verbatim.
QC_RULE = (
    "a cell survives when the surviving-cycle count is at least the failed "
    "count, n_ok >= n_bad, in the contributing rot-0 run; a cell excluded in "
    "every contributing run is NaN.  Applied at ALL radii, so the band the "
    "trust model now treats as authoritative is gated by the same rule as the "
    "core rather than by inspection.  n_ok and n_bad are the per-cell severity "
    "counts from evaluate_langmuir_quality; cells where every cycle only "
    "warned (n_ok = n_bad = 0) survive, as do ties."
)


def _qc_surviving_cells(n_ok: np.ndarray, n_bad: np.ndarray) -> np.ndarray:
    """Return the boolean mask of cells that pass QC.

    The single definition of which cells the fill is allowed to read, applied
    at every radius.  A cell passes when its surviving-cycle count is at least
    its failed count; see ``QC_RULE`` for the exact statement, including what
    happens to ties and to cells whose cycles only warned.
    """
    return np.asarray(n_ok) >= np.asarray(n_bad)


def _load_experiment_set(hf: h5py.File, es_id: str) -> dict:
    """Load QC-masked T_e(n_z, n_x, n_cycles) for one experiment set.

    Only rot=0 runs are used.  Probe shadowing makes rot=180 measurements
    unreliable for T_e (the downstream face sits in the probe's own shadow).
    The rotation_deg attribute reflects any known swap corrections (e.g. ES3
    p21 runs 32/33).

    Which cells survive is ``_qc_surviving_cells``; the rule is ``QC_RULE``.
    """
    x_cm = hf["x_cm"][:]
    eg   = hf["experiment_sets"][es_id]

    z_groups: dict[float, list] = {}
    port_by_z: dict[float, int] = {}
    cycle_time_s = None

    for run_id in sorted(eg.keys()):
        rg  = eg[run_id]
        rot = float(rg.attrs.get("rotation_deg", 0))
        if rot != 0.0:
            continue                              # skip rot=180 runs
        z  = float(rg.attrs["z_cm"])
        port = int(rg.attrs["port"])
        if z in port_by_z and port_by_z[z] != port:
            raise ValueError(f"Inconsistent port metadata at z={z:g} cm")
        port_by_z[z] = port
        if z not in z_groups:
            z_groups[z] = []
        te    = rg["te_best_ev"][:]
        n_ok  = rg["n_ok"][:]
        n_bad = rg["n_bad"][:]
        z_groups[z].append((te, n_ok, n_bad))
        if cycle_time_s is None:
            cycle_time_s = rg["cycle_time_s"][:]

    z_vals   = sorted(z_groups.keys())
    n_z      = len(z_vals)
    n_x      = len(x_cm)
    n_cycles = len(cycle_time_s)
    te_grid  = np.full((n_z, n_x, n_cycles), np.nan)
    n_qc_excluded = 0

    for zi, z in enumerate(z_vals):
        te_sum = np.zeros((n_x, n_cycles))
        weight = np.zeros((n_x, n_cycles))
        mask_bad = np.zeros((n_x, n_cycles), dtype=bool)

        for te, n_ok, n_bad in z_groups[z]:
            cell_bad = ~_qc_surviving_cells(n_ok, n_bad)
            n_qc_excluded += int(cell_bad.sum())
            mask_bad |= cell_bad
            valid     = ~cell_bad
            te_sum   += np.where(valid, te, 0.0)
            weight   += valid.astype(float)

        with np.errstate(invalid="ignore"):
            te_avg = np.where(weight > 0, te_sum / weight, np.nan)
        all_bad = mask_bad & (weight == 0)
        te_avg[all_bad] = np.nan
        te_grid[zi] = te_avg

    return {
        "te":             te_grid,
        "x_cm":           x_cm,
        "z_cm":           np.array(z_vals),
        "port":           np.array([port_by_z[z] for z in z_vals], dtype=np.int16),
        "cycle_time_ms":  cycle_time_s * 1e3,
        "es_label":       eg.attrs.get("label", f"set {es_id}"),
        "v_bank":         eg.attrs.get("v_bank_v", "?"),
        "es_id":          es_id,
        "n_qc_excluded":  n_qc_excluded,
    }


def _load_filled_source_experiment_set(hf: h5py.File, es_id: str) -> dict:
    """Load te_masked from a previous te_filled.hdf5-style product."""
    grp = hf["experiment_sets"][es_id]
    z_cm = grp["z_cm"][()]
    ports = (
        grp["port"][()]
        if "port" in grp
        else np.rint((z_cm - PORT_2_Z_CM) / PORT_SPACING_CM + 2).astype(np.int16)
    )
    return {
        "te":             grp["te_masked"][()],
        "x_cm":           grp["x_cm"][()],
        "z_cm":           z_cm,
        "port":           ports,
        "cycle_time_ms":  grp["cycle_time_ms"][()],
        "es_label":       grp.attrs.get("label", f"set {es_id}"),
        "v_bank":         grp.attrs.get("v_bank_v", "?"),
        "es_id":          es_id,
        "n_qc_excluded":  -1,   # the QC gate ran when the source was built
    }


def _load_input_experiment_set(hf: h5py.File, es_id: str, input_mode: str) -> dict:
    if input_mode == "filled":
        return _load_filled_source_experiment_set(hf, es_id)
    if input_mode == "langmuir":
        return _load_experiment_set(hf, es_id)
    grp = hf["experiment_sets"][es_id]
    if "te_masked" in grp:
        return _load_filled_source_experiment_set(hf, es_id)
    return _load_experiment_set(hf, es_id)


# ---------------------------------------------------------------------------
# 2-D RBF fill
# ---------------------------------------------------------------------------
def _build_sentinel_points(
    x_cm: np.ndarray,
    z_cm: np.ndarray,
    *,
    x_wall: float,
    z_lo: float,
    z_hi: float,
    te_boundary: float,
    n_x_wall: int = 30,
    n_z_end: int  = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (sentinel_coords, sentinel_values) arrays.

    Sentinel points are placed:
      - Along x = ±x_wall from z_lo to z_hi  (drift-tube wall)
      - Along z = z_lo and z = z_hi from -x_wall to +x_wall (end plates)
    All carry T_e = te_boundary.
    """
    z_wall = np.linspace(z_lo, z_hi, n_x_wall)
    x_ends = np.linspace(-x_wall, x_wall, n_z_end)

    coords = []
    # x walls
    for zw in z_wall:
        coords.append((-x_wall, zw))
        coords.append(( x_wall, zw))
    # z end plates
    for xe in x_ends:
        coords.append((xe, z_lo))
        coords.append((xe, z_hi))

    coords  = np.array(coords)          # (N, 2)  columns: [x, z]
    values  = np.full(len(coords), te_boundary)
    return coords, values


def _build_edge_anchor_points(
    *,
    x_anchor: float,
    z_cm: np.ndarray,
    z_lo: float,
    z_hi: float,
    te_edge: float,
    n_z: int = N_EDGE_ANCHOR_Z,
) -> tuple[np.ndarray, np.ndarray]:
    """Return low-Te SOL anchor points inside the measured scan envelope."""
    z_min = max(float(np.nanmin(z_cm)), z_lo)
    z_max = min(float(np.nanmax(z_cm)), z_hi)
    z_vals = np.linspace(z_min, z_max, n_z)
    coords = []
    for z in z_vals:
        coords.append((-x_anchor, z))
        coords.append((x_anchor, z))
    return np.array(coords), np.full(len(coords), te_edge)


def _normalise(x: np.ndarray, z: np.ndarray,
               x_wall: float, z_lo: float, z_hi: float
               ) -> np.ndarray:
    """Stack and normalise (x, z) to the unit square."""
    x_n = (x + x_wall) / (2 * x_wall)
    z_n = (z - z_lo)   / (z_hi - z_lo)
    return np.column_stack([x_n, z_n])


def _core_transition(
    x_pts: np.ndarray,
    x_core: float,
    x_edge: float,
) -> np.ndarray:
    """Return 0 inside the trusted radius, 1 at and beyond the outer end.

    The single definition of how far a radius sits from a trusted region into
    the noisy scrape-off layer, used for two separate ramps that are read off
    it with DIFFERENT radii:

    * ``_point_smoothing`` asks how tightly the RBF should fit a data point.
      That is a statement about the surface and about how noisy a point is as
      an interpolation constraint, and it runs on the fixed ``X_CORE_CM`` to
      ``X_EDGE_CM`` zone at every port.
    * ``fill_te_cycle`` asks how far a MEASURED value is carried into the
      filled product before the scrape-off-layer prior takes over.  That is a
      statement about which of the two the product should report where they
      disagree, it is answered per port by ``TRUST_MODEL_BY_SET_PORT``, and it
      runs on that port's own radii.

    The two were one ramp until the aperture trust radius was adopted, on the
    argument that they could not then drift apart.  They are separated here
    because they answer different questions and the evidence that moved one
    -- a measured, per-port, band-wide fit-window sensitivity -- says nothing
    about the other.  Keeping the smoothing ramp fixed also keeps the fitted
    surface itself untouched, so a port that did not adopt is not moved by a
    neighbour that did.
    """
    return np.clip((np.abs(x_pts) - x_core) / (x_edge - x_core), 0.0, 1.0)


def _trust_model_for_ports(
    es_id: str,
    ports: np.ndarray,
    *,
    x_core: float = X_CORE_CM,
    x_edge: float = X_EDGE_CM,
    trust_model: dict[tuple[str, int], tuple[float, float]] = TRUST_MODEL_BY_SET_PORT,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the per-z-row trusted radius and outer blend radius, in cm.

    A ``(set, port)`` listed in *trust_model* takes its declared pair; every
    other row falls back to ``(x_core, x_edge)``.  The lookup is by port rather
    than by z so the adoption list reads as the ruling does.
    """
    radii = np.empty(len(ports), dtype=np.float64)
    blends = np.empty(len(ports), dtype=np.float64)
    for index, port in enumerate(ports):
        radius, blend = trust_model.get((str(es_id), int(port)), (x_core, x_edge))
        if not blend > radius:
            raise ValueError(
                f"ES {es_id} p{int(port)}: trust blend radius {blend:g} cm must "
                f"be outside the trusted radius {radius:g} cm"
            )
        radii[index] = radius
        blends[index] = blend
    return radii, blends


def _point_smoothing(
    x_pts: np.ndarray,
    *,
    x_core:          float = X_CORE_CM,
    x_edge:          float = X_EDGE_CM,
    smoothing_core:  float = SMOOTHING_CORE,
    smoothing_edge:  float = SMOOTHING_EDGE,
) -> np.ndarray:
    """Per-point smoothing that increases from core to edge.

    |x| ≤ x_core            → smoothing_core  (tight: surface must fit closely)
    x_core < |x| < x_edge   → linearly interpolated
    |x| ≥ x_edge            → smoothing_edge  (loose: noisier edge data)
    """
    t = _core_transition(x_pts, x_core, x_edge)
    return smoothing_core + t * (smoothing_edge - smoothing_core)


def fill_te_cycle(
    te_2d: np.ndarray,          # (n_z, n_x)  NaN where masked
    x_cm: np.ndarray,
    z_cm: np.ndarray,
    *,
    x_wall:          float = X_WALL_CM,
    z_lo:            float = Z_CATHODE_CM,
    z_hi:            float = Z_ANODE_CM,
    te_boundary:     float = TE_BOUNDARY_EV,
    x_core:          float = X_CORE_CM,
    x_edge:          float = X_EDGE_CM,
    smoothing_core:  float = SMOOTHING_CORE,
    smoothing_edge:  float = SMOOTHING_EDGE,
    edge_anchor:     float = EDGE_ANCHOR_CM,
    te_edge_anchor:  float = TE_EDGE_ANCHOR_EV,
    smoothing_edge_anchor: float = SMOOTHING_EDGE_ANCHOR,
    preserve_measured: bool = True,
    trust_radius_cm: np.ndarray | float | None = None,
    trust_blend_cm: np.ndarray | float | None = None,
) -> np.ndarray:
    """Return filled (n_z, n_x) T_e array with no NaN cells.

    Uses a thin-plate-spline RBF with position-dependent smoothing:
      |x| ≤ x_core  → smoothing_core (fit closely; trusted core data)
      |x| ≥ x_edge  → smoothing_edge (fit loosely; noisy edge data)
    Sentinel boundary points use smoothing=0 (exact enforcement).

    With *preserve_measured* (the default) a measured cell inside the trusted
    radius keeps its measured value and the surface is used only where there is
    no measurement; from there out to the blend radius the two are mixed on a
    linear ramp, so beyond it the scrape-off-layer prior (the edge anchors and
    ``smoothing_edge``) is left entirely intact and there is no step anywhere in
    the profile.

    *trust_radius_cm* and *trust_blend_cm* are that ramp's two radii.  Each may
    be a scalar or one value per z-row, which is how a port that adopted the
    aperture trust radius sits in the same fit as one that did not; they default
    to ``x_core`` and ``x_edge``, the historical model.  They do NOT enter the
    RBF: the per-point smoothing keeps its own fixed zone, so changing a port's
    trusted radius does not move the fitted surface and cannot move a
    neighbouring port.  See ``_core_transition``.

    This is what the boundary sentinels make necessary: they are enforced
    near-exactly (``SMOOTHING_SENTINEL``) while data points carry
    ``smoothing_core``, so the fitted surface is free to sit below a measured
    row that lies near an end plate in order to reach the imposed T_e there.
    The effect falls off with distance from the boundary and is therefore a
    z-dependent bias, not a uniform offset: it does not cancel in a ratio or a
    difference between rows, and it can invert the ordering of two rows whose
    measured separation is smaller than the bias.

    The blend is deliberately not carried out to the wall.  Outside the core
    the fill is not merely interpolating: the edge anchors and the loose edge
    smoothing are a prior that exists precisely because scan-edge measurements
    are fluctuation-prone, and it holds the surface 1-2 eV below the measured
    cells at EVERY axial row, including rows the sentinels barely touch.  That
    is a separate, intended choice and is left alone here.

    Setting *preserve_measured* to False restores the surface-everywhere
    behaviour for comparison.
    """
    zz, xx = np.meshgrid(z_cm, x_cm, indexing="ij")  # (n_z, n_x) each
    valid = np.isfinite(te_2d)

    radius = np.reshape(
        np.asarray(x_core if trust_radius_cm is None else trust_radius_cm, dtype=float),
        (-1, 1),
    )
    blend = np.reshape(
        np.asarray(x_edge if trust_blend_cm is None else trust_blend_cm, dtype=float),
        (-1, 1),
    )
    if radius.shape[0] not in (1, te_2d.shape[0]) or blend.shape != radius.shape:
        raise ValueError(
            f"trust radii shaped {radius.shape}/{blend.shape} do not broadcast "
            f"onto the {te_2d.shape[0]} z-rows of the T_e grid"
        )

    # Data coordinates and values
    data_coords = np.column_stack([xx[valid], zz[valid]])   # [x, z]
    data_values = te_2d[valid]

    if data_values.size < 4:
        flat = np.full_like(te_2d, te_boundary)
        if preserve_measured:
            trust = 1.0 - _core_transition(xx, radius, blend)
            flat = np.where(valid, trust * te_2d + (1.0 - trust) * flat, flat)
        return np.clip(flat, te_boundary, None)

    # Per-point smoothing for data points
    data_smoothing = _point_smoothing(
        data_coords[:, 0],
        x_core=x_core, x_edge=x_edge,
        smoothing_core=smoothing_core, smoothing_edge=smoothing_edge,
    )

    # Sentinel coordinates, values, and smoothing (exact = 0)
    sent_coords, sent_values = _build_sentinel_points(
        x_cm, z_cm, x_wall=x_wall, z_lo=z_lo, z_hi=z_hi, te_boundary=te_boundary
    )
    sent_smoothing = np.full(len(sent_values), SMOOTHING_SENTINEL)

    edge_coords, edge_values = _build_edge_anchor_points(
        x_anchor=edge_anchor,
        z_cm=z_cm,
        z_lo=z_lo,
        z_hi=z_hi,
        te_edge=te_edge_anchor,
    )
    edge_smoothing = np.full(len(edge_values), smoothing_edge_anchor)

    all_coords    = np.vstack([data_coords,    edge_coords,    sent_coords])
    all_values    = np.concatenate([data_values,    edge_values,    sent_values])
    all_smoothing = np.concatenate([data_smoothing, edge_smoothing, sent_smoothing])

    # Normalise to unit square (avoids ~25× z/x aspect ratio skewing kernel)
    all_norm = _normalise(all_coords[:, 0], all_coords[:, 1], x_wall, z_lo, z_hi)

    rbf = RBFInterpolator(
        all_norm, all_values,
        kernel="thin_plate_spline",
        smoothing=all_smoothing,
    )

    # Evaluate on full grid
    grid_norm = _normalise(xx.ravel(), zz.ravel(), x_wall, z_lo, z_hi)
    te_out = rbf(grid_norm).reshape(te_2d.shape)
    if preserve_measured:
        trust = 1.0 - _core_transition(xx, radius, blend)
        te_out = np.where(valid, trust * te_2d + (1.0 - trust) * te_out, te_out)
    return np.clip(te_out, te_boundary, None)


def fill_te_grid(
    te_grid: np.ndarray,        # (n_z, n_x, n_cycles)
    x_cm: np.ndarray,
    z_cm: np.ndarray,
    *,
    min_z_coverage: float = MIN_Z_COVERAGE,
    **kwargs,                   # forwarded to fill_te_cycle
) -> np.ndarray:
    """Fill every cycle in the grid; prints progress.

    Before fitting, any z-row whose fraction of finite cells (pooled over all
    x-positions and cycles) is below *min_z_coverage* is blanked to NaN in
    every cycle.  This prevents sparse, noisy outlier cells at poorly-sampled
    ports from corrupting the RBF interpolant.  The blank rows are then filled
    by smooth extrapolation from neighbouring valid z-rows and the boundary
    sentinel points (T_e = te_boundary at the cathode/anode end plates).
    """
    n_z = te_grid.shape[0]
    z_coverage = np.array([np.isfinite(te_grid[zi]).mean() for zi in range(n_z)])
    sparse = z_coverage < min_z_coverage
    if sparse.any():
        te_grid = te_grid.copy()
        te_grid[sparse] = np.nan
        for zi in np.flatnonzero(sparse):
            print(
                f"    z = {z_cm[zi]:.1f} cm excluded: coverage "
                f"{100 * z_coverage[zi]:.1f}% < {100 * min_z_coverage:.0f}% threshold"
            )

    n_cycles = te_grid.shape[2]
    filled = np.empty_like(te_grid)
    for ci in range(n_cycles):
        filled[:, :, ci] = fill_te_cycle(te_grid[:, :, ci], x_cm, z_cm, **kwargs)
        pct = 100 * (ci + 1) / n_cycles
        sys.stdout.write(f"\r    cycle {ci + 1}/{n_cycles} ({pct:.0f}%)")
        sys.stdout.flush()
    sys.stdout.write("\n")
    return filled


def _mask_untrusted_te_ports(
    te_grid: np.ndarray,
    ports: np.ndarray,
    experiment_set_id: str,
    *,
    trusted_ports_by_es: dict[str, tuple[int, ...]] = TRUSTED_TE_PORTS_BY_ES,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Blank later-set port rows whose swept T_e is not diagnostically reliable."""
    trusted = trusted_ports_by_es.get(str(experiment_set_id))
    if trusted is None:
        return te_grid.copy(), ()
    ports = np.asarray(ports, dtype=int)
    untrusted = tuple(int(port) for port in ports if int(port) not in trusted)
    out = te_grid.copy()
    out[~np.isin(ports, trusted)] = np.nan
    return out, untrusted


# ---------------------------------------------------------------------------
# Semi-quantitative marking
# ---------------------------------------------------------------------------
def load_window_band_spreads(path: Path) -> dict[tuple[str, int], dict]:
    """Return the per-cell fit-window spreads, keyed by (set id, port).

    Reads the tracked per-cell CSV written by ``scripts/refit_window_band.py``.
    Each entry carries the cells that were re-fitted (``x_cm`` and their
    ``dln_te_window``) and the port's ``x = 0`` control value, which is the
    same quantity measured at the one radius the published x = 0 product also
    covers.  A set-port the band pass never covered is simply absent, and the
    caller then has no window evidence for it either way.
    """
    spreads: dict[tuple[str, int], dict] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (str(int(row["set_id"])), int(row["port"]))
            entry = spreads.setdefault(
                key, {"x_cm": [], "dln": [], "x0_dln": np.nan}
            )
            dln = float(row["dln_te_window"])
            x_cm = float(row["x_cm"])
            entry["x_cm"].append(x_cm)
            entry["dln"].append(dln)
            if int(row["is_x0_control"]):
                entry["x0_dln"] = dln
    for entry in spreads.values():
        entry["x_cm"] = np.asarray(entry["x_cm"], dtype=np.float64)
        entry["dln"] = np.asarray(entry["dln"], dtype=np.float64)
    return spreads


#: Where a port's x = 0 window control came from, as stored in
#: ``te_window_dln_core_control_source``.
CONTROL_SOURCE_NONE = 0
CONTROL_SOURCE_BAND = 1
CONTROL_SOURCE_LEGACY = 2


def _legacy_dln(grid: np.ndarray) -> float:
    """Return ``ln(max/min)`` over a 5 x 5 window grid -- the shared definition."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        lowest, highest = np.nanmin(grid), np.nanmax(grid)
    if not np.isfinite(lowest) or lowest <= 0:
        return float("nan")
    return float(np.log(highest / lowest))


def merge_legacy_x0_controls(
    spreads: dict[tuple[str, int], dict],
    legacy_path: Path,
    metadata_path: Path,
    *,
    rtol: float = WINDOW_METRIC_RTOL,
) -> dict[tuple[str, int], dict]:
    """Add the older product's x = 0 window controls, after proving them equal.

    The core-control marking asks one question -- does this port's ``T_e`` move
    by half a natural log or more when the retarding fit window is varied
    inside its family -- and two products answer it, over different coverage.
    Consuming their union is what stops a port being marked because it happened
    to be measured twice.  It is only legitimate if the two really do measure
    the same thing, so this REFUSES to read the older product unless that is
    demonstrated rather than assumed:

    * the window family (``p_low`` x ``f_high``) and the plateau window in the
      older product must equal the band product's recorded protocol exactly;
    * its stored grid must have the shape that family implies;
    * at every set-port the two products share, ``ln(max/min)`` RECOMPUTED from
      the older product's own stored grid must reproduce the value it recorded,
      and must agree with the band product's independently measured x = 0
      control to *rtol*;
    * there must be at least one shared set-port, since a guard that checks
      nothing proves nothing.

    Names are never compared.  Where both products carry a control the band
    product's value is kept -- it was computed on the platform the fill runs
    on -- and the source is recorded per port either way.
    """
    metadata = json.loads(metadata_path.read_text())
    with h5py.File(legacy_path, "r") as legacy:
        for name, expected in (
            ("p_low", metadata["window_p_low_percent"]),
            ("f_high", metadata["window_f_high_fraction"]),
            ("plateau_ms", metadata["plateau_ms"]),
        ):
            if name not in legacy.attrs:
                raise ValueError(
                    f"{legacy_path} records no {name}; it cannot be shown to "
                    "measure the same window sensitivity as "
                    f"{metadata_path} and will not be read"
                )
            if not np.array_equal(
                np.asarray(legacy.attrs[name], dtype=np.float64),
                np.asarray(expected, dtype=np.float64),
            ):
                raise ValueError(
                    f"{legacy_path} {name} = {np.asarray(legacy.attrs[name])} "
                    f"does not match the band product's {expected}; the two "
                    "are not the same window sensitivity"
                )
        expected_shape = (
            len(metadata["window_p_low_percent"]),
            len(metadata["window_f_high_fraction"]),
        )

        merged = dict(spreads)
        shared = 0
        for set_name in sorted(legacy.keys()):
            es_id = set_name.removeprefix("set")
            for port_name in sorted(legacy[set_name].keys()):
                port = int(port_name.removeprefix("port"))
                group = legacy[f"{set_name}/{port_name}"]
                grid = np.asarray(group["te_window_ev"][()], dtype=np.float64)
                if grid.shape != expected_shape:
                    raise ValueError(
                        f"{legacy_path} {set_name}/{port_name} grid is "
                        f"{grid.shape}, not the {expected_shape} the declared "
                        "window family implies"
                    )
                recomputed = _legacy_dln(grid)
                recorded = float(group.attrs["dln_te_window"])
                if not np.isclose(recomputed, recorded, rtol=rtol, atol=0.0):
                    raise ValueError(
                        f"{legacy_path} {set_name}/{port_name} records "
                        f"dln_te_window = {recorded!r} but its own grid gives "
                        f"{recomputed!r}; the recorded metric is not "
                        "ln(max/min) over that grid"
                    )
                key = (es_id, port)
                existing = merged.get(key)
                if existing is not None and np.isfinite(existing["x0_dln"]):
                    shared += 1
                    if not np.isclose(
                        recomputed, existing["x0_dln"], rtol=rtol, atol=0.0
                    ):
                        raise ValueError(
                            f"ES {es_id} p{port}: the two window products "
                            f"disagree on the x = 0 control, {recomputed!r} vs "
                            f"{existing['x0_dln']!r} (rtol {rtol:g}); they are "
                            "not measuring the same quantity"
                        )
                    existing["x0_source"] = CONTROL_SOURCE_BAND
                    continue
                merged[key] = {
                    "x_cm": np.empty(0, dtype=np.float64),
                    "dln": np.empty(0, dtype=np.float64),
                    "x0_dln": recomputed,
                    "x0_source": CONTROL_SOURCE_LEGACY,
                }
    if shared == 0:
        raise ValueError(
            f"{legacy_path} shares no set-port with {metadata_path}, so the "
            "metric-identity guard has nothing to check it against"
        )
    return merged


def _window_spread_grids(
    x_cm: np.ndarray,
    ports: np.ndarray,
    es_id: str,
    spreads: dict[tuple[str, int], dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the (n_z, n_x) window spreads, the per-port control, and its source."""
    dln_grid = np.full((len(ports), len(x_cm)), np.nan)
    core_control = np.full(len(ports), np.nan)
    control_source = np.full(len(ports), CONTROL_SOURCE_NONE, dtype=np.int8)
    for z_idx, port in enumerate(ports):
        entry = spreads.get((str(es_id), int(port)))
        if entry is None:
            continue
        core_control[z_idx] = entry["x0_dln"]
        if np.isfinite(entry["x0_dln"]):
            control_source[z_idx] = entry.get("x0_source", CONTROL_SOURCE_BAND)
        for x_value, dln in zip(entry["x_cm"], entry["dln"]):
            x_idx = int(np.argmin(np.abs(x_cm - x_value)))
            if not np.isclose(x_cm[x_idx], x_value, atol=1e-6):
                raise ValueError(
                    f"ES {es_id} p{int(port)}: window-spread cell at "
                    f"x = {x_value:g} cm is not on the T_e x grid"
                )
            dln_grid[z_idx, x_idx] = dln
    return dln_grid, core_control, control_source


def _semi_quantitative_marks(
    te_filled: np.ndarray,
    x_cm: np.ndarray,
    dln_grid: np.ndarray,
    core_control: np.ndarray,
    *,
    core_x_cm: float = X_CORE_CM,
    te_ev: float = SEMI_QUANTITATIVE_TE_EV,
    criterion: float = SEMI_QUANTITATIVE_DLN,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the semi-quantitative reason bitmask and the inflating spread.

    One rule, applied everywhere, with three ways a cell can enter the class:

    ``SEMI_QUANT_SUB_EV``
        the filled value is below *te_ev*, the swept diagnostic's
        low-temperature limit.  This is the standing convention and it is
        cycle-dependent, since a cell can be sub-eV at some times and not
        others.
    ``SEMI_QUANT_WINDOW``
        the cell's OWN measured fit-window spread is at or above *criterion*:
        moving the retarding fit window inside its declared family moves this
        cell's T_e by at least that much, so the value is convention-dependent.
    ``SEMI_QUANT_CORE_CONTROL``
        the port's ``x = 0`` control spread is at or above *criterion*, which
        indicts the whole core of that port rather than one cell.  This is the
        same rule as the cell rule, read at the one radius where the control
        was measured and applied to the region the control speaks for.

    A marked cell KEEPS its measured value: the class is an uncertainty
    statement.  The second return value is the spread that DID the marking, per
    cell, and is NaN where nothing did -- including for a cell marked only for
    being sub-eV, which has no measured window spread to inflate by.
    """
    n_z, n_x, _ = te_filled.shape
    reason = np.zeros(te_filled.shape, dtype=np.uint8)

    with np.errstate(invalid="ignore"):
        sub_ev = np.isfinite(te_filled) & (te_filled < te_ev)
    reason[sub_ev] |= SEMI_QUANT_SUB_EV

    cell_fails = np.isfinite(dln_grid) & (dln_grid >= criterion)
    reason[np.broadcast_to(cell_fails[:, :, None], reason.shape)] |= SEMI_QUANT_WINDOW

    core = np.abs(x_cm) <= core_x_cm
    control_fails = np.isfinite(core_control) & (core_control >= criterion)
    core_fails = control_fails[:, None] & core[None, :]
    reason[np.broadcast_to(core_fails[:, :, None], reason.shape)] |= (
        SEMI_QUANT_CORE_CONTROL
    )

    inflating = np.full((n_z, n_x), np.nan)
    inflating = np.where(cell_fails, dln_grid, inflating)
    control_grid = np.where(core_fails, core_control[:, None], np.nan)
    inflating = np.fmax(inflating, control_grid)
    return reason, inflating


def _core_window_sem(
    te_filled: np.ndarray,
    x_cm: np.ndarray,
    inflating_dln: np.ndarray,
    *,
    x_min: float,
    x_max: float,
) -> np.ndarray:
    """Return the (n_z, n_cycles) window term for the core-band T_e uncertainty.

    A cell whose window spread is ``d = ln(T_max / T_min)`` has a window family
    spanning a factor ``exp(d)``, so about the family's geometric centre its
    half-spread is ``T * sinh(d / 2)``; that is the per-cell uncertainty the
    window convention carries.

    The exported core T_e is the unweighted mean of the core-band cells, and
    the window convention is ONE choice applied to every cell at once, so the
    per-cell half-spreads are fully correlated and add LINEARLY inside that
    mean rather than in quadrature.  A cell with no inflating spread
    contributes zero, which is what makes the term vanish where nothing is
    marked.  The result is combined with the radial SEM in quadrature by the
    consumer, since the two are independent: one is where the window sits, the
    other is how much the profile scatters.
    """
    band = (x_cm >= x_min) & (x_cm <= x_max)
    if not np.any(band):
        raise ValueError(f"No x samples in the core band {x_min:g} to {x_max:g} cm")
    half_spread = np.sinh(np.where(np.isfinite(inflating_dln), inflating_dln, 0.0) / 2.0)
    weighted = te_filled[:, band, :] * half_spread[:, band, None]
    count = np.sum(np.isfinite(te_filled[:, band, :]), axis=1)
    total = np.sum(np.where(np.isfinite(weighted), weighted, 0.0), axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _enforce_core_mean_monotonic_z(
    te_grid: np.ndarray,
    x_cm: np.ndarray,
    *,
    core_x_cm: float = X_CORE_CM,
    te_floor: float = TE_BOUNDARY_EV,
    measured_grid: np.ndarray | None = None,
) -> tuple[np.ndarray, int, float, np.ndarray, np.ndarray]:
    """Scale downstream rows so core-mean T_e is non-increasing with z.

    Each row keeps its radial shape. Scaling is performed about ``te_floor``
    so the imposed wall/end temperature is not pushed below its boundary.

    ``measured_grid`` is the masked measurement the filled grid was built from,
    on the same axes.  When it is supplied, a row-cycle is only rescaled where
    the prior is not contradicted by the measurement it would be rewriting:

    * a row with no finite measured core cell at that cycle is reconstructed
      from neighbouring rows and the end boundaries, so the monotonic prior is
      the only axial constraint it has and the clamp applies as before;
    * a row that does carry measured core cells is rescaled only when the
      MEASURED core means are themselves non-monotonic there, i.e. when the row
      is warmer than the coolest measured row upstream of it.

    Without that restriction the clamp also fires on inversions the fill
    invented (a row whose fitted surface was pulled down by the end-plate
    sentinels reads cooler than the row below it), and the row it then rewrites
    is a measured one.  The prior is an axial statement about the plasma, so it
    must be judged on the measurement, not on the interpolant.

    Returns the adjusted grid, the number of rescaled row-cycles, the largest
    fractional reduction applied, the per-row-cycle boolean record of where the
    clamp acted, and the per-row-cycle scale factor (1.0 where it did not).
    """
    out = np.asarray(te_grid, dtype=np.float64).copy()
    core = np.abs(np.asarray(x_cm)) <= core_x_cm
    adjusted = 0
    maximum_fractional_reduction = 0.0
    if not np.any(core):
        raise ValueError("No x positions fall inside the monotonic core region")

    n_z, _, n_cycles = out.shape
    clamped = np.zeros((n_z, n_cycles), dtype=bool)
    scales = np.ones((n_z, n_cycles), dtype=np.float64)

    if measured_grid is None:
        measured_core_mean = np.full((n_z, n_cycles), np.nan)
    else:
        measured = np.asarray(measured_grid, dtype=np.float64)
        if measured.shape != out.shape:
            raise ValueError(
                f"measured grid shape {measured.shape} does not match the "
                f"filled T_e grid {out.shape}"
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            measured_core_mean = np.nanmean(measured[:, core, :], axis=1)

    for cycle_idx in range(n_cycles):
        previous_mean = np.inf
        previous_measured = np.inf
        for z_idx in range(n_z):
            row = out[z_idx, :, cycle_idx]
            mean = float(np.nanmean(row[core]))
            measured_mean = float(measured_core_mean[z_idx, cycle_idx])
            if not np.isfinite(mean):
                continue
            if mean > previous_mean and _monotonic_clamp_permitted(
                measured_mean, previous_measured
            ):
                if mean <= te_floor or previous_mean <= te_floor:
                    scale = 0.0
                else:
                    scale = (previous_mean - te_floor) / (mean - te_floor)
                scale = float(np.clip(scale, 0.0, 1.0))
                out[z_idx, :, cycle_idx] = te_floor + (row - te_floor) * scale
                adjusted += 1
                clamped[z_idx, cycle_idx] = True
                scales[z_idx, cycle_idx] = scale
                maximum_fractional_reduction = max(
                    maximum_fractional_reduction,
                    1.0 - scale,
                )
                mean = previous_mean
            previous_mean = min(previous_mean, mean)
            if np.isfinite(measured_mean):
                previous_measured = min(previous_measured, measured_mean)
    return out, adjusted, maximum_fractional_reduction, clamped, scales


def _monotonic_clamp_permitted(
    measured_mean: float,
    previous_measured_mean: float,
) -> bool:
    """Return whether the monotonic-z clamp may rewrite this row-cycle.

    ``measured_mean`` is the row's own measured core mean and
    ``previous_measured_mean`` the smallest measured core mean upstream of it,
    both ``NaN``/``inf`` when no such measurement exists.  Permission is granted
    when the row is reconstructed rather than measured, when nothing measured
    sits upstream to judge it against, or when the measurements themselves run
    the wrong way in z.
    """
    if not np.isfinite(measured_mean):
        return True
    if not np.isfinite(previous_measured_mean):
        return True
    return measured_mean > previous_measured_mean


# ---------------------------------------------------------------------------
# Core hot-spot preservation from -I_SWEEP dead-time current
# ---------------------------------------------------------------------------
def _robust_current_spike_mask(
    current_x_cycle: np.ndarray,
    x_cm: np.ndarray,
    *,
    core_x_cm: float = X_CORE_CM,
    sigma: float = CORE_HOTSPOT_CURRENT_SIGMA,
    ratio: float = CORE_HOTSPOT_CURRENT_RATIO,
) -> np.ndarray:
    """Return mask of core current spikes, robustly thresholded per cycle."""
    current = np.asarray(current_x_cycle, dtype=np.float64)
    mask = np.zeros(current.shape, dtype=bool)
    core = np.abs(x_cm) <= core_x_cm
    if not core.any():
        return mask

    core_current = current[core]
    finite_core = np.isfinite(core_current)
    with np.errstate(all="ignore"):
        median = np.nanmedian(core_current, axis=0)
        mad = np.nanmedian(np.abs(core_current - median[np.newaxis, :]), axis=0)
    robust_sigma = 1.4826 * mad

    finite_counts = finite_core.sum(axis=0)
    has_baseline = (finite_counts >= 3) & np.isfinite(median)
    if not has_baseline.any():
        return mask

    sigma_threshold = median + sigma * robust_sigma
    ratio_threshold = np.where(
        median > 0.0,
        median * ratio,
        median + np.abs(median) * (ratio - 1.0),
    )
    threshold = np.maximum(sigma_threshold, ratio_threshold)
    core_spikes = (
        np.isfinite(core_current)
        & has_baseline[np.newaxis, :]
        & (core_current > threshold[np.newaxis, :])
    )
    mask[np.flatnonzero(core), :] = core_spikes
    return mask


def _interp_current_to_te_cycles(
    current_x_cycle: np.ndarray,
    current_time_ms: np.ndarray,
    te_time_ms: np.ndarray,
) -> np.ndarray:
    if (
        current_x_cycle.shape[1] == te_time_ms.size
        and current_time_ms.shape == te_time_ms.shape
        and np.allclose(current_time_ms, te_time_ms)
    ):
        return current_x_cycle
    return np.vstack([
        np.interp(te_time_ms, current_time_ms, current_x_cycle[xi, :])
        for xi in range(current_x_cycle.shape[0])
    ])


def _load_current_hotspot_mask(
    hf: h5py.File,
    es_id: str,
    *,
    x_cm: np.ndarray,
    z_cm: np.ndarray,
    cycle_time_ms: np.ndarray,
    core_x_cm: float,
    sigma: float,
    ratio: float,
) -> np.ndarray:
    """Build a (n_z, n_x, n_cycles) mask of corroborated core hot spots."""
    mask = np.zeros((len(z_cm), len(x_cm), len(cycle_time_ms)), dtype=bool)
    if "x_cm" not in hf or "experiment_sets" not in hf or es_id not in hf["experiment_sets"]:
        return mask

    current_x_cm = hf["x_cm"][()]
    if not np.allclose(x_cm, current_x_cm):
        raise ValueError(f"-I_SWEEP dead-time x grid does not match T_e grid for ES {es_id}")

    eg = hf[f"experiment_sets/{es_id}"]
    for run_id in sorted(eg.keys()):
        rg = eg[run_id]
        source_channel = str(rg.attrs.get("deadtime_source_channel", ""))
        source_invert = bool(rg.attrs.get("deadtime_source_invert_polarity", False))
        if source_channel != "i_sweep" or not source_invert:
            continue

        z_idx = int(np.argmin(np.abs(z_cm - float(rg.attrs["z_cm"]))))
        if not np.isclose(z_cm[z_idx], float(rg.attrs["z_cm"]), atol=1e-3):
            continue

        current_time_ms = rg["inter_sweep_time_s"][()] * 1000.0
        current = _interp_current_to_te_cycles(
            rg["isat_a"][()],
            current_time_ms,
            cycle_time_ms,
        )
        mask[z_idx] |= _robust_current_spike_mask(
            current,
            x_cm,
            core_x_cm=core_x_cm,
            sigma=sigma,
            ratio=ratio,
        )
    return mask


# ---------------------------------------------------------------------------
# Comparison figure
# ---------------------------------------------------------------------------
def _plot_comparison(data: dict, te_filled: np.ndarray, output_dir: Path) -> None:
    """Four-cycle comparison: masked T_e (top) vs filled T_e (bottom)."""
    te_masked = data["te"]          # (n_z, n_x, n_cycles)
    x_cm      = data["x_cm"]
    z_cm      = data["z_cm"]
    times     = data["cycle_time_ms"]
    es_id     = data["es_id"]
    es_label  = data["es_label"]
    v_bank    = data["v_bank"]
    n_cycles  = te_masked.shape[2]

    # Representative cycle indices
    cyc_indices = [
        0,
        n_cycles // 3,
        2 * n_cycles // 3,
        n_cycles - 1,
    ]

    # Colour scale from the masked finite values — matches plot_te_contours.py.
    # (Using te_filled would compress the scale because boundary-fill adds many
    # 0.1 eV points, making the masked top row look darker than in te_contours.)
    finite = te_masked[np.isfinite(te_masked)]
    if finite.size == 0:
        finite = te_filled.ravel()
    vmin = max(0.0, float(np.percentile(finite, 2)))
    vmax = float(np.percentile(finite, 98))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    def edges(arr):
        mid = (arr[:-1] + arr[1:]) / 2
        lo  = arr[0]  - (arr[1]  - arr[0])  / 2
        hi  = arr[-1] + (arr[-1] - arr[-2]) / 2
        return np.concatenate([[lo], mid, [hi]])

    x_edges = edges(x_cm)
    z_edges = edges(z_cm)

    fig, axes = plt.subplots(
        2, len(cyc_indices),
        figsize=(4.5 * len(cyc_indices), 7),
        constrained_layout=True,
    )

    row_labels = ["masked", "filled"]
    grids      = [te_masked, te_filled]

    for row, (label, grid) in enumerate(zip(row_labels, grids)):
        for col, ci in enumerate(cyc_indices):
            ax = axes[row, col]
            frame = grid[:, :, ci]
            ax.pcolormesh(x_edges, z_edges, frame,
                          cmap=CMAP, norm=norm, rasterized=True)
            ax.set_title(f"t = {times[ci]:.1f} ms", fontsize=8)
            ax.set_xlabel("x (cm)", fontsize=7)
            ax.set_ylabel("z (cm)", fontsize=7)
            ax.tick_params(labelsize=6)
            if col == 0:
                ax.set_ylabel(f"{label}\nz (cm)", fontsize=7)

    sm   = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("$T_e$ (eV)  [best-fit]", fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"$T_e$ spatial fill — ES {es_id}: {es_label}  (V_bank = {v_bank} V)\n"
        f"Top: strict-masked data   Bottom: RBF-filled (boundary = {TE_BOUNDARY_EV} eV)",
        fontsize=10,
    )

    out = output_dir / f"te_filled_expset{es_id}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Monotonic-z upper bound
# ---------------------------------------------------------------------------
def _monotonic_z_bound(
    te_grid: np.ndarray,        # (n_z, n_x, n_cycles)
    z_cm: np.ndarray,
    *,
    padding: float = MONO_Z_PADDING,
    preserve_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, int, int]:
    """Mask downstream cells that violate the expected z-monotonicity of T_e.

    For each z-row z_i (i > 0), the upper bound is a **per-cycle global
    maximum** over all upstream z-rows and all x positions:

        bound(cycle) = max( T_e[z_j, x, cycle]  for j < i, all x, finite )
                       × (1 + padding)

    Using the spatial maximum (rather than a per-x value) correctly handles
    radial heat spreading: if a hot spot at x=0 upstream spreads outward by
    the time it reaches z_i, a position at x=6 can legitimately exceed its
    own upstream value as long as it stays within the global upstream maximum.
    The bound is only applied where at least one upstream finite value exists.
    The most-upstream z-row (i=0) is never touched.

    Parameters
    ----------
    padding : float
        Fractional allowance above the upstream peak.  0.25 → 25 % above
        the highest reliable upstream measurement anywhere in the profile.

    Returns
    -------
    te_bounded : (n_z, n_x, n_cycles) — copy of te_grid with outliers masked
    n_masked   : total number of cells set to NaN by this step
    n_preserved: total number of outlier cells preserved by preserve_mask
    """
    te_out   = te_grid.copy()
    n_masked = 0
    n_preserved = 0
    if preserve_mask is None:
        preserve_mask = np.zeros(te_grid.shape, dtype=bool)
    elif preserve_mask.shape != te_grid.shape:
        raise ValueError(
            f"preserve_mask shape {preserve_mask.shape} does not match T_e grid {te_grid.shape}"
        )

    for zi in range(1, len(z_cm)):
        # Global maximum T_e across all upstream z-rows and all x, per cycle.
        # Shape: (n_cycles,)  — one bound value per time point.
        with np.errstate(all="ignore"):   # nanmax of all-NaN slice → NaN, handled below
            upstream_max = np.nanmax(
                te_grid[:zi].reshape(-1, te_grid.shape[2]), axis=0
            )
        has_upstream = np.isfinite(upstream_max)          # (n_cycles,)
        upper_bound  = upstream_max * (1.0 + padding)     # (n_cycles,)

        # Broadcast bound from (n_cycles,) → (n_x, n_cycles) for comparison.
        exceeds = (
            np.isfinite(te_grid[zi])
            & has_upstream[np.newaxis, :]
            & (te_grid[zi] > upper_bound[np.newaxis, :])
        )
        preserved = exceeds & preserve_mask[zi]
        mask_out = exceeds & ~preserve_mask[zi]
        if exceeds.any():
            te_out[zi][mask_out] = np.nan
            n_masked += int(mask_out.sum())
            n_preserved += int(preserved.sum())
            n_ref = int(
                (np.isfinite(te_grid[zi]) & has_upstream[np.newaxis, :]).sum()
            )
            print(
                f"    z = {z_cm[zi]:.1f} cm: {mask_out.sum()} / {n_ref} cells "
                f"capped, {preserved.sum()} preserved by -I_SWEEP hot spot "
                f"(bound = upstream_max × {1+padding:.2f})"
            )

    return te_out, n_masked, n_preserved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input",       type=Path,  default=HDF5_INPUT)
    parser.add_argument("--output",      type=Path,  default=HDF5_OUTPUT)
    parser.add_argument("--output-dir",  type=Path,  default=OUTPUT_DIR)
    parser.add_argument("--input-mode", choices=["auto", "langmuir", "filled"], default="auto",
                        help="Input layout: process_langmuir_sweeps output, previous te_filled output, or auto-detect.")
    parser.add_argument("--x-wall",      type=float, default=X_WALL_CM,
                        help="x position of drift-tube wall boundary (cm)")
    parser.add_argument("--z-cathode",   type=float, default=Z_CATHODE_CM,
                        help="z position of cathode end-plate boundary (cm)")
    parser.add_argument("--z-anode",     type=float, default=Z_ANODE_CM,
                        help="z position of anode end-plate boundary (cm)")
    parser.add_argument("--te-boundary",    type=float, default=TE_BOUNDARY_EV,
                        help="T_e boundary value at machine walls (eV)")
    parser.add_argument("--x-core",         type=float, default=X_CORE_CM,
                        help="Inner edge of weighting transition zone (cm). "
                             "Points with |x| ≤ x-core get smoothing-core.")
    parser.add_argument("--x-edge",         type=float, default=X_EDGE_CM,
                        help="Outer edge of weighting transition zone (cm). "
                             "Points with |x| ≥ x-edge get smoothing-edge.")
    parser.add_argument("--smoothing-core", type=float, default=SMOOTHING_CORE,
                        help="RBF smoothing for core region (low = exact fit).")
    parser.add_argument("--smoothing-edge", type=float, default=SMOOTHING_EDGE,
                        help="RBF smoothing for edge region (high = loose fit).")
    parser.add_argument("--edge-anchor", type=float, default=EDGE_ANCHOR_CM,
                        help="|x| position for low-Te SOL anchor points (cm).")
    parser.add_argument("--te-edge-anchor", type=float, default=TE_EDGE_ANCHOR_EV,
                        help="T_e value assigned to SOL edge anchor points (eV).")
    parser.add_argument("--smoothing-edge-anchor", type=float, default=SMOOTHING_EDGE_ANCHOR,
                        help="RBF smoothing for SOL edge anchor points.")
    parser.add_argument("--mono-z-padding", type=float, default=MONO_Z_PADDING,
                        help="Fractional allowance above the upstream T_e envelope for the "
                             "monotonic-z upper bound.  Set to a large value (e.g. 1e6) to "
                             f"disable.  Default: {MONO_Z_PADDING}.")
    parser.add_argument("--window-band", type=Path, default=WINDOW_BAND_INPUT,
                        help="Per-cell fit-window spreads across the measurement "
                             "band (scripts/refit_window_band.py).  The trust "
                             "model conditions on it and the semi-quantitative "
                             "marking is read from it.")
    parser.add_argument("--window-band-metadata", type=Path,
                        default=WINDOW_BAND_METADATA,
                        help="Band-product protocol record; supplies the window "
                             "family the metric-identity guard checks the "
                             "original x = 0 product against.")
    parser.add_argument("--legacy-window-refits", type=Path,
                        default=LEGACY_WINDOW_REFITS,
                        help="The original x = 0 window-refit product "
                             "(scripts/refit_sweep_windows.py).  Read for the "
                             "x = 0 core control only, and only once it is "
                             "proved to measure the same window sensitivity.")
    parser.add_argument("--core-band-cm", nargs=2, type=float, default=(-10.0, 10.0),
                        help="Core band the exported per-port T_e statistics are "
                             "taken over, and hence the band whose window-spread "
                             "uncertainty is accumulated into "
                             "te_core_window_sem_ev.  Must match the overlay "
                             "exporter's core band.")
    parser.add_argument("--isweep-deadtime", type=Path, default=ISWEEP_DEADTIME_INPUT,
                        help="Optional -I_SWEEP dead-time profile HDF5 used to preserve "
                             "core T_e hot spots that coincide with current spikes.")
    parser.add_argument("--no-current-hotspot-preservation", action="store_true",
                        help="Do not preserve monotonic-z outliers corroborated by core "
                             "-I_SWEEP dead-time current spikes.")
    parser.add_argument("--core-hotspot-current-sigma", type=float,
                        default=CORE_HOTSPOT_CURRENT_SIGMA,
                        help="Robust per-cycle sigma threshold for core current spikes.")
    parser.add_argument("--core-hotspot-current-ratio", type=float,
                        default=CORE_HOTSPOT_CURRENT_RATIO,
                        help="Minimum current/median ratio for core current spikes.")
    parser.add_argument("--min-z-coverage", type=float, default=MIN_Z_COVERAGE,
                        help="Minimum fraction of finite cells (across all x and cycles) "
                             "for a z-row to be included in the RBF fit. Rows below this "
                             "threshold are blanked and filled by boundary extrapolation. "
                             f"Default: {MIN_Z_COVERAGE}.")
    parser.add_argument("--no-preserve-measured-cells", action="store_true",
                        help="Let the RBF surface overwrite core cells that carry a "
                             "finite measurement, instead of filling only the NaN cells "
                             "there.  This is the pre-2026-08-20 behaviour and lets the "
                             "near-exactly enforced boundary sentinels bias measured "
                             "rows near the end plates; kept for comparison only.")
    parser.add_argument("--no-measured-monotonic-gate", action="store_true",
                        help="Let the core-mean monotonic-z clamp rewrite a measured row "
                             "even where the measured core means are themselves "
                             "monotonic.  This is the pre-2026-08-20 behaviour, in which "
                             "the clamp also fires on inversions the fill invented; kept "
                             "for comparison only.")
    parser.add_argument("--no-plots",       action="store_true",
                        help="skip comparison figures")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"{args.input} not found — run process_langmuir_sweeps.py first."
        )
    if not args.window_band.exists():
        raise FileNotFoundError(
            f"{args.window_band} not found — run scripts/refit_window_band.py "
            "first.  The trust model conditions on the measured band-wide "
            "fit-window spreads, so the fill is not defined without them."
        )
    if not args.window_band_metadata.exists():
        raise FileNotFoundError(
            f"{args.window_band_metadata} not found — it carries the window "
            "family the metric-identity guard checks the original x = 0 "
            "product against; run scripts/refit_window_band.py first."
        )
    if not args.legacy_window_refits.exists():
        raise FileNotFoundError(
            f"{args.legacy_window_refits} not found — run "
            "scripts/refit_sweep_windows.py first.  It is the only x = 0 "
            "window control there is for experiment set 3, and the "
            "semi-quantitative marking conditions on it."
        )
    window_spreads = merge_legacy_x0_controls(
        load_window_band_spreads(args.window_band),
        args.legacy_window_refits,
        args.window_band_metadata,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fill_kwargs = dict(
        x_wall          = args.x_wall,
        z_lo            = args.z_cathode,
        z_hi            = args.z_anode,
        te_boundary     = args.te_boundary,
        x_core          = args.x_core,
        x_edge          = args.x_edge,
        smoothing_core  = args.smoothing_core,
        smoothing_edge  = args.smoothing_edge,
        edge_anchor     = args.edge_anchor,
        te_edge_anchor  = args.te_edge_anchor,
        smoothing_edge_anchor = args.smoothing_edge_anchor,
        min_z_coverage  = args.min_z_coverage,
        preserve_measured = not args.no_preserve_measured_cells,
    )

    current_hf = (
        None
        if args.no_current_hotspot_preservation or not args.isweep_deadtime.exists()
        else h5py.File(args.isweep_deadtime, "r")
    )

    try:
        with h5py.File(args.input, "r") as hf_in, \
             h5py.File(args.output, "w") as hf_out:

            es_ids = sorted(hf_in["experiment_sets"].keys(), key=int)
            hf_out.create_group("experiment_sets")

            for es_id in es_ids:
                print(f"\nES {es_id}:")
                data = _load_input_experiment_set(hf_in, es_id, args.input_mode)

                te_masked = data["te"]
                x_cm      = data["x_cm"]
                z_cm      = data["z_cm"]
                ports     = data["port"]

                n_finite = int(np.isfinite(te_masked).sum())
                n_total  = int(te_masked.size)
                print(f"  {n_finite}/{n_total} cells finite ({100*n_finite/n_total:.1f}%) before fill")
                if data["n_qc_excluded"] >= 0:
                    print(
                        f"  QC gate (n_ok >= n_bad, all radii): "
                        f"{data['n_qc_excluded']} run-cells excluded"
                    )

                trust_radius_cm, trust_blend_cm = _trust_model_for_ports(
                    es_id,
                    ports,
                    x_core=args.x_core,
                    x_edge=args.x_edge,
                )
                adopted = trust_radius_cm > args.x_core
                if adopted.any():
                    print(
                        "  Measurement-authoritative to the aperture at: "
                        + ", ".join(
                            f"p{int(ports[i])} (|x| <= {trust_radius_cm[i]:g} cm, "
                            f"blend to {trust_blend_cm[i]:g} cm)"
                            for i in np.flatnonzero(adopted)
                        )
                    )
                if (~adopted).any():
                    print(
                        "  Historical core trust kept at: "
                        + ", ".join(
                            f"p{int(ports[i])} (|x| <= {trust_radius_cm[i]:g} cm)"
                            for i in np.flatnonzero(~adopted)
                        )
                    )

                te_masked, untrusted_ports = _mask_untrusted_te_ports(
                    te_masked,
                    ports,
                    es_id,
                )
                if untrusted_ports:
                    print(
                        "  Low-T_e/unreliable downstream rows excluded before fill: "
                        + ", ".join(f"p{port}" for port in untrusted_ports)
                    )

                preserve_mask = None
                if current_hf is not None:
                    preserve_mask = _load_current_hotspot_mask(
                        current_hf,
                        es_id,
                        x_cm=x_cm,
                        z_cm=z_cm,
                        cycle_time_ms=data["cycle_time_ms"],
                        core_x_cm=args.x_core,
                        sigma=args.core_hotspot_current_sigma,
                        ratio=args.core_hotspot_current_ratio,
                    )
                    print(
                        f"  Core -I_SWEEP hot-spot preservation: "
                        f"{int(preserve_mask.sum())} candidate cells"
                    )

                # Apply monotonic-z upper bound: downstream cells whose Te exceeds
                # the upstream Te envelope are masked before the RBF, unless a
                # core -I_SWEEP dead-time spike corroborates a real hot spot.
                print(f"  Monotonic-z bound (padding={args.mono_z_padding:.0%}) …")
                te_masked, n_mono, n_preserved = _monotonic_z_bound(
                    te_masked, z_cm, padding=args.mono_z_padding, preserve_mask=preserve_mask
                )
                if n_mono:
                    print(f"    {n_mono} cells masked by monotonicity bound")
                else:
                    print(f"    No cells exceeded the upstream bound")
                if n_preserved:
                    print(f"    {n_preserved} core hot-spot cells preserved")

                print(f"  Fitting {te_masked.shape[2]} cycles …")
                te_filled = fill_te_grid(
                    te_masked,
                    x_cm,
                    z_cm,
                    trust_radius_cm=trust_radius_cm,
                    trust_blend_cm=trust_blend_cm,
                    **fill_kwargs,
                )
                (
                    te_filled,
                    n_monotonic_rows,
                    max_monotonic_reduction,
                    monotonic_clamped,
                    monotonic_scale,
                ) = _enforce_core_mean_monotonic_z(
                    te_filled,
                    x_cm,
                    core_x_cm=args.x_core,
                    te_floor=args.te_boundary,
                    measured_grid=None if args.no_measured_monotonic_gate else te_masked,
                )
                print(
                    "  Core-mean monotonic-z enforcement: "
                    f"{n_monotonic_rows} row-cycles adjusted, "
                    f"maximum reduction {max_monotonic_reduction:.1%}"
                )
                for z_idx in np.flatnonzero(monotonic_clamped.any(axis=1)):
                    cycles = np.flatnonzero(monotonic_clamped[z_idx])
                    print(
                        f"    z = {z_cm[z_idx]:.1f} cm (p{int(ports[z_idx])}): "
                        f"{cycles.size} row-cycles clamped at t = "
                        + ", ".join(
                            f"{data['cycle_time_ms'][ci]:.2f}" for ci in cycles
                        )
                        + " ms"
                    )

                (
                    window_dln,
                    window_core_control,
                    window_control_source,
                ) = _window_spread_grids(x_cm, ports, es_id, window_spreads)
                semi_quant_reason, window_inflating = _semi_quantitative_marks(
                    te_filled,
                    x_cm,
                    window_dln,
                    window_core_control,
                    core_x_cm=args.x_core,
                )
                semi_quantitative = semi_quant_reason != 0
                core_band = (x_cm >= args.core_band_cm[0]) & (
                    x_cm <= args.core_band_cm[1]
                )
                band_cells = (np.abs(x_cm) > args.x_core) & (
                    np.abs(x_cm) <= X_TRUST_APERTURE_CM
                )
                semi_quant_core_count = np.sum(
                    semi_quantitative[:, core_band, :], axis=1
                ).astype(np.int16)
                semi_quant_band_count = np.sum(
                    semi_quantitative[:, band_cells, :], axis=1
                ).astype(np.int16)
                core_window_sem = _core_window_sem(
                    te_filled,
                    x_cm,
                    window_inflating,
                    x_min=args.core_band_cm[0],
                    x_max=args.core_band_cm[1],
                )
                print(
                    "  Semi-quantitative cells: "
                    f"{int((semi_quant_reason & SEMI_QUANT_SUB_EV).astype(bool).sum())}"
                    " sub-eV, "
                    f"{int((semi_quant_reason & SEMI_QUANT_WINDOW).astype(bool).sum())}"
                    " window-spread, "
                    f"{int((semi_quant_reason & SEMI_QUANT_CORE_CONTROL).astype(bool).sum())}"
                    " core-control"
                )
                for z_idx in np.flatnonzero(
                    np.isfinite(window_core_control)
                    & (window_core_control >= SEMI_QUANTITATIVE_DLN)
                ):
                    source = {
                        CONTROL_SOURCE_BAND: "band product",
                        CONTROL_SOURCE_LEGACY: "original x = 0 product",
                    }[int(window_control_source[z_idx])]
                    print(
                        f"    p{int(ports[z_idx])}: x = 0 window control "
                        f"{window_core_control[z_idx]:.3f} >= "
                        f"{SEMI_QUANTITATIVE_DLN:g} ({source}); its whole core "
                        "is semi-quantitative"
                    )

                # Write to HDF5
                grp = hf_out["experiment_sets"].create_group(es_id)
                grp.attrs["label"]    = data["es_label"]
                grp.attrs["v_bank_v"] = float(data["v_bank"])
                grp.attrs["current_hotspot_preservation_enabled"] = bool(current_hf is not None)
                grp.attrs["current_hotspot_preserved_cells"] = int(n_preserved)
                grp.attrs["current_hotspot_current_sigma"] = float(args.core_hotspot_current_sigma)
                grp.attrs["current_hotspot_current_ratio"] = float(args.core_hotspot_current_ratio)
                grp.attrs["trusted_te_ports"] = ",".join(
                    str(port) for port in TRUSTED_TE_PORTS_BY_ES.get(es_id, tuple(ports))
                )
                grp.attrs["excluded_unreliable_te_ports"] = ",".join(
                    str(port) for port in untrusted_ports
                )
                grp.attrs["core_mean_te_monotonic_z_enforced"] = True
                grp.attrs["core_mean_te_monotonic_adjusted_row_cycles"] = int(
                    n_monotonic_rows
                )
                grp.attrs["core_mean_te_monotonic_max_fractional_reduction"] = float(
                    max_monotonic_reduction
                )
                grp.attrs["core_mean_te_monotonic_measured_gate"] = bool(
                    not args.no_measured_monotonic_gate
                )
                grp.attrs["measured_cells_preserved_in_fill"] = bool(
                    not args.no_preserve_measured_cells
                )
                grp.attrs["qc_rule"] = QC_RULE
                grp.attrs["qc_excluded_run_cells"] = int(data["n_qc_excluded"])
                grp.attrs["te_trust_model"] = TRUST_MODEL_STATEMENT
                grp.attrs["te_trust_adopted_ports"] = ",".join(
                    str(int(ports[i])) for i in np.flatnonzero(adopted)
                )
                grp.attrs["te_trust_default_radius_cm"] = float(args.x_core)
                grp.attrs["te_trust_default_blend_cm"] = float(args.x_edge)
                grp.attrs["te_semi_quantitative_rule"] = SEMI_QUANTITATIVE_STATEMENT
                grp.attrs["te_semi_quantitative_te_ev"] = float(SEMI_QUANTITATIVE_TE_EV)
                grp.attrs["te_semi_quantitative_dln"] = float(SEMI_QUANTITATIVE_DLN)
                grp.attrs["te_core_window_sem_definition"] = CORE_WINDOW_SEM_STATEMENT
                grp.attrs["te_core_window_sem_x_min_cm"] = float(args.core_band_cm[0])
                grp.attrs["te_core_window_sem_x_max_cm"] = float(args.core_band_cm[1])
                grp.attrs["te_window_band_source"] = str(args.window_band)
                grp.attrs["te_window_legacy_source"] = str(args.legacy_window_refits)
                grp.attrs["te_window_core_control_source_codes"] = (
                    f"{CONTROL_SOURCE_NONE} = no x = 0 control measured; "
                    f"{CONTROL_SOURCE_BAND} = the band product; "
                    f"{CONTROL_SOURCE_LEGACY} = the original x = 0 product.  "
                    "The two are proved to measure the same window sensitivity "
                    "before their union is taken: same window family, same "
                    "plateau, and ln(max/min) recomputed from the original "
                    "product's own stored grid reproduces both its recorded "
                    "value and the band product's independent measurement at "
                    f"every shared set-port to {WINDOW_METRIC_RTOL:g} relative."
                )
                grp.attrs["te_window_metric_rtol"] = float(WINDOW_METRIC_RTOL)
                grp.create_dataset("x_cm",           data=x_cm,                compression="gzip")
                grp.create_dataset("z_cm",           data=z_cm,                compression="gzip")
                grp.create_dataset("port",           data=ports,               compression="gzip")
                grp.create_dataset("cycle_time_ms",  data=data["cycle_time_ms"], compression="gzip")
                grp.create_dataset("te_masked",      data=te_masked,            compression="gzip", compression_opts=4)
                grp.create_dataset("te_filled",      data=te_filled,            compression="gzip", compression_opts=4)
                grp.create_dataset(
                    "core_mean_te_monotonic_clamped",
                    data=monotonic_clamped,
                    compression="gzip",
                )
                grp.create_dataset(
                    "core_mean_te_monotonic_scale",
                    data=monotonic_scale,
                    compression="gzip",
                )
                grp.create_dataset(
                    "te_trust_radius_cm", data=trust_radius_cm, compression="gzip"
                )
                grp.create_dataset(
                    "te_trust_blend_cm", data=trust_blend_cm, compression="gzip"
                )
                grp.create_dataset(
                    "te_window_dln", data=window_dln, compression="gzip"
                )
                grp.create_dataset(
                    "te_window_dln_core_control",
                    data=window_core_control,
                    compression="gzip",
                )
                grp.create_dataset(
                    "te_window_dln_core_control_source",
                    data=window_control_source,
                    compression="gzip",
                )
                grp.create_dataset(
                    "te_window_dln_inflating",
                    data=window_inflating,
                    compression="gzip",
                )
                grp.create_dataset(
                    "te_semi_quantitative",
                    data=semi_quantitative,
                    compression="gzip",
                )
                grp.create_dataset(
                    "te_semi_quantitative_reason",
                    data=semi_quant_reason,
                    compression="gzip",
                )
                grp.create_dataset(
                    "te_semi_quantitative_core_count",
                    data=semi_quant_core_count,
                    compression="gzip",
                )
                grp.create_dataset(
                    "te_semi_quantitative_band_count",
                    data=semi_quant_band_count,
                    compression="gzip",
                )
                grp.create_dataset(
                    "te_core_window_sem_ev",
                    data=core_window_sem,
                    compression="gzip",
                )
                hf_out.flush()

                if not args.no_plots:
                    plot_data = dict(data)
                    plot_data["te"] = te_masked
                    _plot_comparison(plot_data, te_filled, args.output_dir)
    finally:
        if current_hf is not None:
            current_hf.close()

    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
