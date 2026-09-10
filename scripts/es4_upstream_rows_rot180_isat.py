"""ES4 upstream density rows built from the rot-180 ISAT face instead of the rot-0 I_SWEEP face.

WHAT THIS MEASURES
------------------
At every experiment-set-4 port the probe was run twice, once at each rotation.
The face that looks UPSTREAM is the I_SWEEP electrode at rot 0 and the ISAT
electrode at rot 180, so the same physical upstream density has two independent
measurements at each port.  The overlay currently carries the rot-0 I_SWEEP row.
This instrument builds the rot-180 ISAT row for the same ports and puts the two
side by side.

The reason to want the ISAT row at set 4 is a convention term the I_SWEEP row
carries and the ISAT row does not.  The swept face's parked (rest) bias is the
sweep's ``voltage_start``, and that moved from -75 V in sets 1 and 3 to -20 V in
set 4 (``config/may2026_run_manifest.toml``), so an ES4 I_SWEEP dead-time
current is collected at a different bias from its ES1/ES3 control and needs a
rest-bias convention term before the sets are comparable.  The ISAT face has no
sweep-derived rest bias: it is separately biased and the manifest moves no bias
parameter on it across the ladder.  What the manifest DOES change on the ISAT
channel between set 3 and set 4 is the attenuation, 2.0 -> 1.0, which is a
calibration constant the dead-time product already applies, not a bias.

This is a ONE-SHOT INSTRUMENT.  It writes a measurement, never a product: the
output path is refused if it resolves inside ``processed/``, and promoting any
row here into the overlay is a separate, registered step.

METHOD
------
The repo's own density chain, cell by cell:

    C_s   = ion_sound_speed_m_s(T_e, m_i)             (bapsf_lapd.density)
    n_e   = electron_density_m3(I_sat, A_p, C_s)      (bapsf_lapd.density)
    n_e   = I_sat / (exp(-1/2) * A_p * e * C_s)

with

  I_sat  the shot-mean dead-time current of the product that owns the face --
         ``processed/isat_rot180_deadtime_profiles.hdf5`` for the rot-180 ISAT
         face, ``processed/isweep_deadtime_profiles.hdf5`` for the rot-0
         I_SWEEP face.  Both products carry the same 51-point, 1 cm scan and the
         same 20 inter-sweep dead-time windows.

  T_e    the ES4 overlay's per-port row, interpolated to the dead-time window
         midpoints.  The SAME T_e the rot-0 row uses, so the two chains differ
         only in which face they read.  See the T_e caveat below.

  A_p    the face's calibrated area from ``processed/probe_area_calibration.toml``,
         by the identity of the ELECTRODE that collected the current:
         ``ap_R_cm2`` for the ISAT electrode, ``ap_L_cm2`` for the I_SWEEP
         electrode.  This is the convention the areas were themselves produced
         under -- ``scripts/calibrate_probe_areas.py`` accumulates ``ap_R_m2``
         from the rot-180 ISAT product and ``ap_L_m2`` from the rot-0 I_SWEEP
         product -- so it is the only key assignment under which a density built
         from these rows reproduces the calibration that defined the areas.

         ``density_area_key_for_deadtime_source`` in
         ``src/bapsf_lapd/corrections.py`` now answers by the same electrode
         rule, so the two agree at every port and in both rotations.  A rot-180
         ISAT product PLACED BEFORE that change stamps the older port-keyed
         answer (``ap_L_cm2`` for the ISAT face at ports 21/29/41) into each
         run's ``density_area_key`` attribute, and a stale stamp is not what
         this instrument builds its density from.  Both readings are printed for
         every row, so a product that predates the fix is visible rather than
         silently trusted.

  window the SCORING PLATEAU WINDOW, 15.0-19.5 ms, which is
         ``RAW_PLATEAU_WINDOW_MS`` in ``scripts/export_es1_sim1d_overlay.py`` and
         the window the banked raw dead-time read used.  On this 1 ms dead-time
         grid it selects the four windows centred at 15.75, 16.75, 17.75 and
         18.75 ms.

CELL ADMISSION
--------------
A cell is admitted when it is finite and positive on both faces and carries no
exclusion mask on the ISAT product.  The rot-180 ISAT product masks per cell:
``rail_mask`` (a raw sample at a uint16 converter rail) and ``state_mask`` (the
position falls in a run's registered channel-state shot range).  Only run 43 is
masked on either -- it rails, and it carries the registered channel state whose
shot range covers scan positions 12-34.  The rot-0 I_SWEEP product predates both
masks and carries neither.

PORT COVERAGE
-------------
p29 (runs 44/45) and p41 (runs 46/47) are the ports this instrument is for.
p21 (runs 42/43) is reported for its EDGE cells only and labelled: the run-43
state mask removes scan positions 12-34, which is the whole core band, so p21
has no admitted core cell and its core gates refuse rather than report a number.
p50 (run 48) is rot-0 only -- there is no rot-180 partner and no ISAT upstream
row to build -- and is not touched.

T_e CAVEAT
----------
The ES4 overlay's p21, p29, p41 and p50 T_e rows are PRIOR-DERIVED at every
sample (``te_row_measured`` is false; only p11 is measured), so every density
here inherits that.  Because n_e is proportional to 1/sqrt(T_e), a later change
to a port's T_e rescales its whole row by sqrt(T_e_old/T_e_new) with no
re-reading of the raw data.  That factor is printed per port.

GATES
-----
(a) CHORD DEFICIT.  The probe row's line integral over the scan against the
    interferometer chord's line-integrated density at the same port, set 4, over
    the same plateau window: deficit = chord / probe.  The convention is the
    repo's (``scripts/plot_es3_scaled_density_te.py`` takes chord/probe and
    corrects T_e by 1/ratio^2) and the pairing is probe p21 -> chord p20,
    p29 -> p29, p41 -> p40.  The gate asks whether the rot-180 ISAT rows sit in
    the same 1.7-1.9 deficit bin.  Before any set-4 chord is read the instrument
    reproduces the banked set-3 chord numbers as a control, in the banked read's
    own convention (line-average cm^-3, i.e. line-integrated cm^-2 over the 40 cm
    plasma diameter, averaged over the plateau window).

(b) RADIAL SHAPE.  The per-cell density ratio ISAT_rot180(x)/isweep_rot0(x) over
    the core against R_up(x), the upstream two-channel ratio profile from the
    banked raw dead-time read.  Under the electrode-identity area convention the
    two are the same quantity computed two ways -- R_up divides the raw current
    ratio by A_R/A_L, and the density ratio divides by the same areas while the
    shared per-port T_e cancels -- so this is a cross-check of the placed
    products against an independent extraction of the same raw records.  The
    tolerance is the pair's shot-to-shot spread, cell by cell.

A gate whose port has no admitted cell REFUSES rather than reporting a number,
and a value outside its bin is printed as "outside the bins" with the value.  No
gate line ever carries a nan.
"""

from __future__ import annotations

import argparse
import csv
import sys
import tomllib
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bapsf_lapd.corrections import density_area_key_for_deadtime_source  # noqa: E402
from bapsf_lapd.config import ChannelKind  # noqa: E402
from bapsf_lapd.density import electron_density_m3, ion_sound_speed_m_s  # noqa: E402

# Directory this instrument must never write into: it is a one-shot measurement,
# not a member of the product chain.
FORBIDDEN_OUTPUT_DIR = "processed"

REPO_DEFAULT = Path(".")
ISAT_ROT180_HDF5 = Path("processed/isat_rot180_deadtime_profiles.hdf5")
ISWEEP_ROT0_HDF5 = Path("processed/isweep_deadtime_profiles.hdf5")
OVERLAY_NPZ = Path("processed/es4_sim1d_overlay.npz")
INTERF_NPZ = Path("processed/interferometer_experiment_set_stats.npz")
AREA_TOML = Path("processed/probe_area_calibration.toml")

#: The SCORING PLATEAU WINDOW in ms, ``RAW_PLATEAU_WINDOW_MS`` of
#: ``scripts/export_es1_sim1d_overlay.py``, and the window of the banked raw
#: dead-time read this instrument controls against.
PLATEAU_MS = (15.0, 19.5)

#: Core band in cm, the overlay's ``core_x_min_cm``/``core_x_max_cm``.
CORE_CM = 10.0

#: He-4, the ion the whole May 2026 dataset was taken in.
M_I_AMU = 4.003

#: Chord line-integrated cm^-2 is the line average times this diameter in cm
#: (``scripts/plot_interferometer_by_experiment_set.py``).  Used only to restate
#: a chord in the banked read's line-average convention for the control.
PLASMA_DIAMETER_CM = 40.0

#: Physical probe identity from the second digit of the run id
#: (``scripts/compute_density_mach.py``).
PROBE_FROM_DIGIT = {1: "A", 8: "A", 2: "B", 3: "B", 4: "C", 5: "C", 6: "D", 7: "D"}

#: port -> (rot-0 I_SWEEP run, rot-180 ISAT run).  p50 has no rot-180 partner.
PORT_RUNS = {21: ("42", "43"), 29: ("44", "45"), 41: ("46", "47")}

#: The ports whose core rows this instrument is for.  p21 is carried for its
#: edge cells only; its core is removed by the run-43 state mask.
CORE_PORTS = (29, 41)

#: probe port -> interferometer chord port.
CHORD_PORT = {21: 20, 29: 29, 41: 40}

#: The chord deficit bin the rot-180 ISAT rows are asked to reproduce.
DEFICIT_BIN = (1.7, 1.9)

#: Banked set-3 chord line averages in cm^-3 over the plateau window, in the
#: convention of the banked raw dead-time read.  Reproducing these is the
#: control that the chord read here is the same read.
CONTROL_ES3_CHORD_CM3 = {20: 7.986e12, 29: 1.126e13, 40: 1.485e13}
CONTROL_ES4_CHORD_CM3 = {20: 1.873e13, 29: 9.560e12, 40: 2.414e12}
#: Relative tolerance for the control.  The banked values are printed to four
#: significant figures, so this is a round-trip check, not a measurement.
CONTROL_RTOL = 2e-3

#: R_up(x), the upstream two-channel ratio profile from the banked raw dead-time
#: read, plateau-median per position over x = -12..+12 cm inclusive (25 cells).
#:     R_up = ISAT(rot180, upstream) / |I_SWEEP(rot0, upstream)| / (A_R / A_L)
#: Under the electrode-identity area convention this is exactly the density
#: ratio this instrument forms, so it is the shape gate's reference.
R_UP_REFERENCE_X_CM = np.arange(-12.0, 12.5, 1.0)
R_UP_REFERENCE = {
    21: np.array([1.862, 1.826, 1.804, 1.775, 1.745, 1.697, 1.674, 1.655, 1.635,
                  1.620, 1.627, 1.616, 1.624, 1.617, 1.553, 1.400, 1.668, 1.691,
                  1.717, 1.736, 1.767, 1.665, 1.030, 1.031, 1.029]),
    29: np.array([1.255, 1.159, 1.125, 1.120, 1.115, 1.092, 1.090, 1.085, 1.013,
                  1.015, 1.038, 1.135, 0.993, 0.981, 0.989, 0.972, 0.986, 0.995,
                  1.077, 1.186, 1.209, 1.272, 1.241, 1.237, 1.249]),
    41: np.array([1.463, 1.381, 1.292, 1.514, 1.475, 1.263, 1.346, 1.310, 1.430,
                  1.350, 1.326, 1.318, 1.495, 1.503, 1.463, 1.225, 1.523, 1.357,
                  1.311, 1.472, 1.393, 1.303, 1.487, 1.480, 1.502]),
}


# --------------------------------------------------------------------------
# conventions
# --------------------------------------------------------------------------

def area_key_for_electrode(source_channel: ChannelKind) -> str:
    """Return the calibration TOML area key of the electrode that collected.

    ``ap_R_cm2`` is the ISAT electrode and ``ap_L_cm2`` the I_SWEEP electrode,
    which is how ``scripts/calibrate_probe_areas.py`` produced them: it
    accumulates ``ap_R_m2`` from the rot-180 ISAT product and ``ap_L_m2`` from
    the rot-0 I_SWEEP product.  A density built from either product reproduces
    that calibration only under this assignment.

    This is the NOMINAL wiring, which is the only wiring ES4 carries: the one
    run whose cables were crossed at the connector is an ES3 port-11 run and
    never reaches this instrument.  ``density_area_key_for_deadtime_source``
    is the swap-aware answer and is what the stamp comparison below reads.
    """
    if source_channel == ChannelKind.ISAT:
        return "ap_R_cm2"
    if source_channel == ChannelKind.I_SWEEP:
        return "ap_L_cm2"
    raise ValueError(f"no calibrated face area for channel {source_channel!r}")


def plateau_window_mask(t_ms: np.ndarray, window_ms=PLATEAU_MS) -> np.ndarray:
    """Boolean mask of the dead-time windows inside the plateau window."""
    t_ms = np.asarray(t_ms, dtype=np.float64)
    return (t_ms >= window_ms[0]) & (t_ms <= window_ms[1])


def sqrt_te_rescale(te_old_ev: float, te_new_ev: float) -> float:
    """Factor a density row is multiplied by when its T_e moves old -> new.

    n_e is proportional to 1/C_s and C_s to sqrt(T_e), so the row rescales by
    sqrt(T_e_old / T_e_new) with no re-reading of the raw data.
    """
    if not (te_old_ev > 0 and te_new_ev > 0):
        raise ValueError("T_e must be positive on both sides of a rescale")
    return float(np.sqrt(te_old_ev / te_new_ev))


def line_integral_m2(profile_m3: np.ndarray, x_m: np.ndarray) -> float:
    """Trapezoidal line integral of a radial density profile, in m^-2.

    Matches ``_line_integral_cm2`` of ``scripts/plot_es3_scaled_density_te.py``
    (finite and positive cells only) without its cm^-2 conversion.
    """
    profile = np.asarray(profile_m3, dtype=np.float64)
    x = np.asarray(x_m, dtype=np.float64)
    valid = np.isfinite(profile) & (profile > 0)
    if valid.sum() < 2:
        return float("nan")
    return float(np.trapezoid(profile[valid], x[valid]))


# --------------------------------------------------------------------------
# gate arithmetic
# --------------------------------------------------------------------------

def deficit_gate(chord_m2: float, probe_m2: float, bin_=DEFICIT_BIN):
    """Chord-over-probe deficit and its verdict.

    Returns ``(verdict, value)``.  ``verdict`` is "PASS" inside the bin,
    "outside the bins" outside it, and "REFUSED" when either side is not a
    usable positive number -- never a nan reported as a measurement.
    """
    if not (np.isfinite(chord_m2) and np.isfinite(probe_m2)) or probe_m2 <= 0 or chord_m2 <= 0:
        return "REFUSED", float("nan")
    value = float(chord_m2 / probe_m2)
    return ("PASS" if bin_[0] <= value <= bin_[1] else "outside the bins"), value


def shape_gate(ratio: np.ndarray, r_up: np.ndarray, spread: np.ndarray):
    """Per-cell agreement of a density ratio profile with R_up(x).

    Returns ``(verdict, max_dev_in_spreads, max_abs_dev, n_cells)``.  A cell
    agrees when ``|ratio - R_up| <= spread`` there.  With no admitted cell the
    verdict is "REFUSED"; otherwise "PASS" when every admitted cell agrees and
    "outside the bins" when any does not.
    """
    ratio = np.asarray(ratio, dtype=np.float64)
    r_up = np.asarray(r_up, dtype=np.float64)
    spread = np.asarray(spread, dtype=np.float64)
    usable = np.isfinite(ratio) & np.isfinite(r_up) & np.isfinite(spread) & (spread > 0)
    n_cells = int(usable.sum())
    if n_cells == 0:
        return "REFUSED", float("nan"), float("nan"), 0
    dev = np.abs(ratio[usable] - r_up[usable])
    in_spreads = dev / spread[usable]
    verdict = "PASS" if np.all(in_spreads <= 1.0) else "outside the bins"
    return verdict, float(np.max(in_spreads)), float(np.max(dev)), n_cells


# --------------------------------------------------------------------------
# readers
# --------------------------------------------------------------------------

def load_areas_cm2(path: Path) -> dict[str, dict[str, float]]:
    """Probe face areas in cm^2, keyed ``{"A".."D"}[{"ap_L_cm2","ap_R_cm2"}]``."""
    with open(path, "rb") as handle:
        toml = tomllib.load(handle)
    return {
        key.replace("probe_", ""): {
            "ap_L_cm2": float(block["ap_L_cm2"]),
            "ap_R_cm2": float(block["ap_R_cm2"]),
        }
        for key, block in toml.items()
        if key.startswith("probe_")
    }


def read_face(path: Path, set_id: str, run_id: str) -> dict:
    """One run group of a dead-time product: currents, masks, axes and attrs."""
    with h5py.File(path, "r") as hf:
        x_cm = hf["x_cm"][()]
        grp = hf[f"experiment_sets/{set_id}/{run_id}"]
        out = {
            "x_cm": np.asarray(x_cm, dtype=np.float64),
            "t_ms": np.asarray(grp["inter_sweep_time_s"][()], dtype=np.float64) * 1e3,
            "isat_a": np.asarray(grp["isat_a"][()], dtype=np.float64),
            "isat_a_std": np.asarray(grp["isat_a_std"][()], dtype=np.float64),
            "n_shots_used": np.asarray(grp["n_shots_used"][()], dtype=np.float64),
            "attrs": dict(grp.attrs),
            "root_attrs": dict(hf.attrs),
        }
        for mask_name in ("rail_mask", "state_mask"):
            out[mask_name] = (
                np.asarray(grp[mask_name][()], dtype=bool)
                if mask_name in grp
                else np.zeros(out["isat_a"].shape, dtype=bool)
            )
    return out


def read_overlay_te(path: Path, port: int, t_ms: np.ndarray):
    """Per-port T_e in eV at the given times, and the row's measured flag."""
    with np.load(path, allow_pickle=True) as npz:
        ports = [int(p) for p in npz["port"]]
        index = ports.index(int(port))
        te_time_ms = np.asarray(npz["te_time_ms"], dtype=np.float64)
        te_row = np.asarray(npz["te_mean_ev"], dtype=np.float64)[index]
        # te_row_measured is per (port, sample); a row with zero measured cells
        # is PRIOR-DERIVED, so the row is measured when any cell is.
        measured = bool(np.asarray(npz["te_row_measured"])[index].any())
    return np.interp(np.asarray(t_ms, dtype=np.float64), te_time_ms, te_row), measured


def chord_line_integrated_m2(path: Path, set_id: int, chord_port: int, window_ms=PLATEAU_MS) -> float:
    """Chord line-integrated density in m^-2, averaged over the plateau window."""
    with np.load(path, allow_pickle=True) as npz:
        prefix = f"set{set_id}_p{chord_port}"
        t_ms = np.asarray(npz[f"{prefix}_time_ms"], dtype=np.float64)
        line_cm2 = np.asarray(npz[f"{prefix}_line_integrated_mean_cm2"], dtype=np.float64)
    window = (t_ms >= window_ms[0]) & (t_ms <= window_ms[1])
    if not window.any():
        return float("nan")
    return float(np.nanmean(line_cm2[window]) * 1e4)


def chord_line_average_cm3(path: Path, set_id: int, chord_port: int, window_ms=PLATEAU_MS) -> float:
    """The same chord in the banked read's convention: line average in cm^-3."""
    line_m2 = chord_line_integrated_m2(path, set_id, chord_port, window_ms)
    return line_m2 / 1e4 / PLASMA_DIAMETER_CM


# --------------------------------------------------------------------------
# the row build
# --------------------------------------------------------------------------

def build_port(port: int, repo: Path, window_ms=PLATEAU_MS) -> dict:
    """Both upstream density rows at one port, with masks, ratio and spread."""
    run0, run180 = PORT_RUNS[port]
    isweep = read_face(repo / ISWEEP_ROT0_HDF5, "4", run0)
    isat = read_face(repo / ISAT_ROT180_HDF5, "4", run180)

    if not np.array_equal(isweep["x_cm"], isat["x_cm"]):
        raise ValueError(f"port {port}: the two products disagree on the scan axis")
    if not np.array_equal(isweep["t_ms"], isat["t_ms"]):
        raise ValueError(f"port {port}: the two products disagree on the dead-time axis")
    for face, run_id in ((isweep, run0), (isat, run180)):
        applied = float(face["attrs"].get("probe_a_area_factor_applied", 1.0))
        if applied != 1.0:
            raise ValueError(
                f"run {run_id} carries a probe-A area factor of {applied}; this "
                "instrument reads the delivered current and assumes no factor"
            )

    x_cm = isat["x_cm"]
    x_m = x_cm / 100.0
    t_ms = isat["t_ms"]
    window = plateau_window_mask(t_ms, window_ms)
    if not window.any():
        raise ValueError(f"port {port}: no dead-time window inside {window_ms} ms")

    probe = PROBE_FROM_DIGIT[int(run180[1])]
    if PROBE_FROM_DIGIT[int(run0[1])] != probe:
        raise ValueError(f"port {port}: the rotation pair is not the same probe")
    areas = load_areas_cm2(repo / AREA_TOML)[probe]
    area_isat_m2 = areas[area_key_for_electrode(ChannelKind.ISAT)] * 1e-4
    area_isweep_m2 = areas[area_key_for_electrode(ChannelKind.I_SWEEP)] * 1e-4
    # The area key the placed product stamps, against the helper's live answer.
    stamped_isat = str(isat["attrs"].get("density_area_key", ""))
    helper_isat = density_area_key_for_deadtime_source(run180, ChannelKind.ISAT)

    te_ev, te_measured = read_overlay_te(repo / OVERLAY_NPZ, port, t_ms)
    cs = ion_sound_speed_m_s(te_ev, M_I_AMU)

    n_isat = electron_density_m3(isat["isat_a"], area_isat_m2, cs[None, :])
    n_isweep = electron_density_m3(isweep["isat_a"], area_isweep_m2, cs[None, :])
    n_isat_alt = electron_density_m3(
        isat["isat_a"], areas[helper_isat] * 1e-4, cs[None, :]
    )

    excluded = isat["rail_mask"] | isat["state_mask"]
    positive = (isat["isat_a"] > 0) & (isweep["isat_a"] > 0)
    finite = np.isfinite(n_isat) & np.isfinite(n_isweep)
    admitted = positive & finite & ~excluded

    def plateau_profile(values, keep):
        """Per-position median over the plateau windows of the admitted cells."""
        block = np.where(keep[:, window], values[:, window], np.nan)
        with np.errstate(invalid="ignore"):
            return np.array([
                np.nan if not np.isfinite(row).any() else np.nanmedian(row)
                for row in block
            ])

    n_isat_row = plateau_profile(n_isat, admitted)
    n_isat_alt_row = plateau_profile(n_isat_alt, admitted)
    n_isweep_row = plateau_profile(n_isweep, admitted)

    ratio_cells = np.divide(
        n_isat, n_isweep, out=np.full(n_isat.shape, np.nan), where=n_isweep > 0
    )
    ratio_row = plateau_profile(ratio_cells, admitted)

    # The pair's shot-to-shot spread on the ratio: the two faces' relative
    # per-shot standard deviations added in quadrature, carried onto the ratio.
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.sqrt(
            (isat["isat_a_std"] / isat["isat_a"]) ** 2
            + (isweep["isat_a_std"] / isweep["isat_a"]) ** 2
        )
    spread_row = plateau_profile(np.abs(ratio_cells) * rel, admitted)

    core = np.abs(x_cm) <= CORE_CM
    admitted_core_cells = admitted[:, window][core]
    return {
        "port": port,
        "run_isweep": run0,
        "run_isat": run180,
        "probe": probe,
        "x_cm": x_cm,
        "x_m": x_m,
        "core": core,
        "window_t_ms": t_ms[window],
        "te_ev_window": te_ev[window],
        "te_measured": te_measured,
        "area_isat_key": area_key_for_electrode(ChannelKind.ISAT),
        "area_isat_cm2": areas[area_key_for_electrode(ChannelKind.ISAT)],
        "area_isweep_key": area_key_for_electrode(ChannelKind.I_SWEEP),
        "area_isweep_cm2": areas[area_key_for_electrode(ChannelKind.I_SWEEP)],
        "stamped_isat_key": stamped_isat,
        "helper_isat_key": helper_isat,
        "n_isat_row": n_isat_row,
        "n_isat_alt_row": n_isat_alt_row,
        "n_isweep_row": n_isweep_row,
        "ratio_row": ratio_row,
        "spread_row": spread_row,
        "rail_excluded": int(isat["rail_mask"][:, window].sum()),
        "state_excluded": int(isat["state_mask"][:, window].sum()),
        "core_cells_admitted": int(admitted_core_cells.sum()),
        "core_cells_total": int(admitted_core_cells.size),
        "core_positions_admitted": int(np.isfinite(n_isat_row[core]).sum()),
        "core_positions_total": int(core.sum()),
    }


def r_up_on_core(port: int, x_cm: np.ndarray, core: np.ndarray) -> np.ndarray:
    """The banked R_up(x) sampled onto this port's core positions."""
    reference = R_UP_REFERENCE[port]
    out = np.full(x_cm.shape, np.nan)
    for i, x in enumerate(x_cm):
        if not core[i]:
            continue
        hit = np.isclose(R_UP_REFERENCE_X_CM, x, atol=1e-9)
        if hit.any():
            out[i] = reference[int(np.argmax(hit))]
    return out


# --------------------------------------------------------------------------
# output policy
# --------------------------------------------------------------------------

def checked_output_path(path: Path) -> Path:
    """Refuse an output path that resolves inside the product directory."""
    resolved = path.expanduser().resolve()
    if FORBIDDEN_OUTPUT_DIR in resolved.parts:
        raise ValueError(
            f"--output {path} resolves to {resolved}, inside a {FORBIDDEN_OUTPUT_DIR}/ "
            "directory; this instrument writes measurements, not products, and must "
            "not write into the product chain"
        )
    return resolved


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_DEFAULT,
        help="checkout holding processed/ and config/ (default: the working directory)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("es4_upstream_rows_rot180_isat.csv"),
        help="per-cell CSV; defaults to the working directory, and a path inside "
        f"{FORBIDDEN_OUTPUT_DIR}/ is refused",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def _fmt(value, spec=".4g"):
    """Format a number, or say so when there is not one."""
    return "n/a" if value is None or not np.isfinite(value) else format(value, spec)


def run_control(repo: Path) -> bool:
    """Reproduce the banked chord numbers before any set-4 chord is read."""
    print("CONTROL -- the banked chord read, reproduced from the placed npz")
    print(f"  convention: line average cm^-3 = line-integrated cm^-2 / {PLASMA_DIAMETER_CM:g} cm,")
    print(f"              averaged over the {PLATEAU_MS[0]}-{PLATEAU_MS[1]} ms plateau window")
    print(f"  {'chord':>6} {'set':>4} {'here (cm^-3)':>14} {'banked (cm^-3)':>15} {'rel dev':>9}  verdict")
    ok = True
    for set_id, banked in ((3, CONTROL_ES3_CHORD_CM3), (4, CONTROL_ES4_CHORD_CM3)):
        for chord_port, expected in sorted(banked.items()):
            here = chord_line_average_cm3(repo / INTERF_NPZ, set_id, chord_port)
            dev = abs(here - expected) / expected if np.isfinite(here) else float("nan")
            good = np.isfinite(dev) and dev <= CONTROL_RTOL
            ok = ok and good
            print(f"  {'p' + str(chord_port):>6} {set_id:>4} {here:>14.4e} {expected:>15.4e} "
                  f"{_fmt(dev, '.2e'):>9}  {'match' if good else 'DIFFERS'}")
    print(f"  control: {'PASS' if ok else 'FAIL'} (set 3 is the control; set 4 is printed with it)")
    return ok


def report_port(record: dict, repo: Path) -> list[dict]:
    """Print one port's rows and gates; return its per-cell CSV rows."""
    port = record["port"]
    x_cm, core = record["x_cm"], record["core"]
    is_core_port = port in CORE_PORTS

    print()
    print(f"PORT {port}   rot-0 I_SWEEP run {record['run_isweep']}   "
          f"rot-180 ISAT run {record['run_isat']}   probe {record['probe']}")
    print(f"  areas      ISAT {record['area_isat_key']} = {record['area_isat_cm2']:.6f} cm^2   "
          f"I_SWEEP {record['area_isweep_key']} = {record['area_isweep_cm2']:.6f} cm^2")
    stamp_state = (
        "agrees" if record["stamped_isat_key"] == record["helper_isat_key"]
        else "PREDATES the electrode-keyed helper"
    )
    print(f"  stamp      the placed product stamps density_area_key = "
          f"'{record['stamped_isat_key']}' for the ISAT face and the helper returns "
          f"'{record['helper_isat_key']}' ({stamp_state}); the alternate row under "
          f"the stamped key is printed below")
    te = record["te_ev_window"]
    print(f"  T_e        {te.min():.4f}-{te.max():.4f} eV over the plateau windows "
          f"({'MEASURED' if record['te_measured'] else 'PRIOR-DERIVED'} row)")
    print(f"  masks      rail-excluded cells {record['rail_excluded']}, "
          f"state-excluded cells {record['state_excluded']} (plateau windows only)")
    print(f"  core       {record['core_cells_admitted']} of {record['core_cells_total']} cells "
          f"admitted, {record['core_positions_admitted']} of {record['core_positions_total']} positions")
    if not is_core_port:
        print("  LABEL      p21 is reported for its EDGE cells only: the run-43 state mask "
              "removes the whole core band, so the core gates refuse.")

    # A T_e move rescales the whole row; print the sensitivity as a worked factor.
    te_mid = float(np.median(te))
    print(f"  sqrt(T_e)  n_e scales by sqrt(T_e_old/T_e_new); at T_e = {te_mid:.4f} eV a "
          f"+10% T_e move multiplies the row by {sqrt_te_rescale(te_mid, 1.1 * te_mid):.4f}, "
          f"a -10% move by {sqrt_te_rescale(te_mid, 0.9 * te_mid):.4f}")

    # --- profiles -------------------------------------------------------
    n_isat, n_isweep = record["n_isat_row"], record["n_isweep_row"]
    ratio, spread = record["ratio_row"], record["spread_row"]
    r_up = r_up_on_core(port, x_cm, core)
    shown = core if is_core_port else np.isfinite(n_isat) & ~core
    label = "core" if is_core_port else "edge"
    if shown.any():
        with np.printoptions(precision=3, suppress=True, linewidth=200):
            print(f"  x_cm ({label})            {x_cm[shown]}")
            print(f"  n_ISAT180 (1e18 m^-3)  {n_isat[shown] / 1e18}")
            print(f"  n_isweep0 (1e18 m^-3)  {n_isweep[shown] / 1e18}")
            print(f"  ratio ISAT180/isweep0  {ratio[shown]}")
            print(f"  shot-to-shot spread    {spread[shown]}")
            if is_core_port:
                print(f"  R_up (banked)          {r_up[shown]}")
    else:
        print(f"  no {label} position carries an admitted cell")

    # --- gate (a) chord deficit ----------------------------------------
    chord_m2 = chord_line_integrated_m2(repo / INTERF_NPZ, 4, CHORD_PORT[port])
    probe_m2 = line_integral_m2(n_isat, record["x_m"])
    probe_alt_m2 = line_integral_m2(record["n_isat_alt_row"], record["x_m"])
    isweep_m2 = line_integral_m2(n_isweep, record["x_m"])
    if is_core_port and record["core_positions_admitted"] > 0:
        verdict, value = deficit_gate(chord_m2, probe_m2)
        _, value_alt = deficit_gate(chord_m2, probe_alt_m2)
        # The gate asks for the SAME deficit, so the rot-0 row the overlay
        # currently carries is printed beside it as the comparison.
        _, value_isweep = deficit_gate(chord_m2, isweep_m2)
        print(f"  GATE (a) chord deficit p{port} (chord p{CHORD_PORT[port]}): {verdict} "
              f"deficit = {_fmt(value, '.3f')} (bin {DEFICIT_BIN[0]}-{DEFICIT_BIN[1]}); "
              f"chord {chord_m2:.4e} m^-2, probe {probe_m2:.4e} m^-2; "
              f"under the stamped key deficit = {_fmt(value_alt, '.3f')}; "
              f"the rot-0 I_SWEEP row the overlay carries reads "
              f"{_fmt(value_isweep, '.3f')}")
    else:
        print(f"  GATE (a) chord deficit p{port}: REFUSED "
              f"({record['core_positions_admitted']} of {record['core_positions_total']} cells)")

    # --- gate (b) radial shape ------------------------------------------
    if is_core_port:
        verdict, in_spreads, abs_dev, n_cells = shape_gate(
            np.where(core, ratio, np.nan), r_up, np.where(core, spread, np.nan)
        )
        if n_cells == 0:
            print(f"  GATE (b) radial shape p{port}: REFUSED "
                  f"(0 of {record['core_positions_total']} cells)")
        else:
            print(f"  GATE (b) radial shape p{port}: {verdict} over {n_cells} core cells; "
                  f"max deviation {abs_dev:.4f} in ratio, {in_spreads:.3f} of the "
                  f"pair's shot-to-shot spread")
    else:
        print(f"  GATE (b) radial shape p{port}: REFUSED "
              f"(0 of {record['core_positions_total']} cells)")

    rows = []
    for i, x in enumerate(x_cm):
        rows.append({
            "port": port,
            "run_isat_rot180": record["run_isat"],
            "run_isweep_rot0": record["run_isweep"],
            "probe": record["probe"],
            "x_cm": f"{x:.1f}",
            "in_core": int(bool(core[i])),
            "n_isat_rot180_m3": _fmt(n_isat[i], ".6e"),
            "n_isat_rot180_stamped_key_m3": _fmt(record["n_isat_alt_row"][i], ".6e"),
            "n_isweep_rot0_m3": _fmt(n_isweep[i], ".6e"),
            "ratio": _fmt(ratio[i], ".6f"),
            "shot_to_shot_spread": _fmt(spread[i], ".6f"),
            "r_up_banked": _fmt(r_up[i], ".6f"),
        })
    return rows


def main(argv=None) -> int:
    args = parse_args(argv)
    output = checked_output_path(args.output)
    repo = args.repo_root.expanduser().resolve()

    print("ES4 upstream density rows from the rot-180 ISAT face")
    print(f"  repo root  {repo}")
    print(f"  ISAT face  {ISAT_ROT180_HDF5}")
    print(f"  I_SWEEP    {ISWEEP_ROT0_HDF5}")
    print(f"  T_e        {OVERLAY_NPZ} (per-port row, interpolated to the window midpoints)")
    print(f"  chords     {INTERF_NPZ}")
    print(f"  areas      {AREA_TOML}, by collecting electrode: ISAT -> ap_R_cm2, "
          "I_SWEEP -> ap_L_cm2")
    print(f"  plateau    {PLATEAU_MS[0]}-{PLATEAU_MS[1]} ms, the scoring plateau window")
    print(f"  core band  |x| <= {CORE_CM:g} cm")
    print()

    control_ok = run_control(repo)

    rows: list[dict] = []
    for port in sorted(PORT_RUNS):
        rows.extend(report_port(build_port(port, repo), repo))

    print()
    print(f"control {'PASS' if control_ok else 'FAIL'}; p50 (run 48) is rot-0 only and is not touched.")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
