"""Probe-A p11 area factor implied by the port-20 interferometer chord.

What this measures
------------------
Probes B, C and D are calibrated against interferometer chords by
``scripts/calibrate_probe_areas.py``: for a probe sitting one port from a chord
it solves ``A_p`` so that the probe's own line-integrated density matches the
chord's, at every dead-time cycle of the stable plasma plateau.  Probe A has no
chord within nine ports of either of its stations (p11 or p50), so its effective
area is NOT calibrated that way.  Instead it inherits probe B's area and is
divided by one empirical factor fixed by an ordering prior (see below).

The only data-in-hand bound on that factor is the transfer from p11 to the
port-20 chord, and this script computes it.  For each experiment set it asks:
what area factor would ``calibrate_probe_area_m2``'s method assign to probe A at
p11 if the port-20 chord were treated as p11's calibration chord?

  probe line integral   P11 = trapz(I_sat(x) / [e * C_s(x) * exp(-1/2)], x)
  chord line density    L20 = interferometer int(n_e dl) at port 20
  declared transfer     r   = n(p11) / n(p20)          <- a MODEL input
  implied p11 chord     int(n_e dl)|p11 = r * L20
  chord-implied area    A_A = P11 / (r * L20)
  chord-implied factor  f_chord = A_ref / A_A

``A_ref`` is probe B's experiment-set-1 calibrated area for the same face, which
is exactly the area probe A inherits.  ``f_chord`` is therefore on the same
footing as the canonical factor and directly comparable with it.

The transfer ratio is a MODEL input
-----------------------------------
p11 and the port-20 chord are nine port spacings (287.55 cm) apart, so nothing
in the data says how density changes between them.  ``r`` is supplied by the 1D
transport model, is declared per experiment set on the command line, and is the
only non-measured ingredient.  Because ``f_chord`` is exactly proportional to
``r``, its logarithmic sensitivity ``d ln f / d ln r`` is 1 by construction: a
ten percent error in the model's axial ratio is a ten percent error in
``f_chord``, and the script prints the sensitivity so the reader can see how
much of any verdict is the model's.  One experiment set has no declared ratio;
for it the ratio is swept as a free axis instead.

The same one-for-one dependence sits, unstated, inside the existing B/C/D
calibration: probe B at p21 is matched to the port-20 chord one port spacing
away with an implicit transfer ratio of 1.

The prior being tested
----------------------
``config/may2026_probe_a_area_calibration.toml`` holds the canonical factor.  It
is the geometric mean of a bracket fixed on experiment set 1 by two ordering
priors on FWHM-core mean density, ``n(p11) <= n(p21)`` and ``n(p50) >= n(p41)``.
It is an effective current-to-area normalization, not a geometric claim, and it
is applied to both probe-A faces in every experiment set.

Control
-------
The canonical factor is fixed by an ordering prior, not by a chord, so it does
not have to reproduce ``f_chord`` at any declared ratio.  What it does have is
an IMPLIED transfer ratio: the ``r`` at which ``f_chord`` equals it.  The script
inverts for that ratio at every set, then feeds it back through the same
arithmetic to confirm the canonical factor is recovered.  Reading the implied
ratio beside the declared one shows the disagreement as an axial-ratio
statement rather than an area one.

Pre-registered bins
-------------------
With ``|d| = |ln(f_chord / f_canonical)|`` evaluated at the declared ratio:

  d < 0.15 at BOTH set 2 and set 3  ->  "the factor transfers across days"
  d >= 0.15 at either               ->  "day-dependent"

Set 1 is the control and set 4 is reported alongside; neither decides the bin.
The script prints the numbers and the bin each set lands in.  It does not
interpret them.

Inputs
------
  processed/isweep_deadtime_profiles.hdf5     rot-0 dead-time current profiles
  processed/te_filled.hdf5                    filled T_e, per experiment set
  processed/interferometer_experiment_set_stats.npz
  processed/probe_area_calibration.toml       probe B reference areas
  config/may2026_probe_a_area_calibration.toml

Reading them from a checkout other than the one holding this script is what
``--repo-root`` is for; the products are opened read-only.

Usage
-----
  PYTHONPATH=src python scripts/probe_a_p20_chord_transfer.py \
      --ratio 1:1.08 --ratio 3:0.98 --ratio 4:0.89 \
      --free-axis 2 --free-axis-ratios 1.00,1.08,0.98 \
      --output processed/probe_a_p20_chord_transfer.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import tomllib
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from bapsf_lapd.density import (
    calibrate_probe_area_m2,
    ion_sound_speed_m_s,
    load_probe_a_area_calibration,
)


PROFILE_HDF5 = Path("processed/isweep_deadtime_profiles.hdf5")
TE_FILLED_HDF5 = Path("processed/te_filled.hdf5")
INTERF_NPZ = Path("processed/interferometer_experiment_set_stats.npz")
AREA_CALIB_TOML = Path("processed/probe_area_calibration.toml")
PROBE_A_CALIB_TOML = Path("config/may2026_probe_a_area_calibration.toml")

# He-4 ion mass in amu; the value scripts/calibrate_probe_areas.py calibrates with.
M_I_AMU = 4.003

# Stable plasma plateau, mirroring the window the B/C/D calibration uses.
CALIB_T_MIN_MS = 10.0
CALIB_T_MAX_MS = 19.0

# Probe scan positions (51 points, 1 cm spacing, centred at x = 0).
X_CM = np.linspace(-25.0, 25.0, 51)
X_M = X_CM * 1e-2

# Probe A's upstream station, and the chord whose transfer bounds it.
PROBE_A_PORT = 11
TRANSFER_CHORD_PORT = 20

# Probe A inherits this probe's calibrated area.
REFERENCE_PROBE = "B"

# Pre-registered bin threshold on |ln(f_chord / f_canonical)|.
BIN_THRESHOLD = 0.15

# The experiment sets whose bins decide the pre-registered verdict.
DECIDING_SETS = (2, 3)

# The set the canonical factor was fixed on; it is the arithmetic control.
CONTROL_SET = 1


def _experiment_sets(profile_hdf: h5py.File) -> list[int]:
    return sorted(int(key) for key in profile_hdf["experiment_sets"].keys())


def _probe_a_run(profile_hdf: h5py.File, set_id: int) -> str:
    """Return the run id carrying probe A's p11 dead-time profile for a set."""
    group = profile_hdf[f"experiment_sets/{set_id}"]
    matches = [
        run_id for run_id in sorted(group.keys())
        if int(group[run_id].attrs["port"]) == PROBE_A_PORT
    ]
    if len(matches) != 1:
        raise ValueError(
            f"experiment set {set_id} has {len(matches)} port-{PROBE_A_PORT} "
            f"runs in {PROFILE_HDF5}; expected exactly one"
        )
    return matches[0]


def _reference_areas_m2(path: Path) -> dict[str, float]:
    with open(path, "rb") as stream:
        data = tomllib.load(stream)[f"probe_{REFERENCE_PROBE}"]
    return {
        "ap_L_cm2": float(data["ap_L_cm2"]) * 1e-4,
        "ap_R_cm2": float(data["ap_R_cm2"]) * 1e-4,
    }


def _chord_line_integrated_m2(
    npz: dict[str, np.ndarray],
    set_id: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (time_ms, line_integrated_m2, n_shots) for the transfer chord."""
    prefix = f"set{set_id}_p{TRANSFER_CHORD_PORT}"
    time_ms = np.asarray(npz[f"{prefix}_time_ms"], dtype=np.float64)
    # The NPZ stores line-integrated density in cm^-2; convert to m^-2.
    line_integrated_m2 = np.asarray(
        npz[f"{prefix}_line_integrated_mean_cm2"], dtype=np.float64
    ) * 1e4
    n_shots = int(np.atleast_1d(npz[f"{prefix}_n"]).ravel()[0])
    return time_ms, line_integrated_m2, n_shots


def _te_rows_for_port(
    te_hdf: h5py.File,
    set_id: int,
    port_z_cm: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (te_filled(x, cycle), cycle_time_ms, matched z) at a port row."""
    group = te_hdf[f"experiment_sets/{set_id}"]
    if not np.allclose(group["x_cm"][()], X_CM):
        raise ValueError(f"{TE_FILLED_HDF5} x grid does not match the probe x grid")
    z_cm = group["z_cm"][()]
    z_idx = int(np.argmin(np.abs(z_cm - port_z_cm)))
    return group["te_filled"][()][z_idx], group["cycle_time_ms"][()], float(z_cm[z_idx])


def chord_transfer_area_m2(
    isat_profile_a: np.ndarray,
    cs_profile_m_s: np.ndarray,
    x_m: np.ndarray,
    chord_line_integrated_m2: float,
    transfer_ratio: float,
) -> float:
    """Probe area implied by a chord one transfer ratio away from the probe.

    The chord measures ``int(n_e dl)`` at its own port.  ``transfer_ratio`` is
    the declared density ratio ``n(probe port) / n(chord port)``, so the line
    integral the chord implies AT THE PROBE is ``transfer_ratio`` times the
    measured one.  Feeding that to ``calibrate_probe_area_m2`` gives the area
    the chord method would assign at the probe.

    Returns NaN when the underlying calibration cannot be formed.
    """
    if not np.isfinite(transfer_ratio) or transfer_ratio <= 0.0:
        raise ValueError("transfer_ratio must be finite and positive")
    return calibrate_probe_area_m2(
        isat_profile_a,
        cs_profile_m_s,
        x_m,
        transfer_ratio * chord_line_integrated_m2,
    )


def probe_a_factor(reference_area_m2: float, probe_a_area_m2: float) -> float:
    """Factor by which the inherited reference area is divided at probe A.

    Probe A's effective area is ``reference_area_m2 / factor``, equivalently the
    factor is applied once to raw probe-A current.
    """
    if not np.isfinite(probe_a_area_m2) or probe_a_area_m2 <= 0.0:
        return float("nan")
    return float(reference_area_m2 / probe_a_area_m2)


def collect_set(
    profile_hdf: h5py.File,
    te_hdf: h5py.File,
    interf_npz: dict[str, np.ndarray],
    reference_areas_m2: dict[str, float],
    set_id: int,
) -> dict[str, Any]:
    """Gather the per-cycle probe and chord line integrals for one set."""
    run_id = _probe_a_run(profile_hdf, set_id)
    group = profile_hdf[f"experiment_sets/{set_id}/{run_id}"]
    if not np.allclose(profile_hdf["x_cm"][()], X_CM):
        raise ValueError(f"{PROFILE_HDF5} x grid does not match the probe x grid")

    area_key = str(group.attrs["density_area_key"])
    reference_area_m2 = reference_areas_m2[area_key]

    profile_time_ms = group["inter_sweep_time_s"][()] * 1000.0
    # isat_a_raw carries the measured current; isat_a has the canonical probe-A
    # factor already multiplied in, and using it would make this circular.
    isat_a = group["isat_a_raw"][()]
    n_shots_used = group["n_shots_used"][()]

    te_filled, te_time_ms, te_z_cm = _te_rows_for_port(
        te_hdf, set_id, float(group.attrs["z_cm"])
    )
    chord_time_ms, chord_m2, chord_n_shots = _chord_line_integrated_m2(interf_npz, set_id)

    in_window = (profile_time_ms >= CALIB_T_MIN_MS) & (profile_time_ms <= CALIB_T_MAX_MS)
    probe_line_integrals: list[float] = []
    chord_line_integrals: list[float] = []
    cycle_times_ms: list[float] = []
    used_cycles: list[int] = []

    for k, time_ms in enumerate(profile_time_ms):
        if not in_window[k]:
            continue
        te_k = np.array([
            np.interp(time_ms, te_time_ms, te_filled[i, :])
            for i in range(te_filled.shape[0])
        ])
        with np.errstate(invalid="ignore"):
            cs_k = ion_sound_speed_m_s(te_k, M_I_AMU)
        chord_k = float(np.interp(time_ms, chord_time_ms, chord_m2))
        if not np.isfinite(chord_k) or chord_k <= 0.0:
            continue
        # Unit denominator returns the probe's own line integral unchanged.
        probe_k = calibrate_probe_area_m2(isat_a[:, k], cs_k, X_M, 1.0)
        if not np.isfinite(probe_k):
            continue
        probe_line_integrals.append(probe_k)
        chord_line_integrals.append(chord_k)
        cycle_times_ms.append(float(time_ms))
        used_cycles.append(k)

    if not used_cycles:
        raise ValueError(
            f"experiment set {set_id} run {run_id} has no usable dead-time cycle "
            f"in {CALIB_T_MIN_MS:g}-{CALIB_T_MAX_MS:g} ms"
        )

    shots = n_shots_used[:, used_cycles]
    chord_in_window = int(
        ((chord_time_ms >= CALIB_T_MIN_MS) & (chord_time_ms <= CALIB_T_MAX_MS)).sum()
    )
    return {
        "set_id": set_id,
        "run_id": run_id,
        "port": int(group.attrs["port"]),
        "z_cm": float(group.attrs["z_cm"]),
        "te_z_cm": te_z_cm,
        "area_key": area_key,
        "reference_area_m2": reference_area_m2,
        "source_channel": str(group.attrs["deadtime_source_channel"]),
        "source_inverted": bool(group.attrs["deadtime_source_invert_polarity"]),
        "source_overridden": bool(group.attrs["deadtime_source_overridden"]),
        "probe_line_integrals_m2": np.array(probe_line_integrals),
        "chord_line_integrals_m2": np.array(chord_line_integrals),
        "cycle_times_ms": np.array(cycle_times_ms),
        "n_cycles_used": len(used_cycles),
        "n_cycles_in_window": int(in_window.sum()),
        "n_shots_used_min": int(np.min(shots)),
        "n_shots_used_max": int(np.max(shots)),
        "n_shots_used_median": float(np.median(shots)),
        "chord_n_shots": chord_n_shots,
        "chord_samples_in_window": chord_in_window,
    }


def evaluate_ratio(record: dict[str, Any], transfer_ratio: float) -> dict[str, float]:
    """Mean chord-implied area and factor for one declared transfer ratio."""
    if not np.isfinite(transfer_ratio) or transfer_ratio <= 0.0:
        raise ValueError("transfer_ratio must be finite and positive")
    areas = record["probe_line_integrals_m2"] / (
        transfer_ratio * record["chord_line_integrals_m2"]
    )
    # The B/C/D calibration averages the per-cycle area, so this does too.
    mean_area = float(np.mean(areas))
    factors = record["reference_area_m2"] / areas
    return {
        "transfer_ratio": float(transfer_ratio),
        "probe_a_area_m2": mean_area,
        "f_chord": probe_a_factor(record["reference_area_m2"], mean_area),
        "f_chord_cycle_std": float(np.std(factors, ddof=1)) if len(factors) > 1 else float("nan"),
        "f_chord_cycle_min": float(np.min(factors)),
        "f_chord_cycle_max": float(np.max(factors)),
    }


def implied_transfer_ratio(record: dict[str, Any], target_factor: float) -> float:
    """Transfer ratio at which f_chord equals a target factor.

    ``f_chord`` is exactly proportional to the ratio, so one evaluation inverts it.
    """
    unit = evaluate_ratio(record, 1.0)["f_chord"]
    return float(target_factor / unit)


def log_sensitivity(record: dict[str, Any], transfer_ratio: float) -> float:
    """Measured ``d ln f_chord / d ln r`` by central difference about a ratio."""
    step = 0.01
    hi = evaluate_ratio(record, transfer_ratio * (1.0 + step))["f_chord"]
    lo = evaluate_ratio(record, transfer_ratio * (1.0 - step))["f_chord"]
    return float(
        (np.log(hi) - np.log(lo)) / (np.log(1.0 + step) - np.log(1.0 - step))
    )


def bin_label(f_chord: float, canonical_factor: float) -> tuple[float, str]:
    deviation = float(abs(np.log(f_chord / canonical_factor)))
    return deviation, "in-bin" if deviation < BIN_THRESHOLD else "out-of-bin"


def in_bin_ratio_interval(
    record: dict[str, Any],
    canonical_factor: float,
) -> tuple[float, float]:
    """Transfer-ratio interval over which a set lands inside the bin."""
    unit = evaluate_ratio(record, 1.0)["f_chord"]
    return (
        float(canonical_factor * np.exp(-BIN_THRESHOLD) / unit),
        float(canonical_factor * np.exp(BIN_THRESHOLD) / unit),
    )


def _parse_ratios(items: list[str]) -> dict[int, float]:
    ratios: dict[int, float] = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"--ratio expects SET:VALUE, got {item!r}")
        set_text, value_text = item.split(":", 1)
        ratios[int(set_text)] = float(value_text)
    return ratios


def _parse_float_list(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="checkout holding the processed products and config (default: cwd); "
             "the products are opened read-only",
    )
    parser.add_argument(
        "--ratio",
        action="append",
        default=[],
        metavar="SET:VALUE",
        help="declared model transfer ratio n(p11)/n(p20) for an experiment set; "
             "repeatable, and a MODEL input rather than a measurement",
    )
    parser.add_argument(
        "--free-axis",
        action="append",
        type=int,
        default=[],
        metavar="SET",
        help="experiment set with no declared ratio; its ratio is swept instead",
    )
    parser.add_argument(
        "--free-axis-ratios",
        default="1.00,1.08,0.98",
        help="comma-separated ratios to evaluate on every free axis",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("processed/probe_a_p20_chord_transfer.csv"),
        help="path for the result table",
    )
    return parser


CSV_FIELDS = (
    "experiment_set",
    "role",
    "run_id",
    "port",
    "z_cm",
    "source_channel",
    "area_key",
    "reference_area_m2",
    "transfer_ratio",
    "ratio_source",
    "probe_line_integrated_m2",
    "chord_line_integrated_m2",
    "probe_a_area_m2",
    "f_chord",
    "f_chord_cycle_std",
    "ln_f_over_canonical",
    "bin",
    "dln_f_dln_ratio",
    "implied_ratio_at_canonical",
    "n_cycles_used",
    "n_shots_used_median",
    "chord_n_shots",
)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.repo_root
    declared = _parse_ratios(args.ratio)
    free_axis_sets = set(args.free_axis)
    free_axis_ratios = _parse_float_list(args.free_axis_ratios)

    missing = [
        path for path in (
            PROFILE_HDF5, TE_FILLED_HDF5, INTERF_NPZ, AREA_CALIB_TOML, PROBE_A_CALIB_TOML
        )
        if not (root / path).exists()
    ]
    if missing:
        sys.exit(
            "Missing required input(s) under "
            f"{root}:\n  " + "\n  ".join(str(path) for path in missing)
        )

    canonical = load_probe_a_area_calibration(root / PROBE_A_CALIB_TOML)
    reference_areas_m2 = _reference_areas_m2(root / AREA_CALIB_TOML)
    interf_npz = dict(np.load(root / INTERF_NPZ))

    print("Probe-A p11 factor implied by the port-20 chord")
    print("=" * 72)
    print(f"repo root                 {root}")
    print(f"probe profiles            {PROFILE_HDF5}  dataset 'isat_a_raw'")
    print(f"filled T_e                {TE_FILLED_HDF5}  dataset 'te_filled'")
    print(f"chord                     {INTERF_NPZ}")
    print(f"  keys                    set<N>_p{TRANSFER_CHORD_PORT}_line_integrated_mean_cm2 "
          f"(x 1e4 -> m^-2), _time_ms, _n")
    print(f"reference areas           {AREA_CALIB_TOML}  probe {REFERENCE_PROBE}")
    print(f"canonical probe-A factor  {PROBE_A_CALIB_TOML}  {canonical.factor:.6f}")
    print(f"  bracket                 [{canonical.lower_bound:.6f}, {canonical.upper_bound:.6f}]"
          "  from n(p11) <= n(p21) and n(p50) >= n(p41)")
    print(f"plateau window            {CALIB_T_MIN_MS:g}-{CALIB_T_MAX_MS:g} ms, "
          "the window the B/C/D calibration uses")
    print(f"integral                  trapz(I_sat / [e * C_s * exp(-1/2)], x) over "
          f"{X_CM[0]:g}..{X_CM[-1]:g} cm, C_s from filled T_e, m_i = {M_I_AMU} amu")
    print(f"pre-registered bin        |ln(f_chord / {canonical.factor:.6f})| < {BIN_THRESHOLD} "
          f"at sets {' and '.join(str(s) for s in DECIDING_SETS)}")
    print()

    with h5py.File(root / PROFILE_HDF5, "r") as profile_hdf, \
            h5py.File(root / TE_FILLED_HDF5, "r") as te_hdf:
        records = {
            set_id: collect_set(profile_hdf, te_hdf, interf_npz, reference_areas_m2, set_id)
            for set_id in _experiment_sets(profile_hdf)
        }

    rows: list[dict[str, Any]] = []
    # Per deciding set: the bin label at each ratio it was evaluated at.
    deciding_bins: dict[int, dict[float, str]] = {}

    for set_id in sorted(records):
        record = records[set_id]
        implied = implied_transfer_ratio(record, canonical.factor)
        if set_id in DECIDING_SETS:
            role = "deciding"
        elif set_id == CONTROL_SET:
            role = "control"
        else:
            role = "reported"
        if set_id in free_axis_sets:
            ratios = [(value, "free-axis") for value in free_axis_ratios]
        elif set_id in declared:
            ratios = [(declared[set_id], "declared-model")]
        else:
            ratios = []

        print(f"experiment set {set_id}  ({role})")
        print(f"  run {record['run_id']}  port {record['port']}  z {record['z_cm']:g} cm  "
              f"face channel {record['source_channel']}"
              f"{' (source overridden)' if record['source_overridden'] else ''}")
        print(f"  reference area key {record['area_key']}  "
              f"A_ref = {record['reference_area_m2']:.6e} m^2")
        print(f"  T_e row matched at z {record['te_z_cm']:g} cm")
        print(f"  dead-time cycles used {record['n_cycles_used']} of "
              f"{record['n_cycles_in_window']} in window; "
              f"shots per cell median {record['n_shots_used_median']:g} "
              f"(min {record['n_shots_used_min']}, max {record['n_shots_used_max']})")
        print(f"  chord shots {record['chord_n_shots']}, "
              f"{record['chord_samples_in_window']} chord samples span the window, "
              "sampled at each dead-time midpoint")
        print(f"  P11 mean {np.mean(record['probe_line_integrals_m2']):.6e} m^-2   "
              f"L20 mean {np.mean(record['chord_line_integrals_m2']):.6e} m^-2")
        low, high = in_bin_ratio_interval(record, canonical.factor)
        print(f"  implied ratio at the canonical factor  {implied:.4f}"
              "   (the axial ratio the prior is equivalent to)")
        print(f"  ratio interval landing inside the bin  [{low:.4f}, {high:.4f}]")

        if not ratios:
            print("  no ratio declared and no free axis requested; nothing evaluated")
        for ratio, source in ratios:
            result = evaluate_ratio(record, ratio)
            deviation, label = bin_label(result["f_chord"], canonical.factor)
            sensitivity = log_sensitivity(record, ratio)
            print(f"    r = {ratio:.4f} ({source})   A_A = {result['probe_a_area_m2']:.6e} m^2   "
                  f"f_chord = {result['f_chord']:.4f}"
                  f" +/- {result['f_chord_cycle_std']:.4f} (cycle sd)   "
                  f"ln(f/f_can) = {np.log(result['f_chord'] / canonical.factor):+.4f}   "
                  f"|d| = {deviation:.4f}   {label}")
            if set_id in DECIDING_SETS:
                deciding_bins.setdefault(set_id, {})[ratio] = label
            rows.append({
                "experiment_set": set_id,
                "role": role,
                "run_id": record["run_id"],
                "port": record["port"],
                "z_cm": f"{record['z_cm']:.2f}",
                "source_channel": record["source_channel"],
                "area_key": record["area_key"],
                "reference_area_m2": f"{record['reference_area_m2']:.6e}",
                "transfer_ratio": f"{ratio:.4f}",
                "ratio_source": source,
                "probe_line_integrated_m2": f"{np.mean(record['probe_line_integrals_m2']):.6e}",
                "chord_line_integrated_m2": f"{np.mean(record['chord_line_integrals_m2']):.6e}",
                "probe_a_area_m2": f"{result['probe_a_area_m2']:.6e}",
                "f_chord": f"{result['f_chord']:.4f}",
                "f_chord_cycle_std": f"{result['f_chord_cycle_std']:.4f}",
                "ln_f_over_canonical": f"{np.log(result['f_chord'] / canonical.factor):+.4f}",
                "bin": label,
                "dln_f_dln_ratio": f"{sensitivity:.6f}",
                "implied_ratio_at_canonical": f"{implied:.4f}",
                "n_cycles_used": record["n_cycles_used"],
                "n_shots_used_median": f"{record['n_shots_used_median']:g}",
                "chord_n_shots": record["chord_n_shots"],
            })
        print()

    print("control: the canonical factor recovered from its own implied ratio")
    print("-" * 72)
    for set_id in sorted(records):
        record = records[set_id]
        implied = implied_transfer_ratio(record, canonical.factor)
        recovered = evaluate_ratio(record, implied)["f_chord"]
        print(f"  set {set_id}: r_implied = {implied:.6f}  ->  f_chord = {recovered:.6f}  "
              f"(canonical {canonical.factor:.6f}, "
              f"residual {recovered - canonical.factor:+.2e})")
    print()

    print("pre-registered bin verdict")
    print("-" * 72)
    missing_sets = [set_id for set_id in DECIDING_SETS if set_id not in deciding_bins]
    if missing_sets:
        print("  deciding set(s) not evaluated: "
              + ", ".join(str(set_id) for set_id in missing_sets))
    else:
        # A deciding set on a free axis is read at each axis point in turn; a
        # deciding set with a declared ratio is held at it.
        axis_points = sorted({
            ratio
            for set_id in DECIDING_SETS if set_id in free_axis_sets
            for ratio in deciding_bins[set_id]
        }) or [None]
        for axis_point in axis_points:
            labels = {}
            for set_id in DECIDING_SETS:
                evaluated = deciding_bins[set_id]
                key = axis_point if set_id in free_axis_sets else next(iter(evaluated))
                labels[set_id] = (evaluated[key], key)
            verdict = (
                "the factor transfers across days"
                if all(label == "in-bin" for label, _ in labels.values())
                else "day-dependent"
            )
            detail = "  ".join(
                f"set {s} r={labels[s][1]:.4f} {labels[s][0]}" for s in DECIDING_SETS
            )
            print(f"  {detail}  ->  {verdict}")
    print()
    print("sensitivity of f_chord to the transfer ratio")
    print("-" * 72)
    print("  f_chord is exactly proportional to r, so d ln f_chord / d ln r = 1 "
          "by construction;")
    print("  the whole of any disagreement with the canonical factor is carried "
          "one-for-one by")
    print("  the model's axial ratio.  Measured by central difference:")
    for set_id in sorted(records):
        print(f"    set {set_id}: d ln f_chord / d ln r = "
              f"{log_sensitivity(records[set_id], 1.0):.6f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
