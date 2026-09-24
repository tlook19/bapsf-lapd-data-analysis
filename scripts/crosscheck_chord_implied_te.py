"""Cross-check the Langmuir density against the interferometer chord through T_e.

What this measures
------------------
The probe density is an Isat-to-density conversion that carries ``T_e`` inside
it::

    n_e = I_sat / [exp(-1/2) * A_p * e * C_s],    C_s = sqrt(k_B T_e / m_i)

so the probe density scales as ``n propto I_sat / sqrt(T_e)``.  The
interferometer chord measures a phase shift, and its line density
``int n_e dl`` carries no ``T_e`` at all.  Comparing the probe's own radial
line integral with the chord's line density at the same axial position
therefore reads back on the ``T_e`` that went into the conversion:

    r          = L_probe / L_chord
    T_e,implied = T_e,used * r^2

``T_e,implied`` is the temperature at which the probe's line integral would
equal the chord's.  The exponent is exact and is a UNIFORM rescale of the whole
``T_e(x)`` profile: substituting ``T_e(x) -> r^2 T_e(x)`` divides every probe
density by ``r`` and therefore divides ``L_probe`` by ``r``.  Because the move
is a pure scale factor, the choice of representative scalar for ``T_e,used``
only sets the level the pair is quoted at, never the ratio.

This is a CROSS-CHECK and nothing else.  No fit is performed, no correction is
applied here or anywhere else, and no product of this script is read by
``scripts/export_es1_sim1d_overlay.py`` or by any other member of the product
chain.  A disagreement is a disclosure, not a calibration.

Ports
-----
Each interferometer chord is paired with its nearest Langmuir port:

    chord p20 (z = 757.60 cm)  <->  probe p21 (z = 789.55 cm)
    chord p29 (z = 1045.15 cm) <->  probe p29 (z = 1045.15 cm)  -- coincident
    chord p40 (z = 1396.60 cm) <->  probe p41 (z = 1428.55 cm)

Both z values come from the products: the probe z from the axial rows of
``te_filled.hdf5`` / ``density_profiles_isweep.hdf5`` and the chord z from the
package's port ladder, which is what the overlay exporter uses for the same
chords.  Only p29 is coincident; the p20 and p40 pairs are one port spacing
(31.95 cm) apart, and the axial density gradient across that spacing is a
systematic on ``r`` that this script does not model.

Which density
-------------
``processed/density_profiles_isweep.hdf5`` (``n_e_m3``): the ``i_sweep``
channel, which is the UPSTREAM probe face at rot-0, built by
``scripts/plot_density_profiles.py`` from the dead-time Isat profiles, the
filled ``T_e`` and the calibrated face areas.  It is chosen over
``processed/density_mach.hdf5``'s ``n_e_L_m3`` because it is the product the
overlay exporter's density family reads (its ``DENSITY_HDF5``), so the ``T_e``
this cross-check reads back on is the one standing behind the exported
comparands; and because it holds exactly one rot-0 run per port, where the
``n_e_L`` face of the Mach product is the upstream face only at rot-0.  The
density is SIGNED: a negative far-skirt cell is noise about zero and is kept
with its sign, per that product's convention.

The ``T_e`` this reads back on
------------------------------
``T_e,used`` is recomputed the way the density product built it: the
``te_filled`` row at the port's z, linearly interpolated in time onto the
dead-time sample grid, cell by cell in x.  It is reduced to one number as the
DENSITY-WEIGHTED MEAN over the scan and over the samples in the window,

    T_e,used = sum_k int n(x,t_k) T_e(x,t_k) dx / sum_k int n(x,t_k) dx

with the x integrals by the trapezoid rule on the cells where both the density
and the temperature are finite.  The weight is the SIGNED density, which is the
same convention the overlay's ``te_column_ev`` uses; where negative skirt cells
carry weight the result is not a convex combination of its own nodes and can
sit outside the range of the row's ``T_e``.

A row with ``te_row_measured_fraction = 0`` reads back on a RECONSTRUCTED
``T_e`` -- that port's filled row carries no measured cell at any sample in the
window, so it was built from its neighbours, the scrape-off-layer anchors and
the end-plate sentinels -- and its ``pull`` is therefore not a measurement of
that probe.  The fraction is the share of the window's samples at which the row
carries at least one measured cell (``te_row_measured_cells`` in
``te_filled.hdf5``), so a row that loses its measurement part-way through the
window degrades to an intermediate value rather than flipping.  It is a
PROVENANCE statement and is independent of ``te_semiquant_fraction``, which is
an uncertainty class a measured cell can carry and a reconstructed one can
lack: the two do not substitute for each other.

Scan extent LIMITATION
----------------------
``L_probe`` is an integral over the MEASURED SCAN ONLY: 51 points at 1 cm
spacing spanning x = -25 to +25 cm, a 50 cm chord of the column.  The chord's
line density integrates the whole column the beam crosses, and its cm^-2
normalization assumes a 40 cm plasma diameter.  The probe integral therefore
stops at the scan edge while the chord's does not, and whatever density lies
beyond +/-25 cm is missing from ``L_probe`` and present in ``L_chord``.  This
biases ``r`` DOWN and ``T_e,implied`` down with it.  So a reader can judge how
much is missing, every row carries ``n_edge_left_over_axis`` and
``n_edge_right_over_axis``: the density at each end of the scan divided by the
density at x = 0.  A ratio near zero means the scan reaches the skirt and the
truncation is small; a ratio of a few percent or more means it does not.

Uncertainty
-----------
``te_sigma_sys_ev = 0.25 * T_e,used + 0.20 eV`` is a declared systematic band on
the filled ``T_e``, not a measured error bar, and ``pull = (T_e,used -
T_e,implied) / te_sigma_sys_ev`` reports the disagreement on it.  The band is an
input to this script, not a result of it.

Inputs
------
  processed/density_profiles_isweep.hdf5
  processed/te_filled.hdf5
  processed/interferometer_experiment_set_stats.npz

Outputs
-------
  processed/chord_implied_te.csv
  figures/chord_implied_te.png

Rows whose ``window`` field spans a range (``15.000-19.500ms``) are the window
aggregates; rows whose ``window`` field is a single time (``15.375ms``) are the
per-sample detail behind them.

Usage
-----
  MPLCONFIGDIR=.matplotlib PYTHONPATH=src \\
      ~/miniforge3/envs/bapsf-da/bin/python scripts/crosscheck_chord_implied_te.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from bapsf_lapd.config import z_from_port


DENSITY_HDF5 = Path("processed/density_profiles_isweep.hdf5")
TE_FILLED_HDF5 = Path("processed/te_filled.hdf5")
INTERF_NPZ = Path("processed/interferometer_experiment_set_stats.npz")
OUTPUT_CSV = Path("processed/chord_implied_te.csv")
OUTPUT_FIGURE = Path("figures/chord_implied_te.png")

#: Stable plasma plateau, in ms, inclusive at both ends.
WINDOW_MS = (15.0, 19.5)

#: Interferometer chord port paired with its nearest Langmuir probe port.
CHORD_PROBE_PORTS: tuple[tuple[int, int], ...] = ((20, 21), (29, 29), (40, 41))

#: Declared systematic band on the filled T_e: sigma = SLOPE * T_e + FLOOR.
TE_SIGMA_SYS_SLOPE = 0.25
TE_SIGMA_SYS_FLOOR_EV = 0.20

#: Largest axial mismatch, in cm, tolerated between the port z the te product
#: reports and the z of the density row matched to it.
Z_MATCH_TOL_CM = 1.0

CSV_FIELDS = (
    "es",
    "chord_port",
    "probe_port",
    "z_chord_cm",
    "z_probe_cm",
    "window",
    "L_probe_cm2",
    "L_chord_cm2",
    "ratio",
    "te_used_ev",
    "te_implied_ev",
    "te_semiquant_fraction",
    "te_sigma_sys_ev",
    "pull",
    "n_edge_left_over_axis",
    "n_edge_right_over_axis",
    "te_row_measured_fraction",
)


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


def implied_te_ev(te_used_ev: float, ratio: float) -> float:
    """Temperature at which the probe line integral would equal the chord's.

    ``n propto 1 / sqrt(T_e)``, so scaling the whole ``T_e(x)`` profile by
    ``ratio ** 2`` scales the probe's line integral by ``1 / ratio``.  With
    ``ratio = L_probe / L_chord`` that lands the probe on the chord exactly.

    Returns NaN for a non-finite or non-positive ratio: a negative probe line
    integral (a column that has decayed into the noise) implies no temperature.
    """
    if not np.isfinite(ratio) or ratio <= 0.0:
        return float("nan")
    return float(te_used_ev) * float(ratio) ** 2


def te_sigma_sys_ev(te_used_ev: float) -> float:
    """Declared systematic band on a filled ``T_e``, in eV.

    ``0.25 * T_e + 0.20 eV``: a fractional term plus a floor that keeps the
    band finite where the diagnostic reaches its low-temperature limit.  It is
    a declared convention supplied to this script, not a measurement made by
    it.
    """
    return TE_SIGMA_SYS_SLOPE * float(te_used_ev) + TE_SIGMA_SYS_FLOOR_EV


def te_pull(te_used_ev: float, te_implied_ev: float, sigma_ev: float) -> float:
    """``(T_e,used - T_e,implied) / sigma``; positive where the probe reads hot."""
    if not np.isfinite(sigma_ev) or sigma_ev <= 0.0:
        return float("nan")
    return (float(te_used_ev) - float(te_implied_ev)) / float(sigma_ev)


def line_integral_cm2(profile_m3: np.ndarray, x_m: np.ndarray) -> float:
    """Integrate a SIGNED radial density profile across the scan, in cm^-2.

    Every cell carrying a measurement enters with its measured sign, matching
    ``scripts/plot_line_integrated_density_vs_z.py``: a negative far-skirt cell
    is noise about zero, and dropping it would keep only the upward half of
    that noise.  NaN means no usable measurement and is the only exclusion.

    Returns NaN when fewer than two cells are finite.
    """
    profile = np.asarray(profile_m3, dtype=np.float64)
    x = np.asarray(x_m, dtype=np.float64)
    valid = np.isfinite(profile)
    if valid.sum() < 2:
        return float("nan")
    return float(np.trapezoid(profile[valid], x[valid]) / 1e4)


def density_weighted_te_ev(
    density_m3: np.ndarray,
    te_ev: np.ndarray,
    x_m: np.ndarray,
) -> tuple[float, float]:
    """Density-weighted mean ``T_e`` over one or more scans, in eV.

    ``density_m3`` and ``te_ev`` are ``(n_positions,)`` or
    ``(n_positions, n_samples)`` on the same grid.  Returns
    ``(numerator, denominator)`` of

        sum_k trapz(n * T_e, x) , sum_k trapz(n, x)

    so that several scans can be pooled by summing the pairs before dividing.
    Cells where either array is non-finite are excluded from both integrals, so
    numerator and denominator always span the same cells.  The weight is the
    SIGNED density.
    """
    density = np.atleast_2d(np.asarray(density_m3, dtype=np.float64).T).T
    te = np.atleast_2d(np.asarray(te_ev, dtype=np.float64).T).T
    x = np.asarray(x_m, dtype=np.float64)
    numerator = 0.0
    denominator = 0.0
    for column in range(density.shape[1]):
        n_k = density[:, column]
        te_k = te[:, column]
        valid = np.isfinite(n_k) & np.isfinite(te_k)
        if valid.sum() < 2:
            continue
        numerator += float(np.trapezoid(n_k[valid] * te_k[valid], x[valid]))
        denominator += float(np.trapezoid(n_k[valid], x[valid]))
    return numerator, denominator


def interp_te_to_profile_times(
    te_x_cycle: np.ndarray,
    te_time_ms: np.ndarray,
    profile_time_ms: np.ndarray,
) -> np.ndarray:
    """Interpolate a filled ``T_e`` row onto the dead-time sample grid.

    The same cell-by-cell linear interpolation in time that
    ``scripts/plot_density_profiles.py`` applies when it builds the density
    product, so the ``T_e`` recovered here is the one that density was computed
    with.
    """
    return np.vstack([
        np.interp(profile_time_ms, te_time_ms, te_x_cycle[xi, :])
        for xi in range(te_x_cycle.shape[0])
    ])


def edge_over_axis(profile_m3: np.ndarray, x_cm: np.ndarray) -> tuple[float, float]:
    """Scan-edge density divided by on-axis density, for each end of the scan.

    The truncation indicator: ``L_probe`` stops where the scan stops, so a
    reader needs to know how much density is still standing at the edge.
    Returns ``(left, right)``, NaN where the axial cell is not finite or zero.
    """
    profile = np.asarray(profile_m3, dtype=np.float64)
    axis_index = int(np.argmin(np.abs(np.asarray(x_cm, dtype=np.float64))))
    axis_value = profile[axis_index]
    if not np.isfinite(axis_value) or axis_value == 0.0:
        return float("nan"), float("nan")
    return float(profile[0] / axis_value), float(profile[-1] / axis_value)


# ---------------------------------------------------------------------------
# Product reads
# ---------------------------------------------------------------------------


def _chord_line_integrated_cm2(
    npz: Any,
    set_id: int,
    chord_port: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(time_ms, line_integrated_cm2)`` for one chord of one set.

    The NPZ is ``scripts/plot_interferometer_by_experiment_set.py``'s pooled
    per-chord shot average; its ``line_integrated_*_cm2`` arrays are already
    the line density (the line-average density times the 40 cm calibration
    diameter) and are NOT divided by that diameter again.
    """
    prefix = f"set{set_id}_p{chord_port}"
    time_key = f"{prefix}_time_ms"
    value_key = f"{prefix}_line_integrated_mean_cm2"
    if time_key not in npz or value_key not in npz:
        raise KeyError(
            f"{INTERF_NPZ} carries no chord p{chord_port} for experiment set "
            f"{set_id} (looked for {value_key!r})"
        )
    return (
        np.asarray(npz[time_key], dtype=np.float64),
        np.asarray(npz[value_key], dtype=np.float64),
    )


def _te_row_index(port_ids: np.ndarray, probe_port: int, set_id: int) -> int:
    matches = np.flatnonzero(np.asarray(port_ids, dtype=int) == probe_port)
    if matches.size != 1:
        raise ValueError(
            f"experiment set {set_id} has {matches.size} port-{probe_port} rows "
            f"in {TE_FILLED_HDF5}; expected exactly one"
        )
    return int(matches[0])


def _density_row_index(z_density_cm: np.ndarray, z_probe_cm: float, set_id: int) -> int:
    index = int(np.argmin(np.abs(np.asarray(z_density_cm) - z_probe_cm)))
    offset = abs(float(z_density_cm[index]) - z_probe_cm)
    if offset > Z_MATCH_TOL_CM:
        raise ValueError(
            f"experiment set {set_id}: nearest {DENSITY_HDF5} row to z = "
            f"{z_probe_cm:g} cm is {float(z_density_cm[index]):g} cm, "
            f"{offset:g} cm away"
        )
    return index


def collect_port(
    density_hdf: h5py.File,
    te_hdf: h5py.File,
    npz: Any,
    set_id: int,
    chord_port: int,
    probe_port: int,
    window_ms: tuple[float, float],
) -> list[dict[str, Any]]:
    """Return the window row and the per-sample rows for one chord/probe pair."""
    density_set = density_hdf[f"experiment_sets/{set_id}"]
    te_set = te_hdf[f"experiment_sets/{set_id}"]

    x_cm = np.asarray(density_hdf["x_cm"][()], dtype=np.float64)
    if not np.allclose(x_cm, np.asarray(te_set["x_cm"][()], dtype=np.float64)):
        raise ValueError(
            f"experiment set {set_id}: {DENSITY_HDF5} and {TE_FILLED_HDF5} do "
            f"not share an x grid"
        )
    x_m = x_cm * 1e-2

    te_index = _te_row_index(te_set["port"][()], probe_port, set_id)
    z_probe_cm = float(te_set["z_cm"][()][te_index])
    z_chord_cm = float(z_from_port(chord_port))

    density_index = _density_row_index(density_set["z_cm"][()], z_probe_cm, set_id)
    density = np.asarray(density_set["n_e_m3"][()][density_index], dtype=np.float64)
    sample_time_ms = np.asarray(density_set["inter_sweep_time_s"][()]) * 1000.0

    te_cycle_time_ms = np.asarray(te_set["cycle_time_ms"][()], dtype=np.float64)
    te_row = np.asarray(te_set["te_filled"][()][te_index], dtype=np.float64)
    te_on_samples = interp_te_to_profile_times(
        te_row, te_cycle_time_ms, sample_time_ms
    )
    semiquant_row = np.asarray(te_set["te_semi_quantitative"][()][te_index], dtype=bool)
    # Per-sample count of cells this port's row carries a MEASUREMENT for; zero
    # means the row at that sample was reconstructed from its neighbours and
    # the boundary anchors, and is not a measurement of this port.
    measured_row = np.asarray(
        te_set["te_row_measured_cells"][()][te_index], dtype=np.int64
    )

    chord_time_ms, chord_cm2 = _chord_line_integrated_cm2(npz, set_id, chord_port)

    in_window = (sample_time_ms >= window_ms[0]) & (sample_time_ms <= window_ms[1])
    sample_indices = np.flatnonzero(in_window)
    if sample_indices.size == 0:
        raise ValueError(
            f"experiment set {set_id} port {probe_port}: no dead-time sample in "
            f"{window_ms[0]:g}-{window_ms[1]:g} ms"
        )

    common = {
        "es": set_id,
        "chord_port": chord_port,
        "probe_port": probe_port,
        "z_chord_cm": z_chord_cm,
        "z_probe_cm": z_probe_cm,
    }

    sample_rows: list[dict[str, Any]] = []
    probe_values: list[float] = []
    chord_values: list[float] = []
    for k in sample_indices:
        time_ms = float(sample_time_ms[k])
        probe_cm2 = line_integral_cm2(density[:, k], x_m)
        chord_value_cm2 = float(np.interp(time_ms, chord_time_ms, chord_cm2))
        numerator, denominator = density_weighted_te_ev(
            density[:, k], te_on_samples[:, k], x_m
        )
        te_used = numerator / denominator if denominator != 0.0 else float("nan")
        ratio = (
            probe_cm2 / chord_value_cm2
            if np.isfinite(chord_value_cm2) and chord_value_cm2 != 0.0
            else float("nan")
        )
        te_implied = implied_te_ev(te_used, ratio)
        sigma = te_sigma_sys_ev(te_used)
        left, right = edge_over_axis(density[:, k], x_cm)
        cycle = int(np.argmin(np.abs(te_cycle_time_ms - time_ms)))
        sample_rows.append({
            **common,
            "window": f"{time_ms:.3f}ms",
            "L_probe_cm2": probe_cm2,
            "L_chord_cm2": chord_value_cm2,
            "ratio": ratio,
            "te_used_ev": te_used,
            "te_implied_ev": te_implied,
            "te_semiquant_fraction": float(np.mean(semiquant_row[:, cycle])),
            "te_sigma_sys_ev": sigma,
            "pull": te_pull(te_used, te_implied, sigma),
            "n_edge_left_over_axis": left,
            "n_edge_right_over_axis": right,
            "te_row_measured_fraction": float(measured_row[cycle] >= 1),
        })
        probe_values.append(probe_cm2)
        chord_values.append(chord_value_cm2)

    probe_window_cm2 = float(np.mean(probe_values))
    chord_window_cm2 = float(np.mean(chord_values))
    ratio_window = (
        probe_window_cm2 / chord_window_cm2 if chord_window_cm2 != 0.0 else float("nan")
    )
    numerator, denominator = density_weighted_te_ev(
        density[:, sample_indices], te_on_samples[:, sample_indices], x_m
    )
    te_used_window = numerator / denominator if denominator != 0.0 else float("nan")
    te_implied_window = implied_te_ev(te_used_window, ratio_window)
    sigma_window = te_sigma_sys_ev(te_used_window)
    mean_profile = np.nanmean(density[:, sample_indices], axis=1)
    left_window, right_window = edge_over_axis(mean_profile, x_cm)
    te_cycles_in_window = (te_cycle_time_ms >= window_ms[0]) & (
        te_cycle_time_ms <= window_ms[1]
    )
    window_row = {
        **common,
        "window": f"{window_ms[0]:.3f}-{window_ms[1]:.3f}ms",
        "L_probe_cm2": probe_window_cm2,
        "L_chord_cm2": chord_window_cm2,
        "ratio": ratio_window,
        "te_used_ev": te_used_window,
        "te_implied_ev": te_implied_window,
        "te_semiquant_fraction": float(
            np.mean(semiquant_row[:, te_cycles_in_window])
        ),
        "te_sigma_sys_ev": sigma_window,
        "pull": te_pull(te_used_window, te_implied_window, sigma_window),
        "n_edge_left_over_axis": left_window,
        "n_edge_right_over_axis": right_window,
        "te_row_measured_fraction": float(
            np.mean(measured_row[te_cycles_in_window] >= 1)
        ),
    }
    return [window_row, *sample_rows]


def collect_rows(
    density_path: Path,
    te_path: Path,
    interferometer_path: Path,
    window_ms: tuple[float, float],
) -> list[dict[str, Any]]:
    """Collect every chord/probe row of every experiment set that has both."""
    rows: list[dict[str, Any]] = []
    with (
        h5py.File(density_path, "r") as density_hdf,
        h5py.File(te_path, "r") as te_hdf,
        np.load(interferometer_path) as npz,
    ):
        density_sets = set(density_hdf["experiment_sets"].keys())
        te_sets = set(te_hdf["experiment_sets"].keys())
        for key in sorted(density_sets & te_sets, key=int):
            set_id = int(key)
            for chord_port, probe_port in CHORD_PROBE_PORTS:
                try:
                    _chord_line_integrated_cm2(npz, set_id, chord_port)
                except KeyError:
                    continue
                rows.extend(
                    collect_port(
                        density_hdf,
                        te_hdf,
                        npz,
                        set_id,
                        chord_port,
                        probe_port,
                        window_ms,
                    )
                )
    if not rows:
        raise ValueError(
            f"no experiment set carries both a density row and a chord; checked "
            f"{density_path} and {interferometer_path}"
        )
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: (
                    f"{row[field]:.6g}"
                    if isinstance(row[field], float)
                    else row[field]
                )
                for field in CSV_FIELDS
            })
    return path


def window_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if "-" in row["window"]]


def print_table(rows: list[dict[str, Any]], window_ms: tuple[float, float]) -> None:
    print(
        f"Chord-implied T_e cross-check, plateau window "
        f"{window_ms[0]:g}-{window_ms[1]:g} ms"
    )
    print(
        "  r = L_probe / L_chord; T_e,implied = T_e,used * r^2; "
        "sigma_sys = 0.25 T_e + 0.20 eV"
    )
    print(
        "  L_probe integrates the MEASURED SCAN ONLY "
        "(x = -25 to +25 cm); the chord integrates the whole column, so a "
        "column wider than the scan biases r down.  n(edge)/n(0) says how "
        "much density is still standing at the scan edge."
    )
    print(
        "  f_meas is the share of the window's samples at which this port's "
        "filled T_e row carries a measured cell.  A row marked PRIOR "
        "(f_meas = 0) reads back on a RECONSTRUCTED T_e and its pull is not a "
        "measurement of that probe."
    )
    header = (
        f"{'ES':>3} {'chord':>6} {'probe':>6} {'z_ch':>8} {'z_pr':>8} "
        f"{'L_probe':>11} {'L_chord':>11} {'r':>7} {'Te_used':>8} "
        f"{'Te_impl':>8} {'sig_sys':>8} {'pull':>7} {'semiq':>6} "
        f"{'n_L/n0':>7} {'n_R/n0':>7} {'f_meas':>7} {'prov':>5}"
    )
    print()
    print(header)
    print("-" * len(header))
    for row in window_rows(rows):
        print(
            f"{row['es']:>3d} {row['chord_port']:>6d} {row['probe_port']:>6d} "
            f"{row['z_chord_cm']:>8.2f} {row['z_probe_cm']:>8.2f} "
            f"{row['L_probe_cm2']:>11.4e} {row['L_chord_cm2']:>11.4e} "
            f"{row['ratio']:>7.3f} {row['te_used_ev']:>8.3f} "
            f"{row['te_implied_ev']:>8.3f} {row['te_sigma_sys_ev']:>8.3f} "
            f"{row['pull']:>7.2f} {row['te_semiquant_fraction']:>6.2f} "
            f"{row['n_edge_left_over_axis']:>7.3f} "
            f"{row['n_edge_right_over_axis']:>7.3f} "
            f"{row['te_row_measured_fraction']:>7.2f} "
            f"{'PRIOR' if row['te_row_measured_fraction'] == 0.0 else '':>5}"
        )


def make_figure(
    rows: list[dict[str, Any]],
    path: Path,
    window_ms: tuple[float, float],
) -> Path:
    """Per experiment set: the T_e pair with its band, and the ratio."""
    import matplotlib.pyplot as plt

    aggregates = window_rows(rows)
    set_ids = sorted({row["es"] for row in aggregates})
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2,
        len(set_ids),
        figsize=(3.4 * len(set_ids), 6.2),
        sharex="col",
        squeeze=False,
    )
    for column, set_id in enumerate(set_ids):
        set_rows = [row for row in aggregates if row["es"] == set_id]
        labels = [f"p{row['probe_port']}\n(p{row['chord_port']})" for row in set_rows]
        positions = np.arange(len(set_rows), dtype=float)
        te_used = np.array([row["te_used_ev"] for row in set_rows])
        te_implied = np.array([row["te_implied_ev"] for row in set_rows])
        sigma = np.array([row["te_sigma_sys_ev"] for row in set_rows])
        ratio = np.array([row["ratio"] for row in set_rows])

        top = axes[0, column]
        top.errorbar(
            positions,
            te_used,
            yerr=sigma,
            fmt="o",
            color="tab:blue",
            capsize=4,
            label=r"$T_e$ used $\pm\,\sigma_{\rm sys}$",
        )
        top.plot(
            positions,
            te_implied,
            "s",
            color="tab:red",
            label=r"$T_e$ implied by the chord",
        )
        top.set_title(f"ES{set_id}")
        top.grid(True, alpha=0.25)
        if column == 0:
            top.set_ylabel(r"$T_e$ (eV)")
            top.legend(loc="best", fontsize=8, frameon=False)

        bottom = axes[1, column]
        bottom.plot(positions, ratio, "D-", color="tab:green")
        bottom.axhline(1.0, color="0.4", lw=1.0, ls="--")
        bottom.set_xticks(positions)
        bottom.set_xticklabels(labels, fontsize=8)
        bottom.set_xlim(positions[0] - 0.5, positions[-1] + 0.5)
        bottom.grid(True, alpha=0.25)
        if column == 0:
            bottom.set_ylabel(r"$r = L_{\rm probe} / L_{\rm chord}$")

    fig.suptitle(
        "Langmuir density vs interferometer chord: the implied "
        rf"$T_e$ ({window_ms[0]:g}-{window_ms[1]:g} ms)"
        "\n"
        r"$L_{\rm probe}$ integrates the measured scan only "
        r"($|x| \leq 25$ cm); a wider column biases $r$ down",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--density", type=Path, default=DENSITY_HDF5)
    parser.add_argument("--te-filled", type=Path, default=TE_FILLED_HDF5)
    parser.add_argument("--interferometer", type=Path, default=INTERF_NPZ)
    parser.add_argument("--csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--figure", type=Path, default=OUTPUT_FIGURE)
    parser.add_argument(
        "--window-ms",
        type=float,
        nargs=2,
        default=list(WINDOW_MS),
        metavar=("START", "STOP"),
        help="plateau window in ms, inclusive at both ends",
    )
    parser.add_argument(
        "--no-figure",
        action="store_true",
        help="write the CSV and print the table without rendering the figure",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    window_ms = (float(args.window_ms[0]), float(args.window_ms[1]))
    if window_ms[1] <= window_ms[0]:
        raise ValueError(f"--window-ms must be increasing, got {window_ms}")

    rows = collect_rows(args.density, args.te_filled, args.interferometer, window_ms)
    print_table(rows, window_ms)
    print()
    print(write_csv(rows, args.csv))
    if not args.no_figure:
        print(make_figure(rows, args.figure, window_ms))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
