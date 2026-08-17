"""Edge-robust shot-ensemble T_e refit of the raw Langmuir sweeps (edgete_ products).

MOTIVATION
The primary pipeline (``scripts/process_langmuir_sweeps.py``) fits every shot
individually and averages the survivors. At the column edge the per-shot
electron-retarding window is selected from current percentiles of a low
signal-to-noise sweep, so the window wanders, QC rejects most shots, and whole
(x, cycle) cells come back as ``NaN``. This script recovers those cells by
fitting an ensemble-averaged sweep instead of averaging per-shot fits.

METHOD (per ``(run, x, cycle)`` cell, over the ~20 shots of that cell)
  1. Load the raw ``v_sweep``/``i_sweep`` ramps and low-pass filter them with the
     pipeline's own position-dependent Butterworth cutoff (``cutoff_edge_khz``
     for ``|x| > 10 cm``, else ``cutoff_core_khz``, both read from
     ``processed/langmuir_sweeps.hdf5``).
  2. PEER-OUTLIER SHOT REJECTION: robust z-score (MAD-scaled) on the
     electron-saturation level and on the derivative-peak plasma potential,
     cut at ``--z-cut``, plus an absolute ``|V_p - median(V_p)| <= 5 V``
     coherence gate. Rejection is capped so at least ``--min-keep`` shots
     survive; arc-like sweeps are what this removes.
  3. Vp-ALIGNED MEDIAN ENSEMBLE: each shot's voltage axis is shifted by
     ``V_p,shot - median(V_p)`` and the kept sweeps are interpolated onto a
     common voltage grid and combined with a per-point MEDIAN. The alignment is
     the disclosed deviation from AGENTS.md's "preserve per-shot fitting"
     guidance: naive pre-fit averaging smears the I-V transition because V_p
     fluctuates shot to shot, and aligning on V_p first is what makes an
     ensemble average admissible here.
  4. FIT: a robust (soft-L1) straight line to the ion branch below the floating
     potential is subtracted, then ``ln(I_e)`` is fitted (soft-L1 again) over
     the self-consistent window ``[max(V_p - 4.45 T_e, V_f), V_p]`` and the
     window is ITERATED TO A FIXED POINT in ``T_e``. Anchoring the window on
     ``V_p`` and ``V_f`` rather than on current percentiles is what removes the
     per-shot edge failure mode. Non-converged cells stay ``NaN`` (honest
     failure, never a fallback estimate).
  5. ERRORS: bootstrap over shots -- resample the kept shots with replacement,
     re-average, re-fit; report the bootstrap median and the 16/84% half-widths.
     Cells whose bootstrap does not converge often enough keep the direct fit
     value with ``NaN`` errors.

KNOWN FIT-WINDOW OFFSET VERSUS THE PIPELINE (disclose in any comparison)
The self-consistent ``[V_f, V_p]`` window is not the pipeline's
current-percentile window, so this estimator carries a small MULTIPLICATIVE
offset against ``te_best_ev`` even where both methods work. Measured over the
ES1 plateau core (``|x| <= 14 cm``, QC-clean cells) the per-run median ratio
edgete/pipeline is 1.017, 1.041, 1.035, 1.054, 1.012 for runs 01/02/04/06/08
(all cells pooled: 1.032, 16-84% 1.005-1.063) -- i.e. ~1-5%, high by
convention. This is the same systematic family that
``scripts/refit_sweep_windows.py`` maps in ``processed/sweep_window_refits.hdf5``
(and its ``*_es3_*`` / edge-suppressed relatives): over the 5x5 window family
the ES1/ES2 median T_e moves by roughly -10% to +7% about the pipeline default
purely from the window convention. The offset is nearly x-independent, so it
cancels in ratio metrics such as the half-core-T_e width, but it must NOT be
ignored when quoting absolute T_e.

PRODUCTS
All outputs carry the ``edgete_`` prefix -- ``processed/edgete_ensemble_te.hdf5``
by default. These are a SEPARATE, edge-focused product; they are NOT the
pipeline's ``te_filled`` family (``processed/te_filled*.hdf5``), do not feed the
probe-area calibration or the density/Mach chain, and must not be substituted
for ``processed/langmuir_sweeps.hdf5`` downstream.

SCOPE
ES1 rot-0 runs only by default (01, 02, 04, 06, 08 = ports 11, 21, 29, 41, 50).
rot-180 swept T_e is probe-shadowed and excluded, matching the primary
pipeline's policy.

Usage:
    MPLCONFIGDIR=.matplotlib ./.venv/bin/python scripts/edgete_ensemble_refit.py
        [--runs 01,02,04,06,08] [--cycles 20,21 | --all-cycles] [--n-boot 30]
        [--workers 10] [--output processed/edgete_ensemble_te.hdf5]

Single-cell gate (loads, rejects, ensembles, fits, bootstraps; writes nothing):

    MPLCONFIGDIR=.matplotlib ./.venv/bin/python scripts/edgete_ensemble_refit.py --selftest
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy import optimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bapsf_lapd import ChannelKind, LapdDataset  # noqa: E402
from bapsf_lapd.filtering import butterworth_lowpass  # noqa: E402

MANIFEST = ROOT / "config" / "may2026_run_manifest.toml"
SWEEPS_H5 = ROOT / "processed" / "langmuir_sweeps.hdf5"
DEFAULT_OUTPUT = ROOT / "processed" / "edgete_ensemble_te.hdf5"

# ES1 rot-0 runs: ports 11, 21, 29, 41, 50.
ES1_ROT0_RUNS = ("01", "02", "04", "06", "08")
# 10.0-19.5 ms plateau on the 0.5 ms cycle grid (the density-calibration window).
PLATEAU_CYCLES = tuple(range(20, 40))
N_CYCLES = 40

# Bootstrap resampling is seeded per (run, cycle) so the product is reproducible
# and independent of --workers.
BOOTSTRAP_SEED = 20260817

# Self-consistent retarding window width, in e-folds of T_e below V_p.
WINDOW_EFOLDS = 4.45
# Positions beyond this |x| use the run's edge filter cutoff (pipeline policy).
EDGE_X_CM = 10.0

SELFTEST_RUN = "04"
SELFTEST_CYCLE = 24
SELFTEST_X_CM = 0.0


def _require_inputs() -> None:
    """Fail early and specifically when a required input file is absent."""
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"run manifest not found: {MANIFEST}")
    if not SWEEPS_H5.is_file():
        raise FileNotFoundError(
            f"{SWEEPS_H5} not found; run scripts/process_langmuir_sweeps.py "
            "--run-ids all first (its per-run filter cutoffs and cycle times "
            "are inputs here)"
        )


def robust_z(values: np.ndarray) -> np.ndarray:
    """MAD-scaled z-score, falling back to the standard deviation if MAD is 0."""
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    scale = 1.4826 * mad if mad > 0 else (np.nanstd(values) or 1.0)
    return (values - median) / scale


def derivative_peak_vp(voltage: np.ndarray, current: np.ndarray) -> float:
    """Plasma potential from the peak of dI/dV (the pipeline's estimator).

    Searched between the 15th and 92nd voltage percentiles so that the ramp
    endpoints cannot win. Returns ``NaN`` when no interior sample qualifies.
    """
    order = np.argsort(voltage)
    v, i = voltage[order], current[order]
    with np.errstate(invalid="ignore"):
        derivative = np.gradient(i, v)
    region = (v > np.nanquantile(v, 0.15)) & (v < np.nanquantile(v, 0.92))
    index = np.flatnonzero(region)
    if index.size == 0:
        return np.nan
    return float(v[index[np.nanargmax(derivative[index])]])


def reject_shots(
    voltage: np.ndarray,
    current: np.ndarray,
    vp: np.ndarray,
    *,
    z_cut: float = 3.5,
    min_keep: int = 10,
) -> np.ndarray:
    """Peer-outlier shot mask from electron-saturation level and V_p.

    ``voltage``/``current`` are ``(n_shots, n_samples)`` filtered sweeps and
    ``vp`` the per-shot derivative-peak potential. Shots whose worse robust
    z-score exceeds ``z_cut``, or whose V_p sits more than 5 V from the cell
    median (unalignable, arc-like), are dropped. If that would leave fewer than
    ``min_keep`` shots the ``min_keep`` best-scoring shots are kept instead --
    rejection never runs away on a genuinely noisy cell.
    """
    n_shots = current.shape[0]
    esat = np.array(
        [
            np.nanmedian(current[s][voltage[s] > np.nanquantile(voltage[s], 0.9)])
            for s in range(n_shots)
        ]
    )
    z_esat = np.abs(robust_z(esat))
    z_vp = np.abs(robust_z(vp))
    score = np.maximum(z_esat, np.where(np.isfinite(z_vp), z_vp, 0.0))
    keep = score < z_cut
    vp_median = np.nanmedian(vp)
    coherent = np.abs(np.where(np.isfinite(vp), vp, vp_median) - vp_median) <= 5.0
    keep &= coherent
    if keep.sum() < min_keep:
        keep = np.zeros(n_shots, dtype=bool)
        keep[np.argsort(score)[:min_keep]] = True
    return keep


def ensemble_average(
    voltage: np.ndarray,
    current: np.ndarray,
    keep: np.ndarray,
    align_vp: np.ndarray | None,
    *,
    n_grid: int = 1500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """V_p-aligned median ensemble average of the kept shots.

    Each kept sweep is shifted by ``V_p,shot - median(V_p)`` (skipped when
    ``align_vp`` is ``None``) and interpolated onto the common voltage grid
    spanned by all kept shots. Returns ``(v_grid, i_median, i_spread)`` where
    ``i_spread`` is the per-point robust sigma (1.4826 x MAD) of the shot
    ensemble, used only to set the fit's noise floor.
    """
    kept = np.flatnonzero(keep)
    v_min = max(float(voltage[s].min()) for s in kept)
    v_max = min(float(voltage[s].max()) for s in kept)
    grid = np.linspace(v_min, v_max, n_grid)
    stack = np.empty((kept.size, grid.size))
    vp_median = np.nanmedian(align_vp[kept]) if align_vp is not None else 0.0
    for j, s in enumerate(kept):
        shift = 0.0
        if align_vp is not None and np.isfinite(align_vp[s]):
            shift = align_vp[s] - vp_median
        order = np.argsort(voltage[s])
        stack[j] = np.interp(grid, voltage[s][order] - shift, current[s][order])
    i_median = np.median(stack, axis=0)
    i_spread = 1.4826 * np.median(np.abs(stack - i_median), axis=0)
    return grid, i_median, i_spread


def _soft_l1_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Robust soft-L1 straight-line fit; returns ``(intercept, slope)``.

    Seeded from an ordinary least-squares fit and scaled by the MAD of its
    residuals so that ``f_scale = 1`` is the robust transition point.
    """
    seed = np.polyfit(x, y, 1)
    residual = y - np.polyval(seed, x)
    mad = np.median(np.abs(residual - np.median(residual)))
    scale = 1.4826 * mad if mad > 0 else max(np.std(residual), 1e-12)

    def cost(p: np.ndarray) -> np.ndarray:
        return (p[0] + p[1] * x - y) / scale

    result = optimize.least_squares(
        cost, [seed[1], seed[0]], loss="soft_l1", f_scale=1.0
    )
    return float(result.x[0]), float(result.x[1])


def floating_potential(grid: np.ndarray, current: np.ndarray, vp: float) -> float:
    """Last zero crossing of the total current below ``vp``.

    Anchors the lower edge of the retarding window; without it the window can
    grow into the ion-branch noise floor and return a spurious large T_e.
    Falls back to the lowest swept voltage when there is no crossing.
    """
    below = grid < vp
    negative = np.signbit(current[below])
    crossings = np.flatnonzero(negative[:-1] & ~negative[1:])
    if crossings.size == 0:
        return float(grid.min())
    return float(grid[below][crossings[-1]])


def fit_ensemble_te(
    grid: np.ndarray,
    i_median: np.ndarray,
    noise_a: float,
    *,
    te_init: float = 2.0,
    window_efolds: float = WINDOW_EFOLDS,
    n_iter: int = 12,
    te_bounds: tuple[float, float] = (0.05, 100.0),
) -> dict:
    """Fixed-point T_e fit of one ensemble-averaged sweep.

    Iterates: fit the ion line robustly well below V_f, subtract it, fit
    ``ln(I_e)`` robustly over ``[max(V_p - window_efolds*T_e, V_f), V_p]``, and
    re-enter with the new T_e until the window stops moving (2% tolerance) or
    ``n_iter`` is exhausted. Points below a ``3 sigma`` ensemble-noise cut are
    excluded; the cut is relaxed only if fewer than 8 points survive.

    Returns a dict with ``te`` (``NaN`` unless the window converged), ``vp``,
    ``vf``, ``ion`` = ``(intercept, slope)`` of the subtracted ion line,
    ``converged``, ``n_pts``, ``window_v`` and ``efolds`` -- the e-folds of
    electron current above the noise floor at V_p, i.e. how resolvable the cell
    is (below ~1.5 the cell is semi-quantitative).
    """
    vp = derivative_peak_vp(grid, i_median)
    out = {
        "te": np.nan,
        "vp": vp,
        "vf": np.nan,
        "ion": None,
        "converged": False,
        "n_pts": 0,
        "window_v": np.nan,
        "efolds": np.nan,
    }
    if not np.isfinite(vp):
        return out

    vf = floating_potential(grid, i_median, vp)
    out["vf"] = vf
    te = float(te_init)
    noise_floor = max(3.0 * noise_a, 1e-6)
    converged = False
    electron_current = None
    n_pts = 0
    for _ in range(n_iter):
        # Ion line: below the floating-potential region and >~7 T_e below V_p,
        # but always keep at least the lowest 15% of the sweep.
        ion_hi = min(vp - 7.0 * te, vf - 2.0, np.quantile(grid, 0.35))
        ion_mask = grid <= max(ion_hi, np.quantile(grid, 0.15))
        intercept, slope = _soft_l1_line(grid[ion_mask], i_median[ion_mask])
        slope = max(slope, 0.0)  # sheath expansion cannot make the ion line fall
        electron_current = i_median - (intercept + slope * grid)

        window_lo = max(vp - window_efolds * te, vf)
        in_window = (grid >= window_lo) & (grid <= vp)
        mask = in_window & (electron_current > noise_floor)
        if mask.sum() < 8:
            mask = in_window & (electron_current > 0)
        if mask.sum() < 8:
            return out

        # only the slope carries T_e (T_e = 1 / d ln(I_e)/dV)
        _, log_slope = _soft_l1_line(grid[mask], np.log(electron_current[mask]))
        if log_slope <= 1e-4:
            return out
        te_new = 1.0 / log_slope
        if not (te_bounds[0] <= te_new <= te_bounds[1]):
            return out

        out["ion"] = (intercept, slope)
        n_pts = int(mask.sum())
        converged = abs(te_new - te) < 0.02 * te
        te = te_new
        if converged:
            break

    if not converged:
        return out

    ie_at_vp = float(np.interp(vp, grid, electron_current))
    efolds = float(np.log(ie_at_vp / noise_floor)) if ie_at_vp > noise_floor else 0.0
    out.update(
        te=float(te),
        converged=True,
        n_pts=n_pts,
        window_v=float(window_efolds * te),
        efolds=efolds,
    )
    return out


def fit_cell(
    voltage: np.ndarray,
    current: np.ndarray,
    rng: np.random.Generator,
    *,
    n_boot: int = 30,
    align: bool = True,
) -> dict:
    """Full ensemble fit of one ``(x, cycle)`` cell, with bootstrap errors.

    ``voltage``/``current`` are the ``(n_shots, n_samples)`` filtered sweeps of
    that cell. Returns a dict with ``te``, ``err_lo``/``err_hi`` (16/84%
    bootstrap half-widths), ``n_kept``, ``vp``, ``converged``, the ensemble
    curve (``grid``, ``i_med``, ``i_spread``), ``noise_a`` and the raw ``fit``
    dict. ``te`` is ``NaN`` when the direct fit does not converge.
    """
    n_shots = current.shape[0]
    vp = np.array(
        [derivative_peak_vp(voltage[s], current[s]) for s in range(n_shots)]
    )
    keep = reject_shots(voltage, current, vp)
    align_vp = vp if align else None
    grid, i_median, i_spread = ensemble_average(voltage, current, keep, align_vp)
    noise_a = float(np.median(i_spread) / np.sqrt(max(keep.sum(), 1)))

    out = {
        "n_kept": int(keep.sum()),
        "te": np.nan,
        "err_lo": np.nan,
        "err_hi": np.nan,
        "vp": float(np.nanmedian(vp[keep])) if np.isfinite(vp[keep]).any() else np.nan,
        "grid": grid,
        "i_med": i_median,
        "i_spread": i_spread,
        "noise_a": noise_a,
        "fit": None,
        "boot": None,
        "converged": False,
    }
    fit = fit_ensemble_te(grid, i_median, noise_a)
    out["fit"] = fit
    out["converged"] = fit["converged"]
    if not np.isfinite(fit["te"]):
        return out

    kept = np.flatnonzero(keep)
    boot_te = []
    for _ in range(n_boot):
        sample = rng.choice(kept, size=kept.size, replace=True)
        b_grid, b_med, b_spread = ensemble_average(
            voltage,
            current[sample],
            np.ones(sample.size, bool),
            vp[sample] if align else None,
        )
        b_noise = float(np.median(b_spread) / np.sqrt(sample.size))
        b_fit = fit_ensemble_te(b_grid, b_med, b_noise, te_init=fit["te"])
        if np.isfinite(b_fit["te"]):
            boot_te.append(b_fit["te"])

    boot_te = np.asarray(boot_te)
    if boot_te.size >= max(8, n_boot // 3):
        lo, median, hi = np.percentile(boot_te, [16, 50, 84])
        out["te"] = float(median)
        out["err_lo"] = float(median - lo)
        out["err_hi"] = float(hi - median)
        out["boot"] = boot_te
    else:
        # Direct fit converged but the bootstrap did not: keep the point
        # estimate and leave the errors NaN rather than inventing a spread.
        out["te"] = float(fit["te"])
    return out


def load_run_sweeps(
    dataset: LapdDataset, run_id: str, cycles: list[int]
) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Load and pipeline-filter every sweep of ``run_id`` for the given cycles.

    Returns ``(x_cm, cycle_time_ms, {cycle: (voltage, current)})`` with each
    array shaped ``(n_positions, n_shots, n_samples)``. The per-run core/edge
    Butterworth cutoffs and the cycle time base are read from ``SWEEPS_H5`` so
    that this instrument filters exactly as the primary pipeline did.
    """
    run = dataset.run(run_id)
    sample_rate_hz = run.sample_rate_hz()
    ramp_slices = run.sweep_ramp_sample_slices()

    with h5py.File(SWEEPS_H5, "r") as f:
        x_cm = f["x_cm"][:]
        group = None
        for set_id in f["experiment_sets"]:
            if run_id in f["experiment_sets"][set_id]:
                group = f["experiment_sets"][set_id][run_id]
                break
        if group is None:
            raise KeyError(f"run {run_id!r} is not present in {SWEEPS_H5}")
        cutoff_core_hz = float(group.attrs["cutoff_core_khz"]) * 1e3
        cutoff_edge_hz = float(group.attrs["cutoff_edge_khz"]) * 1e3
        cycle_time_ms = group["cycle_time_s"][:] * 1e3

    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for cycle in cycles:
        if not 0 <= cycle < len(ramp_slices):
            raise ValueError(
                f"cycle {cycle} out of range for run {run_id} "
                f"({len(ramp_slices)} ramps)"
            )
        ramp = ramp_slices[cycle]
        voltage = run.langmuir_traces(ChannelKind.V_SWEEP, ramp)
        current = run.langmuir_traces(ChannelKind.I_SWEEP, ramp)
        v_filtered = np.empty_like(voltage)
        i_filtered = np.empty_like(current)
        for ix in range(voltage.shape[0]):
            cutoff_hz = (
                cutoff_edge_hz if abs(x_cm[ix]) > EDGE_X_CM else cutoff_core_hz
            )
            v_filtered[ix] = butterworth_lowpass(
                voltage[ix], sample_rate_hz=sample_rate_hz, cutoff_hz=cutoff_hz
            )
            i_filtered[ix] = butterworth_lowpass(
                current[ix], sample_rate_hz=sample_rate_hz, cutoff_hz=cutoff_hz
            )
        out[cycle] = (v_filtered, i_filtered)
    return x_cm, cycle_time_ms, out


def half_max_radius(
    x_cm: np.ndarray,
    te_ev: np.ndarray,
    side: int,
    *,
    core_halfwidth_cm: float = 5.0,
) -> float:
    """Radius where T_e falls to half the core value, on one side.

    ``side`` is ``+1`` (x >= 0) or ``-1`` (x <= 0); the core reference is the
    mean over ``|x| <= core_halfwidth_cm``. ``NaN`` cells are dropped before the
    crossing search, so the interpolation bridges them (disclosed). Returns
    ``NaN`` when the profile never drops below half the core.
    """
    core = np.nanmean(te_ev[np.abs(x_cm) <= core_halfwidth_cm])
    if not np.isfinite(core):
        return np.nan
    half = core / 2.0
    selected = (x_cm * side) >= 0
    xs, ts = x_cm[selected] * side, te_ev[selected]
    order = np.argsort(xs)
    xs, ts = xs[order], ts[order]
    finite = np.isfinite(ts)
    xs, ts = xs[finite], ts[finite]
    below = np.flatnonzero(ts < half)
    if below.size == 0:
        return np.nan
    j = below[0]
    if j == 0:
        return float(xs[0])
    x0, x1, t0, t1 = xs[j - 1], xs[j], ts[j - 1], ts[j]
    return float(x0 + (t0 - half) * (x1 - x0) / (t0 - t1))


def _cell_rng(run_id: str, cycle: int) -> np.random.Generator:
    """Bootstrap generator keyed on the cell, so results do not depend on
    how tasks were distributed across worker processes."""
    return np.random.default_rng(
        np.random.SeedSequence([BOOTSTRAP_SEED, int(run_id), int(cycle)])
    )


def _fit_run_cycle(task: tuple[str, int, int]) -> dict:
    """Fit every position of one ``(run_id, cycle)``. One HDF5 open per call."""
    run_id, cycle, n_boot = task
    dataset = LapdDataset.from_manifest(MANIFEST)
    x_cm, cycle_time_ms, data = load_run_sweeps(dataset, run_id, [cycle])
    voltage, current = data[cycle]
    rng = _cell_rng(run_id, cycle)

    n_x = x_cm.size
    result = {
        "run_id": run_id,
        "cycle": cycle,
        "time_ms": float(cycle_time_ms[cycle]),
        "x_cm": x_cm,
        "te": np.full(n_x, np.nan),
        "err_lo": np.full(n_x, np.nan),
        "err_hi": np.full(n_x, np.nan),
        "efolds": np.full(n_x, np.nan),
        "vp": np.full(n_x, np.nan),
        "n_kept": np.zeros(n_x, dtype=np.int16),
    }
    for ix in range(n_x):
        cell = fit_cell(voltage[ix], current[ix], rng, n_boot=n_boot)
        result["te"][ix] = cell["te"]
        result["err_lo"][ix] = cell["err_lo"]
        result["err_hi"][ix] = cell["err_hi"]
        result["n_kept"][ix] = cell["n_kept"]
        if cell["fit"] is not None:
            result["efolds"][ix] = cell["fit"]["efolds"]
            result["vp"][ix] = cell["fit"]["vp"]
    return result


def write_output(
    path: Path, run_ids: list[str], results: dict[tuple[str, int], dict]
) -> None:
    """Write the ``edgete_`` HDF5 product, one group per run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["method"] = (
            "peer-outlier shot rejection; Vp-aligned shot-ensemble median "
            "average; self-consistent [max(Vp-4.45Te, Vf), Vp] soft-L1 "
            "ln(Ie) fit iterated to a fixed point; bootstrap-over-shots errors"
        )
        f.attrs["rot_policy"] = "ES1 rot-0 runs only (probe-shadow pitfall)"
        f.attrs["generator"] = "scripts/edgete_ensemble_refit.py"
        f.attrs["window_offset_note"] = (
            "fit-window convention offset vs the pipeline's te_best_ev: "
            "~1-5% high in the ES1 plateau core; same systematic family as "
            "processed/sweep_window_refits.hdf5. NOT a te_filled product."
        )
        f.attrs["bootstrap_seed"] = BOOTSTRAP_SEED
        for run_id in run_ids:
            cycles = sorted(c for (r, c) in results if r == run_id)
            if not cycles:
                continue
            x_cm = results[(run_id, cycles[0])]["x_cm"]
            shape = (x_cm.size, len(cycles))
            te = np.full(shape, np.nan)
            err_lo = np.full(shape, np.nan)
            err_hi = np.full(shape, np.nan)
            efolds = np.full(shape, np.nan)
            vp = np.full(shape, np.nan)
            n_kept = np.zeros(shape, dtype=np.int16)
            time_ms = np.full(len(cycles), np.nan)
            for j, cycle in enumerate(cycles):
                cell = results[(run_id, cycle)]
                time_ms[j] = cell["time_ms"]
                te[:, j] = cell["te"]
                err_lo[:, j] = cell["err_lo"]
                err_hi[:, j] = cell["err_hi"]
                efolds[:, j] = cell["efolds"]
                vp[:, j] = cell["vp"]
                n_kept[:, j] = cell["n_kept"]
            group = f.create_group(run_id)
            group.attrs["run_id"] = run_id
            group.attrs["cycles"] = cycles
            group.create_dataset("x_cm", data=x_cm)
            group.create_dataset("cycle_time_ms", data=time_ms)
            group.create_dataset("te_ens_ev", data=te)
            group.create_dataset("te_ens_err_lo", data=err_lo)
            group.create_dataset("te_ens_err_hi", data=err_hi)
            group.create_dataset("n_kept", data=n_kept)
            group.create_dataset("efolds_above_noise", data=efolds)
            group.create_dataset("vp_ens_v", data=vp)
            f.flush()


def selftest(n_boot: int) -> int:
    """End-to-end single-cell run: load, reject, ensemble, fit, bootstrap.

    Writes nothing. Returns a process exit code (0 = pass).
    """
    _require_inputs()
    dataset = LapdDataset.from_manifest(MANIFEST)
    x_cm, cycle_time_ms, data = load_run_sweeps(
        dataset, SELFTEST_RUN, [SELFTEST_CYCLE]
    )
    voltage, current = data[SELFTEST_CYCLE]
    ix = int(np.abs(x_cm - SELFTEST_X_CM).argmin())
    cell = fit_cell(
        voltage[ix],
        current[ix],
        _cell_rng(SELFTEST_RUN, SELFTEST_CYCLE),
        n_boot=n_boot,
    )
    fit = cell["fit"]

    print(
        f"selftest cell: run {SELFTEST_RUN}, cycle {SELFTEST_CYCLE} "
        f"({cycle_time_ms[SELFTEST_CYCLE]:.1f} ms), x = {x_cm[ix]:+.1f} cm"
    )
    print(f"  shots kept        : {cell['n_kept']} / {current.shape[1]}")
    print(f"  V_p (ensemble)    : {fit['vp']:.2f} V")
    print(f"  V_f (ensemble)    : {fit['vf']:.2f} V")
    print(f"  fit window        : {fit['window_v']:.2f} V, {fit['n_pts']} points")
    print(f"  converged         : {fit['converged']}")
    print(f"  T_e (direct fit)  : {fit['te']:.3f} eV")
    print(
        f"  T_e (bootstrap)   : {cell['te']:.3f} "
        f"-{cell['err_lo']:.3f} +{cell['err_hi']:.3f} eV "
        f"({0 if cell['boot'] is None else cell['boot'].size}/{n_boot} resamples)"
    )
    print(f"  e-folds above noise: {fit['efolds']:.2f}")

    failures = []
    if cell["n_kept"] < 10:
        failures.append("fewer than 10 shots kept")
    if not fit["converged"]:
        failures.append("fixed-point window did not converge")
    if not (0.5 <= cell["te"] <= 30.0):
        failures.append(f"T_e = {cell['te']} outside the 0.5-30 eV sanity band")
    if not np.isfinite(cell["err_lo"]) or not np.isfinite(cell["err_hi"]):
        failures.append("bootstrap errors are not finite")
    if failures:
        print("SELFTEST FAIL: " + "; ".join(failures))
        return 1
    print("SELFTEST PASS")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Edge-robust shot-ensemble T_e refit of the ES1 Langmuir sweeps."
    )
    parser.add_argument(
        "--runs",
        default=",".join(ES1_ROT0_RUNS),
        help="comma-separated run ids (default: the ES1 rot-0 runs)",
    )
    parser.add_argument(
        "--all-cycles",
        action="store_true",
        help=f"fit all {N_CYCLES} cycles instead of the 10-19.5 ms plateau",
    )
    parser.add_argument(
        "--cycles",
        default=None,
        help="comma-separated cycle indices (overrides the plateau default; "
        "use one cycle to smoke-test a run)",
    )
    parser.add_argument("--n-boot", type=int, default=30, help="bootstrap resamples")
    parser.add_argument(
        "--workers", type=int, default=10, help="worker processes (1 = serial)"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="fit one core cell end to end, print the result, write nothing",
    )
    args = parser.parse_args(argv)
    if args.n_boot < 1:
        parser.error("--n-boot must be >= 1")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.cycles is not None and args.all_cycles:
        parser.error("--cycles and --all-cycles are mutually exclusive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.selftest:
        return selftest(args.n_boot)

    _require_inputs()
    run_ids = [r.strip() for r in args.runs.split(",") if r.strip()]
    if not run_ids:
        raise ValueError("--runs selected no runs")
    if args.cycles is not None:
        cycles = [int(c) for c in args.cycles.split(",") if c.strip()]
        if not cycles:
            raise ValueError("--cycles selected no cycles")
    elif args.all_cycles:
        cycles = list(range(N_CYCLES))
    else:
        cycles = list(PLATEAU_CYCLES)

    tasks = [(run_id, cycle, args.n_boot) for run_id in run_ids for cycle in cycles]
    results: dict[tuple[str, int], dict] = {}

    def record(cell: dict) -> None:
        results[(cell["run_id"], cell["cycle"])] = cell
        print(
            f"run {cell['run_id']} cycle {cell['cycle']} "
            f"({cell['time_ms']:.1f} ms) done",
            flush=True,
        )

    if args.workers == 1:
        for task in tasks:
            record(_fit_run_cycle(task))
    else:
        with futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            for cell in pool.map(_fit_run_cycle, tasks):
                record(cell)

    write_output(args.output, run_ids, results)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
