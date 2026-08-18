"""Export one experiment set for the bapsf-transport sim1d notebook.

The NPZ product is self-contained and uses simulation-facing units:

* core density and total SEM in cm^-3;
* flux-tube-averaged density in cm^-3 and flux-tube-averaged ion-saturation
  current in A, the second radial-averaging convention (see below);
* core electron temperature and radial SEM in eV;
* offset-corrected upstream ion-saturation current at x=0 in A;
* offset-corrected discharge current in A and cathode-anode voltage in V,
  each with the shot-to-shot standard deviation and the SEM of the mean;
* per-port fractional spread of the fit-window T_e re-fits, dimensionless;
* all time axes in ms relative to the experimental SIS trigger.

Probe-A density SEM includes the propagated area-calibration uncertainty.
The ion-current decay uses the same one-sided high-current detector as the
profile pipeline.  A shot is excluded from the continuous decay trace if it is
flagged in any pre-afterglow inter-sweep cell at x=0.  The retained traces are
100 kHz low-pass filtered and reduced to approximately 10 us time bins before
the shot mean and SEM are computed.
Each run's discharge current has its own additive channel zero offset
subtracted before smoothing; the per-run values are exported alongside the
traces.  Discharge traces are smoothed with the same 9-sample moving average
used by ``process_discharge_experiment_summary.py`` and averaged over both
stored traces from all runs in the selected experiment set.  The discharge
standard deviation is the spread of those same single shots about that mean,
taken on the shared trigger-referenced time grid without per-shot alignment,
so it carries the machine's breakdown-timing jitter.  Raw cathode-anode
voltage is negative, so the exported overlay voltage is multiplied by -1;
dispersion is sign-invariant and is exported unnegated.

Two radial-averaging conventions
--------------------------------
The overlay carries the measured radial average in BOTH conventions, because
the comparison convention is otherwise implicit and is the same order as the
residuals it is used to judge:

``density_mean_cm3`` (and its SEM fields) is the LEGACY convention: an
unweighted arithmetic mean of the 51-point line scan over the core band
``X_MIN_CM <= x <= X_MAX_CM``.  It is a line cut through the column, so it
carries no radial area weighting at all.  Nothing about it changes here.

``density_ftavg_cm3``, ``isat_ftavg_upstream_a`` and ``isat_ftavg_a`` are the
FLUX-TUBE convention:
``int_0^R 2 pi r f(r) dr / (pi R^2)`` with ``R = FLUX_TUBE_RADIUS_CM``, the
measured cathode frame opening.  This is the quantity a 1D transport model
that carries one radial cell of radius R reports, so it is the convention in
which a model cell and a measurement are the same quantity.  Getting there
from a diameter line scan requires assuming the column is axisymmetric about
its own centroid, which is an ASSUMPTION and not a measurement -- see
``ftavg_axisymmetry`` in the exported product.

The two conventions are exported side by side and are NOT interchangeable; a
consumer must state which one a number came from.

Which probe face the Isat flux-tube average comes from
------------------------------------------------------
The Mach probe has two planar faces on opposite sides of the body.  At rot-0
the ``i_sweep`` channel collects on the UPSTREAM face and the ``isat`` channel
on the DOWNSTREAM one; at rot-180 the assignment reverses.  A downstream face
sits in the probe body's own flow shadow, so it under-reads, and on 2026-08-18
the upstream face was RULED the Isat truth channel.

``isat_ftavg_upstream_a`` is therefore the correction-bearing field: it comes
from the ``i_sweep`` product, which is also the chain behind ``n_e_m3`` and
behind ``isat_decay_*`` / ``isat_drive_*``, so it pairs with the density and
core-band fields without a face change anywhere.

``isat_ftavg_a`` is the downstream ``isat`` face, kept because it is the
effective-width ledger's rot-0 primary and the face the 2026-08-18 paper read
used.  It carries the shadowing caveat in ``isat_ftavg_face`` and the two
faces must not be ratioed against each other -- their flux-tube corrections
run in opposite directions with z.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import h5py
import numpy as np
from scipy.ndimage import uniform_filter1d

from bapsf_lapd import ChannelKind, LapdDataset
from bapsf_lapd.filtering import butterworth_lowpass
from plot_core_density_temperature_timeseries import (
    DENSITY_SCALE_M3,
    _load_density_stats,
    _load_te_stats,
    _plot_uncertainty,
)
from plot_decay_rates_x0 import (
    _load_zero_offsets,
    _padded_sample_window,
    _zero_offset_v,
)
from plot_isat_profiles import _deadtime_shot_means, _high_shot_outlier_mask


MANIFEST = Path("config/may2026_run_manifest.toml")
DENSITY_HDF5 = Path("processed/density_profiles_isweep.hdf5")
TE_HDF5 = Path("processed/te_filled.hdf5")
#: Dead-time line-scan profiles from the ``i_sweep`` channel: the UPSTREAM
#: probe face at rot-0, ruled the Isat truth channel 2026-08-18.  Feeds both
#: the x=0 decay trace and the ``isat_ftavg_upstream_*`` flux-tube family.
ISAT_PROFILE_HDF5 = Path("processed/isweep_deadtime_profiles.hdf5")
#: The same scans from the ``isat`` channel: the DOWNSTREAM face at rot-0,
#: which reads low because it sits in the probe body's flow shadow.  Feeds the
#: ``isat_ftavg_*`` family only.
ROT0_ISAT_PROFILE_HDF5 = Path("processed/isat_profiles.hdf5")
ZERO_OFFSETS = Path("processed/trace_zero_offsets.toml")
WINDOW_REFITS_HDF5 = Path("processed/sweep_window_refits.hdf5")
PORTS = np.array([11, 21, 29, 41, 50], dtype=np.int16)
X_MIN_CM = -10.0
X_MAX_CM = 10.0
DISCHARGE_SMOOTHING_SAMPLES = 9
DENSITY_SCALE_CM3 = DENSITY_SCALE_M3 * 1.0e-6
M3_TO_CM3 = 1.0e-6  # the flux-tube fields work on raw n_e_m3, not the scaled grid
ISAT_DECAY_STOP_S = 47.5e-3
ISAT_DECAY_FILTER_PAD_S = 0.1e-3
ISAT_DECAY_CUTOFF_HZ = 100.0e3
ISAT_DECAY_BIN_S = 10.0e-6

#: Radius of the flux tube the measured profiles are averaged over, in cm.
#: Direct caliper reading of the LAPD cathode assembly (2026-08-17): the frame
#: opening is a 14.5 in aperture, 36.830 cm diameter, in a 15.0 in x 0.25 in
#: disc whose outer radius is 19.050 cm.  Half the aperture diameter is
#: 18.415 cm.  This is hardware, not a fit, and it is the radius the transport
#: model's single radial cell uses, which is why the measured target has to use
#: it too for the two to be the same quantity.
FLUX_TUBE_RADIUS_CM = 18.415

#: Collection area of the probe face each electrical channel sits on, as the
#: attribute name carrying it.  Per ``bapsf_lapd.density``: at rot-0 the left
#: face (toward the cathode, upstream) is the Isweep channel with area A_p_L,
#: and the right face (toward the anode, downstream) is the Isat channel with
#: area A_p_R.  The pairing is keyed off each run's OWN recorded channel rather
#: than off which product file it came from, which is what keeps ES3 run 31 --
#: whose two channels are exchanged between the products -- normalized by the
#: right areas.
CHANNEL_AREA_ATTR = {"i_sweep": "ap_L_cm2", "isat": "ap_R_cm2"}

#: Number of points at each end of the scan that set the background baseline.
#: Convention transcribed from the effective-width ledger
#: (``figures/effwidth_analysis.py``, ``width_metrics``): the scalar baseline is
#: ``min(median of the outer BACKGROUND_EDGE_POINTS on each side)``, clipped at
#: zero, subtracted, and the profile is then clipped at zero.
BACKGROUND_EDGE_POINTS = 3

#: Spatial-coherence despike, transcribed from the edge-T_e ledger's gate
#: (``figures/edgete_early_analyze.py``): a cell that sits further than
#: ``DESPIKE_TOLERANCE`` (fractional) from the median of its finite neighbours
#: within +/- ``DESPIKE_HALF_WIDTH`` cells is an isolated single-cell spike and
#: is replaced by that median.  At least two finite neighbours are required
#: before the gate will judge a cell, and they must BRACKET it -- at least one
#: on each side.  Bracketing is what makes a cell "isolated"; without it the
#: gate judges the two scan endpoints against a one-sided window, and those are
#: exactly the points that set the background baseline below, so a one-sided
#: misjudgement moves the baseline and with it the whole average.
DESPIKE_HALF_WIDTH = 2
DESPIKE_TOLERANCE = 0.5

#: Amplitude floor below which the despike gate does not judge a cell, as a
#: fraction of the profile peak.  The gate's premise is that the profile is
#: locally flat on the scale of DESPIKE_HALF_WIDTH cells; in the outer skirt the
#: profile legitimately falls by more than DESPIKE_TOLERANCE over two cells, so
#: the gate is not valid there and an unrestricted gate flags ordinary skirt
#: points.  The threshold reuses the effective-width ledger's own 0.2 x peak
#: level, the amplitude at which that ledger stops treating the profile as
#: closed.
DESPIKE_MIN_PEAK_FRACTION = 0.2


def _despike_profile(profile: np.ndarray) -> tuple[np.ndarray, int]:
    """Return the profile with isolated single-cell spikes repaired, and a count.

    A cell is judged only if it is finite, at or above
    ``DESPIKE_MIN_PEAK_FRACTION`` of the profile peak, and bracketed by finite
    neighbours within +/- ``DESPIKE_HALF_WIDTH`` cells -- at least one on each
    side and at least two in total.  A judged cell whose value differs from the
    median of those neighbours by more than ``DESPIKE_TOLERANCE`` is replaced by
    that median.  Replacement rather than deletion keeps the sample grid intact,
    so the quadrature below does not have to bridge a hole.
    """
    values = np.asarray(profile, dtype=np.float64)
    repaired = values.copy()
    finite = np.isfinite(values)
    if not np.any(finite):
        return repaired, 0
    peak = float(np.max(values[finite]))
    if not np.isfinite(peak) or peak <= 0.0:
        return repaired, 0
    n_replaced = 0
    for index in range(values.size):
        if not finite[index] or values[index] < DESPIKE_MIN_PEAK_FRACTION * peak:
            continue
        low = max(0, index - DESPIKE_HALF_WIDTH)
        high = index + 1 + DESPIKE_HALF_WIDTH
        left = values[low:index]
        right = values[index + 1 : high]
        left = left[np.isfinite(left)]
        right = right[np.isfinite(right)]
        if left.size == 0 or right.size == 0:
            continue
        neighbours = np.concatenate([left, right])
        if neighbours.size < 2:
            continue
        median = float(np.median(neighbours))
        if median == 0.0:
            continue
        if abs(values[index] / median - 1.0) > DESPIKE_TOLERANCE:
            repaired[index] = median
            n_replaced += 1
    return repaired, n_replaced


def _subtract_background(profile: np.ndarray) -> np.ndarray:
    """Return the profile with the effective-width ledger's baseline removed.

    The baseline is the smaller of the two outer-edge medians, clipped at zero.
    A side whose outer points are all non-finite contributes no median and is
    skipped rather than poisoning the minimum; the profile is only rejected
    outright when neither side yields one.
    """
    values = np.asarray(profile, dtype=np.float64)
    if np.count_nonzero(np.isfinite(values)) < 5:
        return np.full_like(values, np.nan)
    edge = BACKGROUND_EDGE_POINTS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        baseline = float(
            np.nanmin(
                [np.nanmedian(values[:edge]), np.nanmedian(values[-edge:])]
            )
        )
    if not np.isfinite(baseline):
        return np.full_like(values, np.nan)
    return values - max(baseline, 0.0)


def _flux_tube_weights(radius_cm: np.ndarray, radius_limit_cm: float) -> np.ndarray:
    """Return quadrature weights ``w`` with the flux-tube average = ``w @ f``.

    ``radius_cm`` holds the folded radius of each retained sample.  The weights
    implement ``int_0^R 2 r f(r) dr / R^2`` as a trapezoidal rule over the
    sorted sample radii, closed at ``r = 0`` -- where the integrand ``2 r f``
    vanishes, so the value there never enters -- and at ``r = R``, where ``f``
    is linearly interpolated between the two samples that bracket ``R``.  The
    weights sum to one for a uniform profile, exactly, because ``2 r`` is
    linear and the trapezoidal rule is exact for it.

    Raises ``ValueError`` if the samples do not reach ``R``; the average is not
    defined without extrapolating the profile past its own scan extent.
    """
    order = np.argsort(radius_cm)
    sorted_radius = radius_cm[order]
    if sorted_radius[-1] < radius_limit_cm:
        raise ValueError(
            f"folded profile reaches only r = {sorted_radius[-1]:g} cm, "
            f"short of the flux-tube radius {radius_limit_cm:g} cm"
        )
    inside = np.flatnonzero(sorted_radius < radius_limit_cm)
    first_outside = int(inside[-1]) + 1 if inside.size else 0
    nodes = np.concatenate(([0.0], sorted_radius[inside], [radius_limit_cm]))
    coefficient = np.empty(nodes.size, dtype=np.float64)
    coefficient[0] = (nodes[1] - nodes[0]) / 2.0
    coefficient[1:-1] = (nodes[2:] - nodes[:-2]) / 2.0
    coefficient[-1] = (nodes[-1] - nodes[-2]) / 2.0
    node_weight = 2.0 * nodes * coefficient / radius_limit_cm**2

    weights = np.zeros(radius_cm.size, dtype=np.float64)
    weights[order[inside]] = node_weight[1:-1]
    lower = sorted_radius[first_outside - 1] if first_outside else 0.0
    upper = sorted_radius[first_outside]
    fraction = 0.0 if upper == lower else (radius_limit_cm - lower) / (upper - lower)
    if first_outside:
        weights[order[first_outside - 1]] += node_weight[-1] * (1.0 - fraction)
    weights[order[first_outside]] += node_weight[-1] * fraction
    return weights


def _flux_tube_profile_stats(
    profile: np.ndarray,
    x_cm: np.ndarray,
    *,
    point_sem: np.ndarray | None = None,
    radius_cm: float = FLUX_TUBE_RADIUS_CM,
) -> dict[str, float | int]:
    """Return the flux-tube average of one radial profile and its companions.

    The pipeline is, in order: repair single-cell spikes; subtract the
    effective-width ledger's scalar background; take the core-band mean of that
    same background-subtracted profile (the numerator of the convention ratio,
    reported so the ratio is reconstructible from the product alone); clip the
    profile at zero and drop non-finite cells; fold about the profile centroid,
    treating ``r = |x - x_c|`` as radius; and integrate to ``radius_cm``.

    ``point_sem`` is an optional per-point uncertainty on the SAME samples; it
    is propagated through the quadrature weights in quadrature, treating the
    points as independent.  The background's own sampling uncertainty is not
    propagated -- it is a small, fully correlated term.
    """
    despiked, n_despiked = _despike_profile(profile)
    subtracted = _subtract_background(despiked)
    core_band = (x_cm >= X_MIN_CM) & (x_cm <= X_MAX_CM)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        core_mean = float(np.nanmean(subtracted[core_band]))

    empty = dict(
        ftavg=np.nan,
        ftavg_sem=np.nan,
        core=core_mean,
        centroid=np.nan,
        n_despiked=n_despiked,
    )
    finite = np.isfinite(subtracted)
    if np.count_nonzero(finite) < 5:
        return empty
    values = np.clip(subtracted[finite], 0.0, None)
    positions = x_cm[finite]
    total = float(np.sum(values))
    if total <= 0.0:
        return empty
    centroid = float(np.sum(values * positions) / total)
    try:
        weights = _flux_tube_weights(np.abs(positions - centroid), radius_cm)
    except ValueError:
        empty["centroid"] = centroid
        return empty

    ftavg = float(weights @ values)
    ftavg_sem = np.nan
    if point_sem is not None:
        sem_values = np.asarray(point_sem, dtype=np.float64)[finite]
        if np.all(np.isfinite(sem_values)):
            ftavg_sem = float(np.sqrt(np.sum((weights * sem_values) ** 2)))
    return dict(
        ftavg=ftavg,
        ftavg_sem=ftavg_sem,
        core=core_mean,
        centroid=centroid,
        n_despiked=n_despiked,
    )


def _flux_tube_series(
    profiles: np.ndarray,
    x_cm: np.ndarray,
    point_sem: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Apply ``_flux_tube_profile_stats`` to every (z, time) profile.

    ``profiles`` is shaped ``(z, x, time)``; every returned array is shaped
    ``(z, time)``.  Each time sample is reduced independently, so the exported
    series is self-contained sample by sample.
    """
    n_z, _, n_t = profiles.shape
    out = {
        name: np.full((n_z, n_t), np.nan, dtype=np.float64)
        for name in ("ftavg", "ftavg_sem", "core", "centroid")
    }
    out["n_despiked"] = np.zeros((n_z, n_t), dtype=np.int16)
    for zi in range(n_z):
        for ti in range(n_t):
            sem = None if point_sem is None else point_sem[zi, :, ti]
            stats = _flux_tube_profile_stats(profiles[zi, :, ti], x_cm, point_sem=sem)
            for name in ("ftavg", "ftavg_sem", "core", "centroid"):
                out[name][zi, ti] = stats[name]
            out["n_despiked"][zi, ti] = stats["n_despiked"]
    return out


def _rot0_isat_profiles(
    path: Path,
    experiment_set_id: int,
    z_cm: np.ndarray,
) -> dict[str, np.ndarray | str]:
    """Return one rot-0 line-scan product ordered onto the overlay z axis.

    Used for both probe faces, since the two dead-time profile products have the
    same layout and differ only in which electrical channel filled them:

    * ``ISAT_PROFILE_HDF5`` -- the ``i_sweep`` channel, the UPSTREAM face at
      rot-0 and the ruled Isat truth channel;
    * ``ROT0_ISAT_PROFILE_HDF5`` -- the ``isat`` channel, the DOWNSTREAM face.

    The per-run source channel is exported with the fields rather than assumed,
    and a set is NOT required to be single-channel: ES3 run 31 falls back to
    ``i_sweep`` in the downstream product, and the mix is disclosed through
    ``isat_ftavg_source_channel`` the same way
    ``augment_sim1d_overlay_isat_drive.py`` discloses it for the drive family.
    """
    with h5py.File(path, "r") as hdf:
        rotation = float(hdf.attrs["rotation_filter_deg"])
        if rotation != 0.0:
            raise ValueError(f"{path} is a rot-{rotation:g} product, not rot-0")
        x_cm = hdf["x_cm"][()]
        set_group = hdf[f"experiment_sets/{experiment_set_id}"]
        run_ids = sorted(set_group.keys())
        z_by_run = np.array(
            [float(set_group[run_id].attrs["z_cm"]) for run_id in run_ids]
        )
        order = np.argsort(z_by_run)
        run_ids = [run_ids[i] for i in order]
        if not np.allclose(z_by_run[order], z_cm):
            raise ValueError(
                f"{path} ES{experiment_set_id} z grid {z_by_run[order]} does not "
                f"match the overlay z grid {z_cm}"
            )
        time_ms = set_group[run_ids[0]]["inter_sweep_time_s"][()] * 1000.0
        isat = []
        sem = []
        channels = []
        for run_id in run_ids:
            run_group = set_group[run_id]
            if not np.allclose(run_group["inter_sweep_time_s"][()] * 1000.0, time_ms):
                raise ValueError(f"{path} run {run_id} dead-time grid differs")
            isat.append(run_group["isat_a"][()])
            shots = np.maximum(
                np.asarray(run_group["n_shots_used"][()], dtype=np.float64), 1.0
            )
            sem.append(run_group["isat_a_std"][()] / np.sqrt(shots))
            channels.append(str(run_group.attrs["deadtime_source_channel"]))
        ports = np.array(
            [int(set_group[run_id].attrs["port"]) for run_id in run_ids],
            dtype=np.int16,
        )
    return {
        "x_cm": x_cm,
        "time_ms": time_ms,
        "isat_a": np.stack(isat, axis=0),
        "sem_a": np.stack(sem, axis=0),
        "port": ports,
        "run_id": np.asarray(run_ids),
        "source_channel": np.asarray(channels),
    }


def _flow_symmetrized_profiles(
    upstream: dict,
    downstream: dict,
    areas_cm2: dict[str, dict[str, float]],
) -> dict[str, np.ndarray]:
    """Return the flow-artifact-cancelled current density from both probe faces.

    For each port, sample and x, the geometric mean ``sqrt(J_up * J_dn)`` of the
    two faces' AREA-NORMALIZED currents, in A cm^-2.  In the Chung two-sided
    probe model the faces carry flow factors ``exp(+K M / 2)`` and
    ``exp(-K M / 2)``; those are reciprocal, so they cancel exactly in the
    geometric mean to first order in ``M`` and what is left is the flow-free
    shape.  Either single face carries the flow artifact with the opposite
    sign, which is why their flux-tube corrections trend oppositely in z.

    The geometric mean is symmetric in its two arguments, so which product
    supplied which face does not matter; only the area normalization is
    face-specific, and that is keyed off each run's own channel.  A cell is NaN
    wherever either face is non-finite or non-positive, since a geometric mean
    of a non-positive current is not defined; ``_flux_tube_profile_stats`` then
    drops those cells from the quadrature.
    """
    if not np.array_equal(upstream["port"], downstream["port"]):
        raise ValueError(
            f"face products disagree on ports: {upstream['port']} vs "
            f"{downstream['port']}"
        )
    if not np.array_equal(upstream["run_id"], downstream["run_id"]):
        raise ValueError(
            f"face products disagree on runs: {upstream['run_id']} vs "
            f"{downstream['run_id']}"
        )
    if not np.allclose(upstream["x_cm"], downstream["x_cm"]):
        raise ValueError("face products disagree on the x grid")
    if not np.allclose(upstream["time_ms"], downstream["time_ms"]):
        raise ValueError("face products disagree on the inter-sweep time grid")

    geomean = np.empty_like(upstream["isat_a"])
    sem = np.empty_like(geomean)
    pairing = []
    used_areas = []
    for index, run_id in enumerate(upstream["run_id"]):
        run_id = str(run_id)
        channels = (
            str(upstream["source_channel"][index]),
            str(downstream["source_channel"][index]),
        )
        if channels[0] == channels[1]:
            raise ValueError(
                f"run {run_id}: both products report the {channels[0]} channel, "
                "so there is no second face to symmetrize against"
            )
        currents = (upstream["isat_a"][index], downstream["isat_a"][index])
        sems = (upstream["sem_a"][index], downstream["sem_a"][index])
        densities = []
        relatives = []
        run_areas = []
        for channel, current, current_sem in zip(channels, currents, sems):
            if channel not in CHANNEL_AREA_ATTR:
                raise ValueError(f"run {run_id}: unknown probe channel {channel}")
            area = float(areas_cm2[run_id][CHANNEL_AREA_ATTR[channel]])
            if area <= 0.0:
                raise ValueError(f"run {run_id}: non-positive {channel} face area")
            run_areas.append(area)
            densities.append(current / area)
            with np.errstate(invalid="ignore", divide="ignore"):
                relatives.append(current_sem / current)
        product = densities[0] * densities[1]
        usable = np.isfinite(product) & (product > 0.0)
        geomean[index] = np.where(usable, np.sqrt(np.abs(product)), np.nan)
        # d(sqrt(ab))/sqrt(ab) = 0.5 * hypot(da/a, db/b)
        sem[index] = geomean[index] * 0.5 * np.hypot(relatives[0], relatives[1])
        pairing.append(
            f"{channels[0]}/{CHANNEL_AREA_ATTR[channels[0]]}={run_areas[0]:.6f} cm2"
            f" x {channels[1]}/{CHANNEL_AREA_ATTR[channels[1]]}="
            f"{run_areas[1]:.6f} cm2"
        )
        used_areas.append(run_areas)
    return {
        "profiles": geomean,
        "sem": sem,
        "pairing": np.asarray(pairing),
        "area_cm2": np.asarray(used_areas, dtype=np.float64),
    }


def _isat_decay_stats(
    dataset: LapdDataset,
    profile_path: Path,
    zero_offsets_path: Path,
    experiment_set_id: int,
) -> dict[str, np.ndarray | float | int | str]:
    """Return offset-corrected x=0 upstream Isat mean and shot SEM."""
    zero_offsets = _load_zero_offsets(zero_offsets_path)
    means = []
    sems = []
    ports = []
    run_ids = []
    n_used = []
    n_rejected = []
    zero_offset_v = []
    source_channels = []
    source_inverted = []
    afterglow_start_ms = []
    reference_time_ms = None
    source_by_run: dict[str, tuple[ChannelKind, bool]] = {}

    with h5py.File(profile_path, "r") as profile_hdf:
        if not bool(profile_hdf.attrs.get("high_shot_rejection_enabled", False)):
            raise ValueError(f"{profile_path} was built without high-current rejection")
        sigma = float(profile_hdf.attrs["high_shot_rejection_sigma"])
        ratio = float(profile_hdf.attrs["high_shot_rejection_ratio"])
        min_shots = int(profile_hdf.attrs["high_shot_rejection_min_shots_used"])
        x_cm = profile_hdf["x_cm"][()]
        x_idx = int(np.argmin(np.abs(x_cm)))
        if not np.isclose(x_cm[x_idx], 0.0):
            raise ValueError("x=0 is not present in the Isat profile grid")
        set_group = profile_hdf[f"experiment_sets/{experiment_set_id}"]
        selected_run_ids = sorted(set_group.keys())
        for run_id in selected_run_ids:
            run_group = set_group[run_id]
            source_by_run[run_id] = (
                ChannelKind(str(run_group.attrs["deadtime_source_channel"])),
                bool(run_group.attrs["deadtime_source_invert_polarity"]),
            )

    for run_id in selected_run_ids:
        run = dataset.run(run_id)
        channel, invert_polarity = source_by_run[run_id]

        # Recreate the profile pipeline's per-cycle outlier flags, then use a
        # fixed shot ensemble for the continuous decay so its SEM is coherent.
        deadtime_means = _deadtime_shot_means(
            run,
            channel_kind=channel,
            invert_polarity=invert_polarity,
        )
        flagged = _high_shot_outlier_mask(
            deadtime_means,
            sigma=sigma,
            ratio=ratio,
            min_shots_used=min_shots,
        )[x_idx]
        reject_shot = np.any(flagged, axis=1)
        if np.sum(~reject_shot) < min_shots:
            flag_counts = np.sum(flagged, axis=1)
            keep = np.argsort(flag_counts)[:min_shots]
            reject_shot[:] = True
            reject_shot[keep] = False

        sweep = run.config.sweep
        decay_start_s = sweep.t0_s + sweep.n_cycles * sweep.tau_cycle_s
        sample, crop = _padded_sample_window(
            run,
            decay_start_s,
            ISAT_DECAY_STOP_S,
            ISAT_DECAY_FILTER_PAD_S,
        )
        channel_config = run.config.channel(channel)
        shot_start = run.flat_shot_index(x_idx, 0)
        shot_stop = shot_start + run.shots_per_position()
        with run.open() as raw_hdf:
            raw = raw_hdf[channel_config.hdf5_path][shot_start:shot_stop, sample].astype(
                np.float64
            )
            headers = raw_hdf[f"{channel_config.hdf5_path} headers"][shot_start:shot_stop]

        offset_v = _zero_offset_v(run, channel, zero_offsets)
        voltage = raw * headers["Scale"][:, None] + headers["Offset"][:, None]
        voltage -= offset_v
        current = channel_config.apply_calibration(voltage)
        if invert_polarity:
            current = -current
        current = butterworth_lowpass(
            current,
            sample_rate_hz=run.sample_rate_hz(),
            cutoff_hz=ISAT_DECAY_CUTOFF_HZ,
            axis=-1,
        )[:, crop]

        samples_per_bin = max(1, int(round(ISAT_DECAY_BIN_S / run.sample_dt_s())))
        n_bins = current.shape[1] // samples_per_bin
        current = current[:, : n_bins * samples_per_bin]
        current = current.reshape(current.shape[0], n_bins, samples_per_bin).mean(axis=2)
        start_index = sample.start + crop.start
        time_s = run.time_axis(n_bins * samples_per_bin, start_index=start_index)
        time_ms = (
            time_s[: n_bins * samples_per_bin]
            .reshape(n_bins, samples_per_bin)
            .mean(axis=1)
            * 1000.0
        )

        retained = current[~reject_shot]
        if reference_time_ms is None:
            reference_time_ms = time_ms
        elif not np.allclose(time_ms, reference_time_ms, rtol=0.0, atol=1e-9):
            raise ValueError(f"Isat decay time grid differs for run {run_id}")
        mean = np.mean(retained, axis=0)
        std = np.std(retained, axis=0, ddof=1)
        means.append(mean)
        sems.append(std / np.sqrt(retained.shape[0]))
        ports.append(int(run.config.probe.port or 0))
        run_ids.append(run_id)
        n_used.append(retained.shape[0])
        n_rejected.append(int(np.sum(reject_shot)))
        zero_offset_v.append(offset_v)
        source_channels.append(channel.value)
        source_inverted.append(invert_polarity)
        afterglow_start_ms.append(decay_start_s * 1000.0)

    if reference_time_ms is None:
        raise RuntimeError(
            f"No experiment-set-{experiment_set_id} upstream Isat decay traces found"
        )
    order = np.argsort(ports)
    return {
        "time_ms": reference_time_ms,
        "mean_a": np.asarray(means)[order],
        "sem_a": np.asarray(sems)[order],
        "port": np.asarray(ports, dtype=np.int16)[order],
        "run_id": np.asarray(run_ids)[order],
        "n_shots_used": np.asarray(n_used, dtype=np.int16)[order],
        "n_shots_rejected": np.asarray(n_rejected, dtype=np.int16)[order],
        "zero_offset_v": np.asarray(zero_offset_v)[order],
        "source_channel": np.asarray(source_channels)[order],
        "source_inverted": np.asarray(source_inverted, dtype=np.bool_)[order],
        "afterglow_start_ms": np.asarray(afterglow_start_ms)[order],
        "cutoff_hz": ISAT_DECAY_CUTOFF_HZ,
        "bin_s": ISAT_DECAY_BIN_S,
        "outlier_sigma": sigma,
        "outlier_ratio": ratio,
        "outlier_min_shots": min_shots,
        "outlier_method": "exclude shot if flagged in any inter-sweep cycle at x=0",
    }


def _te_window_spread_frac(
    refits_path: Path,
    experiment_set_id: int,
    ports: np.ndarray,
) -> np.ndarray:
    """Return the per-port fractional spread of the fit-window ``T_e`` re-fits.

    For each port, ``ptp(te_window_ev) / mean(te_window_ev)`` over the
    ``p_low`` x ``f_high`` fit-window grid stored in
    ``processed/sweep_window_refits.hdf5``.  The result is dimensionless, is
    aligned element-wise with ``ports`` (and hence with the overlay's ``port``
    and ``z_cm`` axes), and measures how far a port's ``T_e`` moves when the
    retarding-region fit window is varied rather than how uncertain its mean is.

    The value is ``NaN`` where the experiment set or the port has no re-fit
    group, and ``NaN`` where a port's grid contains a failed re-fit, since a
    single missing window makes the full spread undetermined rather than
    smaller.
    """
    spread = np.full(len(ports), np.nan, dtype=np.float64)
    with h5py.File(refits_path, "r") as refits_hdf:
        set_group = refits_hdf.get(f"set{experiment_set_id}")
        if set_group is None:
            return spread
        for index, port in enumerate(ports):
            port_group = set_group.get(f"port{int(port)}")
            if port_group is None:
                continue
            grid = np.asarray(port_group["te_window_ev"][()], dtype=np.float64)
            spread[index] = np.ptp(grid) / np.mean(grid)
    return spread


def _discharge_stats(
    dataset: LapdDataset,
    experiment_set_id: int,
) -> dict[str, np.ndarray | int]:
    """Return the offset-corrected experiment-set discharge current and voltage.

    Each run's discharge current has its own additive channel zero offset
    (``LapdRun.discharge_zero_offset_stats``) subtracted before smoothing, so
    the exported trace reads zero where no current flows.

    Two per-sample dispersion measures are returned for both current and
    voltage, computed over the same pooled single-shot ensemble as the mean:

    * ``*_sd_*``  -- the sample standard deviation (``ddof=1``), i.e. the
      shot-to-shot ENVELOPE.  This is the width of the population of real
      shots, and does not shrink as more shots are stored.
    * ``*_sem_*`` -- that standard deviation divided by ``sqrt(n_traces)``,
      i.e. the uncertainty of the plotted MEAN.

    The ensemble is pooled across runs on the raw SIS-trigger sample grid,
    which every run in the set is required to share; no per-shot time
    alignment is applied.  Shot-to-shot breakdown-timing jitter therefore
    appears INSIDE these dispersions rather than being removed from them,
    and dominates them wherever the trace is steep.  Each single trace is
    smoothed by the ``DISCHARGE_SMOOTHING_SAMPLES`` moving average before
    the statistics are taken, so both measures describe smoothed shots.
    """
    currents = []
    voltages_raw = []
    zero_offsets_a = []
    run_ids = []
    reference_time_s = None

    for run_id in dataset.experiment_set_run_ids(experiment_set_id):
        run = dataset.run(run_id)
        current, voltage, time_s = run.discharge_traces()
        zero_offset_a = run.discharge_zero_offset_stats().offset_a
        current = current - zero_offset_a
        if reference_time_s is None:
            reference_time_s = time_s
        elif not np.allclose(time_s, reference_time_s, rtol=0.0, atol=1e-12):
            raise ValueError(f"Discharge time grid differs for run {run_id}")
        zero_offsets_a.append(zero_offset_a)
        run_ids.append(run_id)
        currents.append(
            uniform_filter1d(
                current,
                size=DISCHARGE_SMOOTHING_SAMPLES,
                axis=1,
                mode="nearest",
            )
        )
        voltages_raw.append(
            uniform_filter1d(
                voltage,
                size=DISCHARGE_SMOOTHING_SAMPLES,
                axis=1,
                mode="nearest",
            )
        )

    if reference_time_s is None:
        raise RuntimeError(f"No experiment-set-{experiment_set_id} discharge traces found")
    current_all = np.concatenate(currents, axis=0)
    voltage_all_raw = np.concatenate(voltages_raw, axis=0)
    n_traces = current_all.shape[0]
    current_std = np.std(current_all, axis=0, ddof=1)
    voltage_std = np.std(voltage_all_raw, axis=0, ddof=1)
    return {
        "time_ms": reference_time_s * 1000.0,
        "current_mean_a": np.mean(current_all, axis=0),
        "current_sd_a": current_std,
        "current_sem_a": current_std / np.sqrt(n_traces),
        "voltage_positive_mean_v": -np.mean(voltage_all_raw, axis=0),
        "voltage_sd_v": voltage_std,
        "voltage_sem_v": voltage_std / np.sqrt(n_traces),
        "n_traces": n_traces,
        "zero_offset_a": np.asarray(zero_offsets_a, dtype=np.float64),
        "zero_offset_run_id": np.asarray(run_ids),
    }


def export_overlay(
    density_path: Path,
    te_path: Path,
    isat_profile_path: Path,
    zero_offsets_path: Path,
    manifest_path: Path,
    output_path: Path,
    experiment_set_id: int = 1,
    window_refits_path: Path = WINDOW_REFITS_HDF5,
    rot0_isat_profile_path: Path = ROT0_ISAT_PROFILE_HDF5,
) -> Path:
    experiment_set_key = str(experiment_set_id)
    with h5py.File(density_path, "r") as density_hdf, h5py.File(te_path, "r") as te_hdf:
        density = _load_density_stats(
            density_hdf,
            experiment_set_key,
            X_MIN_CM,
            X_MAX_CM,
        )
        density_x_cm = density_hdf["x_cm"][()]
        density_profiles_m3 = density_hdf[
            f"experiment_sets/{experiment_set_key}/n_e_m3"
        ][()]
        runs_group = density_hdf[f"experiment_sets/{experiment_set_key}/runs"]
        face_areas_cm2 = {
            run_id: {
                attr: float(run_group.attrs[attr])
                for attr in set(CHANNEL_AREA_ATTR.values())
            }
            for run_id, run_group in runs_group.items()
        }
        te = _load_te_stats(
            te_hdf,
            experiment_set_key,
            X_MIN_CM,
            X_MAX_CM,
            dataset="te_filled",
        )
        if not np.allclose(density.z_cm, te.z_cm):
            raise ValueError(
                f"ES{experiment_set_id} density and temperature z grids differ"
            )
        if len(density.z_cm) != len(PORTS):
            raise ValueError(
                f"Expected {len(PORTS)} ES{experiment_set_id} port rows, "
                f"found {len(density.z_cm)}"
            )
        experiment_label = str(
            density_hdf[f"experiment_sets/{experiment_set_key}"].attrs["label"]
        )
        factor = float(density_hdf.attrs["probe_a_factor"])
        factor_lower = float(density_hdf.attrs["probe_a_factor_lower_bound"])
        factor_upper = float(density_hdf.attrs["probe_a_factor_upper_bound"])
    dataset = LapdDataset.from_manifest(manifest_path)
    isat_decay = _isat_decay_stats(
        dataset,
        isat_profile_path,
        zero_offsets_path,
        experiment_set_id,
    )
    if not np.array_equal(isat_decay["port"], PORTS):
        raise ValueError(
            f"ES{experiment_set_id} Isat decay ports {isat_decay['port']} "
            f"do not match expected {PORTS}"
        )
    discharge = _discharge_stats(dataset, experiment_set_id)
    te_window_spread = _te_window_spread_frac(
        window_refits_path,
        experiment_set_id,
        PORTS,
    )

    density_ftavg = _flux_tube_series(density_profiles_m3, density_x_cm)

    def _face_ftavg(path: Path, label: str) -> tuple[dict, dict]:
        scans = _rot0_isat_profiles(path, experiment_set_id, density.z_cm)
        if not np.array_equal(scans["port"], PORTS):
            raise ValueError(
                f"ES{experiment_set_id} {label} Isat ports {scans['port']} "
                f"do not match expected {PORTS}"
            )
        if not np.allclose(scans["x_cm"], density_x_cm):
            raise ValueError(f"{label} Isat and density x grids differ")
        if not np.allclose(scans["time_ms"], density.time_ms):
            raise ValueError(
                f"{label} Isat and density inter-sweep time grids differ"
            )
        return scans, _flux_tube_series(
            scans["isat_a"], scans["x_cm"], scans["sem_a"]
        )

    upstream_scans, upstream_ftavg = _face_ftavg(isat_profile_path, "upstream")
    rot0_isat, isat_ftavg = _face_ftavg(rot0_isat_profile_path, "downstream")
    geomean_scans = _flow_symmetrized_profiles(
        upstream_scans,
        rot0_isat,
        face_areas_cm2,
    )
    geomean_ftavg = _flux_tube_series(
        geomean_scans["profiles"],
        upstream_scans["x_cm"],
        geomean_scans["sem"],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema_version=np.array(13, dtype=np.int16),
        experiment_set_id=np.array(experiment_set_id, dtype=np.int16),
        experiment_label=np.array(experiment_label),
        port=PORTS,
        z_cm=density.z_cm,
        density_time_ms=density.time_ms,
        density_mean_cm3=density.mean * DENSITY_SCALE_CM3,
        density_total_sem_cm3=_plot_uncertainty(density, "sem") * DENSITY_SCALE_CM3,
        density_radial_sem_cm3=density.sem * DENSITY_SCALE_CM3,
        density_core_count=density.count,
        te_time_ms=te.time_ms,
        te_mean_ev=te.mean,
        te_sem_ev=te.sem,
        te_core_count=te.count,
        te_window_spread_frac=te_window_spread,
        core_x_min_cm=np.array(X_MIN_CM),
        core_x_max_cm=np.array(X_MAX_CM),
        probe_a_factor=np.array(factor),
        probe_a_factor_lower_bound=np.array(factor_lower),
        probe_a_factor_upper_bound=np.array(factor_upper),
        isat_decay_time_ms=isat_decay["time_ms"],
        isat_decay_mean_a=isat_decay["mean_a"],
        isat_decay_sem_a=isat_decay["sem_a"],
        isat_decay_port=isat_decay["port"],
        isat_decay_run_id=isat_decay["run_id"],
        isat_decay_n_shots_used=isat_decay["n_shots_used"],
        isat_decay_n_shots_rejected=isat_decay["n_shots_rejected"],
        isat_decay_zero_offset_v=isat_decay["zero_offset_v"],
        isat_decay_source_channel=isat_decay["source_channel"],
        isat_decay_source_inverted=isat_decay["source_inverted"],
        isat_decay_afterglow_start_ms=isat_decay["afterglow_start_ms"],
        isat_decay_cutoff_hz=np.array(isat_decay["cutoff_hz"]),
        isat_decay_bin_s=np.array(isat_decay["bin_s"]),
        isat_decay_outlier_sigma=np.array(isat_decay["outlier_sigma"]),
        isat_decay_outlier_ratio=np.array(isat_decay["outlier_ratio"]),
        isat_decay_outlier_min_shots=np.array(
            isat_decay["outlier_min_shots"],
            dtype=np.int16,
        ),
        isat_decay_outlier_method=np.array(isat_decay["outlier_method"]),
        isat_decay_current_correction=np.array(
            "digitizer scale/offset, measured zero-offset subtraction, channel calibration"
        ),
        isat_decay_face=np.array(
            "upstream-facing electrical channel at x=0; see per-port source metadata"
        ),
        discharge_time_ms=discharge["time_ms"],
        discharge_current_mean_a=discharge["current_mean_a"],
        discharge_current_sd_a=discharge["current_sd_a"],
        discharge_current_sem_a=discharge["current_sem_a"],
        discharge_voltage_positive_mean_v=discharge["voltage_positive_mean_v"],
        discharge_voltage_sd_v=discharge["voltage_sd_v"],
        discharge_voltage_sem_v=discharge["voltage_sem_v"],
        discharge_spread_alignment=np.array(
            "pooled single shots on the shared raw SIS-trigger sample grid; "
            "no per-shot time alignment, so shot-to-shot breakdown-timing "
            "jitter is inside the spread and dominates it where the trace is "
            "steep; each shot smoothed by the moving average first; sd is the "
            "ddof=1 shot envelope, sem is sd/sqrt(discharge_n_traces)"
        ),
        discharge_n_traces=np.array(discharge["n_traces"], dtype=np.int16),
        discharge_zero_offset_a=discharge["zero_offset_a"],
        discharge_zero_offset_run_id=discharge["zero_offset_run_id"],
        discharge_current_correction=np.array(
            "per-run additive zero offset from the post-connect pre-avalanche "
            "window subtracted before smoothing"
        ),
        discharge_smoothing_samples=np.array(
            DISCHARGE_SMOOTHING_SAMPLES,
            dtype=np.int16,
        ),
        discharge_voltage_sign=np.array(
            "positive overlay is -1 times raw cathode-anode voltage"
        ),
        density_uncertainty=np.array(
            "radial SEM plus Probe-A area calibration in quadrature"
        ),
        density_mean_convention=np.array(
            f"density_mean_cm3 and its SEM fields are the LEGACY core-band "
            f"convention: an unweighted arithmetic mean of the line scan over "
            f"{X_MIN_CM:g} <= x <= {X_MAX_CM:g} cm, a diameter line cut with no "
            "radial area weighting.  It is NOT the flux-tube average; see "
            "density_ftavg_cm3 and ftavg_convention."
        ),
        ftavg_radius_cm=np.array(FLUX_TUBE_RADIUS_CM),
        ftavg_convention=np.array(
            "flux-tube average: int_0^R 2 pi r f(r) dr / (pi R^2) with "
            f"R = ftavg_radius_cm = {FLUX_TUBE_RADIUS_CM:g} cm, evaluated per "
            "port and per inter-sweep time sample by trapezoidal quadrature "
            "over the folded line-scan samples.  The quadrature is closed at "
            "r = 0, where the integrand 2 r f vanishes, and at r = R, where f "
            "is linearly interpolated between the two samples bracketing R; a "
            "sample whose folded profile does not reach R is NaN rather than "
            "extrapolated.  R is the caliper-measured cathode frame opening "
            "(14.5 in aperture diameter, 2026-08-17), the same radius the 1D "
            "transport model's single radial cell uses."
        ),
        ftavg_axisymmetry=np.array(
            "ASSUMED, not measured: the 51-point scan is a single diameter in "
            "x at fixed y, folded about its own intensity centroid, and the "
            "folded coordinate r = |x - x_c| is then read as radius.  This "
            "assumes the column is axisymmetric about that centroid.  The "
            "per-sample centroids are exported as density_ftavg_centroid_cm "
            "and isat_ftavg_centroid_cm; a single-diameter scan cannot test "
            "the assumption, and the on-record left/right profile asymmetry is "
            "not captured by it."
        ),
        ftavg_background=np.array(
            "scalar baseline = the smaller of the two medians of the outer "
            f"{BACKGROUND_EDGE_POINTS} points on each side, clipped at zero "
            "and subtracted, profile then clipped at zero; convention taken "
            "from the effective-width ledger.  A side whose outer points are "
            "all non-finite yields no median and is skipped; a sample with "
            "neither side usable is NaN.  The legacy density_mean_cm3 is "
            "NOT background-subtracted, so the ratio of the two exported "
            "fields is not the pure weighting correction; ratio "
            "density_ftavg_core_cm3 / density_ftavg_cm3 for that."
        ),
        ftavg_despike=np.array(
            f"isolated single-cell spikes at or above "
            f"{DESPIKE_MIN_PEAK_FRACTION:g} x peak that differ by more than "
            f"{DESPIKE_TOLERANCE:g} from the median of their finite "
            f"neighbours within +/- {DESPIKE_HALF_WIDTH} cells are replaced by "
            "that median; gate taken from the edge-Te ledger, amplitude floor "
            "from the effective-width ledger.  Counts per sample are exported "
            "as density_ftavg_n_despiked, isat_ftavg_upstream_n_despiked and "
            "isat_ftavg_n_despiked."
        ),
        ftavg_face_ruling=np.array(
            "RULED 2026-08-18: at rot-0 the i_sweep channel collects on the "
            "UPSTREAM probe face and is the Isat truth channel; the isat "
            "channel is the DOWNSTREAM face and under-reads because it sits "
            "in the probe body's flow shadow (the assignment reverses at "
            "rot-180).  isat_ftavg_upstream_* is therefore the "
            "correction-bearing family and is the face the density chain and "
            "the isat_decay_*/isat_drive_* families already use; isat_ftavg_* "
            "is the downstream face, retained as the effective-width ledger's "
            "rot-0 primary and as the face the 2026-08-18 paper read used.  "
            "The two faces' flux-tube corrections run in OPPOSITE directions "
            "with z and must never be ratioed against each other.  "
            "ADJUDICATED 2026-08-18, three estimators with three roles: "
            "isat_ftavg_* is the downstream face, SHADOWED, biased low; "
            "isat_ftavg_upstream_* is the ruled truth channel and the "
            "flow-ENHANCED conjugate of it, biased the other way; "
            "isat_ftavg_geomean_* is the flow-CANCELLED central estimator "
            "built from both, and is the one whose C(z) came out z-flat."
        ),
        density_ftavg_cm3=density_ftavg["ftavg"] * M3_TO_CM3,
        density_ftavg_core_cm3=density_ftavg["core"] * M3_TO_CM3,
        density_ftavg_centroid_cm=density_ftavg["centroid"],
        density_ftavg_n_despiked=density_ftavg["n_despiked"],
        isat_ftavg_geomean_time_ms=upstream_scans["time_ms"],
        isat_ftavg_geomean_a_per_cm2=geomean_ftavg["ftavg"],
        isat_ftavg_geomean_sem_a_per_cm2=geomean_ftavg["ftavg_sem"],
        isat_ftavg_geomean_core_a_per_cm2=geomean_ftavg["core"],
        isat_ftavg_geomean_centroid_cm=geomean_ftavg["centroid"],
        isat_ftavg_geomean_n_despiked=geomean_ftavg["n_despiked"],
        isat_ftavg_geomean_port=upstream_scans["port"],
        isat_ftavg_geomean_run_id=upstream_scans["run_id"],
        isat_ftavg_geomean_area_cm2=geomean_scans["area_cm2"],
        isat_ftavg_geomean_pairing=geomean_scans["pairing"],
        isat_ftavg_geomean_definition=np.array(
            "FLOW-SYMMETRIZED central estimator, in A cm^-2: per port, per "
            "inter-sweep sample and per x, the geometric mean "
            "sqrt(J_up * J_dn) of the two probe faces' area-normalized "
            "currents, then the same despike / background / centroid-fold / "
            "R = ftavg_radius_cm quadrature as the other flux-tube fields.  "
            "In the Chung two-sided model the faces carry reciprocal flow "
            "factors exp(+K M / 2) and exp(-K M / 2), which cancel in the "
            "geometric mean to first order in M, so this is the "
            "flow-artifact-cancelled shape; each single face keeps that "
            "artifact with the opposite sign.  Per-run face-and-area pairing "
            "is in isat_ftavg_geomean_pairing.  A cell is NaN wherever either "
            "face is non-positive or non-finite."
        ),
        isat_ftavg_upstream_time_ms=upstream_scans["time_ms"],
        isat_ftavg_upstream_a=upstream_ftavg["ftavg"],
        isat_ftavg_upstream_sem_a=upstream_ftavg["ftavg_sem"],
        isat_ftavg_upstream_core_a=upstream_ftavg["core"],
        isat_ftavg_upstream_centroid_cm=upstream_ftavg["centroid"],
        isat_ftavg_upstream_n_despiked=upstream_ftavg["n_despiked"],
        isat_ftavg_upstream_port=upstream_scans["port"],
        isat_ftavg_upstream_run_id=upstream_scans["run_id"],
        isat_ftavg_upstream_source_file=np.array(str(isat_profile_path)),
        isat_ftavg_upstream_source_channel=upstream_scans["source_channel"],
        isat_ftavg_upstream_face=np.array(
            "UPSTREAM probe face at rot-0 ('i_sweep' channel; per-port channel "
            "is in isat_ftavg_upstream_source_channel).  RULED the Isat truth "
            "channel 2026-08-18 because the opposite face collects in the "
            "probe body's flow shadow and under-reads.  This is the same face "
            "as the density chain behind density_ftavg_cm3 and as "
            "isat_decay_*/isat_drive_*, so this family is the "
            "correction-bearing one and pairs with them without a face change. "
            "See ftavg_face_ruling."
        ),
        isat_ftavg_time_ms=rot0_isat["time_ms"],
        isat_ftavg_a=isat_ftavg["ftavg"],
        isat_ftavg_sem_a=isat_ftavg["ftavg_sem"],
        isat_ftavg_core_a=isat_ftavg["core"],
        isat_ftavg_centroid_cm=isat_ftavg["centroid"],
        isat_ftavg_n_despiked=isat_ftavg["n_despiked"],
        isat_ftavg_port=rot0_isat["port"],
        isat_ftavg_run_id=rot0_isat["run_id"],
        isat_ftavg_source_file=np.array(str(rot0_isat_profile_path)),
        isat_ftavg_source_channel=rot0_isat["source_channel"],
        isat_ftavg_face=np.array(
            "DOWNSTREAM probe face at rot-0 ('isat' channel; per-port channel "
            "is in isat_ftavg_source_channel).  CAVEAT: this face collects in "
            "the probe body's flow shadow and under-reads, which is why the "
            "2026-08-18 ruling put the truth channel on the upstream face -- "
            "use isat_ftavg_upstream_* for a correction.  Retained because it "
            "is the effective-width ledger's rot-0 primary and the face the "
            "2026-08-18 paper read measured.  It is NOT the face behind "
            "isat_decay_*, isat_drive_* or the density chain, and its "
            "flux-tube correction runs the opposite way with z."
        ),
        isat_ftavg_sem_definition=np.array(
            "per-point shot SEM (isat_a_std / sqrt(n_shots_used)) propagated "
            "through the quadrature weights in quadrature, points treated as "
            "independent; the background's own sampling uncertainty is not "
            "included.  This is a shot SEM and is NOT commensurate with "
            "density_total_sem_cm3, which is a radial-scatter SEM plus the "
            "Probe-A area calibration."
        ),
    )
    print(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--density", type=Path, default=DENSITY_HDF5)
    parser.add_argument("--te-filled", type=Path, default=TE_HDF5)
    parser.add_argument("--isat-profiles", type=Path, default=ISAT_PROFILE_HDF5)
    parser.add_argument(
        "--rot0-isat-profiles",
        type=Path,
        default=ROT0_ISAT_PROFILE_HDF5,
        help="rot-0 Isat-channel line-scan product for the flux-tube fields",
    )
    parser.add_argument("--zero-offsets", type=Path, default=ZERO_OFFSETS)
    parser.add_argument("--window-refits", type=Path, default=WINDOW_REFITS_HDF5)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--experiment-set", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(
        f"processed/es{args.experiment_set}_sim1d_overlay.npz"
    )
    export_overlay(
        args.density,
        args.te_filled,
        args.isat_profiles,
        args.zero_offsets,
        args.manifest,
        output,
        args.experiment_set,
        args.window_refits,
        args.rot0_isat_profiles,
    )


if __name__ == "__main__":
    main()
