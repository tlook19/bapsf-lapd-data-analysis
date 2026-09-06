"""Export one experiment set for the bapsf-transport sim1d notebook.

The NPZ product is self-contained and uses simulation-facing units:

* core density and total SEM in cm^-3;
* flux-tube-averaged density in cm^-3 and flux-tube-averaged ion-saturation
  current in A, the second radial-averaging convention (see below);
* core electron temperature in eV and its total SEM, the radial scatter and
  the fit-window convention added in quadrature;
* the per-port trust model the filled T_e was built under, and the per-port,
  per-sample count of semi-quantitative cells behind each T_e row;
* offset-corrected ion-saturation current at x=0 in A for BOTH Mach-probe
  faces, and the flow-symmetrized geometric mean of the two in A cm^-2;
* shot-averaged interferometer line-integrated density in cm^-2 for the
  three chords, on the raw interferometer clock;
* offset-corrected discharge current in A and cathode-anode voltage in V,
  each with the shot-to-shot standard deviation and the SEM of the mean;
* per-port fractional spread of the fit-window T_e re-fits, dimensionless;
* the per-port, per-sample record of where the filled T_e product's core-mean
  monotonic-z clamp acted, so a consumer can see which T_e samples are a
  neighbouring port's value rather than their own;
* the axial port ladder the product was exported on (``port_map``);
* all time axes in ms relative to the experimental SIS trigger.

Probe-A density SEM includes the propagated area-calibration uncertainty.
The ion-current decay uses the same one-sided high-current detector as the
profile pipeline.  A shot is excluded from the continuous decay trace if it is
flagged in any pre-afterglow inter-sweep cell at x=0.  The retained traces are
100 kHz low-pass filtered and reduced to approximately 10 us time bins before
the shot mean and SEM are computed.  The downstream face runs the identical
pipeline against its own product, and therefore builds its OWN shot ensemble
from its own channel's flags: the two faces reject different shots and both
ensemble sizes are exported.  See ``isat_decay_face_convention``.

The interferometer chords are read with the repo's own reader and alignment
(``plot_interferometer_by_experiment_set``): per-chord Welford statistics over
every stored shot of every run in the set, pooled, on that chord's reference
time grid, then put on one shared grid.  Its clock is NOT the Isat clock; see
``interf_decay_clock_offset``.
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

That smoothing is applied per shot and BEFORE pooling, so the exported
discharge dispersion is the spread of smoothed shots and reads low wherever the
trace is steep: the moving average rounds each shot's own breakdown knee before
the ensemble ever sees it, and the shot-to-shot breakdown-timing jitter is what
the steep part of the spread is made of.  ``--raw-discharge-ensemble`` ADDS the
same statistics taken on the unsmoothed shots, plus each shot's crossing time
through ONE common current level, as a separate ``discharge_current_raw_*`` /
``discharge_raw_t_half_*`` family.  The level is half the ensemble MEDIAN
plateau current, taken over the scoring plateau window, so the crossing times
differ only in when each shot got there
and not in how large it grew; ES1 needs that, because run 05 runs a plateau
about 60 % above the pack and a per-shot half-peak level would report it as
breaking down late for a purely amplitude reason.  No existing field changes,
and the family is ABSENT unless the flag is passed, so a product that lacks
these names was exported without it rather than with it and zeroed.

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
from bapsf_lapd.config import PORT_MAP_DEFAULT, PORT_MAPS, z_from_port
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
from plot_interferometer_by_experiment_set import (
    INTERFEROMETER_PORTS,
    PLASMA_DIAMETER_CM,
    RunningTraceStats,
    _phase_group,
    _reference_time_grid,
    _time_group,
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

#: Why this exporter will not write a product on a non-default port ladder.
#: Only the interferometer chord positions are computed here; the probe-port z
#: grid arrives already baked into the upstream products, so exporting on a
#: ladder those products were not built under would write a mixed-ladder
#: product -- and this product is scored, with nothing in it to say so.
PORT_MAP_REFUSAL = (
    "refusing to export on port_map={port_map!r}: only the interferometer "
    "chord positions are derived here, while the probe-port z grid is copied "
    "from the upstream products, which bake in the ladder they were built "
    "under. Exporting now would write a mixed-ladder product -- chords on the "
    "requested ladder, probe ports on the one the inputs carry. Adopting a "
    "ladder is a rebuild of those inputs first, in order: (1) the "
    "shot-averaged Langmuir product, then the review-flag pass that appends "
    "to it; (2) the four dead-time and line-scan profile products, then the "
    "two rot-180 annotators over the rebuilt rot-180 pair; (3) the filled "
    "T_e product and its ES3 p11/p29 variant, whose own sources transcribe "
    "the ladder and must be edited before they are re-run; (4) the "
    "probe-area calibration, which chooses its T_e row by z; (5) the "
    "density, density-Mach, Mach-velocity and ES3 scaled products. Only then "
    "does this exporter have inputs on the requested ladder, and only then "
    "does the refusal here come out."
)
X_MIN_CM = -10.0
X_MAX_CM = 10.0
DISCHARGE_SMOOTHING_SAMPLES = 9

#: The SCORING PLATEAU WINDOW, in ms on the trigger-referenced grid: the same
#: 15.0-19.5 ms the transport comparison scores the drive plateau over.  It is
#: inherited from that convention rather than chosen here, and is used only to
#: set the ONE crossing level the raw-ensemble timing statistic is referred to.
RAW_PLATEAU_WINDOW_MS = (15.0, 19.5)
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


def _isat_decay_geomean(
    upstream: dict,
    downstream: dict,
    areas_cm2: dict[str, dict[str, float]],
) -> dict[str, np.ndarray]:
    """Return the flow-artifact-cancelled x=0 afterglow current density.

    The x=0 continuous-trace form of ``_flow_symmetrized_profiles``: per port
    and per time sample, the geometric mean ``sqrt(J_up * J_dn)`` of the two
    probe faces' AREA-NORMALIZED currents, in A cm^-2.  The Chung argument and
    the SEM propagation are that function's; only the array rank differs, since
    the decay traces carry no x axis.

    The two faces must come from the same runs in the same order and must agree
    on the time grid, and each run must report a DIFFERENT electrical channel
    for its two faces -- otherwise there is no second face to symmetrize
    against.  The area normalization is keyed off each run's own recorded
    channel, not off which product supplied it.

    A sample is NaN wherever either face is non-finite or non-positive; a
    geometric mean of a non-positive current is not defined, and a face that
    has decayed through zero is data, not an artifact to be clipped away.
    """
    if not np.array_equal(upstream["port"], downstream["port"]):
        raise ValueError(
            f"Isat decay faces disagree on ports: {upstream['port']} vs "
            f"{downstream['port']}"
        )
    if not np.array_equal(upstream["run_id"], downstream["run_id"]):
        raise ValueError(
            f"Isat decay faces disagree on runs: {upstream['run_id']} vs "
            f"{downstream['run_id']}"
        )
    if not np.allclose(
        upstream["time_ms"], downstream["time_ms"], rtol=0.0, atol=1e-9
    ):
        raise ValueError("Isat decay faces disagree on the time grid")

    geomean = np.empty_like(np.asarray(upstream["mean_a"], dtype=np.float64))
    sem = np.empty_like(geomean)
    used_areas = []
    pairing = []
    for index, run_id in enumerate(upstream["run_id"]):
        run_id = str(run_id)
        channels = (
            str(upstream["source_channel"][index]),
            str(downstream["source_channel"][index]),
        )
        if channels[0] == channels[1]:
            raise ValueError(
                f"run {run_id}: both Isat decay faces report the {channels[0]} "
                "channel, so there is no second face to symmetrize against"
            )
        currents = (upstream["mean_a"][index], downstream["mean_a"][index])
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
        "geomean_a_per_cm2": geomean,
        "sem_a_per_cm2": sem,
        "area_cm2": np.asarray(used_areas, dtype=np.float64),
        "pairing": np.asarray(pairing),
    }


def _interferometer_decay_stats(
    dataset: LapdDataset,
    experiment_set_id: int,
    port_map: str = PORT_MAP_DEFAULT,
) -> dict[str, np.ndarray]:
    """Return the shot-averaged interferometer line density for the three chords.

    The reader, the calibration, the ``rigol_missing`` skip, the Welford shot
    statistics, the per-chord reference time grid and the ``np.interp``
    alignment onto it are all
    ``plot_interferometer_by_experiment_set.build_grouped_stats``, restricted
    to one experiment set; the line-integrated normalization is that module's
    ``PLASMA_DIAMETER_CM`` (40 cm) times the line-average density.  Every shot
    of every run in the set enters ONE pooled accumulator per chord, so the
    exported mean is not a mean of per-run means and the exported SEM is the
    pooled standard deviation over ``n_shots``.

    The three chords do not share a digitizer: p20 and p29 sit on the LeCroy
    ``time_array`` and p40 on the Rigol ``time_array_p40``.  Each chord is
    reduced on its own reference grid first, and the result is then
    interpolated once onto the first chord's grid so the exported product has a
    single time axis.  A chord whose own grid does not cover that shared grid is
    REFUSED rather than silently extrapolated by ``np.interp``'s endpoint hold.

    ``MSI/Interferometer array`` is malformed in these files and is never read.
    """
    run_ids = list(dataset.experiment_set_run_ids(experiment_set_id))
    chord_time_ms = []
    chord_mean_cm3 = []
    chord_sem_cm3 = []
    chord_n = []
    for port in INTERFEROMETER_PORTS:
        reference_time_ms = _reference_time_grid(dataset, experiment_set_id, port)
        stats = RunningTraceStats.empty(reference_time_ms)
        phase_group = _phase_group(port)
        time_group = _time_group(port)
        for run_id in run_ids:
            run = dataset.run(run_id)
            with h5py.File(run.path, "r") as hdf:
                calibration = float(
                    hdf[phase_group].attrs["calibration factor (m^-3/rad)"]
                )
                for key in sorted(hdf[phase_group].keys(), key=int):
                    phase_ds = hdf[f"{phase_group}/{key}"]
                    if bool(phase_ds.attrs.get("rigol_missing", False)):
                        continue
                    phase = phase_ds[()].astype(np.float64)
                    time_ms = hdf[f"{time_group}/{key}"][()].astype(np.float64)
                    line_average_cm3 = phase * calibration * M3_TO_CM3
                    if np.array_equal(time_ms, reference_time_ms):
                        aligned = line_average_cm3
                    else:
                        aligned = np.interp(
                            reference_time_ms, time_ms, line_average_cm3
                        )
                    stats.update(aligned)
        if stats.count == 0:
            raise RuntimeError(
                f"No experiment-set-{experiment_set_id} interferometer shots "
                f"for chord p{port}"
            )
        chord_time_ms.append(reference_time_ms)
        chord_mean_cm3.append(stats.mean)
        chord_sem_cm3.append(stats.stderr)
        chord_n.append(stats.count)

    shared_time_ms = chord_time_ms[0]
    mean = []
    sem = []
    for index, port in enumerate(INTERFEROMETER_PORTS):
        own_time_ms = chord_time_ms[index]
        if np.array_equal(own_time_ms, shared_time_ms):
            mean.append(chord_mean_cm3[index])
            sem.append(chord_sem_cm3[index])
            continue
        if (
            own_time_ms[0] > shared_time_ms[0]
            or own_time_ms[-1] < shared_time_ms[-1]
        ):
            raise ValueError(
                f"interferometer chord p{port} spans "
                f"[{own_time_ms[0]:g}, {own_time_ms[-1]:g}] ms and does not "
                f"cover the shared grid "
                f"[{shared_time_ms[0]:g}, {shared_time_ms[-1]:g}] ms"
            )
        mean.append(np.interp(shared_time_ms, own_time_ms, chord_mean_cm3[index]))
        sem.append(np.interp(shared_time_ms, own_time_ms, chord_sem_cm3[index]))

    return {
        "time_ms": shared_time_ms,
        "line_density_cm2": np.stack(mean, axis=0) * PLASMA_DIAMETER_CM,
        "sem_cm2": np.stack(sem, axis=0) * PLASMA_DIAMETER_CM,
        "port": np.asarray(INTERFEROMETER_PORTS, dtype=np.int16),
        "z_cm": np.asarray(
            [z_from_port(port, port_map) for port in INTERFEROMETER_PORTS],
            dtype=np.float64,
        ),
        "n_shots": np.asarray(chord_n, dtype=np.int32),
        "run_id": np.asarray([run_ids for _ in INTERFEROMETER_PORTS]),
        "plasma_diameter_cm": PLASMA_DIAMETER_CM,
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


#: Records the filled T_e product has to carry before it can be exported.  A
#: product without them is REFUSED rather than
#: exported silently without the record, the same way the monotonic-z
#: clamp record has been required since schema v15: a consumer scoring a T_e
#: row must be able to see which cells behind it are semi-quantitative and how
#: far out the measurement was trusted, and a missing record is indistinguish-
#: able from "nothing was marked" once it is gone.
REQUIRED_TE_RECORDS = (
    "core_mean_te_monotonic_clamped",
    "core_mean_te_monotonic_scale",
    "te_trust_radius_cm",
    "te_trust_blend_cm",
    "te_semi_quantitative_core_count",
    "te_semi_quantitative_band_count",
    "te_core_window_sem_ev",
    "te_window_dln_core_control",
    "te_window_dln_core_control_source",
    "te_row_measured_cells",
    "te_row_measured_core_cells",
)


def _te_trust_records(
    te_group: h5py.Group,
    te_path: Path,
    experiment_set_id: int,
) -> dict[str, np.ndarray | str]:
    """Return the filled T_e product's trust and semi-quantitative records.

    Refuses a product that is missing any of ``REQUIRED_TE_RECORDS``, and one
    whose window-spread uncertainty was accumulated over a different core band
    than this exporter's, since the two are then not the same quantity and
    combining them would silently mix bands.
    """
    for required in REQUIRED_TE_RECORDS:
        if required not in te_group:
            raise ValueError(
                f"{te_path} ES{experiment_set_id} carries no {required} "
                "record; rebuild it with scripts/fit_te_spatial.py"
            )
    window_band = (
        float(te_group.attrs["te_core_window_sem_x_min_cm"]),
        float(te_group.attrs["te_core_window_sem_x_max_cm"]),
    )
    if window_band != (X_MIN_CM, X_MAX_CM):
        raise ValueError(
            f"{te_path} ES{experiment_set_id} accumulated its window-spread "
            f"uncertainty over {window_band} cm, not this exporter's core band "
            f"({X_MIN_CM:g}, {X_MAX_CM:g}) cm; the two must agree or the "
            "exported te_sem_ev mixes bands"
        )
    return {
        "clamped": np.asarray(
            te_group["core_mean_te_monotonic_clamped"][()], dtype=np.bool_
        ),
        "clamp_scale": np.asarray(
            te_group["core_mean_te_monotonic_scale"][()], dtype=np.float64
        ),
        "trust_radius_cm": np.asarray(
            te_group["te_trust_radius_cm"][()], dtype=np.float64
        ),
        "trust_blend_cm": np.asarray(
            te_group["te_trust_blend_cm"][()], dtype=np.float64
        ),
        "semi_quant_core_count": np.asarray(
            te_group["te_semi_quantitative_core_count"][()], dtype=np.int16
        ),
        "semi_quant_band_count": np.asarray(
            te_group["te_semi_quantitative_band_count"][()], dtype=np.int16
        ),
        "window_sem_ev": np.asarray(
            te_group["te_core_window_sem_ev"][()], dtype=np.float64
        ),
        "core_control_dln": np.asarray(
            te_group["te_window_dln_core_control"][()], dtype=np.float64
        ),
        "core_control_source": np.asarray(
            te_group["te_window_dln_core_control_source"][()], dtype=np.int8
        ),
        "core_control_source_codes": str(
            te_group.attrs["te_window_core_control_source_codes"]
        ),
        "row_measured_cells": np.asarray(
            te_group["te_row_measured_cells"][()], dtype=np.int16
        ),
        "row_measured_core_cells": np.asarray(
            te_group["te_row_measured_core_cells"][()], dtype=np.int16
        ),
        "row_provenance_definition": str(
            te_group.attrs["te_row_provenance_definition"]
        ),
        "qc_floor_rule": str(te_group.attrs["te_qc_floor_rule"]),
        "trust_model": str(te_group.attrs["te_trust_model"]),
        "semi_quant_rule": str(te_group.attrs["te_semi_quantitative_rule"]),
        "window_sem_definition": str(
            te_group.attrs["te_core_window_sem_definition"]
        ),
    }


def _plateau_current_a(current: np.ndarray, time_ms: np.ndarray) -> np.ndarray:
    """Per-shot plateau current, in A: the mean over the scoring plateau window.

    The window is ``RAW_PLATEAU_WINDOW_MS``, inherited from the scoring
    convention rather than chosen here.  What this returns is an AMPLITUDE, not
    a timing quantity: it is what the shots differ in when one of them runs a
    hotter discharge than the rest.
    """
    window = (
        (time_ms >= RAW_PLATEAU_WINDOW_MS[0])
        & (time_ms <= RAW_PLATEAU_WINDOW_MS[1])
    )
    return np.mean(current[:, window], axis=1)


def _t_half_level_ms(
    current: np.ndarray,
    time_ms: np.ndarray,
    level_a: float,
) -> np.ndarray:
    """Per-shot time of the first upward crossing of ``level_a``, in ms.

    ``current`` is the ``(n_traces, n_samples)`` ensemble on the ``time_ms``
    grid and ``level_a`` is ONE current common to every shot, so the returned
    times differ only in WHEN each shot got there.  Referring each shot to its
    own peak instead would make the level a function of that shot's plateau
    amplitude, and a shot running a higher plateau would then be reported as
    breaking down late purely because its threshold was higher; with a common
    level the statistic is amplitude-independent by construction.

    The crossing is linearly interpolated between the two samples bracketing
    it, which is what makes the estimate finer than the sample pitch; a shot
    already above the level at the first sample is placed at the first sample.
    Raises ``ValueError`` if any shot never reaches the level, since that shot
    has no crossing to report and must not be silently placed at the record
    start.
    """
    reached = current >= level_a
    if not reached.any(axis=1).all():
        missing = np.flatnonzero(~reached.any(axis=1))
        raise ValueError(
            f"discharge shots {missing.tolist()} never reach the common "
            f"crossing level {level_a:g} A and have no crossing time"
        )
    rows = np.arange(current.shape[0])
    idx = np.argmax(reached, axis=1)
    prev = np.maximum(idx - 1, 0)
    y0 = current[rows, prev]
    rise = current[rows, idx] - y0
    frac = np.where(
        rise > 0.0, (level_a - y0) / np.where(rise > 0.0, rise, 1.0), 0.0
    )
    return time_ms[prev] + frac * (time_ms[idx] - time_ms[prev])


def _discharge_stats(
    dataset: LapdDataset,
    experiment_set_id: int,
    raw_ensemble: bool = False,
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

    With ``raw_ensemble`` the same two measures are additionally taken on the
    UNSMOOTHED shots, together with each shot's crossing time through one
    common current level -- half the ensemble MEDIAN plateau -- and returned
    under the ``"raw"`` key.  Nothing else in the result changes.
    """
    currents = []
    currents_raw = []
    raw_run_ids = []
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
        if raw_ensemble:
            currents_raw.append(current)
            raw_run_ids.extend([run_id] * current.shape[0])
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

    raw: dict[str, np.ndarray] = {}
    if raw_ensemble:
        current_raw_all = np.concatenate(currents_raw, axis=0)
        current_raw_std = np.std(current_raw_all, axis=0, ddof=1)
        raw_time_ms = reference_time_s * 1000.0
        plateau_a = _plateau_current_a(current_raw_all, raw_time_ms)
        t_half_level_a = 0.5 * float(np.median(plateau_a))
        t_half_level_ms = _t_half_level_ms(
            current_raw_all, raw_time_ms, t_half_level_a
        )
        raw = {
            "current_mean_a": np.mean(current_raw_all, axis=0),
            "current_sd_a": current_raw_std,
            "current_sem_a": current_raw_std / np.sqrt(n_traces),
            "t_half_level_a": np.asarray(t_half_level_a),
            "t_half_level_ms": t_half_level_ms,
            "t_half_level_run_id": np.asarray(raw_run_ids),
            "t_half_level_mean_ms": np.asarray(np.mean(t_half_level_ms)),
            "t_half_level_sd_ms": np.asarray(np.std(t_half_level_ms, ddof=1)),
        }

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
        "raw": raw,
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
    raw_discharge_ensemble: bool = False,
    port_map: str = PORT_MAP_DEFAULT,
) -> Path:
    if port_map != PORT_MAP_DEFAULT:
        raise ValueError(PORT_MAP_REFUSAL.format(port_map=port_map))
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
        te_group = te_hdf[f"experiment_sets/{experiment_set_key}"]
        te_records = _te_trust_records(te_group, te_path, experiment_set_id)
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
    isat_decay_dn = _isat_decay_stats(
        dataset,
        rot0_isat_profile_path,
        zero_offsets_path,
        experiment_set_id,
    )
    if not np.array_equal(isat_decay_dn["port"], PORTS):
        raise ValueError(
            f"ES{experiment_set_id} downstream Isat decay ports "
            f"{isat_decay_dn['port']} do not match expected {PORTS}"
        )
    isat_decay_geomean = _isat_decay_geomean(
        isat_decay,
        isat_decay_dn,
        face_areas_cm2,
    )
    interf_decay = _interferometer_decay_stats(
        dataset,
        experiment_set_id,
        port_map,
    )
    discharge = _discharge_stats(
        dataset,
        experiment_set_id,
        raw_ensemble=raw_discharge_ensemble,
    )
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
    discharge_raw_fields: dict[str, np.ndarray] = {}
    if raw_discharge_ensemble:
        raw = discharge["raw"]
        discharge_raw_fields = {
            "discharge_current_raw_mean_a": raw["current_mean_a"],
            "discharge_current_raw_sd_a": raw["current_sd_a"],
            "discharge_current_raw_sem_a": raw["current_sem_a"],
            "discharge_raw_t_half_level_a": raw["t_half_level_a"],
            "discharge_raw_t_half_level_ms": raw["t_half_level_ms"],
            "discharge_raw_t_half_level_run_id": raw["t_half_level_run_id"],
            "discharge_raw_t_half_level_mean_ms": raw["t_half_level_mean_ms"],
            "discharge_raw_t_half_level_sd_ms": raw["t_half_level_sd_ms"],
            "discharge_raw_ensemble_definition": np.array(
                "ADDITIVE, opt-in family: the SAME shots and the SAME "
                "trigger-referenced grid as discharge_time_ms and the "
                "discharge_current_* fields, pooled WITHOUT the per-shot "
                "moving average those apply first.  discharge_current_raw_sd_a "
                "is the ddof=1 shot envelope of the unsmoothed shots and "
                "discharge_current_raw_sem_a that divided by "
                "sqrt(discharge_n_traces); the smoothed family's sd is the "
                "same statistic after each shot has been averaged over "
                "discharge_smoothing_samples samples, which rounds every "
                "shot's own breakdown knee and therefore reads LOW where the "
                "trace is steep.  discharge_raw_t_half_level_ms is one time "
                "per shot: the first upward crossing of discharge_raw_t_half_"
                "level_a, linearly interpolated between the bracketing "
                "samples, with discharge_raw_t_half_level_run_id naming the "
                "run each shot came from.  That level is ONE current common "
                "to every shot -- half the MEDIAN of the per-shot plateau "
                "means over the SCORING PLATEAU WINDOW, 15.0-19.5 ms, the "
                "window the transport comparison already scores the drive "
                "plateau over -- so the times are amplitude-independent "
                "by construction and differ only in WHEN each shot got there; "
                "their mean and ddof=1 sd are the breakdown-timing jitter of "
                "the ensemble, measured rather than inferred from the width "
                "of a current spread.  A per-shot level referred to each "
                "shot's own peak would NOT be that statistic: at ES1 run 05 "
                "runs a plateau about 60 % above the rest of the pack, an "
                "amplitude outlier, and its own-peak threshold would place "
                "its crossings late and inflate the exported sd by nearly an "
                "order of magnitude.  Nothing in the smoothed family is "
                "changed or superseded by these fields."
            ),
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema_version=np.array(23, dtype=np.int16),
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
        te_sem_ev=np.hypot(
            te.sem, np.nan_to_num(te_records["window_sem_ev"], nan=0.0)
        ),
        te_radial_sem_ev=te.sem,
        te_window_sem_ev=te_records["window_sem_ev"],
        te_sem_definition=np.array(
            "te_sem_ev is the radial scatter of the core-band cells "
            "(te_radial_sem_ev) and the fit-window convention term "
            "(te_window_sem_ev) added in quadrature.  The second is non-zero "
            "only where a core cell is semi-quantitative for its measured "
            "window spread; see te_window_sem_definition and "
            "te_semi_quantitative_rule.  Schema v15 and earlier carried the "
            "radial term alone under the name te_sem_ev.  Schema v17 read the "
            "x = 0 core control from the band product only, so the two ES3 "
            "ports whose control fails were silent there; v19 reads it from "
            "the union of both window-refit products, and te_core_control_dln "
            "and te_core_control_source say which product spoke per port."
        ),
        te_window_sem_definition=np.array(te_records["window_sem_definition"]),
        te_trust_radius_cm=te_records["trust_radius_cm"],
        te_trust_blend_cm=te_records["trust_blend_cm"],
        te_trust_model=np.array(te_records["trust_model"]),
        te_semi_quantitative_core_count=te_records["semi_quant_core_count"],
        te_semi_quantitative_band_count=te_records["semi_quant_band_count"],
        te_semi_quantitative_rule=np.array(te_records["semi_quant_rule"]),
        te_core_control_dln=te_records["core_control_dln"],
        te_core_control_source=te_records["core_control_source"],
        te_core_control_source_codes=np.array(
            te_records["core_control_source_codes"]
        ),
        te_row_measured=(te_records["row_measured_cells"] > 0),
        te_row_measured_cells=te_records["row_measured_cells"],
        te_row_measured_core_cells=te_records["row_measured_core_cells"],
        te_row_provenance_definition=np.array(
            te_records["row_provenance_definition"]
            + "  te_row_measured[port, sample] is the boolean form: False "
            "means the T_e row at that port and sample is PRIOR-DERIVED and "
            "is not a measurement of that port.  Seven port-rows in this "
            "dataset are prior-derived at every sample -- ES3 p21/p41/p50 and "
            "ES4 p21/p29/p41/p50 -- and anything computed from their T_e, "
            "including the density and the flux-tube average, inherits that."
        ),
        te_qc_floor_rule=np.array(te_records["qc_floor_rule"]),
        te_semi_quantitative_definition=np.array(
            "te_semi_quantitative_core_count[port, sample] counts the cells "
            f"inside the core band ({X_MIN_CM:g} to {X_MAX_CM:g} cm) behind "
            "that te_mean_ev which are semi-quantitative; a non-zero count "
            "means the T_e row is semi-quantitative at that sample.  "
            "te_semi_quantitative_band_count is the same count over the "
            "measurement band outside the core, the cells that reach the "
            "density chain through C_s but never enter te_mean_ev.  A marked "
            "cell keeps its measured value in the filled product."
        ),
        te_core_count=te.count,
        te_window_spread_frac=te_window_spread,
        te_core_mean_clamped=te_records["clamped"],
        te_core_mean_clamp_scale=te_records["clamp_scale"],
        te_core_mean_clamp_definition=np.array(
            "te_core_mean_clamped[port, sample] is True where the filled T_e "
            "product's core-mean monotonic-z clamp rescaled that port's radial "
            "profile so its core mean did not exceed the coolest core mean "
            "upstream of it, and te_core_mean_clamp_scale is the factor it "
            "applied about the 0.1 eV boundary temperature (1.0 where it did "
            "not act).  A clamped sample's te_mean_ev is by construction the "
            "upstream row's value rather than its own, so the two ports are "
            "then one number and not two measurements.  The clamp only rewrites "
            "a port that carries measured core cells where the MEASURED core "
            "means are themselves non-monotonic; a port with no measured core "
            "cells is reconstructed from its neighbours and the end boundaries, "
            "and the prior is its only axial constraint."
        ),
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
        isat_decay_dn_mean_a=isat_decay_dn["mean_a"],
        isat_decay_dn_sem_a=isat_decay_dn["sem_a"],
        isat_decay_dn_n_shots_used=isat_decay_dn["n_shots_used"],
        isat_decay_dn_n_shots_rejected=isat_decay_dn["n_shots_rejected"],
        isat_decay_dn_zero_offset_v=isat_decay_dn["zero_offset_v"],
        isat_decay_dn_source_file=np.array(str(rot0_isat_profile_path)),
        isat_decay_dn_source_channel=isat_decay_dn["source_channel"],
        isat_decay_dn_source_inverted=isat_decay_dn["source_inverted"],
        isat_decay_geomean_a_per_cm2=isat_decay_geomean["geomean_a_per_cm2"],
        isat_decay_geomean_sem_a_per_cm2=isat_decay_geomean["sem_a_per_cm2"],
        isat_decay_geomean_area_cm2=isat_decay_geomean["area_cm2"],
        isat_decay_geomean_pairing=isat_decay_geomean["pairing"],
        isat_decay_face_convention=np.array(
            "BOTH Mach-probe faces at x = 0, on the shared isat_decay_time_ms "
            "grid, at the same ports and the same runs (isat_decay_port, "
            "isat_decay_run_id apply to both).  isat_decay_mean_a / "
            "isat_decay_sem_a are the UPSTREAM face at rot-0 -- the 'i_sweep' "
            "channel, ruled the Isat truth channel 2026-08-18 -- and "
            "isat_decay_dn_mean_a / isat_decay_dn_sem_a are the DOWNSTREAM "
            "face, the 'isat' channel, which collects in the probe body's flow "
            "shadow and under-reads.  Both are raw currents in A with no "
            "Probe-A effective-area factor applied, exactly as before.  Each "
            "face's channel and polarity are read from its own product's run "
            "attrs and exported as isat_decay_source_channel / "
            "isat_decay_dn_source_channel and the matching _source_inverted "
            "arrays; the source products are isat_decay_dn_source_file and the "
            "upstream default of --isat-profiles.  The two faces run the "
            "IDENTICAL pipeline (padded read window, per-shot digitizer "
            "Scale/Offset, cached zero-offset subtraction, channel "
            "calibration, polarity, 100 kHz zero-phase low-pass, 10 us mean "
            "bins) but each builds its OWN fixed shot ensemble from its own "
            "channel's inter-sweep outlier flags, so THE TWO FACES REJECT "
            "DIFFERENT SHOTS: the sizes are isat_decay_n_shots_used / "
            "isat_decay_n_shots_rejected for the upstream face and "
            "isat_decay_dn_n_shots_used / isat_decay_dn_n_shots_rejected for "
            "the downstream one, and they are not equal port by port.  "
            "isat_decay_geomean_a_per_cm2 is sqrt(J_up * J_dn) of the two "
            "faces' AREA-NORMALIZED currents, in A cm^-2: the per-run "
            "(upstream, downstream) face areas actually used are "
            "isat_decay_geomean_area_cm2 and the channel/area pairing is "
            "isat_decay_geomean_pairing, so sqrt(A_up * A_dn) x "
            "isat_decay_geomean_a_per_cm2 recovers a current in A.  In the "
            "Chung two-sided model the faces carry reciprocal flow "
            "factors exp(+K M / 2) and exp(-K M / 2), which cancel in the "
            "geometric mean to first order in M, so the geomean is the "
            "flow-artifact-cancelled central estimator while each single face "
            "keeps that artifact with the opposite sign; this is the same "
            "construction as isat_ftavg_geomean_* (see "
            "isat_ftavg_geomean_definition and ftavg_face_ruling).  "
            "isat_decay_geomean_sem_a_per_cm2 uses that family's "
            "propagation, "
            "geomean x 0.5 x hypot(sem_up / I_up, sem_dn / I_dn).  A sample is "
            "NaN wherever either face is non-finite or non-positive -- the p50 "
            "upstream trace decays through zero late in the afterglow, so the "
            "p50 geomean is NaN from there on.  That is the data and is NOT "
            "clipped."
        ),
        interf_decay_time_ms=interf_decay["time_ms"],
        interf_decay_line_density_cm2=interf_decay["line_density_cm2"],
        interf_decay_sem_cm2=interf_decay["sem_cm2"],
        interf_decay_port=interf_decay["port"],
        interf_decay_z_cm=interf_decay["z_cm"],
        interf_decay_n_shots=interf_decay["n_shots"],
        interf_decay_run_ids=interf_decay["run_id"],
        interf_decay_plasma_diameter_cm=np.array(
            interf_decay["plasma_diameter_cm"]
        ),
        interf_decay_convention=np.array(
            "Shot-averaged interferometer LINE-INTEGRATED density, shaped "
            "(chord, time).  Units cm^-2 = interf_decay_plasma_diameter_cm "
            "(40 cm) x the line-average density in cm^-3, the calibration's "
            "own convention.  Per chord the stored phase is multiplied by that "
            "chord's 'calibration factor (m^-3/rad)' attribute, shots carrying "
            "rigol_missing are skipped, and the shot statistics are the repo's "
            "Welford accumulation over EVERY stored shot of ALL the runs in "
            "interf_decay_run_ids, pooled into one accumulator per chord -- "
            "this is not a mean of per-run means.  interf_decay_n_shots is "
            "that pooled count and interf_decay_sem_cm2 is the pooled sample "
            "standard deviation divided by sqrt(interf_decay_n_shots), so it "
            "carries the shot-to-shot machine jitter and no per-shot time "
            "alignment is applied.  Alignment is the repo's own: each shot is "
            "np.interp-ed onto its chord's reference grid, the first stored "
            "time array of the set's first run.  p20 and p29 sit on the LeCroy "
            "'time_array' and p40 on the Rigol 'time_array_p40', so the p40 "
            "chord mean and SEM are interpolated once more onto the p20 grid "
            "and all three chords share interf_decay_time_ms; p40's own grid "
            "strictly contains that shared grid, so nothing is extrapolated.  "
            "interf_decay_time_ms is the RAW interferometer clock, uncorrected "
            "-- see interf_decay_clock_offset.  MSI/'Interferometer array' is "
            "malformed in these files and is never read."
        ),
        interf_decay_chord_caveat=np.array(
            "A chord is a LINE INTEGRAL across the whole column at its port, "
            "not a core quantity: it weights the edge and the scrape-off layer "
            "the same as the axis, and the 40 cm plasma diameter behind the "
            "cm^-2 normalization is the calibration's fixed assumption, not a "
            "per-shot measured width.  It is therefore NOT commensurate with "
            "the core-band density_mean_cm3 or with the flux-tube "
            "density_ftavg_cm3, and it must not be compared to either as a "
            "level without stating that.  The chords also sit at ports "
            "20/29/40 (interf_decay_z_cm), which are not the probe ports "
            "11/21/29/41/50 -- only p29 coincides.  Use it as a decay SHAPE "
            "and as an independent line-integrated magnitude."
        ),
        interf_decay_clock_offset=np.array(
            "UNCORRECTED clock offset between the two families.  MSI/Discharge "
            "and every isat_decay_* / discharge_* field sit on the SIS trigger "
            "grid; the interferometer sits on the LeCroy (p20/p29) and Rigol "
            "(p40) digitizers.  Measured over the eight ES1 runs by comparing "
            "the 10 % rise of the discharge current with the 10 % rise of the "
            "p29 line density, the interferometer clock runs LATE by +0.26 to "
            "+0.46 ms (run-by-run range; median about +0.42 ms).  NO alignment "
            "is applied: interf_decay_time_ms is the raw interferometer clock "
            "and isat_decay_time_ms is the raw SIS clock.  A consumer putting "
            "the two on one axis must carry this as a systematic."
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
        port_map=np.array(port_map),
        **discharge_raw_fields,
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
    parser.add_argument(
        "--raw-discharge-ensemble",
        action="store_true",
        help="additionally export the unsmoothed per-shot discharge-current "
             "statistics and each shot's half-peak crossing time; off by "
             "default, and off leaves every exported field unchanged",
    )
    parser.add_argument(
        "--port-map",
        choices=PORT_MAPS,
        default=PORT_MAP_DEFAULT,
        help="axial port ladder to export on.  Anything but the default is "
             "REFUSED, with the rebuild the adoption needs spelled out: the "
             "probe-port z grid is copied from the upstream products, so "
             "exporting on a ladder they were not built under would write a "
             "mixed-ladder product.  The ladder actually used is stamped into "
             "the product as port_map",
    )
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
        args.raw_discharge_ensemble,
        args.port_map,
    )


if __name__ == "__main__":
    main()
