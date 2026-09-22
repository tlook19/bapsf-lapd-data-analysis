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
* the same afterglow decay for the same three faces under all three radial
  conventions -- core band, flux tube and whole column -- and the e-fold
  matrix fitted over the scored decay window across the three faces and the
  four conventions, the x=0 point included (see
  ``isat_decay_radial_definition`` and ``isat_decay_matrix_definition``);
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
differ only in when each shot got there and not in what the trace did later.
ES1 needs that: both run-05 shots carry a narrow current spike at the same
sample, t = 20.000 ms, reaching 4.95 and 4.07 kA against a pack peak median of
3.04 kA, while their plateau is normal -- within 2 % of the pack median over
the scoring window, which ends 0.5 ms before the spike.  A level set from each
shot's OWN peak reads that spike as the shot's amplitude and puts run 05's
threshold near the top of its rise, which walked its crossings out and inflated
the ensemble sd nearly tenfold; the common level clears it.  No existing field
changes, and the family is ABSENT unless the flag is passed, so a product that
lacks these names was exported without it rather than with it and zeroed.

Three radial-averaging conventions
----------------------------------
The overlay carries the measured radial average in ALL THREE conventions,
because the comparison convention is otherwise implicit and is the same order
as the residuals it is used to judge:

``density_mean_cm3`` (and its SEM fields) is the LEGACY convention: an
unweighted arithmetic mean of the 51-point line scan over the core band
``X_MIN_CM <= x <= X_MAX_CM``.  It is a line cut through the column, so it
carries no radial area weighting at all.  Nothing about it changes here.

``density_ftavg_cm3``, ``te_ftavg_ev``, ``isat_ftavg_upstream_a`` and
``isat_ftavg_a`` are the FLUX-TUBE convention:
``int_0^R 2 pi r f(r) dr / (pi R^2)`` with ``R = FLUX_TUBE_RADIUS_CM``, the
measured cathode frame opening.  This is the quantity a 1D transport model
that carries one radial cell of radius R reports, so it is the convention in
which a model cell and a measurement are the same quantity.  Getting there
from a diameter line scan requires assuming the column is axisymmetric about
its own centroid, which is an ASSUMPTION and not a measurement -- see
``ftavg_axisymmetry`` in the exported product.

``density_column_cm3``, ``te_column_ev`` and the three ``isat_column_*`` face
rows are the WHOLE-COLUMN convention:
the same quadrature taken from the density centroid out to the COLUMN EDGE --
which is the SCAN LIMIT in the folded frame, not a measured column boundary --
and then, for the density row, divided by ``pi * ftavg_radius_cm^2`` rather
than by the area it was integrated over.  It is therefore the column's
INVENTORY PER UNIT LENGTH expressed as the density the transport model's tube
would carry if all of that plasma were inside it.  Plasma measured outside the
cathode flux tube got there by cross-field transport the 1D model does not
represent, so a model tube that carries the column's whole inventory is the
comparison this convention makes; ``column_over_ftavg_ratio`` is the share of
the column that sits outside the tube, per port and per sample.

NO BACKGROUND IS SUBTRACTED FROM ANY COMPARAND IN THIS PRODUCT.  The Isat
baseline is already taken from the end of the shot, where there is no plasma,
so a line scan carries no second background to remove and the edge-median
subtraction this chain used to apply was removing the cross-field plasma the
column convention counts.  ``density_ftavg_cm3``, the density weight behind
``te_ftavg_ev`` and the ES4 upstream bracket moved at schema v33; the three
``isat_ftavg_*`` families at v35.  Every area-averaged row is therefore on one
convention and they are commensurate with each other.  The retired values
survive under five ``*_subtracted_*`` names, for continuity reads only -- see
``ftavg_background`` and ``subtracted_legacy_definition``.

The three conventions are exported side by side and are NOT interchangeable; a
consumer must state which one a number came from.  ``ftavg_comparand_map``
names the field that carries each scored row in each of them.

T_e is the one row where the conventions differ by more than a weighting.
``te_ftavg_ev`` is the DENSITY-WEIGHTED flux-tube mean, which is what a 1D
cell carries (its ``E_e / (3/2 n)``); ``te_ftavg_plain_ev`` is the unweighted
area mean of the same profile and is printed beside it, not scored.  Both
integrate over a disc the measurement does not always fill: the filled T_e
product reports the MEASUREMENT only out to a per-port trust radius and the
scrape-off-layer prior beyond it, so every T_e flux-tube sample carries
``ftavg_coverage_cm`` and ``ftavg_prior_beyond_coverage`` saying how far the
measurement actually reached.  ``te_column_ev`` is the same density-weighted
mean over the column extent, with ``te_column_plain_ev`` beside it and
``column_coverage_cm`` / ``column_prior_beyond_coverage`` as its own coverage
record -- and that flag is True at EVERY port of every set, because no port's
T_e is trusted all the way to the column edge.  HOW MUCH of each column T_e is
the prior rather than a measurement is ``te_column_prior_weight``: 20-27 % at
the eight ES1/ES2 aperture ports and 42-74 % at every p50 row and across sets
3 and 4, so eleven of the twenty port-rows are substantially a statement about
the prior -- and the twentieth, ES4 p50, carries no column row at all.  That
share is a ratio of two SIGNED sums and is not confined to ``[0, 1]``; it is
inside it at every sample of the scored plateau window, and leaves it only at
early samples.  See ``te_ftavg_definition``,
``te_ftavg_sem_definition``, ``ftavg_coverage_definition``,
``column_definition``, ``column_edge_definition``, ``column_sem_definition``
and ``column_coverage_definition``.

Which probe face the Isat flux-tube average comes from
------------------------------------------------------
The Mach probe has two planar faces on opposite sides of the body.  At rot-0
the ``i_sweep`` channel collects on the UPSTREAM face and the ``isat`` channel
on the DOWNSTREAM one; at rot-180 the assignment reverses.  A downstream face
sits in the probe body's own flow shadow, so it under-reads, and the upstream
face is the Isat truth channel.

``isat_ftavg_upstream_a`` is therefore the correction-bearing field: it comes
from the ``i_sweep`` product, which is also the chain behind ``n_e_m3`` and
behind ``isat_decay_*`` / ``isat_drive_*``, so it pairs with the density and
core-band fields without a face change anywhere.

``isat_ftavg_a`` is the downstream ``isat`` face, kept because it is the
effective-width ledger's rot-0 primary and the face the paper read
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

from bapsf_lapd import ChannelKind, LapdDataset, density_area_key_for_deadtime_source
from bapsf_lapd.annotations import refuse_artifacts_in_window
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
from es4_upstream_rows_rot180_isat import (
    CORE_CM as ES4_UPSTREAM_CORE_CM,
    PORT_RUNS as ES4_UPSTREAM_PORT_RUNS,
    PROBE_FROM_DIGIT as ES4_UPSTREAM_PROBE_FROM_DIGIT,
    area_key_for_electrode as es4_upstream_area_key,
    build_port as es4_upstream_build_port,
    load_areas_cm2 as es4_upstream_load_areas_cm2,
)


#: Schema version stamped into every exported product as ``schema_version``.
#: ODD is an exporter version and EVEN an augmented one
#: (``augment_sim1d_overlay_isat_drive.AUGMENTED_SCHEMA`` maps each exporter
#: version to the augmented version it becomes), so this is bumped by TWO
#: whenever the exported field set changes.  It is a named constant rather
#: than a literal in the ``savez`` call so that a consumer or a test can pin
#: the CURRENT schema by importing it: a literal pinned in one test is a pin
#: on whatever vintage happened to be on disk the day it was written, and
#: goes stale silently the next time the product is placed.
SCHEMA_VERSION = 43

MANIFEST = Path("config/may2026_run_manifest.toml")
DENSITY_HDF5 = Path("processed/density_profiles_isweep.hdf5")
TE_HDF5 = Path("processed/te_filled.hdf5")
#: Dead-time line-scan profiles from the ``i_sweep`` channel: the UPSTREAM
#: probe face at rot-0, and the Isat truth channel.  Feeds both
#: the x=0 decay trace and the ``isat_ftavg_upstream_*`` flux-tube family.
ISAT_PROFILE_HDF5 = Path("processed/isweep_deadtime_profiles.hdf5")
#: The same scans from the ``isat`` channel: the DOWNSTREAM face at rot-0,
#: which reads low because it sits in the probe body's flow shadow.  Feeds the
#: ``isat_ftavg_*`` family only.
ROT0_ISAT_PROFILE_HDF5 = Path("processed/isat_profiles.hdf5")
#: The rot-180 dead-time line scans from the ``isat`` channel: at rot-180 that
#: channel collects on the UPSTREAM probe face, so this product is the second,
#: independent measurement of the same upstream density the ``i_sweep`` chain
#: measures at rot-0.  Read by the ES4 upstream bracket only.
ROT180_ISAT_PROFILE_HDF5 = Path("processed/isat_rot180_deadtime_profiles.hdf5")
#: Per-probe collecting-electrode areas, keyed ``ap_L_cm2`` (the I_SWEEP
#: electrode) and ``ap_R_cm2`` (the ISAT electrode).  Read by the ES4 upstream
#: bracket only; every other family gets its areas from its input product.
AREA_CALIBRATION_TOML = Path("processed/probe_area_calibration.toml")
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

#: Fit window for the afterglow e-fold matrix, in ms on the trigger-referenced
#: grid ``isat_decay_time_ms`` is already on.  It is the transport
#: comparison's stage (iii) decay window, inherited from that convention
#: rather than chosen here: the discharge ends at 20 ms, so this is the first
#: 1.5 ms OF THE AFTERGLOW, the early decay both the model and these traces
#: resolve.  Two windows are not comparable, so this must stay the same
#: interval as the window the transport side fits over; it is exported with
#: the matrix as ``isat_decay_matrix_window_ms`` so a consumer can check
#: rather than assume.
ISAT_DECAY_FIT_WINDOW_MS = (20.0, 21.5)

#: Length of the trace tail the e-fold fit's noise floor is estimated over, in
#: ms, and the multiple of the tail's robust sigma that floor is set at.
ISAT_DECAY_NOISE_TAIL_MS = 5.0
ISAT_DECAY_NOISE_SIGMAS = 5.0

#: The two axes of the afterglow e-fold matrix, in the order they are stored.
#: ``isat_decay_matrix_tau_ms`` is indexed ``[face, convention, port]``.  The
#: conventions run outward: the x = 0 POINT, then the three radial averages
#: the plateau rows are already exported in, so every plateau convention has a
#: decay counterpart.
ISAT_DECAY_MATRIX_FACES = ("upstream", "downstream", "geomean")
ISAT_DECAY_MATRIX_CONVENTIONS = ("x0", "core", "ftavg", "column")

#: Runs whose LATE-AFTERGLOW Isat decay trace (the ``isat_decay_*`` /
#: ``isat_decay_dn_*`` family ``_isat_decay_stats`` builds from
#: ``decay_start_s`` onward) carries a probe-local current the plasma's own
#: light does not see, registered per ``(run_id, source channel)``.
#:
#: Run 22 (experiment set 2, port 21, rot-0): the core ISAT-channel decay
#: fits tau = 22.66 ms against 7.18 ms on the run's OWN reference photodiode
#: and 6.61-9.14 ms on every neighbouring set-2 run's ISAT channel at x=0;
#: 0.9 mA is still undecayed at 47.8 ms, the record's end.  The run's
#: I_SWEEP channel (tau 12.52 ms, unremarkable next to run 21's 10.32 ms)
#: and its rot-180 partner run 23 (isat tau 9.14 ms) are both normal, so the
#: finding is specific to this run's own ISAT channel and not a set-wide or
#: a discharge effect.
#:
#: DISCLOSED, never corrected: a registered run's afterglow trace is
#: NaN-filled from ``decay_start_s`` onward rather than dropped, so a
#: consumer sees an explicit gap (the ``*_excluded`` / ``*_excluded_reason``
#: arrays) and the run stays in the port roster the PORTS-equality checks
#: below require.  Nothing BEFORE ``decay_start_s`` moves: the finding is
#: afterglow-only, and every drive-phase product (``isat_ftavg_*``, the
#: 15.0-19.5 ms plateau window, ``processed/isat_profiles.hdf5`` itself) is
#: built from an entirely separate code path this registry does not touch.
LATE_AFTERGLOW_PROBE_LOCAL_CURRENT: dict[tuple[str, str], dict[str, object]] = {
    ("22", "isat"): {
        "reason": "probe_local_current_suspected",
        "tau_ms": 22.657038696464106,
        "photodiode_tau_ms": 7.1767975767679495,
        "neighbor_tau_range_ms": (6.614529358475285, 9.136996719905085),
        "undecayed_current_ma": 0.9,
        "undecayed_time_ms": 47.8,
        "source": (
            "core ISAT-channel decay time constant 22.66 ms at x=0, against "
            "7.18 ms on this run's own reference-photodiode decay and "
            "6.61-9.14 ms on every neighbouring experiment-set-2 run's ISAT "
            "channel at x=0; this run's I_SWEEP channel (tau 12.52 ms) and "
            "its rot-180 partner run 23's ISAT channel (tau 9.14 ms) are "
            "both unremarkable"
        ),
    },
}

#: Radius of the flux tube the measured profiles are averaged over, in cm.
#: Direct caliper reading of the LAPD cathode assembly (2026-08-17): the frame
#: opening is a 14.5 in aperture, 36.830 cm diameter, in a 15.0 in x 0.25 in
#: disc whose outer radius is 19.050 cm.  Half the aperture diameter is
#: 18.415 cm.  This is hardware, not a fit, and it is the radius the transport
#: model's single radial cell uses, which is why the measured target has to use
#: it too for the two to be the same quantity.
FLUX_TUBE_RADIUS_CM = 18.415

# --------------------------------------------------------------------------
# The ES4 upstream-face bracket (experiment set 4 only)
# --------------------------------------------------------------------------
#
# At every experiment-set-4 port the probe was run twice, once at each
# rotation, so the SAME physical upstream density has two independent
# measurements: the ``i_sweep`` electrode at rot-0 and the ``isat`` electrode
# at rot-180.  The density chain behind density_mean_cm3 / density_ftavg_cm3
# reads the rot-0 ``i_sweep`` face at every set and is unchanged here.  This
# family ADDS the rot-180 ``isat`` face beside it as a labelled two-face
# bracket, so a consumer can see the face spread instead of inheriting one
# face's convention silently.  Nothing in this family replaces an existing
# field, and the family is ABSENT from every set but 4.
#
# The arithmetic is NOT re-implemented here: the rows come from
# ``scripts/es4_upstream_rows_rot180_isat.py``'s build_port, the single
# implementation, called with this exporter's own product paths and its own
# filled-T_e row.

#: The only experiment set that carries the bracket.
ES4_UPSTREAM_SET_ID = 4

#: The ports whose rot-180 partner survives that product's own exclusion
#: masks.  p21's partner (run 43) carries a registered channel-state shot range
#: that removes 20 of the 21 core positions, leaving one admitted core position
#: (x = +10.0 cm, 4 of 84 core cells, ratio ISAT180/isweep0 = 1.017) -- too few
#: to form the core-band mean or the chord line integral, so no rot-180 row is
#: built there; p50 (run 48) is rot-0 only and has no partner.
ES4_UPSTREAM_BRACKET_PORTS = (29, 41)

#: The port whose density row is re-derived at a MEASURED plateau T_e.
ES4_UPSTREAM_TE_REDERIVED_PORT = 21

#: The measured ES4 p21 plateau T_e, in eV, and what it is a measurement of.
#: Produced by ``scripts/es4_te_time_slope.py``: its ``core_family_med_ev``
#: column -- the core-band (|x| <= 10 cm) mean of the per-cell
#: window-family-median sweep T_e -- averaged over the cycles inside the
#: window named beside it.  The primary value is quoted at the SCORING
#: PLATEAU WINDOW, the window the transport comparison scores the drive
#: plateau over and the window the prior it replaces is quoted at.
ES4_UPSTREAM_P21_TE_MEASURED_EV = 0.6507
ES4_UPSTREAM_P21_TE_MEASURED_WINDOW_MS = RAW_PLATEAU_WINDOW_MS
ES4_UPSTREAM_P21_TE_MEASURED_FACE = "rot-0 i_sweep upstream face, run 42"
ES4_UPSTREAM_P21_TE_MEASURED_BASIS = (
    "core-band family-median sweep T_e (scripts/es4_te_time_slope.py, "
    "core_family_med_ev), cycle mean over the window"
)
#: The SAME estimator and face over the instrument's full displayed plateau,
#: 10.0-19.5 ms.  Carried because it is the value the p11->p21 pressure chain
#: on record was computed at; it is a different window, not a different
#: measurement, and is exported so that chain stays reproducible from this
#: product rather than from a transcription.
ES4_UPSTREAM_P21_TE_MEASURED_DISPLAY_EV = 0.6641
ES4_UPSTREAM_P21_TE_MEASURED_DISPLAY_WINDOW_MS = (10.0, 19.5)
#: The opposite face at the same port, same estimator, same two windows: the
#: T_e face spread, which is the uncertainty the re-derivation inherits.
ES4_UPSTREAM_P21_TE_MEASURED_ROT180_EV = 0.6431
ES4_UPSTREAM_P21_TE_MEASURED_ROT180_DISPLAY_EV = 0.6531

#: The sweep rest-bias factor F applied to the retained i_sweep rows.  The
#: placed chain applies NONE -- neither the density product nor this exporter
#: multiplies an ES4 i_sweep row by a rest-bias factor -- so the value stamped
#: on every row of this family is 1.0 and the field exists to say so
#: explicitly rather than leave it inferred.
ES4_UPSTREAM_F_APPLIED = 1.0

#: Collection area of the probe face each electrical channel sits on under the
#: NOMINAL wiring, as the attribute name carrying it.  Per ``bapsf_lapd.density``:
#: at rot-0 the left face (toward the cathode, upstream) is the Isweep channel
#: with area A_p_L, and the right face (toward the anode, downstream) is the
#: Isat channel with area A_p_R.  The pairing is keyed off each run's OWN
#: recorded channel rather than off which product file it came from, which is
#: what keeps ES3 run 31 -- whose two channels are exchanged between the
#: products -- normalized by the right areas.  Run 31's cables are also crossed
#: at the connector, so its channels sit on the OTHER electrodes than this map
#: names; ``_channel_area_attr`` applies that per run and is what every area
#: lookup and pairing string here goes through.
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

    RETIRED FROM THE COMPARAND CHAIN (see ``ftavg_background`` in the exported
    product).  The probe's own zeroing already removes the instrumental
    baseline -- it is taken from the end of the shot, where there is no plasma
    -- so there is no second background left in a line scan to remove, and
    what this function takes off the outer cells is plasma.  It survives for
    the explicitly named legacy rows ``density_ftavg_subtracted_cm3`` and
    ``density_column_subtracted_cm3``, and for the Isat flux-tube families,
    which this member did not re-cut.
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


def _column_edge_cm(folded_radius_cm: np.ndarray) -> float:
    """Return the COLUMN EDGE of one folded profile, in cm.

    The whole-column conventions integrate from the density centroid out to
    this radius: the outermost folded radius the scan still carries a retained
    cell at.  A retained cell is one that carries a measurement -- the density
    chain writes a cell as NaN only where there is no usable measurement, and
    a cell whose measured current is negative is noise about zero and is KEPT
    with its sign -- so this is the outermost radius the scan measured at.
    Because the sign test was retired, that is now essentially the scan extent
    itself, and the column edge no longer contracts toward the core when the
    outer skirt dips below zero.

    It is also where the effective-width ledger reads its own background:
    ``_subtract_background`` takes its baseline from the outermost
    ``BACKGROUND_EDGE_POINTS`` of this same scan, so a profile has reached the
    ledger's background by construction at the edge, and the integral closes on
    the scan rather than on an assumed column radius.  The two candidate edge
    definitions -- the outermost QC-surviving cell and the radius at which the
    profile reaches the background ledger -- are therefore the same radius in
    this chain, and this is it.
    """
    return float(np.max(folded_radius_cm))


def _core_band_point_sem(
    prepared: np.ndarray,
    core_band: np.ndarray,
    point_sem: np.ndarray | None,
) -> float:
    """Return the per-point SEM of one profile's CORE-BAND mean.

    The core-band companion of ``ftavg_sem``, composed the same way and under
    the same assumption: the per-point uncertainties are propagated through
    the average that produced the row and the points are treated as
    INDEPENDENT.  That average is the unweighted ``nanmean`` over the retained
    cells of ``X_MIN_CM <= x <= X_MAX_CM``, so the weights are ``1/N`` and the
    propagation is ``sqrt(sum sem_i^2) / N`` over exactly those cells.

    ``nan`` when no per-point uncertainty was given, when the band retains no
    cell, or when a retained cell carries a non-finite one -- an uncertainty
    is not invented where the profile states none.  This is a SHOT SEM and is
    not commensurate with the radial-scatter SEM the core-band density row
    quotes.
    """
    if point_sem is None:
        return float("nan")
    cells = core_band & np.isfinite(prepared)
    n_cells = int(np.count_nonzero(cells))
    if n_cells == 0:
        return float("nan")
    sem_values = np.asarray(point_sem, dtype=np.float64)[cells]
    if not np.all(np.isfinite(sem_values)):
        return float("nan")
    return float(np.sqrt(np.sum(sem_values**2)) / n_cells)


def _flux_tube_profile_stats(
    profile: np.ndarray,
    x_cm: np.ndarray,
    *,
    point_sem: np.ndarray | None = None,
    radius_cm: float | None = FLUX_TUBE_RADIUS_CM,
    normalize_radius_cm: float | None = None,
    subtract_background: bool = True,
) -> dict[str, float | int]:
    """Return the flux-tube average of one radial profile and its companions.

    The pipeline is, in order: repair single-cell spikes; OPTIONALLY subtract
    the effective-width ledger's scalar background; take the core-band mean of
    whichever profile that leaves (the numerator of the convention ratio,
    reported so the ratio is reconstructible from the product alone); drop
    non-finite cells; fold about the profile centroid, treating
    ``r = |x - x_c|`` as radius; and integrate to ``radius_cm``.

    ``subtract_background=False`` is THE COMPARAND CHAIN and is what every
    density row of record uses.  The probe's own zeroing already removes the
    instrumental baseline, from the end of the shot where there is no plasma,
    so a line scan carries no second background to remove and what the
    subtraction took off the outer cells was plasma.  With it off, the
    centroid, the quadrature nodes and the core companion are all taken from
    the UNSUBTRACTED profile, and the clip at zero -- which existed because
    subtracting a baseline can drive outer cells negative -- is not applied
    either.  THE PROFILE IS SIGNED: the density product stores the measured
    value of every cell it has one for, negative cells included, since a cell
    on noise about zero averages out only if it is kept, and dropping it
    redistributes its ``2 r dr`` weight onto the positive cells -- the same
    upward bias the subtraction had.  ``True`` keeps the historical behaviour
    for the explicitly named legacy rows and for the Isat families this member
    did not re-cut.

    ``radius_cm`` is the quadrature's outer limit.  ``None`` integrates to the
    profile's own COLUMN EDGE instead (``_column_edge_cm``), which is what the
    whole-column convention does; the limit actually used comes back as
    ``edge``.  ``normalize_radius_cm`` is the radius the area average is
    expressed over: ``None`` (the default) divides by the area integrated over,
    which is the flux-tube convention, and passing ``ftavg_radius_cm`` with
    ``radius_cm=None`` divides the column's whole inventory per unit length by
    the tube's area instead.  The returned ``ftavg`` and its two uncertainties
    carry that normalization; nothing else in the dict does.

    ``point_sem`` is an optional per-point uncertainty on the SAME samples; it
    is propagated through the quadrature weights in quadrature, treating the
    points as independent, and through the core-band mean the same way as
    ``core_sem`` (``_core_band_point_sem``).  The background's own sampling
    uncertainty is not
    propagated -- it is a small, fully correlated term.  ``scatter_sem`` is the
    other, always-available uncertainty: the scatter of the retained radial
    cells about their own weighted average (``_weighted_mean_and_sem``), the
    weighted generalization of the core-band radial SEM.

    A NON-POSITIVE COLUMN IS A MEASUREMENT AND IS REPORTED AS ONE.  Where the
    retained cells sum to zero or less the CENTROID does not exist -- an
    intensity-weighted mean position is undefined for a non-positive intensity
    -- but the scan did measure the column, and ``NaN`` would say it did not.
    The row is reported on the one fold centre that needs no intensity: the
    port's GEOMETRIC AXIS, ``x = 0``.  The same ``2 r dr`` trapezoidal
    quadrature is formed about it and ``ftavg`` carries the SIGNED AREA MEAN
    ``sum(w n) / sum(w)``, which is in the units of every other sample of that
    row, so a decayed port reads beside a port that has plasma.  ``centroid``
    stays ``NaN`` because there is none; ``edge`` is the extent the weights
    were formed over -- the scan limit under the whole-column convention, the
    tube radius under the flux-tube one; ``ftavg_sem`` and ``scatter_sem``
    stay ``NaN``.  ``normalize_radius_cm`` is NOT applied: the mean is over the
    disc it was integrated over, so a whole-column consumer multiplying by
    ``pi * ftavg_radius_cm^2`` gets a consistent inventory.  This is the SIGNED
    chain's rule only: ``subtract_background=True`` clips its values at zero,
    so a non-positive total there means the retired subtraction erased the
    profile, and those rows keep returning ``NaN`` because they exist to
    reproduce that method exactly as it behaved.

    ``core`` and ``core_sem`` are taken BEFORE the quadrature and are returned
    on every path, the early exits included, so the core-band row of a profile
    survives wherever the band retains a cell at all -- including the paths
    that come back carrying no area average.
    """
    despiked, n_despiked = _despike_profile(profile)
    prepared = _subtract_background(despiked) if subtract_background else despiked
    core_band = (x_cm >= X_MIN_CM) & (x_cm <= X_MAX_CM)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        core_mean = float(np.nanmean(prepared[core_band]))
    core_sem = _core_band_point_sem(prepared, core_band, point_sem)

    empty = dict(
        ftavg=np.nan,
        ftavg_sem=np.nan,
        scatter_sem=np.nan,
        core=core_mean,
        core_sem=core_sem,
        centroid=np.nan,
        edge=np.nan,
        n_despiked=n_despiked,
    )
    finite = np.isfinite(prepared)
    if np.count_nonzero(finite) < 5:
        return empty
    values = (
        np.clip(prepared[finite], 0.0, None)
        if subtract_background
        else prepared[finite]
    )
    positions = x_cm[finite]
    total = float(np.sum(values))
    if total <= 0.0:
        # A column whose intensity sums non-positive has NO CENTROID -- the
        # intensity-weighted mean position is not defined for it -- but it IS a
        # MEASUREMENT, and NaN would say the opposite.  The row is reported on
        # the ONE fold centre that needs no intensity: the port's GEOMETRIC
        # AXIS, x = 0.  The same 2 r dr quadrature is formed about it and the
        # row carries the SIGNED AREA MEAN sum(w n)/sum(w), in the units of
        # every other sample of that row, so a decayed port can be read beside
        # a port that has plasma.  ``centroid`` stays NaN because there is
        # none; ``edge`` is the extent the weights were formed over.
        # ONLY ON THE SIGNED CHAIN: the clipped legacy chain's values are
        # non-negative, so a non-positive total there means the retired
        # subtraction erased the profile, and those rows exist to reproduce
        # that method exactly as it behaved.
        if not subtract_background:
            axial = np.abs(positions)
            limit = _column_edge_cm(axial) if radius_cm is None else float(radius_cm)
            try:
                axial_weights = _flux_tube_weights(axial, limit)
            except ValueError:
                return empty
            weight_sum = float(np.sum(axial_weights))
            if weight_sum > 0.0:
                empty["ftavg"] = float(axial_weights @ values) / weight_sum
                empty["edge"] = limit
        return empty
    centroid = float(np.sum(values * positions) / total)
    folded = np.abs(positions - centroid)
    limit = _column_edge_cm(folded) if radius_cm is None else float(radius_cm)
    try:
        weights = _flux_tube_weights(folded, limit)
    except ValueError:
        empty["centroid"] = centroid
        return empty

    ftavg = float(weights @ values)
    ftavg_sem = np.nan
    if point_sem is not None:
        sem_values = np.asarray(point_sem, dtype=np.float64)[finite]
        if np.all(np.isfinite(sem_values)):
            ftavg_sem = float(np.sqrt(np.sum((weights * sem_values) ** 2)))
    _, scatter_sem = _weighted_mean_and_sem(values, weights)
    if normalize_radius_cm is not None:
        # The quadrature already divided by the area it integrated over; put
        # the integral back over the requested area instead.
        scale = (limit / float(normalize_radius_cm)) ** 2
        ftavg *= scale
        ftavg_sem *= scale
        scatter_sem *= scale
    return dict(
        ftavg=ftavg,
        ftavg_sem=ftavg_sem,
        scatter_sem=scatter_sem,
        core=core_mean,
        core_sem=core_sem,
        centroid=centroid,
        edge=limit,
        n_despiked=n_despiked,
    )


def _flux_tube_series(
    profiles: np.ndarray,
    x_cm: np.ndarray,
    point_sem: np.ndarray | None = None,
    *,
    radius_cm: float | None = FLUX_TUBE_RADIUS_CM,
    normalize_radius_cm: float | None = None,
    subtract_background: bool = True,
) -> dict[str, np.ndarray]:
    """Apply ``_flux_tube_profile_stats`` to every (z, time) profile.

    ``profiles`` is shaped ``(z, x, time)``; every returned array is shaped
    ``(z, time)``.  Each time sample is reduced independently, so the exported
    series is self-contained sample by sample -- and so is its integration
    extent, which is why ``edge`` comes back per port and per sample.
    ``radius_cm``, ``normalize_radius_cm`` and ``subtract_background`` are
    passed through unchanged; see ``_flux_tube_profile_stats``.
    """
    names = (
        "ftavg",
        "ftavg_sem",
        "scatter_sem",
        "core",
        "core_sem",
        "centroid",
        "edge",
    )
    n_z, _, n_t = profiles.shape
    out = {
        name: np.full((n_z, n_t), np.nan, dtype=np.float64) for name in names
    }
    out["n_despiked"] = np.zeros((n_z, n_t), dtype=np.int16)
    for zi in range(n_z):
        for ti in range(n_t):
            sem = None if point_sem is None else point_sem[zi, :, ti]
            stats = _flux_tube_profile_stats(
                profiles[zi, :, ti],
                x_cm,
                point_sem=sem,
                radius_cm=radius_cm,
                normalize_radius_cm=normalize_radius_cm,
                subtract_background=subtract_background,
            )
            for name in names:
                out[name][zi, ti] = stats[name]
            out["n_despiked"][zi, ti] = stats["n_despiked"]
    return out


def _weighted_mean_and_sem(
    values: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    """Return a weighted mean and the SCATTER SEM of that mean.

    ``weights`` need not be normalized; they are
    normalized here to ``a_i``.  The returned SEM is the weighted
    generalization of the core-band convention (``_nan_core_stats``: the
    sample standard deviation over the retained cells divided by the square
    root of their count):

        s^2  = sum_i a_i (f_i - m)^2 / (1 - sum_i a_i^2)
        sem  = sqrt(s^2 * sum_i a_i^2)

    With equal weights ``a_i = 1/N`` this reduces EXACTLY to
    ``std(f, ddof=1) / sqrt(N)``, so the flux-tube row and the core-band row
    report the same kind of quantity: the scatter of the retained radial
    cells about their own average, not a shot or cycle SEM.

    Returns ``(nan, nan)`` for a non-positive total weight and a finite mean
    with a ``nan`` SEM where one node carries all of it.

    WEIGHTS MAY BE NEGATIVE.  The density weight behind the T_e rows is the
    SIGNED measured density, and a cell sitting on noise about zero carries a
    negative weight.  Two things follow, and NEITHER IS REPAIRED HERE:

    * the weighted mean is no longer a convex combination of the values, so
      it can leave ``[min f, max f]``.  It is still the ratio of the two
      quadrature sums, which is the quantity wanted, and it is NOT clipped
      back into the range of its own nodes -- clipping would reimpose exactly
      the upward bias that deleting the negative cells had;
    * ``s^2`` can come out NEGATIVE, and where it does the scatter of the
      nodes about their own average is NOT DEFINED and the SEM is ``nan``.
      It is deliberately not floored to zero: zero would assert that the
      retained cells agree exactly, which is the opposite of what a negative
      variance says.

    Neither can happen with non-negative weights -- ``s^2 >= 0`` identically
    once ``sum a^2 < 1`` is checked -- so every unsigned caller (the Isat
    families, the clipped legacy rows, the unweighted ``plain`` average) is
    bit-unaffected by this.
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        return float("nan"), float("nan")
    normalized = weights / total
    mean = float(normalized @ values)
    sum_squares = float(normalized @ normalized)
    if not np.isfinite(mean) or sum_squares >= 1.0:
        return mean, float("nan")
    variance = float(normalized @ (values - mean) ** 2) / (1.0 - sum_squares)
    if variance < 0.0:
        return mean, float("nan")
    return mean, float(np.sqrt(variance * sum_squares))


def _flux_tube_te_stats(
    te_profile: np.ndarray,
    density_profile: np.ndarray,
    x_cm: np.ndarray,
    *,
    semi_quantitative: np.ndarray | None = None,
    radius_cm: float | None = FLUX_TUBE_RADIUS_CM,
    normalize_radius_cm: float | None = None,
    subtract_background: bool = True,
    trust_radius_cm: float | None = None,
    blend_radius_cm: float | None = None,
) -> dict[str, float | int]:
    """Return the flux-tube T_e of one radial profile pair, both weightings.

    The DENSITY side runs the identical pipeline as
    ``_flux_tube_profile_stats`` -- despike, optionally subtract the
    effective-width ledger's scalar background and clip at zero, drop
    non-finite cells, fold about the profile centroid, integrate to
    ``radius_cm`` -- so the returned
    ``weight_density`` is the same number that function returns as ``ftavg``,
    and the quadrature nodes and weights are the same ones.  That holds WHERE
    T_e IS FINITE WHEREVER THE DENSITY IS, since a non-finite T_e cell would
    drop a node the density chain keeps; it is true of every filled T_e
    product in this repo, which carry zero non-finite cells by construction of
    the fill.

    The T_e side is despiked on the same gate and then NOT
    background-subtracted and NOT clipped: T_e does not fall to zero at the
    scan edge, and subtracting an edge median from it would remove a real
    pedestal rather than a background.  T_e is folded about the DENSITY
    centroid, because the flux tube is centred on the column and T_e has no
    centroid of its own.

    Two averages come out of the one quadrature:

    ``ftavg``  = sum_i w_i n_i T_i / sum_i w_i n_i, the DENSITY-WEIGHTED area
                 mean, which is the quantity a 1D cell carries by construction
                 (its E_e / (3/2 n));
    ``plain``  = sum_i w_i T_i, the unweighted area mean, reported beside it.

    ``semi_quantitative`` is an optional boolean on the same samples; the
    count of marked cells carrying quadrature weight and the total weight they
    carry are returned so a consumer can see how much of the average is at the
    swept diagnostic's limit.

    ``radius_cm=None`` integrates to the pair's own COLUMN EDGE instead of to
    the flux-tube radius, and the limit used comes back as ``edge``; both
    averages are RATIOS over one node set, so neither moves with the
    normalization and ``normalize_radius_cm`` reaches ``weight_density``
    alone -- see ``_flux_tube_profile_stats`` for what it means there.

    ``trust_radius_cm`` and ``blend_radius_cm`` are that port's filled-T_e
    trust model.  Given either, the returned ``prior_weight`` /
    ``pure_prior_weight`` are the SHARE of the density-weighted quadrature
    weight (``sum w n``) carried by cells at ``|x|`` beyond that radius, and
    ``plain_prior_weight`` / ``plain_pure_prior_weight`` the same share of the
    unweighted ``sum w``.  Beyond the trust radius the filled product mixes
    the measurement into the scrape-off-layer prior; beyond the blend radius
    it reports the prior alone.  These say how much of an average is the prior
    rather than a measurement of that port -- a number, not an adjective.  The
    mask is in SCAN ``|x|``, the frame the trust model is stated in, NOT in
    the quadrature's centroid-folded radius.  ``NaN`` where the radius is not
    given.  The density-weighted pair are RATIOS OF TWO SIGNED SUMS and are
    therefore not confined to ``[0, 1]``: where the cells beyond the radius
    are noise about zero their numerator can go negative, or exceed a
    denominator the same cells have pulled down.  That is reported as
    computed; the unweighted pair, whose weights ``w`` are non-negative, stay
    shares in the ordinary sense.

    ``subtract_background=False`` is the comparand chain, and it reaches the
    T_e rows through the WEIGHT: both averages are taken over the unsubtracted
    density's node set, weights and centroid.  The plain row is not weighted
    by the density but still sits on that node set, so it moves a little too.

    THE WEIGHT IS SIGNED, and the consequences are carried, not repaired.
    ``ftavg`` is the ratio ``sum w n T / sum w n`` of two quadrature sums, and
    on the comparand chain ``n`` is the measured density with its sign, so a
    node on noise about zero enters with a NEGATIVE weight.  Where enough of
    them do, ``ftavg`` stops being a convex combination of its own nodes and
    can read outside ``[min T, max T]``; where they cancel the positive ones,
    ``sum w n <= 0`` and the row is ``NaN``.  Both are disclosed rather than
    clipped -- clipping the weight at zero is the sign test again, by another
    name, and it is what the product just stopped doing.  A row that leaves
    its own bound is a row whose denominator is small and whose T_e therefore
    is not measured to that precision; both happen only at samples where there
    is nearly no plasma to weight with, and the exported row carries
    ``te_ftavg_weight_density_*`` so a consumer can see the denominator.  The
    ``plain`` row is unweighted and stays a convex combination throughout.

    WHERE THE RETAINED DENSITY CELLS SUM TO ZERO OR LESS, BOTH T_e ROWS STAY
    ``NaN``.  The denominator is non-positive and there is no centroid, and a
    ratio reported anyway would assert a temperature at a port with no plasma
    to carry one.  ``weight_density`` is reported instead: it carries the
    SIGNED AREA MEAN of the retained density cells, folded about the GEOMETRIC
    AXIS ``x = 0`` -- the same number ``_flux_tube_profile_stats`` puts in its
    ``ftavg`` there, in the same units -- with ``edge`` giving the extent it
    was formed over, so a consumer reading a ``NaN`` T_e row can see the
    non-positive denominator that produced it.
    """
    despiked_density, _ = _despike_profile(density_profile)
    prepared_density = (
        _subtract_background(despiked_density)
        if subtract_background
        else despiked_density
    )
    despiked_te, n_despiked = _despike_profile(te_profile)

    empty = dict(
        ftavg=np.nan,
        ftavg_sem=np.nan,
        plain=np.nan,
        plain_sem=np.nan,
        weight_density=np.nan,
        centroid=np.nan,
        edge=np.nan,
        n_despiked=n_despiked,
        node_count=0,
        semi_quant_count=0,
        semi_quant_weight=np.nan,
        prior_weight=np.nan,
        pure_prior_weight=np.nan,
        plain_prior_weight=np.nan,
        plain_pure_prior_weight=np.nan,
    )
    finite = np.isfinite(prepared_density) & np.isfinite(despiked_te)
    if np.count_nonzero(finite) < 5:
        return empty
    values = (
        np.clip(prepared_density[finite], 0.0, None)
        if subtract_background
        else prepared_density[finite]
    )
    te_values = despiked_te[finite]
    positions = x_cm[finite]
    total = float(np.sum(values))
    if total <= 0.0:
        # The T_e ROWS STAY NaN.  Both averages are ratios whose denominator is
        # the density weight, and where that weight is non-positive neither
        # ratio exists -- reporting one anyway would be asserting a temperature
        # where there is no plasma to carry it.  The DENOMINATOR is reported
        # instead: ``weight_density`` carries the SIGNED AREA MEAN of the
        # retained density cells, folded about the GEOMETRIC AXIS x = 0 since
        # the intensity centroid is undefined -- the same number the density
        # rows carry there, in the same units -- so a consumer can see why the
        # ratio refused.  Carried on the SIGNED chain only, as on the density
        # side.
        if not subtract_background:
            axial = np.abs(positions)
            limit = _column_edge_cm(axial) if radius_cm is None else float(radius_cm)
            try:
                axial_weights = _flux_tube_weights(axial, limit)
            except ValueError:
                return empty
            weight_sum = float(np.sum(axial_weights))
            if weight_sum > 0.0:
                empty["weight_density"] = (
                    float(axial_weights @ values) / weight_sum
                )
                empty["edge"] = limit
        return empty
    centroid = float(np.sum(values * positions) / total)
    folded = np.abs(positions - centroid)
    limit = _column_edge_cm(folded) if radius_cm is None else float(radius_cm)
    try:
        weights = _flux_tube_weights(folded, limit)
    except ValueError:
        empty["centroid"] = centroid
        return empty

    plain, plain_sem = _weighted_mean_and_sem(te_values, weights)
    ftavg, ftavg_sem = _weighted_mean_and_sem(te_values, weights * values)
    carrying = weights > 0.0
    semi_quant_count = 0
    semi_quant_weight = 0.0
    if semi_quantitative is not None:
        marked = np.asarray(semi_quantitative, dtype=bool)[finite] & carrying
        semi_quant_count = int(np.count_nonzero(marked))
        semi_quant_weight = float(np.sum(weights[marked]))
    weight_density = float(weights @ values)

    def _beyond_share(radius: float | None) -> tuple[float, float]:
        """The (density-weighted, unweighted) weight share past ``radius``."""
        if radius is None:
            return float("nan"), float("nan")
        beyond = np.abs(positions) > float(radius)
        weighted = weights * values
        weighted_total = float(np.sum(weighted))
        plain_total = float(np.sum(weights))
        return (
            float(np.sum(weighted[beyond]) / weighted_total)
            if weighted_total > 0.0
            else float("nan"),
            float(np.sum(weights[beyond]) / plain_total)
            if plain_total > 0.0
            else float("nan"),
        )

    prior_weight, plain_prior_weight = _beyond_share(trust_radius_cm)
    pure_prior_weight, plain_pure_prior_weight = _beyond_share(blend_radius_cm)
    if normalize_radius_cm is not None:
        weight_density *= (limit / float(normalize_radius_cm)) ** 2
    return dict(
        ftavg=ftavg,
        ftavg_sem=ftavg_sem,
        plain=plain,
        plain_sem=plain_sem,
        weight_density=weight_density,
        centroid=centroid,
        edge=limit,
        n_despiked=n_despiked,
        node_count=int(np.count_nonzero(carrying)),
        semi_quant_count=semi_quant_count,
        semi_quant_weight=semi_quant_weight,
        prior_weight=prior_weight,
        pure_prior_weight=pure_prior_weight,
        plain_prior_weight=plain_prior_weight,
        plain_pure_prior_weight=plain_pure_prior_weight,
    )


def _flux_tube_te_series(
    te_profiles: np.ndarray,
    density_profiles: np.ndarray,
    x_cm: np.ndarray,
    semi_quantitative: np.ndarray | None = None,
    *,
    radius_cm: float | None = FLUX_TUBE_RADIUS_CM,
    normalize_radius_cm: float | None = None,
    subtract_background: bool = True,
    trust_radius_cm: np.ndarray | None = None,
    blend_radius_cm: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Apply ``_flux_tube_te_stats`` to every (z, time) profile pair.

    Both grids are shaped ``(z, x, time)`` on the SAME time base; every
    returned array is shaped ``(z, time)``.  ``radius_cm`` and
    ``normalize_radius_cm`` are passed through unchanged; see
    ``_flux_tube_te_stats``.  ``trust_radius_cm`` and ``blend_radius_cm`` are
    PER-PORT, shaped ``(z,)``: the trust model is a per-port statement, so
    each row's prior-weight share is taken against its own radii.
    """
    if te_profiles.shape != density_profiles.shape:
        raise ValueError(
            f"T_e grid {te_profiles.shape} and density grid "
            f"{density_profiles.shape} must be reduced over the same "
            "(z, x, time) grid"
        )
    names = (
        "ftavg",
        "ftavg_sem",
        "plain",
        "plain_sem",
        "weight_density",
        "centroid",
        "edge",
        "semi_quant_weight",
        "prior_weight",
        "pure_prior_weight",
        "plain_prior_weight",
        "plain_pure_prior_weight",
    )
    n_z, _, n_t = te_profiles.shape
    out = {
        name: np.full((n_z, n_t), np.nan, dtype=np.float64) for name in names
    }
    out["n_despiked"] = np.zeros((n_z, n_t), dtype=np.int16)
    out["node_count"] = np.zeros((n_z, n_t), dtype=np.int16)
    out["semi_quant_count"] = np.zeros((n_z, n_t), dtype=np.int16)
    for zi in range(n_z):
        for ti in range(n_t):
            marks = (
                None if semi_quantitative is None else semi_quantitative[zi, :, ti]
            )
            stats = _flux_tube_te_stats(
                te_profiles[zi, :, ti],
                density_profiles[zi, :, ti],
                x_cm,
                semi_quantitative=marks,
                radius_cm=radius_cm,
                normalize_radius_cm=normalize_radius_cm,
                subtract_background=subtract_background,
                trust_radius_cm=(
                    None if trust_radius_cm is None else float(trust_radius_cm[zi])
                ),
                blend_radius_cm=(
                    None if blend_radius_cm is None else float(blend_radius_cm[zi])
                ),
            )
            for name in names:
                out[name][zi, ti] = stats[name]
            out["n_despiked"][zi, ti] = stats["n_despiked"]
            out["node_count"][zi, ti] = stats["node_count"]
            out["semi_quant_count"][zi, ti] = stats["semi_quant_count"]
    return out


def _interp_onto_time_grid(
    profiles: np.ndarray,
    source_time_ms: np.ndarray,
    target_time_ms: np.ndarray,
) -> np.ndarray:
    """Return ``(z, x, time)`` profiles put on ``target_time_ms``, cell by cell.

    A target sample takes a value only where the source samples BRACKETING it
    are both finite; anywhere else the cell stays non-finite.  That is what
    keeps the source product's own "this cell is unusable here" pattern
    intact: filling a non-finite cell from a distant finite sample of the same
    cell would invent a measurement, and at the scan edges -- where these
    products' non-finite cells actually are -- it would also move the scalar
    background baseline and with it the whole flux-tube average.  The
    bracketing test is done by interpolating the finite MASK: a target between
    a finite and a non-finite source sample reads below one and is dropped.

    ``np.interp`` holds the end values, so a target sample outside the source
    span takes the nearest source value rather than an extrapolation, and is
    kept when that end sample is finite.
    """
    n_z, n_x, _ = profiles.shape
    out = np.full((n_z, n_x, target_time_ms.size), np.nan, dtype=np.float64)
    for zi in range(n_z):
        for xi in range(n_x):
            series = profiles[zi, xi]
            finite = np.isfinite(series)
            if np.count_nonzero(finite) < 2:
                continue
            bracketed = np.interp(
                target_time_ms, source_time_ms, finite.astype(np.float64)
            )
            out[zi, xi] = np.where(
                bracketed >= 1.0 - 1e-12,
                np.interp(target_time_ms, source_time_ms[finite], series[finite]),
                np.nan,
            )
    return out


def _measured_coverage_cm(
    measured_te: np.ndarray,
    x_cm: np.ndarray,
    trust_radius_cm: np.ndarray,
    row_measured_cells: np.ndarray,
) -> np.ndarray:
    """Return how far out each T_e row reports its own MEASUREMENT, in cm.

    ``measured_te`` is the filled product's ``te_masked`` grid, ``(z, x,
    time)``, finite exactly where a QC-surviving measurement stands behind the
    cell.  The answer per port and sample is the outermost ``|x|`` carrying
    such a cell, CAPPED at that port's ``te_trust_radius_cm``: beyond the trust
    radius the filled product mixes the measurement into the scrape-off-layer
    prior and past the blend radius it reports the prior alone, so a measured
    cell out there is not what the product reports.  A row the product itself
    calls prior-derived at that sample (``te_row_measured_cells == 0``) has no
    coverage at all and is ``NaN``.

    The radius is in SCAN ``|x|``, the frame the trust model and the line scan
    are both stated in, NOT in the flux-tube quadrature's centroid-folded
    radius; the two differ by the per-sample centroid offset, which is exported
    beside them.
    """
    finite = np.isfinite(measured_te)
    radius = np.abs(np.asarray(x_cm, dtype=np.float64))[np.newaxis, :, np.newaxis]
    reach = np.where(finite, radius, -np.inf).max(axis=1)
    coverage = np.minimum(reach, np.asarray(trust_radius_cm)[:, np.newaxis])
    return np.where(
        np.isfinite(reach) & (np.asarray(row_measured_cells) > 0),
        coverage,
        np.nan,
    )


def _es4_upstream_definitions() -> dict[str, np.ndarray]:
    """The prose fields of the ES4 upstream bracket, kept out of the row build."""
    return {
        "es4_upstream_definition": np.array(
            "ADDITIVE family, experiment set 4 ONLY: labelled alternative "
            "UPSTREAM density rows, exported as a row table.  It ADDS to the "
            "product and replaces nothing -- density_mean_cm3, "
            "density_ftavg_cm3, density_total_sem_cm3, te_mean_ev, "
            "te_row_measured and every other pre-existing field are "
            "bit-unchanged by its presence, and a consumer that does not know "
            "these names reads exactly the product it read before.  Row r is "
            "named by es4_upstream_row_key[r]; its port is "
            "es4_upstream_row_port[r]; its per-sample values are "
            "es4_upstream_density_mean_cm3[r] (the unweighted core-band mean, "
            "the LEGACY convention of density_mean_cm3) and "
            "es4_upstream_density_ftavg_cm3[r] (the FLUX-TUBE convention of "
            "density_ftavg_cm3, same radius, same despike, same "
            "centroid-folding and, since schema v33, the same NO-background-"
            "subtraction ruling -- see ftavg_background; both rows of the "
            "bracket moved together, so the face spread is unaffected by the "
            "ruling), both in cm^-3 on es4_upstream_time_ms, which "
            "is density_time_ms.  es4_upstream_density_core_count[r] counts "
            "the core cells behind each sample.  NO SEM is exported for these "
            "rows: the pre-existing density_total_sem_cm3 is the rot-0 "
            "i_sweep chain's radial-scatter SEM plus the Probe-A area "
            "calibration and is NOT the uncertainty of a rot-180 ISAT row.  "
            "The face spread between a primary row and its "
            "es4_upstream_row_bracket_partner is the quantity this family "
            "exists to expose; it is a bracket, not an error bar."
        ),
        "es4_upstream_bracket_definition": np.array(
            "THE TWO-FACE BRACKET.  At every ES4 port the probe was run twice, "
            "once at each rotation.  The face that looks UPSTREAM is the "
            "i_sweep electrode at rot-0 and the isat electrode at rot-180, so "
            "the same physical upstream density has two independent "
            "measurements.  Rows with es4_upstream_row_chain == "
            "'deadtime_face_pair' are that pair: es4_upstream_row_face is "
            "'rot180_isat' on the PRIMARY row (es4_upstream_row_role == "
            "'primary') and 'rot0_isweep' on its partner "
            "(es4_upstream_row_bracket_partner names it, and 'bracket_partner' "
            "is its role).  Both come from ONE implementation of the row "
            "arithmetic, scripts/es4_upstream_rows_rot180_isat.py's "
            "build_port, called with this export's own products and its own "
            "filled-T_e core-band row, so the two rows differ ONLY in which "
            "face they read: identical cell-admission mask (finite and "
            "positive on both faces, and carrying neither the rot-180 "
            "product's rail_mask nor its state_mask), identical T_e, "
            "identical plateau, identical area convention.  THREE LEVELS, NOT "
            "TWO.  The partner row is NOT the SCORED row: density_mean_cm3 and "
            "density_ftavg_cm3 at that port come from the full density "
            "product, with its own per-cell T_e and no dead-time admission "
            "mask, and this family does not touch them.  Over the scoring "
            "plateau window the three read, as primary / partner / scored: at "
            "p29, flux-tube 2.982e12 / 2.466e12 / 2.703e12 cm^-3 (primary "
            "1.103 and partner 0.912 times the scored row) and core-band "
            "4.709e12 / 4.434e12 / 4.416e12 cm^-3 (1.066 and 1.004); at p41, "
            "flux-tube 2.784e11 / 2.550e11 / 2.420e11 cm^-3 (1.150 and 1.054) "
            "and core-band 9.133e11 / 6.722e11 / 6.686e11 cm^-3 (1.366 and "
            "1.005).  The partner and the scored row agree to 0.4-0.5 percent "
            "in the core band and part by -8.8 percent at p29 and +5.4 percent "
            "at p41 under the flux-tube convention, which reduces the whole "
            "scan and so is where the dead-time admission mask and the "
            "despiking part company; a consumer must say which of the three it "
            "quoted.  AREAS are keyed by the ELECTRODE that "
            "collected, es4_upstream_row_area_key with the value in "
            "es4_upstream_row_area_cm2: ap_R_cm2 is the RIGHT electrode, which "
            "is what the isat channel sits on under the nominal wiring every "
            "ES4 run carries, and ap_L_cm2 the LEFT one, which is the i_sweep "
            "channel's.  That is the assignment the areas were themselves "
            "produced under (scripts/calibrate_probe_areas.py accumulates "
            "ap_R_m2 from the rot-180 isat product and ap_L_m2 from the rot-0 "
            "i_sweep product), and it is the swap-aware answer of "
            "density_area_key_for_deadtime_source for these runs.  PORTS: "
            "only p29 (runs 44/45) and p41 (runs 46/47) carry the pair.  p21's "
            "rot-180 partner (run 43) carries a registered channel-state shot "
            "range that removes 20 of the 21 core positions, leaving one "
            "admitted core position (x = +10.0 cm, 4 of 84 core cells, ratio "
            "ISAT180/isweep0 = 1.017) -- too few to form the core-band mean or "
            "the chord line integral, so NO rot-180 row is built there; p11 "
            "and p50 have no rot-180 partner run at all.  A consumer must not "
            "read the absence of a row as a null result -- it is an absent "
            "measurement."
        ),
        "es4_upstream_te_definition": np.array(
            "WHICH T_e EACH ROW WAS DERIVED AT.  n_e is proportional to "
            "I_sat / (A_p e C_s) and C_s to sqrt(T_e), so every density row "
            "carries the T_e it was built with: es4_upstream_te_ev[r] is that "
            "T_e per sample, and a row moves to a different T_e by the factor "
            "sqrt(T_e_old / T_e_new) with no re-reading of the raw data.  "
            "es4_upstream_row_te_measured[r] is True only where that T_e is a "
            "MEASUREMENT of that port, and then es4_upstream_row_te_measured_"
            "ev[r] is its value, es4_upstream_row_te_face[r] the probe face it "
            "was measured on, es4_upstream_row_te_window_ms[r] the window it "
            "is a mean over, and es4_upstream_row_te_basis[r] the estimator.  "
            "False means the row was derived under the filled-T_e product's "
            "PRIOR-DERIVED row for that port -- reconstructed from its "
            "neighbours, the SOL anchors and the end-plate sentinels, and not "
            "a measurement of that port (see te_row_provenance_definition; "
            "ES4 p21/p29/p41/p50 are prior-derived at every sample).  Every "
            "rot-180 ISAT row and every bracket partner is False: the face "
            "bracket moves the FACE, not the T_e."
        ),
        "es4_upstream_p21_te_rederivation": np.array(
            "THE p21 RE-DERIVATION.  The row this product has always carried "
            "at p21 is the rot-0 i_sweep row derived under the filled-T_e "
            "product's PRIOR-DERIVED T_e for that port, whose mean over the "
            "scoring plateau window is te_mean_ev at p21.  The plateau T_e at "
            "p21 has since been MEASURED from that port's own sweeps "
            "(scripts/es4_te_time_slope.py), and it is far below the prior.  "
            "Row 'p21_rot0_isweep_te_measured' is that same row re-derived at "
            "the measured value: the primary, with "
            "es4_upstream_row_te_measured True and the value, face, window and "
            "estimator stamped on it.  Row 'p21_rot0_isweep_te_prior' is the "
            "row as it was derived under the prior, kept beside it as the "
            "record of what was scored before; the two differ by exactly "
            "sqrt(T_e_prior / T_e_measured) per sample and by nothing else.  "
            "The measured value is quoted at the SCORING PLATEAU WINDOW, the "
            "window the prior it replaces is quoted at and the window the "
            "transport comparison scores.  es4_upstream_p21_te_measured_"
            "display_ev is the SAME estimator on the SAME face over the "
            "instrument's full displayed plateau, es4_upstream_p21_te_"
            "measured_display_window_ms -- a different window, not a different "
            "measurement, exported because the p11->p21 electron-pressure "
            "chain on record was computed at it.  es4_upstream_p21_te_"
            "measured_rot180_ev and its _display counterpart are the opposite "
            "face at the same port under the same estimator and the same two "
            "windows.  WHAT THE RE-DERIVATION INHERITS, largest term first.  "
            "(1) THE ESTIMATOR SPREAD.  The instrument reports four estimators "
            "per face -- core band or x = 0, each under the window-family "
            "median or the default window -- and on run 42 they read "
            "0.6507-0.7144 eV over the scoring window and 0.6641-0.7355 eV "
            "over the display window: 9.8 and 10.7 percent in T_e, which is "
            "4.6 and 5.0 percent on the density.  The value used here is the "
            "COOLEST of the four at both windows, so the spread is ONE-SIDED: "
            "every other estimator reads hotter, and a hotter T_e gives a "
            "smaller density, so the re-derived row is biased HIGH by up to "
            "that amount and not symmetrically uncertain.  (2) THE FACE "
            "SPREAD, the second and smaller term: the same estimator on the "
            "opposite face reads 0.6431 eV against 0.6507 eV over the scoring "
            "window, 1.2 percent in T_e and 0.6 percent on the density.  The "
            "pre-existing te_mean_ev, te_row_measured and te_row_measured_"
            "cells at p21 are UNCHANGED and still describe the filled-T_e "
            "product, which is still prior-derived there: this family does not "
            "restate that product, it records a density row built at a "
            "different T_e."
        ),
        "es4_upstream_f_convention": np.array(
            "SWEEP REST-BIAS FACTOR: NOT APPLIED.  es4_upstream_row_f_applied "
            "is 1.0 on every row, and it is exported to say so explicitly.  "
            "ES4 runs 42-48 -- p21, p29, p41 and p50 -- sweep a nominal +-20 V "
            "ramp and park the swept face at that shallower rest bias, while "
            "ES3 parks at -75 V nominal, and a dead-time ion-saturation "
            "current is collected at the parked bias, so comparing one of "
            "those i_sweep rows with its ES3 control needs a channel-scale "
            "factor F.  ES4 p11 (run 41) is the EXCEPTION and needs no F: it "
            "sweeps +-75 V, identical to its ES3 p11 control (run 31), so the "
            "two are already on one bias scale.  Neither the density product "
            "behind "
            "density_mean_cm3 nor this exporter has ever multiplied an ES4 row "
            "by one, so the retained i_sweep rows here carry exactly what the "
            "placed product carries and no factor is invented at export.  F is "
            "MEASURED by scripts/es4_sweep_rest_bias_factor.py, which reports "
            "it per port and rotation as a BRACKET between a direct read off "
            "the ES3 ramp and an extrapolation of the ES4 one; the two ends "
            "disagree by more than either one's spread, so the bracket is the "
            "claim and a single central value is not available to apply.  The "
            "ISAT face has no sweep-derived rest bias at all, which is one "
            "reason the rot-180 ISAT row is worth having: F does not enter it."
        ),
    }


def _es4_upstream_rows(
    *,
    te_core_mean_ev: np.ndarray,
    te_time_ms: np.ndarray,
    density_mean_cm3: np.ndarray,
    density_ftavg_cm3: np.ndarray,
    density_core_count: np.ndarray,
    density_time_ms: np.ndarray,
    isweep_path: Path,
    isat_rot180_path: Path,
    area_toml_path: Path,
) -> dict[str, np.ndarray]:
    """Build the ES4 upstream two-face bracket and the p21 T_e re-derivation.

    Returns the ``es4_upstream_*`` fields as a row table: one row per labelled
    density row, with the per-row provenance in the ``es4_upstream_row_*``
    arrays and the per-sample values in ``es4_upstream_density_*``.  Every row
    sits on ``density_time_ms``.

    The two-face rows at ``ES4_UPSTREAM_BRACKET_PORTS`` come from
    ``es4_upstream_build_port``, the single implementation of that arithmetic,
    so the ISAT row and its bracket partner differ ONLY in which face they
    read: same admission mask, same T_e, same areas convention, same window.
    The partner is therefore NOT the same number as ``density_mean_cm3`` at
    that port, which comes from the full density product with its own per-cell
    T_e and no dead-time admission mask; both are exported and both are
    labelled.

    The p21 rows come from this export's own density row instead.  The primary
    is that row re-derived at the MEASURED plateau T_e -- n_e is proportional
    to 1/C_s and C_s to sqrt(T_e), so a row derived under a prior T_e rescales
    by sqrt(T_e_prior / T_e_measured) per sample with no re-reading of the raw
    data -- and the row as it was derived under the prior is kept beside it.

    T_e for the two-face rows is this export's own filled-T_e core-band row,
    NOT the placed overlay: an exporter may not read the product it is
    writing.  It is the same quantity the product exports as ``te_mean_ev``.
    """
    ports = [int(value) for value in PORTS]
    keys: list[str] = []
    row_port: list[int] = []
    role: list[str] = []
    face: list[str] = []
    chain: list[str] = []
    run_id: list[str] = []
    area_key: list[str] = []
    area_cm2: list[float] = []
    te_measured_flag: list[bool] = []
    te_value: list[float] = []
    te_basis: list[str] = []
    te_face: list[str] = []
    te_window: list[tuple[float, float]] = []
    partner: list[str] = []
    mean_cm3: list[np.ndarray] = []
    ftavg_cm3: list[np.ndarray] = []
    core_count: list[np.ndarray] = []
    te_series: list[np.ndarray] = []

    def te_provider(port_id, times_ms):
        row = te_core_mean_ev[ports.index(int(port_id))]
        return np.interp(np.asarray(times_ms, dtype=np.float64), te_time_ms, row), False

    for port in ES4_UPSTREAM_BRACKET_PORTS:
        built = es4_upstream_build_port(
            port,
            Path("."),
            isweep_path=isweep_path,
            isat_path=isat_rot180_path,
            area_toml_path=area_toml_path,
            te_provider=te_provider,
        )
        if not np.allclose(built["t_ms"], density_time_ms):
            raise ValueError(
                f"ES4 upstream bracket at p{port}: the dead-time axis "
                f"{built['t_ms']} does not match the exported density axis"
            )
        x_cm = built["x_cm"]
        admitted = built["admitted"]
        core = np.abs(x_cm) <= ES4_UPSTREAM_CORE_CM
        run_isweep, run_isat = ES4_UPSTREAM_PORT_RUNS[port]
        faces = (
            (
                "rot180_isat",
                built["n_isat_cells"],
                run_isat,
                built["area_isat_key"],
                built["area_isat_cm2"],
                "primary",
            ),
            (
                "rot0_isweep",
                built["n_isweep_cells"],
                run_isweep,
                built["area_isweep_key"],
                built["area_isweep_cm2"],
                "bracket_partner",
            ),
        )
        for face_label, cells, run, key, area, row_role in faces:
            masked = np.where(admitted, cells, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                core_mean_m3 = np.nanmean(masked[core, :], axis=0)
            # The comparand convention: NO background subtraction, the same as
            # density_ftavg_cm3, which this family brackets and which the p21
            # row below is a rescale of.  A bracket whose two rows were reduced
            # under different conventions would report the convention change as
            # a face spread.
            ftavg = _flux_tube_series(
                masked[None, :, :], x_cm, subtract_background=False
            )["ftavg"][0]
            keys.append(f"p{port}_{face_label}")
            row_port.append(port)
            role.append(row_role)
            face.append(face_label)
            chain.append("deadtime_face_pair")
            run_id.append(run)
            area_key.append(key)
            area_cm2.append(float(area))
            te_measured_flag.append(False)
            te_value.append(float("nan"))
            te_basis.append("filled-T_e core-band row, prior-derived at this port")
            te_face.append("")
            te_window.append((float("nan"), float("nan")))
            partner.append(
                f"p{port}_rot0_isweep" if face_label == "rot180_isat"
                else f"p{port}_rot180_isat"
            )
            mean_cm3.append(core_mean_m3 * M3_TO_CM3)
            ftavg_cm3.append(ftavg * M3_TO_CM3)
            core_count.append(np.sum(np.isfinite(masked[core, :]), axis=0).astype(np.int64))
            te_series.append(np.asarray(built["te_ev"], dtype=np.float64))

    # ---- the p21 T_e re-derivation -----------------------------------
    port = ES4_UPSTREAM_TE_REDERIVED_PORT
    index = ports.index(port)
    te_prior_row = np.interp(density_time_ms, te_time_ms, te_core_mean_ev[index])
    rescale = np.sqrt(te_prior_row / ES4_UPSTREAM_P21_TE_MEASURED_EV)
    run_isweep = ES4_UPSTREAM_PORT_RUNS[port][0]
    p21_area_key = es4_upstream_area_key(ChannelKind.I_SWEEP)
    p21_probe = ES4_UPSTREAM_PROBE_FROM_DIGIT[int(run_isweep[1])]
    p21_area_cm2 = es4_upstream_load_areas_cm2(area_toml_path)[p21_probe][p21_area_key]
    p21_rows = (
        (
            f"p{port}_rot0_isweep_te_measured",
            "primary",
            rescale,
            True,
            ES4_UPSTREAM_P21_TE_MEASURED_EV,
            ES4_UPSTREAM_P21_TE_MEASURED_BASIS,
            ES4_UPSTREAM_P21_TE_MEASURED_FACE,
            ES4_UPSTREAM_P21_TE_MEASURED_WINDOW_MS,
            np.full(density_time_ms.shape, ES4_UPSTREAM_P21_TE_MEASURED_EV),
            f"p{port}_rot0_isweep_te_prior",
        ),
        (
            f"p{port}_rot0_isweep_te_prior",
            "retained_prior_derived",
            np.ones_like(rescale),
            False,
            float("nan"),
            "filled-T_e core-band row, prior-derived at this port",
            "",
            (float("nan"), float("nan")),
            te_prior_row,
            f"p{port}_rot0_isweep_te_measured",
        ),
    )
    for (
        key,
        row_role,
        factor,
        measured,
        value,
        basis,
        measured_face,
        window,
        te_row,
        partner_key,
    ) in p21_rows:
        keys.append(key)
        row_port.append(port)
        role.append(row_role)
        face.append("rot0_isweep")
        chain.append("overlay_density_isweep")
        run_id.append(run_isweep)
        area_key.append(p21_area_key)
        area_cm2.append(float(p21_area_cm2))
        te_measured_flag.append(measured)
        te_value.append(value)
        te_basis.append(basis)
        te_face.append(measured_face)
        te_window.append(window)
        partner.append(partner_key)
        mean_cm3.append(density_mean_cm3[index] * factor)
        ftavg_cm3.append(density_ftavg_cm3[index] * factor)
        core_count.append(np.asarray(density_core_count[index], dtype=np.int64))
        te_series.append(te_row)

    return {
        "es4_upstream_row_key": np.array(keys),
        "es4_upstream_row_port": np.array(row_port, dtype=np.int16),
        "es4_upstream_row_role": np.array(role),
        "es4_upstream_row_face": np.array(face),
        "es4_upstream_row_chain": np.array(chain),
        "es4_upstream_row_run_id": np.array(run_id),
        "es4_upstream_row_area_key": np.array(area_key),
        "es4_upstream_row_area_cm2": np.array(area_cm2, dtype=np.float64),
        "es4_upstream_row_bracket_partner": np.array(partner),
        "es4_upstream_row_te_measured": np.array(te_measured_flag, dtype=bool),
        "es4_upstream_row_te_measured_ev": np.array(te_value, dtype=np.float64),
        "es4_upstream_row_te_basis": np.array(te_basis),
        "es4_upstream_row_te_face": np.array(te_face),
        "es4_upstream_row_te_window_ms": np.array(te_window, dtype=np.float64),
        "es4_upstream_row_f_applied": np.full(
            len(keys), ES4_UPSTREAM_F_APPLIED, dtype=np.float64
        ),
        "es4_upstream_density_mean_cm3": np.asarray(mean_cm3, dtype=np.float64),
        "es4_upstream_density_ftavg_cm3": np.asarray(ftavg_cm3, dtype=np.float64),
        "es4_upstream_density_core_count": np.asarray(core_count, dtype=np.int64),
        "es4_upstream_te_ev": np.asarray(te_series, dtype=np.float64),
        "es4_upstream_time_ms": np.asarray(density_time_ms, dtype=np.float64),
        "es4_upstream_p21_te_measured_display_ev": np.array(
            ES4_UPSTREAM_P21_TE_MEASURED_DISPLAY_EV
        ),
        "es4_upstream_p21_te_measured_display_window_ms": np.array(
            ES4_UPSTREAM_P21_TE_MEASURED_DISPLAY_WINDOW_MS, dtype=np.float64
        ),
        "es4_upstream_p21_te_measured_rot180_ev": np.array(
            ES4_UPSTREAM_P21_TE_MEASURED_ROT180_EV
        ),
        "es4_upstream_p21_te_measured_rot180_display_ev": np.array(
            ES4_UPSTREAM_P21_TE_MEASURED_ROT180_DISPLAY_EV
        ),
        "es4_upstream_source_file_rot180_isat": np.array(str(isat_rot180_path)),
        "es4_upstream_source_file_rot0_isweep": np.array(str(isweep_path)),
        "es4_upstream_source_file_area_calibration": np.array(str(area_toml_path)),
    }


def _check_density_convention_pair(
    density_mean_cm3: np.ndarray,
    density_ftavg_cm3: np.ndarray,
    ports: np.ndarray,
    time_ms: np.ndarray,
) -> None:
    """Refuse a sample whose core-band density is unusable but flux-tube one is not.

    The two exported density conventions come from two chains over the SAME
    ``n_e_m3`` grid: ``density_mean_cm3`` is the unweighted core-band mean of
    that grid, and ``density_ftavg_cm3`` is the despiked, background-subtracted,
    centroid-folded flux-tube quadrature over the whole scan.  A cell of the
    grid is either a SIGNED measurement or NaN, and NaN means only that there
    is no usable measurement there; a negative cell is a real reading on noise
    about zero and is kept.  The core-band mean is therefore a signed number,
    and at a port whose column has decayed into the noise it can legitimately
    come out negative -- which is a measurement, not a defect, and is not
    refused here.

    What IS refused is a core-band mean of EXACTLY ZERO, or a non-finite one,
    under a finite flux-tube average of the same profile.  Exact zero is not
    something a mean of measured cells reaches; it is what a zero-filled input
    grid produces.  A non-finite one under a finite flux-tube value means the
    core band is empty while the off-core cells still carry signal.  Both are
    defects of the input product, not the sample-level "unusable" condition the
    flux-tube fields already express as NaN (that one fails on the flux-tube
    side, which here succeeded).

    Emitting the pair anyway would ship a flux-tube value whose only exported
    uncertainty cannot be formed: the overlay carries no SEM under the flux-tube
    convention, so a consumer transfers the core-band FRACTIONAL error
    ``density_total_sem_cm3 / density_mean_cm3`` onto it, which is NaN or a
    division by zero exactly on these samples.  Raises ``ValueError`` naming the
    offending port and time instead.
    """
    mean = np.asarray(density_mean_cm3, dtype=np.float64)
    ftavg = np.asarray(density_ftavg_cm3, dtype=np.float64)
    if mean.shape != ftavg.shape:
        raise ValueError(
            f"density_mean_cm3 shape {mean.shape} and density_ftavg_cm3 shape "
            f"{ftavg.shape} differ; the two conventions must be reduced over "
            "the same (port, sample) grid"
        )
    unusable = (~np.isfinite(mean)) | (mean == 0.0)
    offending = unusable & np.isfinite(ftavg)
    if not np.any(offending):
        return
    port_index, time_index = np.nonzero(offending)
    shown = [
        f"port {int(ports[zi])} at t = {float(time_ms[ti]):g} ms "
        f"(density_mean_cm3 = {mean[zi, ti]:g}, "
        f"density_ftavg_cm3 = {ftavg[zi, ti]:g})"
        for zi, ti in zip(port_index[:5], time_index[:5])
    ]
    if port_index.size > len(shown):
        shown.append(f"and {port_index.size - len(shown)} more")
    raise ValueError(
        f"{port_index.size} density sample(s) carry a finite flux-tube average "
        "over a core-band mean that is zero or non-finite: "
        + "; ".join(shown)
        + ".  The core-band mean of the density grid is a signed average of "
        "measured cells and does not reach exactly zero; a zero or non-finite "
        "one under a finite flux-tube average means the input density product "
        "and the flux-tube reduction disagree about whether the sample carries "
        "measurements at all.  The flux-tube row's uncertainty is "
        "carried as the core-band fractional error "
        "density_total_sem_cm3 / density_mean_cm3, which cannot be formed here; "
        "rebuild the density product rather than exporting the pair."
    )


def _rot0_isat_profiles(
    path: Path,
    experiment_set_id: int,
    z_cm: np.ndarray,
) -> dict[str, np.ndarray | str]:
    """Return one rot-0 line-scan product ordered onto the overlay z axis.

    Used for both probe faces, since the two dead-time profile products have the
    same layout and differ only in which electrical channel filled them:

    * ``ISAT_PROFILE_HDF5`` -- the ``i_sweep`` channel, the UPSTREAM face at
      rot-0 and the Isat truth channel;
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


def _channel_area_attr(run_id: str, channel: str) -> str:
    """Return the area attribute of the electrode ``run_id``'s ``channel`` collected on.

    ``CHANNEL_AREA_ATTR`` names the nominal wiring; a run whose cables were
    crossed at the connector reads the other electrode on each channel, and
    ``density_area_key_for_deadtime_source`` is the single place that knows
    which runs those are.
    """
    if channel not in CHANNEL_AREA_ATTR:
        raise ValueError(f"run {run_id}: unknown probe channel {channel}")
    return density_area_key_for_deadtime_source(run_id, ChannelKind(channel))


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
        run_area_attrs = []
        for channel, current, current_sem in zip(channels, currents, sems):
            area_attr = _channel_area_attr(run_id, channel)
            area = float(areas_cm2[run_id][area_attr])
            if area <= 0.0:
                raise ValueError(f"run {run_id}: non-positive {channel} face area")
            run_area_attrs.append(area_attr)
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
            f"{channels[0]}/{run_area_attrs[0]}={run_areas[0]:.6f} cm2"
            f" x {channels[1]}/{run_area_attrs[1]}="
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
    *,
    radial: bool = False,
) -> dict[str, np.ndarray | float | int | str]:
    """Return offset-corrected x=0 upstream Isat mean and shot SEM.

    ``radial=True`` additionally reduces EVERY scanned x position of the same
    afterglow traces and returns the full line scan as ``profile_mean_a`` /
    ``profile_sem_a``, shaped ``(port, x, time)`` on ``x_cm`` and on the same
    ``time_ms`` grid, so the radial-averaging conventions can be taken on the
    decay the way they are already taken on the dead-time line scans.  Each
    position is reduced by the SAME statements as ``x = 0``, against its OWN
    per-position high-current shot rejection -- which at ``x = 0`` is the mask
    the x=0 rows already use -- so the ``x = 0`` column of the returned profile
    is the exported ``mean_a`` / ``sem_a`` bit for bit.  It costs one raw read
    and one filter pass per position, so it is off by default and the x=0-only
    caller is unaffected.
    """
    zero_offsets = _load_zero_offsets(zero_offsets_path)
    means = []
    sems = []
    profile_means = []
    profile_sems = []
    ports = []
    run_ids = []
    n_used = []
    n_rejected = []
    zero_offset_v = []
    source_channels = []
    source_inverted = []
    afterglow_start_ms = []
    excluded = []
    excluded_reason = []
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
        flag_grid = _high_shot_outlier_mask(
            deadtime_means,
            sigma=sigma,
            ratio=ratio,
            min_shots_used=min_shots,
        )

        sweep = run.config.sweep
        decay_start_s = sweep.t0_s + sweep.n_cycles * sweep.tau_cycle_s
        sample, crop = _padded_sample_window(
            run,
            decay_start_s,
            ISAT_DECAY_STOP_S,
            ISAT_DECAY_FILTER_PAD_S,
        )
        channel_config = run.config.channel(channel)
        offset_v = _zero_offset_v(run, channel, zero_offsets)
        samples_per_bin = max(1, int(round(ISAT_DECAY_BIN_S / run.sample_dt_s())))

        def _reduce_position(position: int, raw_hdf):
            """Shot mean and SEM of one scanned position's afterglow trace."""
            flagged = flag_grid[position]
            reject_shot = np.any(flagged, axis=1)
            if np.sum(~reject_shot) < min_shots:
                flag_counts = np.sum(flagged, axis=1)
                keep = np.argsort(flag_counts)[:min_shots]
                reject_shot[:] = True
                reject_shot[keep] = False

            shot_start = run.flat_shot_index(position, 0)
            shot_stop = shot_start + run.shots_per_position()
            raw = raw_hdf[channel_config.hdf5_path][
                shot_start:shot_stop, sample
            ].astype(np.float64)
            headers = raw_hdf[f"{channel_config.hdf5_path} headers"][
                shot_start:shot_stop
            ]

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

            n_bins = current.shape[1] // samples_per_bin
            current = current[:, : n_bins * samples_per_bin]
            current = current.reshape(
                current.shape[0], n_bins, samples_per_bin
            ).mean(axis=2)

            retained = current[~reject_shot]
            mean = np.mean(retained, axis=0)
            std = np.std(retained, axis=0, ddof=1)
            return (
                mean,
                std / np.sqrt(retained.shape[0]),
                retained.shape[0],
                int(np.sum(reject_shot)),
                n_bins,
            )

        scanned = range(x_cm.size) if radial else (x_idx,)
        position_means: dict[int, np.ndarray] = {}
        position_sems: dict[int, np.ndarray] = {}
        with run.open() as raw_hdf:
            for position in scanned:
                (
                    position_means[position],
                    position_sems[position],
                    kept,
                    dropped,
                    n_bins,
                ) = _reduce_position(position, raw_hdf)
                if position == x_idx:
                    x0_kept, x0_dropped = kept, dropped

        start_index = sample.start + crop.start
        time_s = run.time_axis(n_bins * samples_per_bin, start_index=start_index)
        time_ms = (
            time_s[: n_bins * samples_per_bin]
            .reshape(n_bins, samples_per_bin)
            .mean(axis=1)
            * 1000.0
        )

        if reference_time_ms is None:
            reference_time_ms = time_ms
        elif not np.allclose(time_ms, reference_time_ms, rtol=0.0, atol=1e-9):
            raise ValueError(f"Isat decay time grid differs for run {run_id}")
        mean = position_means[x_idx]
        sem = position_sems[x_idx]
        profile_mean = (
            np.stack([position_means[i] for i in scanned], axis=0) if radial else None
        )
        profile_sem = (
            np.stack([position_sems[i] for i in scanned], axis=0) if radial else None
        )
        registered = LATE_AFTERGLOW_PROBE_LOCAL_CURRENT.get((run_id, channel.value))
        if registered is not None:
            mean = np.full_like(mean, np.nan)
            sem = np.full_like(sem, np.nan)
            if radial:
                # The registered exclusion is a property of the RUN's channel,
                # not of one scanned position, so it takes the whole line scan
                # with it: a face excluded at x=0 is excluded at every radius
                # and therefore under every radial-averaging convention.
                profile_mean = np.full_like(profile_mean, np.nan)
                profile_sem = np.full_like(profile_sem, np.nan)
            excluded.append(True)
            excluded_reason.append(str(registered["source"]))
        else:
            excluded.append(False)
            excluded_reason.append("")
        means.append(mean)
        sems.append(sem)
        if radial:
            profile_means.append(profile_mean)
            profile_sems.append(profile_sem)
        ports.append(int(run.config.probe.port or 0))
        run_ids.append(run_id)
        n_used.append(x0_kept)
        n_rejected.append(x0_dropped)
        zero_offset_v.append(offset_v)
        source_channels.append(channel.value)
        source_inverted.append(invert_polarity)
        afterglow_start_ms.append(decay_start_s * 1000.0)

    if reference_time_ms is None:
        raise RuntimeError(
            f"No experiment-set-{experiment_set_id} upstream Isat decay traces found"
        )
    order = np.argsort(ports)
    radial_fields = (
        {
            "x_cm": x_cm,
            "profile_mean_a": np.asarray(profile_means)[order],
            "profile_sem_a": np.asarray(profile_sems)[order],
        }
        if radial
        else {}
    )
    return {
        "time_ms": reference_time_ms,
        "mean_a": np.asarray(means)[order],
        "sem_a": np.asarray(sems)[order],
        **radial_fields,
        "port": np.asarray(ports, dtype=np.int16)[order],
        "run_id": np.asarray(run_ids)[order],
        "n_shots_used": np.asarray(n_used, dtype=np.int16)[order],
        "n_shots_rejected": np.asarray(n_rejected, dtype=np.int16)[order],
        "zero_offset_v": np.asarray(zero_offset_v)[order],
        "source_channel": np.asarray(source_channels)[order],
        "source_inverted": np.asarray(source_inverted, dtype=np.bool_)[order],
        "afterglow_start_ms": np.asarray(afterglow_start_ms)[order],
        # DISCLOSED late-afterglow exclusion, per LATE_AFTERGLOW_PROBE_LOCAL_CURRENT
        # above: True/non-empty exactly where mean_a/sem_a were NaN-filled.
        "excluded": np.asarray(excluded, dtype=np.bool_)[order],
        "excluded_reason": np.asarray(excluded_reason)[order],
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
        run_area_attrs = []
        for channel, current, current_sem in zip(channels, currents, sems):
            area_attr = _channel_area_attr(run_id, channel)
            area = float(areas_cm2[run_id][area_attr])
            if area <= 0.0:
                raise ValueError(f"run {run_id}: non-positive {channel} face area")
            run_area_attrs.append(area_attr)
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
            f"{channels[0]}/{run_area_attrs[0]}={run_areas[0]:.6f} cm2"
            f" x {channels[1]}/{run_area_attrs[1]}="
            f"{run_areas[1]:.6f} cm2"
        )
        used_areas.append(run_areas)
    return {
        "geomean_a_per_cm2": geomean,
        "sem_a_per_cm2": sem,
        "area_cm2": np.asarray(used_areas, dtype=np.float64),
        "pairing": np.asarray(pairing),
    }


def _decay_noise_floor_a(trace: np.ndarray, tail_mask: np.ndarray) -> float:
    """Return the e-fold fit's noise floor for one decay trace.

    ``ISAT_DECAY_NOISE_SIGMAS`` times the robust sigma (1.4826 x MAD) of the
    trace's OWN final ``ISAT_DECAY_NOISE_TAIL_MS``, where the plasma is gone.
    Each trace carries its own floor, so a convention that averages more of
    the column -- and therefore more of the outer noise -- is not fitted
    against another convention's floor.

    Returns 0.0 -- positivity only, which is ``_decay_efold_ms``'s own default
    mode -- when the tail carries NO finite sample, because then the trace
    states no noise level and one must not be manufactured from another
    trace's.  That is not a rare corner on the AREA-AVERAGED rows: the radial
    quadrature reports nothing for a sample whose signed line scan sums to a
    non-positive total, and once the plasma is gone the upstream face's scan
    does that at every sample, so its flux-tube and whole-column traces have
    no tail at all while their fit window is fully populated.  The
    substitution is visible in the product -- the floor actually used is
    ``isat_decay_matrix_noise_floor_a`` and the sample count that entered each
    fit is ``isat_decay_matrix_n_fit``.
    """
    tail = np.asarray(trace, dtype=np.float64)[tail_mask]
    if not np.any(np.isfinite(tail)):
        return 0.0
    return float(
        ISAT_DECAY_NOISE_SIGMAS
        * 1.4826
        * np.nanmedian(np.abs(tail - np.nanmedian(tail)))
    )


def _decay_efold_ms(
    t_ms: np.ndarray,
    y: np.ndarray,
    floor: float = 0.0,
) -> float:
    """Return the log-linear e-folding decay time [ms] of ``y`` over ``t_ms``.

    Positive for a decaying signal.  NaN when fewer than 8 samples survive the
    positivity/noise-floor mask, or when the fitted slope is not a decay.  One
    unweighted least-squares line through ``log(y)``; the SAME estimator the
    transport comparison's stage (iii) applies to ``isat_decay_mean_a``, so a
    tau exported here and a tau quoted there are the same number computed the
    same way rather than two fits that happen to be described alike.
    """
    t_ms = np.asarray(t_ms, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    good = np.isfinite(t_ms) & np.isfinite(y) & (y > max(floor, 0.0))
    if np.count_nonzero(good) < 8:
        return np.nan
    slope = np.polyfit(t_ms[good], np.log(y[good]), 1)[0]
    return -1.0 / slope if slope < 0.0 else np.nan


def _decay_efold_sigma_ms(
    t_ms: np.ndarray,
    y: np.ndarray,
    sem: np.ndarray,
    floor: float = 0.0,
) -> float:
    """Return the 1-sigma measurement uncertainty [ms] of ``_decay_efold_ms``.

    ESTIMATOR.  The per-sample SEM is propagated through the very fit
    ``_decay_efold_ms`` performs.  That fit is an unweighted least-squares line
    through ``log(y)`` over the same mask, so the fitted slope is a fixed
    linear combination of the fitted samples::

        slope = sum_i c_i log(y_i),   c_i = (t_i - tbar) / sum_j (t_j - tbar)^2

    each ``log(y_i)`` carries ``sem_i / y_i`` by the delta method, and
    ``tau = -1 / slope`` carries::

        sigma_tau = tau^2 * sqrt( sum_i c_i^2 (sem_i / y_i)^2 )

    Samples are treated as INDEPENDENT, which is what a per-sample SEM states
    on its own; any sample-to-sample correlation left by the 100 kHz anti-alias
    filter would make this an UNDER-estimate, so the returned sigma is a lower
    bound on the measurement uncertainty in that sense.

    NaN under exactly the conditions that make ``_decay_efold_ms`` NaN, and
    additionally when a fitted sample carries a non-finite or non-positive
    SEM -- an uncertainty is not invented where the trace states none.
    """
    t_ms = np.asarray(t_ms, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    sem = np.asarray(sem, dtype=np.float64)
    good = np.isfinite(t_ms) & np.isfinite(y) & (y > max(floor, 0.0))
    if np.count_nonzero(good) < 8:
        return np.nan
    t_fit, y_fit, sem_fit = t_ms[good], y[good], sem[good]
    if not np.all(np.isfinite(sem_fit)) or np.any(sem_fit <= 0.0):
        return np.nan
    slope = np.polyfit(t_fit, np.log(y_fit), 1)[0]
    if not slope < 0.0:
        return np.nan
    dt = t_fit - t_fit.mean()
    s_tt = float(np.sum(dt * dt))
    if not s_tt > 0.0:
        return np.nan
    var_slope = float(np.sum((dt / s_tt) ** 2 * (sem_fit / y_fit) ** 2))
    tau = -1.0 / slope
    return float(tau * tau * np.sqrt(var_slope))


def _isat_decay_matrix(
    time_ms: np.ndarray,
    cells: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    ports: np.ndarray,
    excluded: dict[str, np.ndarray],
    excluded_reason: dict[str, np.ndarray],
    window_ms: tuple[float, float] = ISAT_DECAY_FIT_WINDOW_MS,
) -> dict[str, np.ndarray]:
    """Return the afterglow e-fold matrix over face x convention x port.

    ``cells`` maps ``(face, convention)`` -- one key per entry of
    ``ISAT_DECAY_MATRIX_FACES`` x ``ISAT_DECAY_MATRIX_CONVENTIONS`` -- to that
    cell's ``(mean, sem)`` traces, each shaped ``(port, time)`` on ``time_ms``.
    EVERY cell is fitted by the one recipe: a noise floor from its own final
    ``ISAT_DECAY_NOISE_TAIL_MS``, then ``_decay_efold_ms`` over
    ``window_ms`` on this grid's own clock.  Nothing about the fit knows which
    face or which radial average it was handed, which is what makes the nine
    numbers comparable with each other.

    ``excluded`` / ``excluded_reason`` are per FACE, keyed by face label: an
    exclusion is a property of the run's channel, so a face excluded on one
    convention is excluded on all three and the reason travels with it.
    """
    t = np.asarray(time_ms, dtype=np.float64)
    t0, t1 = float(window_ms[0]), float(window_ms[1])
    window = (t >= t0) & (t <= t1)
    tail = t >= t.max() - ISAT_DECAY_NOISE_TAIL_MS
    n_face = len(ISAT_DECAY_MATRIX_FACES)
    n_conv = len(ISAT_DECAY_MATRIX_CONVENTIONS)
    n_port = len(ports)

    tau = np.full((n_face, n_conv, n_port), np.nan, dtype=np.float64)
    tau_sem = np.full((n_face, n_conv, n_port), np.nan, dtype=np.float64)
    noise_floor = np.full((n_face, n_conv, n_port), np.nan, dtype=np.float64)
    n_fit = np.zeros((n_face, n_conv, n_port), dtype=np.int16)
    for fi, face in enumerate(ISAT_DECAY_MATRIX_FACES):
        for ci, convention in enumerate(ISAT_DECAY_MATRIX_CONVENTIONS):
            mean, sem = cells[(face, convention)]
            mean = np.asarray(mean, dtype=np.float64)
            sem = np.asarray(sem, dtype=np.float64)
            if mean.shape != (n_port, t.size) or sem.shape != mean.shape:
                raise ValueError(
                    f"decay matrix cell ({face}, {convention}) is shaped "
                    f"{mean.shape}/{sem.shape}, not {(n_port, t.size)}"
                )
            for p in range(n_port):
                with warnings.catch_warnings():
                    # An all-NaN tail is an expected state of an area-averaged
                    # row; _decay_noise_floor_a answers it explicitly.
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    floor = _decay_noise_floor_a(mean[p], tail)
                windowed = mean[p, window]
                noise_floor[fi, ci, p] = floor
                n_fit[fi, ci, p] = int(
                    np.count_nonzero(
                        np.isfinite(windowed) & (windowed > max(floor, 0.0))
                    )
                )
                tau[fi, ci, p] = _decay_efold_ms(t[window], windowed, floor)
                tau_sem[fi, ci, p] = _decay_efold_sigma_ms(
                    t[window], windowed, sem[p, window], floor
                )

    return {
        "tau_ms": tau,
        "tau_sem_ms": tau_sem,
        "noise_floor_a": noise_floor,
        "n_fit": n_fit,
        "window_ms": np.asarray([t0, t1], dtype=np.float64),
        "face": np.asarray(ISAT_DECAY_MATRIX_FACES),
        "convention": np.asarray(ISAT_DECAY_MATRIX_CONVENTIONS),
        "port": np.asarray(ports, dtype=np.int16),
        "excluded": np.stack(
            [np.asarray(excluded[f], dtype=np.bool_) for f in ISAT_DECAY_MATRIX_FACES],
            axis=0,
        ),
        "excluded_reason": np.stack(
            [np.asarray(excluded_reason[f]) for f in ISAT_DECAY_MATRIX_FACES],
            axis=0,
        ),
        "n_window": int(np.count_nonzero(window)),
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
    # The two per-CELL grids the flux-tube T_e rows stand on: the strict
    # measured mask, which says how far out each row reports its own
    # measurement, and the per-cell semi-quantitative marks, which say how
    # much of the flux-tube average sits at the diagnostic's limit.  Required
    # for the same reason as the rest of this list: a product exported without
    # them is indistinguishable from one where nothing was measured or marked.
    "te_masked",
    "te_semi_quantitative",
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
        "measured_te": np.asarray(te_group["te_masked"][()], dtype=np.float64),
        "semi_quantitative": np.asarray(
            te_group["te_semi_quantitative"][()], dtype=bool
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


def _plateau_current_a(
    current: np.ndarray,
    time_ms: np.ndarray,
    window_ms: tuple[float, float] = RAW_PLATEAU_WINDOW_MS,
) -> np.ndarray:
    """Per-shot plateau current, in A: the mean over the scoring plateau window.

    The window defaults to ``RAW_PLATEAU_WINDOW_MS``, inherited from the
    scoring convention rather than chosen here; it is a parameter only so a
    caller can state the window it is actually averaging over, which is what
    the artifact guard in ``_discharge_stats`` checks.  What this returns is an
    AMPLITUDE, not a timing quantity: it is what the shots differ in when one
    of them runs a hotter discharge than the rest.
    """
    window = (time_ms >= window_ms[0]) & (time_ms <= window_ms[1])
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
    plateau_window_ms: tuple[float, float] = RAW_PLATEAU_WINDOW_MS,
    run_artifacts=None,
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

    Before that plateau is taken, ``plateau_window_ms`` is checked against the
    declared run artifacts in ``config/may2026_run_artifacts.toml`` and the
    whole statistic is REFUSED if the window covers one.  ES1 run 05 carries a
    one-sample discharge-current excursion at t = 20.000 ms, 0.5 ms past the
    end of the default window, so the default passes; a window widened past it
    would otherwise average a discharge-termination switching transient into
    the drive plateau without saying so.  ``run_artifacts`` overrides the
    registry for tests and offline callers.
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
        refuse_artifacts_in_window(
            raw_run_ids,
            plateau_window_ms,
            "the raw discharge ensemble's plateau current, and the common "
            "crossing level derived from it",
            channel="discharge_current",
            artifacts=run_artifacts,
        )
        plateau_a = _plateau_current_a(
            current_raw_all, raw_time_ms, plateau_window_ms
        )
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
    rot180_isat_profile_path: Path = ROT180_ISAT_PROFILE_HDF5,
    area_calibration_path: Path = AREA_CALIBRATION_TOML,
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
        te_x_cm = te_group["x_cm"][()]
        te_filled_profiles = np.asarray(te_group["te_filled"][()], dtype=np.float64)
        if not np.allclose(te_x_cm, density_x_cm):
            raise ValueError(
                f"ES{experiment_set_id} T_e radial grid {te_x_cm} and density "
                f"radial grid {density_x_cm} differ; the flux-tube T_e rows "
                "weight one profile by the other and need one grid"
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
        radial=True,
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
        radial=True,
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
    # The afterglow line scans, reduced under the two AREA conventions.  The
    # geometric mean is taken on the profiles by the same function the
    # dead-time scans use, so the decay geomean and the drive geomean are the
    # same construction on two clocks.
    if not np.allclose(isat_decay["x_cm"], isat_decay_dn["x_cm"]):
        raise ValueError("Isat decay faces disagree on the x grid")
    decay_geomean_scans = _flow_symmetrized_profiles(
        {
            "port": isat_decay["port"],
            "run_id": isat_decay["run_id"],
            "x_cm": isat_decay["x_cm"],
            "time_ms": isat_decay["time_ms"],
            "isat_a": isat_decay["profile_mean_a"],
            "sem_a": isat_decay["profile_sem_a"],
            "source_channel": isat_decay["source_channel"],
        },
        {
            "port": isat_decay_dn["port"],
            "run_id": isat_decay_dn["run_id"],
            "x_cm": isat_decay_dn["x_cm"],
            "time_ms": isat_decay_dn["time_ms"],
            "isat_a": isat_decay_dn["profile_mean_a"],
            "sem_a": isat_decay_dn["profile_sem_a"],
            "source_channel": isat_decay_dn["source_channel"],
        },
        face_areas_cm2,
    )
    decay_face_profiles = {
        "upstream": (isat_decay["profile_mean_a"], isat_decay["profile_sem_a"]),
        "downstream": (
            isat_decay_dn["profile_mean_a"],
            isat_decay_dn["profile_sem_a"],
        ),
        "geomean": (decay_geomean_scans["profiles"], decay_geomean_scans["sem"]),
    }
    decay_reduced: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for face, (profiles, profile_sem) in decay_face_profiles.items():
        decay_reduced[(face, "ftavg")] = _flux_tube_series(
            profiles,
            isat_decay["x_cm"],
            profile_sem,
            subtract_background=False,
        )
        decay_reduced[(face, "column")] = _flux_tube_series(
            profiles,
            isat_decay["x_cm"],
            profile_sem,
            radius_cm=None,
            normalize_radius_cm=FLUX_TUBE_RADIUS_CM,
            subtract_background=False,
        )
    decay_matrix_cells = {
        ("upstream", "x0"): (isat_decay["mean_a"], isat_decay["sem_a"]),
        ("downstream", "x0"): (isat_decay_dn["mean_a"], isat_decay_dn["sem_a"]),
        ("geomean", "x0"): (
            isat_decay_geomean["geomean_a_per_cm2"],
            isat_decay_geomean["sem_a_per_cm2"],
        ),
    }
    for (face, convention), reduced in decay_reduced.items():
        decay_matrix_cells[(face, convention)] = (
            reduced["ftavg"],
            reduced["ftavg_sem"],
        )
        if convention == "ftavg":
            # The core band is taken before the quadrature's outer limit is
            # chosen, so it is the same number under both reductions; read it
            # off the flux-tube pass rather than reducing the profiles twice.
            decay_matrix_cells[(face, "core")] = (
                reduced["core"],
                reduced["core_sem"],
            )
    decay_matrix = _isat_decay_matrix(
        isat_decay["time_ms"],
        decay_matrix_cells,
        isat_decay["port"],
        {
            "upstream": isat_decay["excluded"],
            "downstream": isat_decay_dn["excluded"],
            "geomean": isat_decay["excluded"] | isat_decay_dn["excluded"],
        },
        {
            "upstream": isat_decay["excluded_reason"],
            "downstream": isat_decay_dn["excluded_reason"],
            "geomean": np.asarray(
                [
                    " | ".join(part for part in (up, dn) if part)
                    for up, dn in zip(
                        isat_decay["excluded_reason"],
                        isat_decay_dn["excluded_reason"],
                    )
                ]
            ),
        },
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

    # THE DENSITY ROWS OF RECORD ARE TAKEN ON THE UNSUBTRACTED PROFILE.  The
    # probe's zeroing already removed the instrumental baseline, from the end
    # of the shot where there is no plasma, so the ledger's edge-median
    # subtraction was removing plasma; see ftavg_background.
    density_ftavg = _flux_tube_series(
        density_profiles_m3, density_x_cm, subtract_background=False
    )
    # The WHOLE-COLUMN convention: the same reduction of the same profiles,
    # integrated to each sample's own column edge and then divided by the
    # tube's area rather than by the area integrated over.
    density_column = _flux_tube_series(
        density_profiles_m3,
        density_x_cm,
        radius_cm=None,
        normalize_radius_cm=FLUX_TUBE_RADIUS_CM,
        subtract_background=False,
    )
    # The same two reductions WITH the retired subtraction, exported under
    # explicit legacy names so a pre-ruling result stays reproducible.
    density_ftavg_legacy = _flux_tube_series(density_profiles_m3, density_x_cm)
    density_column_legacy = _flux_tube_series(
        density_profiles_m3,
        density_x_cm,
        radius_cm=None,
        normalize_radius_cm=FLUX_TUBE_RADIUS_CM,
    )

    # The flux-tube T_e rows.  T_e lives on the sweep-cycle clock and the
    # density on the inter-sweep dead-time clock, so the DENSITY -- which is
    # only the weight here -- is the grid that moves: it is interpolated onto
    # te_time_ms cell by cell, and the T_e rows stay on the clock every other
    # te_* field is already exported on.
    te_clock_density_m3 = _interp_onto_time_grid(
        density_profiles_m3, density.time_ms, te.time_ms
    )
    te_ftavg = _flux_tube_te_series(
        te_filled_profiles,
        te_clock_density_m3,
        density_x_cm,
        te_records["semi_quantitative"],
        subtract_background=False,
    )
    te_column = _flux_tube_te_series(
        te_filled_profiles,
        te_clock_density_m3,
        density_x_cm,
        te_records["semi_quantitative"],
        radius_cm=None,
        normalize_radius_cm=FLUX_TUBE_RADIUS_CM,
        subtract_background=False,
        trust_radius_cm=te_records["trust_radius_cm"],
        blend_radius_cm=te_records["trust_blend_cm"],
    )
    ftavg_coverage_cm = _measured_coverage_cm(
        te_records["measured_te"],
        density_x_cm,
        te_records["trust_radius_cm"],
        te_records["row_measured_cells"],
    )
    ftavg_prior_beyond_coverage = ~(ftavg_coverage_cm >= FLUX_TUBE_RADIUS_CM)
    # The column rows measure the SAME thing -- how far out the filled T_e
    # product reports its own measurement -- so the coverage radius is the same
    # number under both conventions.  Only what it is compared against moves:
    # the column integrates to te_column_edge_cm, not to the tube radius.
    column_coverage_cm = ftavg_coverage_cm
    column_prior_beyond_coverage = ~(column_coverage_cm >= te_column["edge"])

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
        # The comparand convention, as for the density rows: no background
        # subtraction (see ftavg_background), with the retired reduction kept
        # beside it under a _subtracted name.
        return (
            scans,
            _flux_tube_series(
                scans["isat_a"],
                scans["x_cm"],
                scans["sem_a"],
                subtract_background=False,
            ),
            _flux_tube_series(scans["isat_a"], scans["x_cm"], scans["sem_a"]),
        )

    upstream_scans, upstream_ftavg, upstream_ftavg_legacy = _face_ftavg(
        isat_profile_path, "upstream"
    )
    rot0_isat, isat_ftavg, isat_ftavg_legacy = _face_ftavg(
        rot0_isat_profile_path, "downstream"
    )
    geomean_scans = _flow_symmetrized_profiles(
        upstream_scans,
        rot0_isat,
        face_areas_cm2,
    )
    geomean_ftavg = _flux_tube_series(
        geomean_scans["profiles"],
        upstream_scans["x_cm"],
        geomean_scans["sem"],
        subtract_background=False,
    )
    geomean_ftavg_legacy = _flux_tube_series(
        geomean_scans["profiles"],
        upstream_scans["x_cm"],
        geomean_scans["sem"],
    )
    # The WHOLE-COLUMN convention for the same three Isat faces: the same
    # quadrature over the same comparand profiles, integrated to each sample's
    # own column edge and expressed over the tube's area, so the Isat rows
    # exist in all three conventions the density rows already do.
    upstream_column = _flux_tube_series(
        upstream_scans["isat_a"],
        upstream_scans["x_cm"],
        upstream_scans["sem_a"],
        radius_cm=None,
        normalize_radius_cm=FLUX_TUBE_RADIUS_CM,
        subtract_background=False,
    )
    isat_column = _flux_tube_series(
        rot0_isat["isat_a"],
        rot0_isat["x_cm"],
        rot0_isat["sem_a"],
        radius_cm=None,
        normalize_radius_cm=FLUX_TUBE_RADIUS_CM,
        subtract_background=False,
    )
    geomean_column = _flux_tube_series(
        geomean_scans["profiles"],
        upstream_scans["x_cm"],
        geomean_scans["sem"],
        radius_cm=None,
        normalize_radius_cm=FLUX_TUBE_RADIUS_CM,
        subtract_background=False,
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
                "shot's own peak would NOT be that statistic, and ES1 shows "
                "why: both run-05 shots carry a narrow current spike at the "
                "SAME sample, t = 20.000 ms, reaching 4.95 and 4.07 kA "
                "against a pack peak median of 3.04 kA -- a fixed-time "
                "excursion, not a hot discharge, since their plateau is "
                "normal, within 2 % of the pack median over the scoring "
                "window, which ends 0.5 ms before the spike.  An own-peak "
                "level reads that spike as the shot's amplitude and sets "
                "run 05's threshold near the top of its rise, which walks "
                "its crossings out and inflates the exported sd by nearly an "
                "order of magnitude; the common level is below the spike and "
                "the plateau window excludes it.  Nothing in the smoothed "
                "family is changed or superseded by these fields."
            ),
        }
    density_mean_cm3 = density.mean * DENSITY_SCALE_CM3
    density_ftavg_cm3 = density_ftavg["ftavg"] * M3_TO_CM3
    density_column_cm3 = density_column["ftavg"] * M3_TO_CM3
    density_ftavg_subtracted_cm3 = density_ftavg_legacy["ftavg"] * M3_TO_CM3
    density_column_subtracted_cm3 = density_column_legacy["ftavg"] * M3_TO_CM3
    # The raw inventory per unit length, cm^-1: the column row is that divided
    # by the tube's area, so multiplying it back is the same number the
    # quadrature integrated and not a second reduction.
    column_inventory_per_cm = (
        np.pi * FLUX_TUBE_RADIUS_CM**2 * density_column_cm3
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        column_over_ftavg_ratio = np.where(
            np.isfinite(density_ftavg_cm3) & (density_ftavg_cm3 > 0.0),
            density_column_cm3 / density_ftavg_cm3,
            np.nan,
        )
        # The Probe-A area calibration enters the core-band SEM as
        # |core mean| * relative (see _load_density_stats), so the RELATIVE
        # term is what transfers onto a row at a different level; recovering it
        # here keeps the two-term composition of density_total_sem_cm3 without
        # re-reading the calibration.
        calibration_relative = np.where(
            np.isfinite(density.mean) & (density.mean != 0.0),
            density.calibration_uncertainty / np.abs(density.mean),
            0.0,
        )
    density_column_radial_sem_cm3 = density_column["scatter_sem"] * M3_TO_CM3
    density_column_sem_cm3 = np.hypot(
        density_column_radial_sem_cm3,
        np.abs(density_column_cm3) * calibration_relative,
    )
    # The ES4 upstream-face bracket is exported at experiment set 4 only; every
    # other set writes the same field set it always has.
    es4_upstream_fields: dict[str, np.ndarray] = {}
    if experiment_set_id == ES4_UPSTREAM_SET_ID:
        es4_upstream_fields = _es4_upstream_rows(
            te_core_mean_ev=te.mean,
            te_time_ms=te.time_ms,
            density_mean_cm3=density_mean_cm3,
            density_ftavg_cm3=density_ftavg_cm3,
            density_core_count=density.count,
            density_time_ms=density.time_ms,
            isweep_path=isat_profile_path,
            isat_rot180_path=rot180_isat_profile_path,
            area_toml_path=area_calibration_path,
        )
        es4_upstream_fields.update(_es4_upstream_definitions())
    # The two conventions are reduced independently; refuse the pair that
    # a consumer cannot put an uncertainty on.  See the helper.
    _check_density_convention_pair(
        density_mean_cm3,
        density_ftavg_cm3,
        PORTS,
        density.time_ms,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema_version=np.array(SCHEMA_VERSION, dtype=np.int16),
        experiment_set_id=np.array(experiment_set_id, dtype=np.int16),
        experiment_label=np.array(experiment_label),
        port=PORTS,
        z_cm=density.z_cm,
        density_time_ms=density.time_ms,
        density_mean_cm3=density_mean_cm3,
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
        isat_decay_excluded=isat_decay["excluded"],
        isat_decay_excluded_reason=isat_decay["excluded_reason"],
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
        isat_decay_dn_excluded=isat_decay_dn["excluded"],
        isat_decay_dn_excluded_reason=isat_decay_dn["excluded_reason"],
        isat_decay_geomean_a_per_cm2=isat_decay_geomean["geomean_a_per_cm2"],
        isat_decay_geomean_sem_a_per_cm2=isat_decay_geomean["sem_a_per_cm2"],
        isat_decay_geomean_area_cm2=isat_decay_geomean["area_cm2"],
        isat_decay_geomean_pairing=isat_decay_geomean["pairing"],
        isat_decay_geomean_excluded=isat_decay["excluded"] | isat_decay_dn["excluded"],
        isat_decay_face_convention=np.array(
            "BOTH Mach-probe faces at x = 0, on the shared isat_decay_time_ms "
            "grid, at the same ports and the same runs (isat_decay_port, "
            "isat_decay_run_id apply to both).  isat_decay_mean_a / "
            "isat_decay_sem_a are the UPSTREAM face at rot-0 -- the 'i_sweep' "
            "channel, which is the Isat truth channel -- and "
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
        isat_decay_core_upstream_a=decay_reduced[("upstream", "ftavg")]["core"],
        isat_decay_core_upstream_sem_a=decay_reduced[("upstream", "ftavg")][
            "core_sem"
        ],
        isat_decay_core_dn_a=decay_reduced[("downstream", "ftavg")]["core"],
        isat_decay_core_dn_sem_a=decay_reduced[("downstream", "ftavg")]["core_sem"],
        isat_decay_core_geomean_a_per_cm2=decay_reduced[("geomean", "ftavg")]["core"],
        isat_decay_core_geomean_sem_a_per_cm2=decay_reduced[("geomean", "ftavg")][
            "core_sem"
        ],
        isat_decay_ftavg_upstream_a=decay_reduced[("upstream", "ftavg")]["ftavg"],
        isat_decay_ftavg_upstream_sem_a=decay_reduced[("upstream", "ftavg")][
            "ftavg_sem"
        ],
        isat_decay_ftavg_dn_a=decay_reduced[("downstream", "ftavg")]["ftavg"],
        isat_decay_ftavg_dn_sem_a=decay_reduced[("downstream", "ftavg")]["ftavg_sem"],
        isat_decay_ftavg_geomean_a_per_cm2=decay_reduced[("geomean", "ftavg")][
            "ftavg"
        ],
        isat_decay_ftavg_geomean_sem_a_per_cm2=decay_reduced[("geomean", "ftavg")][
            "ftavg_sem"
        ],
        isat_decay_column_upstream_a=decay_reduced[("upstream", "column")]["ftavg"],
        isat_decay_column_upstream_sem_a=decay_reduced[("upstream", "column")][
            "ftavg_sem"
        ],
        isat_decay_column_upstream_edge_cm=decay_reduced[("upstream", "column")][
            "edge"
        ],
        isat_decay_column_dn_a=decay_reduced[("downstream", "column")]["ftavg"],
        isat_decay_column_dn_sem_a=decay_reduced[("downstream", "column")][
            "ftavg_sem"
        ],
        isat_decay_column_dn_edge_cm=decay_reduced[("downstream", "column")]["edge"],
        isat_decay_column_geomean_a_per_cm2=decay_reduced[("geomean", "column")][
            "ftavg"
        ],
        isat_decay_column_geomean_sem_a_per_cm2=decay_reduced[("geomean", "column")][
            "ftavg_sem"
        ],
        isat_decay_column_geomean_edge_cm=decay_reduced[("geomean", "column")]["edge"],
        isat_decay_radial_definition=np.array(
            "THE AFTERGLOW DECAY IN ALL THREE RADIAL CONVENTIONS.  The same "
            "continuous afterglow traces behind isat_decay_mean_a are read at "
            "EVERY scanned x position, not only at x = 0, and reduced by the "
            "radial-averaging chain the dead-time line scans already use "
            "(_flux_tube_series on the unsubtracted, despiked profile; see "
            "ftavg_background, ftavg_definition and column_definition).  Each "
            "position gets its own high-current shot rejection by the same "
            "rule the x=0 rows use, so the x = 0 column of the line scan IS "
            "isat_decay_mean_a / isat_decay_sem_a bit for bit.  The families "
            "are: isat_decay_core_* (the CORE BAND -- the unweighted mean "
            "over core_x_min_cm <= x <= core_x_max_cm, a diameter line cut "
            "with no area weighting, the convention density_mean_cm3 and "
            "isat_ftavg_*_core_a are in), isat_decay_ftavg_* (flux-tube area "
            "mean to "
            "ftavg_radius_cm) and isat_decay_column_* (the column's inventory "
            "per unit length out to its own edge, expressed over the tube's "
            "area; the per-sample limit is isat_decay_column_*_edge_cm).  "
            "EVERY radial convention the plateau rows are exported in "
            "therefore has a decay counterpart.  The core band is taken "
            "BEFORE the quadrature chooses its outer limit, so it is one "
            "number under both area reductions and it survives at samples "
            "where the area rows do not; its SEM is the per-point shot SEM "
            "propagated through the unweighted band mean, "
            "sqrt(sum sem^2) / N over the retained band cells, the band "
            "companion of the _sem_a the area rows carry.  All "
            "three faces are carried under all three -- upstream, dn (downstream) "
            "and geomean, the last built from the two faces' AREA-NORMALIZED "
            "line scans by the same function as isat_ftavg_geomean_*, so it "
            "is in A cm^-2 and its core/ftavg/column rows are too.  Every row "
            "is on "
            "isat_decay_time_ms.  A face excluded by the late-afterglow "
            "probe-local-current registry is NaN at EVERY radius, so its "
            "core, flux-tube and column rows are NaN too (isat_decay_excluded "
            "/ "
            "isat_decay_dn_excluded, reasons in the matching _excluded_reason "
            "arrays).  These rows and isat_decay_mean_a / isat_decay_dn_mean_a "
            "/ isat_decay_geomean_a_per_cm2 are the TWELVE time series the "
            "e-fold matrix is fitted to."
        ),
        isat_decay_matrix_tau_ms=decay_matrix["tau_ms"],
        isat_decay_matrix_tau_sem_ms=decay_matrix["tau_sem_ms"],
        isat_decay_matrix_noise_floor_a=decay_matrix["noise_floor_a"],
        isat_decay_matrix_n_fit=decay_matrix["n_fit"],
        isat_decay_matrix_n_window=np.array(decay_matrix["n_window"]),
        isat_decay_matrix_window_ms=decay_matrix["window_ms"],
        isat_decay_matrix_face=decay_matrix["face"],
        isat_decay_matrix_convention=decay_matrix["convention"],
        isat_decay_matrix_port=decay_matrix["port"],
        isat_decay_matrix_excluded=decay_matrix["excluded"],
        isat_decay_matrix_excluded_reason=decay_matrix["excluded_reason"],
        isat_decay_matrix_definition=np.array(
            "THE AFTERGLOW E-FOLD MATRIX, indexed [face, convention, port] by "
            "isat_decay_matrix_face x isat_decay_matrix_convention x "
            "isat_decay_matrix_port, in ms, with isat_decay_matrix_tau_sem_ms "
            "beside it.  TWELVE cells per port: three probe faces (upstream, "
            "downstream, geomean) crossed with four conventions, which run "
            "outward from the axis.  "
            "NAMING, READ THIS FIRST: the convention labelled 'x0' is the "
            "x = 0 POINT trace -- isat_decay_mean_a, isat_decay_dn_mean_a and "
            "isat_decay_geomean_a_per_cm2, the rows the transport "
            "comparison's stage (iii) already fits -- and it is a DIFFERENT "
            "quantity from the repo's CORE-BAND convention, which is the "
            "unweighted mean over "
            "core_x_min_cm <= x <= core_x_max_cm and appears in this product "
            "as density_mean_cm3 and isat_ftavg_*_core_a.  Both are carried: "
            "the core band is the convention labelled 'core' "
            "(isat_decay_core_*), so 'x0' must not be read as it.  'ftavg' is "
            "the flux-tube area mean to "
            "ftavg_radius_cm (isat_decay_ftavg_*) and 'column' the whole-column "
            "inventory over the tube's area (isat_decay_column_*); see "
            "isat_decay_radial_definition for how the three radial rows are "
            "built.  The last three are the SAME three conventions the "
            "plateau rows are exported in, so a decay and a plateau level can "
            "be read in one convention.  "
            "THE FIT IS ONE RECIPE FOR ALL TWELVE CELLS, and it is the recipe "
            "the transport comparison applies to isat_decay_mean_a: a noise "
            "floor of 5 x 1.4826 x MAD over the trace's OWN final 5 ms, then "
            "an unweighted least-squares line through log(I) over "
            "isat_decay_matrix_window_ms on the isat_decay_time_ms clock, "
            "tau = -1/slope.  The floor actually used is "
            "isat_decay_matrix_noise_floor_a and the number of window samples "
            "that entered each fit is isat_decay_matrix_n_fit, against "
            "isat_decay_matrix_n_window in the window; where the two agree "
            "the floor removed nothing.  A cell whose trace has NO FINITE "
            "TAIL states no noise level and is fitted on positivity alone "
            "(floor 0.0) rather than against another trace's floor -- that is "
            "every upstream flux-tube and whole-column cell, whose radial "
            "quadrature reports nothing once the plasma is gone and the "
            "signed line scan sums non-positive, while its fit window stays "
            "fully populated (see _decay_noise_floor_a).  The 'core' row is "
            "not affected: the band mean is taken before the quadrature and "
            "is defined at every sample, so it keeps its own tail and its own "
            "floor.  A cell is NaN where "
            "fewer than 8 samples clear "
            "the positivity/noise-floor mask or where the fitted slope is not "
            "a decay; tau_sem is that fit's per-sample SEM propagated by the "
            "delta method, samples treated as INDEPENDENT, which the "
            "100 kHz anti-alias filter makes a LOWER BOUND, and it is NaN "
            "additionally where a fitted sample carries a non-positive or "
            "non-finite SEM.  Because every cell shares the recipe, the "
            "[:, 0, :] slice reproduces the stage (iii) measured tau of "
            "record exactly, and a difference between two cells is a "
            "difference between the traces and not between two fits.  "
            "isat_decay_matrix_excluded / _excluded_reason are per (face, "
            "port) -- an exclusion belongs to a run's channel, so it takes "
            "all four conventions of that face with it.  COMPARISON RULE: "
            "these are Isat e-folds; the interferometer decay "
            "(interf_decay_*) is a line-integrated DENSITY decay on its own "
            "clock and is not a comparand for them."
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
            "the two on one axis must carry this as a systematic.  This "
            "exporter also runs on every other experiment set: the offset "
            "above is carried into those exports UNMEASURED, not re-derived "
            "from each set's own traces, so a consumer relying on this field "
            "outside ES1 must treat the transfer to that set, not only the "
            "offset itself, as an assumption."
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
        density_sign_convention=np.array(
            "THE DENSITY IS SIGNED (schema v37).  processed/"
            "density_profiles_isweep.hdf5 stores the measured value of every "
            "cell it has a measurement for, negative cells included; NaN in "
            "n_e_m3 means no usable measurement and NEVER a negative one.  The "
            "retired convention wrote every non-positive cell as NaN -- a SIGN "
            "TEST, never a QC gate: the mask was exactly the set of cells whose "
            "Isat is negative (348/347/408/1218 cells of 10,200/10,200/5,100/"
            "5,100 at ES1/ES2/ES3/ES4, identical to the negative-Isat mask at "
            "every set).  Those cells are noise about zero at a radius where "
            "there is little current; deleting them dropped the downward half "
            "of the noise and redistributed each cell's 2 r dr quadrature "
            "weight onto the positive cells, which is the same upward bias as "
            "the background subtraction this product already retired.  The "
            "cells the sign test would have deleted are still identifiable, as "
            "n_e_sign_masked in the density product.  CONSEQUENCES, CARRIED "
            "RATHER THAN CLIPPED: (1) column_edge_cm no longer contracts when "
            "the skirt dips below zero, so the whole-column rows integrate the "
            "full scan; (2) density_mean_cm3, an unweighted core-band mean, is "
            "now a signed average and CAN BE NEGATIVE at a port whose column "
            "has decayed into the noise -- it is at 2 ES3 and 19 ES4 samples, "
            "all at the far end, and a consumer transferring the fractional "
            "error density_total_sem_cm3/density_mean_cm3 must not use it "
            "there; (3) the density is the WEIGHT behind te_ftavg_ev and "
            "te_column_ev, so those ratios are no longer convex combinations "
            "of their own T_e nodes and can read outside [min T, max T], and "
            "where the negative nodes cancel the positive ones the row is NaN. "
            " Measured at this vintage: under the flux-tube convention 0/1/1/1 "
            "samples leave their bound at ES1/ES2/ES3/ES4 and none is in the "
            "SCORING PLATEAU WINDOW, 15.0-19.5 ms (the window the transport "
            "comparison scores the plateau over, and the one "
            "RAW_PLATEAU_WINDOW_MS pins); under the whole-column convention "
            "2/5/1/2 leave it and 2/1/2/1 go NaN on a non-positive total "
            "weight, all at t <= 9 ms where there is nearly no plasma to "
            "weight with.  A FOURTH CONSEQUENCE DOES REACH THE PLATEAU: a "
            "signed weight can make the SEM's weighted variance NEGATIVE, "
            "which is not a scatter and is reported as NaN rather than "
            "floored at zero, and at ES3 p50 te_column_sem_ev is NaN at "
            "t = 16.0 and 17.0 ms -- 2 of the 5 T_e samples that set carries "
            "in 15.0-19.5 ms, its row's mean finite at both.  A consumer "
            "averaging that row's SEM over the plateau therefore averages 3 "
            "samples, not 5, and must say so; every other plateau NaN in "
            "these rows is ES4 p50, which carries no flux-tube or column row "
            "at all.  te_ftavg_weight_density_cm3 and "
            "te_column_weight_density_cm3 are "
            "the denominators, exported so a consumer can see when one is "
            "small.  Nothing is clipped back into range: a clip at zero is the "
            "sign test again under another name.  (4) A WHOLE COLUMN CAN SUM "
            "NON-POSITIVE, and at ES4 p50 it does at every sample (schema "
            "v41).  Such a column HAS NO CENTROID -- an intensity-weighted "
            "mean position is not defined for a non-positive intensity -- but "
            "it IS a measurement, and NaN would say the opposite.  THE ROW IS "
            "THEREFORE REPORTED ON THE ONE FOLD CENTRE THAT NEEDS NO "
            "INTENSITY: THE PORT'S GEOMETRIC AXIS, x = 0.  The same 2 r dr "
            "trapezoidal quadrature is formed about it and the row carries the "
            "SIGNED AREA MEAN sum(w n)/sum(w) -- density_ftavg_cm3 and "
            "density_column_cm3 in cm^-3, column_inventory_per_cm as "
            "pi * ftavg_radius_cm^2 times the second, and the Isat families "
            "isat_ftavg_upstream_a and isat_ftavg_a in A on the same rule.  IT "
            "IS IN THE UNITS OF EVERY OTHER SAMPLE OF ITS OWN ROW, so a "
            "decayed port can be read beside a port that has plasma; at ES4 "
            "p50 the plateau reads of order -1e11 cm^-3 against p41's +3e11, "
            "and density_mean_cm3, the unweighted core-band mean, reads the "
            "same order.  WHAT IT IS NOT: it is not folded about the column's "
            "own centre, because there is not one, so it is an AXIAL area "
            "mean and not the centroid-folded quantity the row carries "
            "elsewhere; and the whole-column row does not carry the tube-area "
            "renormalization there, which is why multiplying it by the tube "
            "area is the consistent inventory.  The companions that need the "
            "centroid or a spread stay NaN: density_ftavg_centroid_cm, "
            "density_column_sem_cm3 and density_column_radial_sem_cm3; "
            "column_edge_cm and te_column_edge_cm give the extent the weights "
            "were formed over.  column_over_ftavg_ratio is NOT a level at "
            "these samples and can be finite and NEGATIVE: the tube stops at "
            "ftavg_radius_cm while the column runs to the scan edge, so where "
            "the negative cells sit outside the tube the flux-tube mean comes "
            "out POSITIVE while the column mean is negative.  Measured at this "
            "vintage that is 2 ES3 and 2 ES4 samples, all at t <= 4.75 ms.  "
            "THE T_e ROWS STAY "
            "NaN: te_ftavg_ev, te_column_ev and their plain and prior-weight "
            "companions are ratios whose denominator is that same non-positive "
            "weight, and a temperature is not measured where there is no "
            "plasma to weight with.  The denominator is exported beside them "
            "-- te_ftavg_weight_density_cm3 and te_column_weight_density_cm3 "
            "carry that same signed area mean -- so a NaN T_e row says WHY it "
            "refused.  The rule is the SIGNED chain's: the legacy "
            "density_ftavg_subtracted_cm3 and density_column_subtracted_cm3 "
            "clip at zero and still read NaN there, because they reproduce the "
            "retired method as it behaved."
        ),
        ftavg_background=np.array(
            "NO BACKGROUND IS SUBTRACTED FROM ANY DENSITY ROW OF RECORD, AND "
            "THAT IS A RULING, NOT AN OMISSION.  The Isat baseline behind "
            "these profiles is already taken by subtracting the signal from "
            "the END OF THE SHOT, where there is no plasma, so a line scan "
            "carries no second background to remove -- and what an edge-median "
            "subtraction takes off the outer cells is therefore PLASMA, which "
            "at a 25 cm scan is exactly the cross-field plasma "
            "density_column_cm3 exists to count.  EVERY AREA-AVERAGED "
            "COMPARAND IN THIS PRODUCT IS NOW ON THAT RULING, so they are all "
            "commensurate with each other: density_ftavg_cm3, "
            "density_ftavg_core_cm3, density_ftavg_centroid_cm and every "
            "density_column_* row; the density WEIGHT behind te_ftavg_ev and "
            "te_column_ev; the ES4 es4_upstream_density_ftavg_cm3 bracket "
            "(all at schema v33); and isat_ftavg_upstream_a, isat_ftavg_a and "
            "isat_ftavg_geomean_a_per_cm2 with their SEM, core and centroid "
            "companions (v35).  All are taken on the UNSUBTRACTED, despiked "
            "profile; the clip at zero that the subtraction motivated is not "
            "applied to them either.  ON THE DENSITY GRID THE CLIP NOW HAS "
            "WORK TO DO AND IS STILL NOT DONE: the density product's own sign "
            "test was RETIRED (schema v37), so its far skirt carries the same "
            "genuinely negative cells the Isat scans do -- noise about zero -- "
            "and they enter the quadrature as measured.  In the vintage "
            "before that, the density product wrote every non-positive cell "
            "as NaN, which dropped it from the quadrature and redistributed "
            "its 2 r dr weight onto the positive cells: the same upward bias "
            "as the clip, by deletion instead of by lifting.  ON THE ISAT "
            "SCANS THE CLIP ALWAYS DID: the far skirt carries "
            "genuinely negative cells (349 of 10,200 on the ES1 upstream "
            "face), noise about zero at a radius where there is little "
            "current, and the retired path lifted every one of them to zero. "
            " That lift was a second, silent positive bias and it is gone "
            "too; a negative cell now enters the quadrature as measured.  It "
            "is also why an isat_ftavg_*_subtracted_* row is not uniformly "
            "below its comparand: where the baseline was itself zero, the "
            "lift is all the retired path did, and the legacy row can sit "
            "slightly above.  "
            "The core-band density_mean_cm3 was never background-subtracted, "
            "so it and density_ftavg_cm3 now differ by the weighting alone.  "
            "THE RETIRED CONVENTION, for continuity reads only: "
            "scalar baseline = the smaller of the two medians of the outer "
            f"{BACKGROUND_EDGE_POINTS} points on each side, clipped at zero "
            "and subtracted, profile then clipped at zero, taken from the "
            "effective-width ledger; a side whose outer points are all "
            "non-finite yields no median and is skipped, and a sample with "
            "neither side usable was NaN.  Its values survive as the five "
            "rows named in subtracted_legacy_definition and are NOT "
            "comparands.  NOTHING IN THIS PRODUCT STILL SUBTRACTS."
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
            "At rot-0 the i_sweep channel collects on the "
            "UPSTREAM probe face and is the Isat truth channel; the isat "
            "channel is the DOWNSTREAM face and under-reads because it sits "
            "in the probe body's flow shadow (the assignment reverses at "
            "rot-180).  isat_ftavg_upstream_* is therefore the "
            "correction-bearing family and is the face the density chain and "
            "the isat_decay_*/isat_drive_* families already use; isat_ftavg_* "
            "is the downstream face, retained as the effective-width ledger's "
            "rot-0 primary and as the face the paper read used.  "
            "The two faces' flux-tube corrections run in OPPOSITE directions "
            "with z and must never be ratioed against each other.  "
            "Three estimators with three roles: "
            "isat_ftavg_* is the downstream face, SHADOWED, biased low; "
            "isat_ftavg_upstream_* is the truth channel and the "
            "flow-ENHANCED conjugate of it, biased the other way; "
            "isat_ftavg_geomean_* is the flow-CANCELLED central estimator "
            "built from both, and is the one whose C(z) came out z-flat."
        ),
        density_ftavg_cm3=density_ftavg_cm3,
        density_ftavg_core_cm3=density_ftavg["core"] * M3_TO_CM3,
        density_ftavg_centroid_cm=density_ftavg["centroid"],
        density_ftavg_n_despiked=density_ftavg["n_despiked"],
        density_ftavg_subtracted_cm3=density_ftavg_subtracted_cm3,
        subtracted_legacy_definition=np.array(
            "LEGACY ROWS, NOT COMPARANDS.  Five rows carry the RETIRED "
            "edge-median background subtraction, one for each area-averaged "
            "comparand: density_ftavg_subtracted_cm3, "
            "density_column_subtracted_cm3, isat_ftavg_upstream_subtracted_a, "
            "isat_ftavg_subtracted_a and "
            "isat_ftavg_geomean_subtracted_a_per_cm2.  Each is its comparand "
            "computed the way it was before the ruling -- the effective-width "
            "ledger's baseline, the smaller of the two outer-edge medians "
            "clipped at zero, subtracted from every cell and the profile then "
            "clipped at zero -- and they exist only so a result quoted before "
            "the ruling can be reproduced and traced.  THE SUBTRACTION WAS AN "
            "ERROR: the Isat baseline is already taken by subtracting the "
            "signal from the END OF THE SHOT, where there is no plasma, so "
            "there is no background left for a line scan to remove and the "
            "edge medians it removed are plasma.  Only the value row of each "
            "family is kept; no legacy SEM, core, centroid or despike "
            "companion is exported, because the retired reduction is not "
            "something a new result may quote.  See ftavg_background.  The "
            "density rows were re-cut at schema v33 and the three Isat "
            "families at v35; a product at v31 or earlier carries the "
            "subtracted values under the COMPARAND names.  THE TWO DENSITY "
            "LEGACY ROWS MOVED AT v37 AND NO LONGER REPRODUCE A PRE-v37 "
            "VALUE: they are the legacy REDUCTION, not a frozen copy, and the "
            "density grid under them stopped deleting its negative cells.  "
            "Those cells shift the edge medians this reduction takes its "
            "baseline from, and the clip at zero then lifts them, so "
            "density_ftavg_subtracted_cm3 and density_column_subtracted_cm3 "
            "reproduce the retired METHOD over the corrected input, not the "
            "numbers the retired method once produced.  The three Isat legacy "
            "rows are unaffected -- their input never had a sign test."
        ),
        te_ftavg_time_ms=te.time_ms,
        te_ftavg_ev=te_ftavg["ftavg"],
        te_ftavg_sem_ev=np.hypot(
            te_ftavg["ftavg_sem"],
            np.nan_to_num(te_records["window_sem_ev"], nan=0.0),
        ),
        te_ftavg_radial_sem_ev=te_ftavg["ftavg_sem"],
        te_ftavg_plain_ev=te_ftavg["plain"],
        te_ftavg_plain_sem_ev=np.hypot(
            te_ftavg["plain_sem"],
            np.nan_to_num(te_records["window_sem_ev"], nan=0.0),
        ),
        te_ftavg_plain_radial_sem_ev=te_ftavg["plain_sem"],
        te_ftavg_weight_density_cm3=te_ftavg["weight_density"] * M3_TO_CM3,
        te_ftavg_centroid_cm=te_ftavg["centroid"],
        te_ftavg_n_despiked=te_ftavg["n_despiked"],
        te_ftavg_node_count=te_ftavg["node_count"],
        te_ftavg_semi_quantitative_count=te_ftavg["semi_quant_count"],
        te_ftavg_semi_quantitative_weight=te_ftavg["semi_quant_weight"],
        te_ftavg_definition=np.array(
            "THE FLUX-TUBE T_e COMPARAND, in eV, shaped (port, sample) on "
            "te_ftavg_time_ms, which IS te_time_ms.  te_ftavg_ev is the "
            "DENSITY-WEIGHTED area mean out to ftavg_radius_cm, "
            "int 2 pi r n T_e dr / int 2 pi r n dr: the quantity a 1D "
            "transport cell of that radius carries by construction, since its "
            "T_e is E_e / (3/2 n).  te_ftavg_plain_ev is the UNWEIGHTED area "
            "mean of the same profile over the same quadrature, "
            "int 2 pi r T_e dr / (pi R^2); it is printed beside the weighted "
            "row as the weighting's size and is NOT the comparand.  WHICH OF "
            "THE TWO IS LARGER IS NOT FIXED.  With w the quadrature weights "
            "and <.> the w-normalized average, te_ftavg_ev - te_ftavg_plain_ev "
            "= cov_w(n, T_e) / <n> exactly, so the weighted row sits ABOVE the "
            "plain one only where n and T_e correlate POSITIVELY across the "
            "disc -- the ordinary peaked-column case -- and BELOW it wherever "
            "they anti-correlate, which happens at a hollow density profile "
            "and wherever the density is at the noise floor and its shape is "
            "no longer the column's.  Both orderings occur in these products, "
            "sample by sample; it is the PLATEAU-WINDOW MEAN that is ordered "
            "plain < weighted < core at every port of every set, and a "
            "per-sample ordering must not be assumed.  The same caveat "
            "applies to te_mean_ev as an upper bound: a sample can read "
            "te_ftavg_ev above its own core-band row where the profile is "
            "flat or inverted across the band edge.  Both are "
            "reduced with the same machinery as density_ftavg_cm3 -- same "
            "despike gate, same trapezoidal quadrature closed at r = 0 and "
            "r = R, same centroid fold -- and BOTH fold about the DENSITY "
            "centroid (te_ftavg_centroid_cm), because the flux tube is "
            "centred on the column and T_e has no centroid of its own.  Two "
            "differences from the density chain, both deliberate: the T_e "
            "profile is NOT background-subtracted and NOT clipped at zero (it "
            "does not fall to zero at the scan edge, and the ledger's edge "
            "baseline would remove a real pedestal), and the quadrature nodes "
            "are the cells where the DENSITY is usable, so the weighted and "
            "the plain average sit on one node set.  te_ftavg_weight_density_"
            "cm3 is the denominator sum_i w_i n_i in cm^-3, the weighting the "
            "row actually used.  CLOCKS: T_e is measured on the sweep cycles "
            "and the density on the inter-sweep dead times, 0.375 ms apart on "
            "the same 0.5 ms pitch.  The WEIGHT is what moves: the density "
            "grid is linearly interpolated cell by cell onto te_time_ms, a "
            "target sample kept only where the source samples bracketing it "
            "are both finite, and the T_e rows stay on the clock te_mean_ev "
            "and te_sem_ev are already on.  te_ftavg_weight_density_cm3 is "
            "therefore the SAME reduction as density_ftavg_cm3 taken over an "
            "interpolated profile, which is not density_ftavg_cm3 "
            "interpolated: the reduction is nonlinear in the profile "
            "(centroid, scalar background, clipping), and the two part "
            "company by a few percent where the scan is clean and by tens of "
            "percent where it is not -- at the far ports of the high-puff "
            "sets, where the density is at the noise floor and the scan "
            "carries non-finite edge cells.  The field is exported so that "
            "disagreement can be MEASURED rather than assumed away; the "
            "weighted mean is a ratio and is insensitive to an overall scale "
            "of the weight, though not to its shape.  "
            "te_ftavg_node_count is the number of quadrature nodes carrying "
            "weight and te_ftavg_n_despiked the T_e cells the despike gate "
            "repaired."
        ),
        te_ftavg_sem_definition=np.array(
            "te_ftavg_sem_ev is te_ftavg_radial_sem_ev and the fit-window "
            "convention term te_window_sem_ev added in quadrature, the SAME "
            "composition as the core-band te_sem_ev, and "
            "te_ftavg_plain_sem_ev is that composition for the unweighted "
            "row.  WHAT THE FIRST TERM IS: the scatter of the retained radial "
            "cells about their own weighted average, propagated through the "
            "quadrature weights -- s^2 = sum a_i (T_i - m)^2 / (1 - sum a_i^2) "
            "and sem = sqrt(s^2 sum a_i^2) with a_i the normalized weights, "
            "which reduces exactly to std(ddof=1)/sqrt(N) at equal weights and "
            "is therefore the weighted generalization of te_radial_sem_ev.  It "
            "is a RADIAL-SCATTER term, not a shot or cycle SEM; the filled T_e "
            "product carries no per-cell uncertainty, so the per-point "
            "propagation the Isat flux-tube rows use (isat_ftavg_sem_"
            "definition) is NOT available here and no cycle-to-cycle term "
            "enters either row.  WHAT THE SECOND TERM IS NOT: te_window_sem_ev "
            "is defined over the CORE BAND only -- the export refuses a filled "
            "product whose window-spread band is not core_x_min_cm to "
            "core_x_max_cm -- so it is TRANSFERRED onto the flux-tube row "
            "unchanged and is not re-derived over the flux tube.  The "
            "sweep-systematics term for the band between the core and "
            "ftavg_radius_cm is MISSING, and the exported total understates "
            "the flux-tube uncertainty by however large it is."
        ),
        ftavg_coverage_cm=ftavg_coverage_cm,
        ftavg_prior_beyond_coverage=ftavg_prior_beyond_coverage,
        ftavg_coverage_definition=np.array(
            "HOW MUCH OF THE FLUX TUBE IS MEASURED, per port and per sample, "
            "for the T_e rows -- the only flux-tube rows with a prior in them. "
            " ftavg_coverage_cm is the outermost scan |x| carrying a "
            "QC-surviving MEASURED T_e cell, capped at that port's "
            "te_trust_radius_cm: beyond the trust radius the filled product "
            "mixes the measurement into the scrape-off-layer prior and past "
            "te_trust_blend_cm it reports the prior alone, so a measured cell "
            "out there is not what the product reports.  It is NaN where the "
            "row is prior-derived at that sample (te_row_measured == False), "
            "and it is stated in scan |x|, NOT in the quadrature's "
            "centroid-folded radius r = |x - x_c|.  THE TWO FRAMES ARE NOT "
            "INTERCHANGEABLE AND THE OFFSET IS NOT SMALL: the shift is "
            "te_ftavg_centroid_cm, whose magnitude runs to several cm at the "
            "clean sets (ES1 median 0.7 cm, 95th percentile 2.3 cm) and to "
            "TENS of cm at the high-puff sets, where the density profile is "
            "at the noise floor and its intensity centroid wanders -- ES3 "
            "reaches a 95th percentile of 6.7 cm and a maximum of 24.0 cm, "
            "past the scan edge itself, and ES4 a 95th percentile of 11.1 cm. "
            " Where the centroid is offset by that much, a coverage of "
            "10.0 cm in scan |x| does not map onto a 10.0 cm annulus of the "
            "integrated disc, and the flag is the reliable statement while "
            "the radius is indicative.  "
            "ftavg_prior_beyond_coverage is True wherever "
            "ftavg_coverage_cm < ftavg_radius_cm, including where the coverage "
            "is NaN: True means part of the disc the T_e average integrates "
            "over is the repo's SOL prior rather than a measurement of that "
            "port.  It is True at EVERY p50 row and across experiment sets 3 "
            "and 4, whose ports keep the historical 10 cm trust radius, and "
            "False only at the ES1/ES2 p11/p21/p29/p41 rows that adopted the "
            "18.415 cm aperture radius -- which is ftavg_radius_cm itself, so "
            "those rows are measured to the edge of the tube and no further.  "
            "The density and Isat flux-tube rows carry no such field because "
            "they are measured at every radius; where their scan does not "
            "reach ftavg_radius_cm the average is NaN rather than "
            "prior-filled.  te_ftavg_semi_quantitative_count and "
            "te_ftavg_semi_quantitative_weight are the second disclosure over "
            "the same disc: how many weight-carrying cells behind a T_e row "
            "are semi-quantitative (sub-eV T_e, or a fit-window spread at or "
            "above the threshold -- see te_semi_quantitative_rule) and what "
            "fraction of the total quadrature weight they carry."
        ),
        density_column_cm3=density_column_cm3,
        density_column_sem_cm3=density_column_sem_cm3,
        density_column_radial_sem_cm3=density_column_radial_sem_cm3,
        density_column_subtracted_cm3=density_column_subtracted_cm3,
        column_inventory_per_cm=column_inventory_per_cm,
        column_edge_cm=density_column["edge"],
        column_over_ftavg_ratio=column_over_ftavg_ratio,
        te_column_ev=te_column["ftavg"],
        te_column_sem_ev=np.hypot(
            te_column["ftavg_sem"],
            np.nan_to_num(te_records["window_sem_ev"], nan=0.0),
        ),
        te_column_radial_sem_ev=te_column["ftavg_sem"],
        te_column_plain_ev=te_column["plain"],
        te_column_plain_sem_ev=np.hypot(
            te_column["plain_sem"],
            np.nan_to_num(te_records["window_sem_ev"], nan=0.0),
        ),
        te_column_plain_radial_sem_ev=te_column["plain_sem"],
        # The whole-column counterpart of te_ftavg_weight_density_cm3: the
        # DENOMINATOR sum w n of te_column_ev, exported because the weight is
        # signed and a consumer has to be able to see when it is small or
        # negative (see density_sign_convention).
        te_column_weight_density_cm3=te_column["weight_density"] * M3_TO_CM3,
        te_column_edge_cm=te_column["edge"],
        te_column_prior_weight=te_column["prior_weight"],
        te_column_pure_prior_weight=te_column["pure_prior_weight"],
        te_column_plain_prior_weight=te_column["plain_prior_weight"],
        te_column_plain_pure_prior_weight=te_column["plain_pure_prior_weight"],
        column_coverage_cm=column_coverage_cm,
        column_prior_beyond_coverage=column_prior_beyond_coverage,
        column_definition=np.array(
            "THE WHOLE-COLUMN COMPARAND, the third radial-averaging convention "
            "beside the core-band line cut and the flux tube.  WHY IT EXISTS: "
            "plasma measured outside the cathode flux tube got there by "
            "cross-field transport a 1D model does not represent, so the "
            "model's single radial cell is compared against the WHOLE column's "
            "plasma rather than against the part of it that happens to lie "
            "inside the tube.  density_column_cm3 (cm^-3, on density_time_ms) "
            "is int_0^edge n(r) 2 pi r dr / (pi * ftavg_radius_cm^2): the "
            "column's inventory per unit length, expressed as the density a "
            "tube of radius ftavg_radius_cm would carry if all of it were "
            "inside.  It is NOT an average over the disc it was integrated "
            "over -- the numerator runs to column_edge_cm and the denominator "
            "is the TUBE's area -- which is exactly why it exceeds "
            "density_ftavg_cm3 wherever the column is wider than the tube.  "
            "THE INTEGRAND IS THE PROFILE AS MEASURED: nothing is subtracted "
            "from it and nothing is clipped.  The Isat baseline was already "
            "taken from the end of the shot, where there is no plasma, so "
            "there is no background left in a line scan to remove, and the "
            "edge-median subtraction this chain used to apply was removing "
            "the very cross-field plasma this convention counts (see "
            "ftavg_background; the retired values survive as "
            "density_column_subtracted_cm3).  density_column_cm3 is therefore "
            "THE MEASURED INVENTORY OUT TO THE SCAN EDGE.  It remains a lower "
            "bound in ONE respect only, and a different one: the scan stops at "
            "|x| = 25 cm, so any plasma beyond the scan edge is not in it, and "
            "this product cannot say how much that is.  "
            "column_inventory_per_cm is the numerator itself, in cm^-1, for "
            "an interferometer cross-check; it is pi * ftavg_radius_cm^2 "
            "times density_column_cm3 by construction and is not a second "
            "reduction.  column_over_ftavg_ratio is "
            "density_column_cm3 / density_ftavg_cm3 PER SAMPLE: the factor by "
            "which the column's inventory exceeds the tube's, one plus the "
            "share outside the tube.  THE LEVEL IS THE RATIO OF THE PLATEAU "
            "MEANS, not the plateau mean of this field; the per-sample field "
            "is for time-resolved inspection, and where the denominator is at "
            "the noise floor NEITHER is a level -- at ES4 p50 the mean of the "
            "per-sample ratios is 1.41 over the one plateau sample that "
            "survives while the ratio of the plateau means is 0.85, an "
            "implied share outside the tube of -18 %.  te_column_ev (eV, on "
            "te_time_ms) is the DENSITY-WEIGHTED mean over the SAME extent, "
            "int n T_e 2 pi r dr / int n 2 pi r dr, which is a 1D cell's "
            "E_e / (3/2 n); te_column_plain_ev is the unweighted area mean "
            "over that extent, int T_e 2 pi r dr / (pi * edge^2), printed "
            "beside it and NOT scored.  Both are ratios over one node set, so "
            "neither carries the tube-area normalization the density row does; "
            "both sit on the UNSUBTRACTED density's nodes and centroid, and "
            "the weighted one is weighted by it.  EVERYTHING ELSE IS THE "
            "FLUX-TUBE CHAIN UNCHANGED: same despike gate, same fold about "
            "the DENSITY centroid "
            "(density_ftavg_centroid_cm and te_ftavg_centroid_cm are the "
            "centroids these rows fold about too), same trapezoidal quadrature "
            "closed at r = 0 and at the outer limit, and the same "
            "interpolation of the density weight onto te_time_ms.  Isat gets "
            "NO column row: the scorer synthesises the Isat comparand as "
            "n sqrt(T_e) from the two rows above, so a third measured Isat "
            "convention would be a fourth way to say the same thing.  The "
            "three conventions are NOT interchangeable and a result must say "
            "which one it quoted -- see ftavg_comparand_map.  WHERE THE COLUMN "
            "SUMS NON-POSITIVE there is no centroid, so the quadrature is "
            "formed about the GEOMETRIC AXIS x = 0 instead, and "
            "density_column_cm3, column_inventory_per_cm and "
            "density_ftavg_cm3 carry the SIGNED AREA MEAN sum(w n)/sum(w) "
            "instead of NaN -- in the row's own units, so it can be read "
            "beside a port that has plasma, but AXIAL rather than "
            "centroid-folded and without the tube-area renormalization -- "
            "while te_column_ev and te_ftavg_ev stay NaN with "
            "te_column_weight_density_cm3 and te_ftavg_weight_density_cm3 "
            "carrying that same signed mean as the denominator that refused.  "
            "This is the ES4 p50 state at every sample; that port is not "
            "scored.  See density_sign_convention."
        ),
        column_edge_definition=np.array(
            "WHERE THE COLUMN INTEGRAL STOPS, per port and per sample.  "
            "column_edge_cm is the extent behind density_column_cm3 and "
            "column_inventory_per_cm, on density_time_ms; te_column_edge_cm is "
            "the extent behind te_column_ev and te_column_plain_ev, on "
            "te_time_ms, and the two differ because the T_e rows are reduced "
            "over the density grid INTERPOLATED onto the T_e clock and over "
            "the nodes where both profiles are usable.  THE DEFINITION, "
            "plainly: THE EDGE IS THE SCAN LIMIT, in the centroid-folded "
            "frame -- the outermost folded radius r = |x - x_c| at which the "
            "scan still carries a retained cell.  Retained means the cell "
            "carries a measurement: the density chain writes a cell as NaN "
            "only where there is no usable measurement, and since the sign "
            "test was retired (schema v37) a negative far-skirt cell is a "
            "measurement and is retained.  The edge is therefore the "
            "|x| = 25 cm scan end at essentially every sample, plus the "
            "per-sample centroid offset: 23 to 28 cm in practice.  Under the "
            "pre-v37 vintage it could instead be the outermost POSITIVE cell, "
            "which pulled the edge inward wherever the skirt had dipped below "
            "zero.  IT IS NOT A MEASURED "
            "COLUMN BOUNDARY.  The profile is NOT required to have fallen to "
            "anything there, and nothing is subtracted to make it look as "
            "though it had: the density at the scan edge is whatever the "
            "probe measured, and it is integrated as plasma (see "
            "ftavg_background).  An earlier vintage of this field claimed the "
            "edge was where the profile 'reaches the background ledger'; that "
            "was circular -- the ledger DEFINED its baseline from the "
            "outermost cells of this same scan, so the subtracted profile sat "
            "at zero there by construction at any scan extent whatever -- and "
            "the ledger is retired from this chain.  WHAT THE SCAN LIMIT DOES "
            "COST: plasma beyond |x| = 25 cm is outside every column row, and "
            "this product cannot say how much there is.  A consumer wanting "
            "the column integrated to a DECLARED radius, rather than to the "
            "scan's own end, cannot get it from this product.  The edge is "
            "stated in the CENTROID-FOLDED radius, not "
            "in scan |x|; column_coverage_cm is stated in scan |x| (see "
            "column_coverage_definition), and the offset between the two "
            "frames is te_ftavg_centroid_cm."
        ),
        column_sem_definition=np.array(
            "THE SAME ERROR MODEL THE FLUX-TUBE ROWS CARRY, over the column "
            "extent.  te_column_sem_ev is te_column_radial_sem_ev and the "
            "fit-window convention term te_window_sem_ev added in quadrature, "
            "and te_column_plain_sem_ev is that composition for the unweighted "
            "row.  The radial term is the scatter of the retained radial cells "
            "about their own weighted average, propagated through the "
            "quadrature weights -- s^2 = sum a_i (T_i - m)^2 / (1 - sum a_i^2), "
            "sem = sqrt(s^2 sum a_i^2) with a_i the normalized weights -- which "
            "reduces exactly to std(ddof=1)/sqrt(N) at equal weights.  IT IS A "
            "RADIAL-SCATTER TERM, not a shot or cycle SEM; the filled T_e "
            "product carries no per-cell uncertainty, so no cycle-to-cycle "
            "term enters either row.  THE WINDOW TERM IS TRANSFERRED AND "
            "UNDERSTATES: te_window_sem_ev is defined over the CORE BAND ONLY "
            "-- the export refuses a filled product whose window-spread band "
            "is not core_x_min_cm to core_x_max_cm -- so it is carried onto "
            "the column row unchanged rather than re-derived over it.  The "
            "sweep-systematics term for everything between the core band and "
            "column_edge_cm is MISSING, and the exported total understates the "
            "column uncertainty by however large it is; the gap is WIDER here "
            "than for the flux-tube rows, because the column extends further.  "
            "density_column_sem_cm3 is the same two-term composition as "
            "density_total_sem_cm3: density_column_radial_sem_cm3, the same "
            "weighted radial-scatter statistic (the weighted generalization of "
            "density_radial_sem_cm3, carrying the tube-area normalization the "
            "row itself carries), in quadrature with the Probe-A area "
            "calibration entered as the RELATIVE uncertainty it is, i.e. the "
            "core-band calibration term divided by the core-band mean and "
            "applied to density_column_cm3.  It is NOT the core-band "
            "fractional total error transferred whole, which is what "
            "ftavg_comparand_map tells a consumer to do with density_ftavg_cm3 "
            "and which gives a different number; a result must say which it "
            "used.  NEITHER SEM CONTAINS THE SCAN-EXTENT CAVEAT: plasma "
            "beyond the |x| = 25 cm scan edge is in no column row and is not "
            "a scatter term (see column_edge_definition)."
        ),
        column_coverage_definition=np.array(
            "HOW MUCH OF THE COLUMN IS MEASURED, per port and per sample, for "
            "the T_e rows -- the only column rows with a prior in them.  "
            "column_coverage_cm IS ftavg_coverage_cm, the same number: the "
            "outermost scan |x| carrying a QC-surviving MEASURED T_e cell, "
            "capped at that port's te_trust_radius_cm, NaN where the row is "
            "prior-derived at that sample.  What the column convention changes "
            "is only what the coverage is compared AGAINST: "
            "column_prior_beyond_coverage is True wherever column_coverage_cm "
            "< te_column_edge_cm, including where the coverage is NaN.  IT IS "
            "TRUE AT EVERY PORT OF EVERY EXPERIMENT SET, without exception: "
            "the highest trust radius in the repo is ftavg_radius_cm = "
            "18.415 cm (ES1/ES2 p11/p21/p29/p41) and every other port keeps "
            "the historical 10 cm, while the column integral runs past 23 cm, "
            "so part of the disc every te_column_ev integrates over is the "
            "repo's scrape-off-layer prior rather than a measurement of that "
            "port.  HOW MUCH OF IT IS THE PRIOR IS A NUMBER, NOT AN "
            "ADJECTIVE, AND IT IS SET- AND PORT-CONDITIONAL: "
            "te_column_prior_weight is the share of the row's own "
            "density-weighted quadrature weight (sum w n) carried by cells at "
            "|x| beyond that port's te_trust_radius_cm, per port and per "
            "sample, and te_column_pure_prior_weight the share beyond "
            "te_trust_blend_cm, where the filled product reports the prior "
            "alone.  THE TWO DENSITY-WEIGHTED SHARES ARE RATIOS OF TWO SIGNED "
            "SUMS (schema v37) and are not confined to [0, 1]: where the cells "
            "beyond the radius are noise about zero the numerator can go "
            "negative.  Measured at this vintage, every sample of the "
            "15.0-19.5 ms scored window is inside [0, 1] at all four sets, and "
            "the excursions are 5 / 8 / 1 / 1 samples at ES1 / ES2 / ES3 / ES4 "
            "of 198 / 199 / 93 / 75 finite, all at t <= 9 ms where there is "
            "nearly no plasma to weight with.  Over the 15.0-19.5 ms plateau "
            "those shares read: ES1 "
            "0.236 / 0.223 / 0.268 / 0.212 / 0.668 at p11 / p21 / p29 / p41 / "
            "p50, ES2 0.270 / 0.201 / 0.254 / 0.228 / 0.688, ES3 0.735 / "
            "0.680 / 0.700 / 0.600 / 0.504, ES4 0.698 / 0.619 / 0.617 / 0.419 "
            "/ NaN (ES4 p50 carries no column row at all: its whole scan sums "
            "non-positive over the plateau); pure-prior at ES1, 0.096 / 0.119 "
            "/ 0.173 / 0.131 / 0.339.  At the eight ES1/ES2 aperture ports the "
            "prior carries "
            "20-27 % of the weight; at every p50 row and across sets 3 and 4 "
            "it carries 42-74 %, and AT ELEVEN OF THE TWENTY PORT-ROWS IT IS "
            "NOT A SMALL CORRECTION -- those rows are substantially a "
            "statement about the repo's SOL prior and must not be read as a "
            "measurement of that port's column T_e.  The unweighted row is "
            "worse again: te_column_plain_prior_weight and "
            "te_column_plain_pure_prior_weight are the same shares of sum w, "
            "and they run 0.42 to 0.85, which is one reason the plain row is "
            "not the comparand.  These shares are taken on the UNSUBTRACTED "
            "density weight (schema v33): the retired background subtraction "
            "held the outer cells down, so every share here is LARGER than "
            "the same statistic taken before the ruling -- ES1 p11 0.236 "
            "against 0.218.  The "
            "flag is the reliable statement and the coverage radius is "
            "indicative, for the same frame reason ftavg_coverage_definition "
            "gives: coverage is in scan |x| and the edge is in the "
            "centroid-folded radius.  The prior-weight masks are in scan |x| "
            "too.  density_column_cm3 and column_inventory_per_cm carry no "
            "such field: the density is measured at every radius of the scan."
        ),
        ftavg_comparand_map=np.array(
            "WHICH FIELD CARRIES EACH SCORED ROW IN EACH OF THE THREE "
            "RADIAL-AVERAGING CONVENTIONS, so a consumer does not have to "
            "choose.  (1) CORE, the legacy line cut, an unweighted arithmetic "
            "mean of the 51-point scan over core_x_min_cm to core_x_max_cm, "
            "carrying no radial area weighting at all: density_mean_cm3 with "
            "density_total_sem_cm3, te_mean_ev with te_sem_ev.  (2) FTAVG, the "
            "flux tube, int 2 pi r f dr / (pi R^2) out to R = ftavg_radius_cm, "
            "the disc the model's single radial cell occupies: "
            "density_ftavg_cm3 (cm^-3), the plain area mean, with the "
            "core-band fractional error density_total_sem_cm3 / "
            "density_mean_cm3 transferred onto it; te_ftavg_ev (eV), the "
            "density-weighted area mean, with te_ftavg_sem_ev; "
            "isat_ftavg_upstream_a (A), the plain area mean of the measured "
            "Isat(r) on the UPSTREAM probe face, with "
            "isat_ftavg_upstream_sem_a -- that is the truth channel and the "
            "face the density chain reads; isat_ftavg_a is the SHADOWED "
            "downstream face and isat_ftavg_geomean_a_per_cm2 the "
            "flow-cancelled estimator in A cm^-2 (see ftavg_face_ruling).  "
            "(3) COLUMN, the whole measured column's inventory per unit "
            "length divided by the TUBE's area, the comparand for a model tube "
            "asked to carry all of the plasma including what cross-field "
            "transport put outside it: density_column_cm3 (cm^-3) with "
            "density_column_sem_cm3, te_column_ev (eV) with te_column_sem_ev, "
            "and NO Isat row -- the scorer synthesises Isat as n sqrt(T_e) "
            "from the two.  column_over_ftavg_ratio is the share of the plasma "
            "outside the tube and column_inventory_per_cm the raw line "
            "inventory in cm^-1.  See column_definition, column_edge_definition, "
            "column_sem_definition and column_coverage_definition.  NO ROW IN "
            "ANY OF THE THREE SUBTRACTS A BACKGROUND (schema v35; see "
            "ftavg_background), so they are commensurate with each other and "
            "differ only by the radial weighting and the extent -- but they "
            "are still NOT interchangeable and a result must say which one it "
            "quoted."
        ),
        isat_ftavg_geomean_time_ms=upstream_scans["time_ms"],
        isat_ftavg_geomean_a_per_cm2=geomean_ftavg["ftavg"],
        isat_ftavg_geomean_subtracted_a_per_cm2=geomean_ftavg_legacy["ftavg"],
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
        isat_ftavg_upstream_subtracted_a=upstream_ftavg_legacy["ftavg"],
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
            "is in isat_ftavg_upstream_source_channel).  Is the Isat truth "
            "channel because the opposite face collects in the "
            "probe body's flow shadow and under-reads.  This is the same face "
            "as the density chain behind density_ftavg_cm3 and as "
            "isat_decay_*/isat_drive_*, so this family is the "
            "correction-bearing one and pairs with them without a face change. "
            "See ftavg_face_ruling."
        ),
        isat_ftavg_time_ms=rot0_isat["time_ms"],
        isat_ftavg_a=isat_ftavg["ftavg"],
        isat_ftavg_subtracted_a=isat_ftavg_legacy["ftavg"],
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
            "truth channel is the upstream face -- "
            "use isat_ftavg_upstream_* for a correction.  Retained because it "
            "is the effective-width ledger's rot-0 primary and the face the "
            "paper read measured.  It is NOT the face behind "
            "isat_decay_*, isat_drive_* or the density chain, and its "
            "flux-tube correction runs the opposite way with z."
        ),
        isat_ftavg_sem_definition=np.array(
            "per-point shot SEM (isat_a_std / sqrt(n_shots_used)) propagated "
            "through the quadrature weights in quadrature, points treated as "
            "independent.  Since schema v35 there is no background term to "
            "carry: nothing is subtracted from these rows (see "
            "ftavg_background).  This is a shot SEM and is NOT commensurate "
            "with "
            "density_total_sem_cm3, which is a radial-scatter SEM plus the "
            "Probe-A area calibration."
        ),
        isat_column_upstream_a=upstream_column["ftavg"],
        isat_column_upstream_sem_a=upstream_column["ftavg_sem"],
        isat_column_upstream_edge_cm=upstream_column["edge"],
        isat_column_a=isat_column["ftavg"],
        isat_column_sem_a=isat_column["ftavg_sem"],
        isat_column_edge_cm=isat_column["edge"],
        isat_column_geomean_a_per_cm2=geomean_column["ftavg"],
        isat_column_geomean_sem_a_per_cm2=geomean_column["ftavg_sem"],
        isat_column_geomean_edge_cm=geomean_column["edge"],
        isat_column_definition=np.array(
            "THE WHOLE-COLUMN CONVENTION FOR THE THREE Isat FACES, the twin of "
            "the isat_ftavg_* families and the Isat counterpart of "
            "density_column_cm3.  Same profiles, same despiking, same "
            "comparand chain (NO background subtraction -- see "
            "ftavg_background), same quadrature; only the outer limit and the "
            "normalization move.  The integral runs from the profile centroid "
            "to that sample's own COLUMN EDGE, which is the scan limit in the "
            "folded frame and not a measured column boundary "
            "(isat_column_upstream_edge_cm, isat_column_edge_cm and "
            "isat_column_geomean_edge_cm, per port and per sample -- the same "
            "rule as column_edge_cm), and the result is divided by "
            "pi * ftavg_radius_cm^2 rather than by the area integrated over.  "
            "Each row is therefore the face's CURRENT PER UNIT LENGTH "
            "expressed as the current density the transport model's tube would "
            "carry if all of it were inside the tube, in A for the two single "
            "faces and A cm^-2 for the geomean, on isat_ftavg_upstream_time_ms "
            "/ isat_ftavg_time_ms.  Their SEMs are composed exactly as their "
            "flux-tube twins' are (see isat_ftavg_sem_definition): the "
            "per-point shot SEM propagated through the quadrature weights in "
            "quadrature, points treated as independent.  Every OTHER companion "
            "of these rows is identical to the flux-tube family's by "
            "construction and is NOT duplicated here -- the port roster, run "
            "ids, source file and channel, the face rulings, the despike "
            "counts, the centroid and the core-band companion are all taken "
            "before the outer limit is chosen, so read them from "
            "isat_ftavg_upstream_*, isat_ftavg_* and isat_ftavg_geomean_*.  "
            "The two faces must not be ratioed against each other "
            "(isat_ftavg_face), and the column and flux-tube rows are NOT "
            "interchangeable: a consumer must state which convention a number "
            "came from (see column_definition, column_edge_definition)."
        ),
        port_map=np.array(port_map),
        **discharge_raw_fields,
        **es4_upstream_fields,
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
    parser.add_argument(
        "--rot180-isat-profiles",
        type=Path,
        default=ROT180_ISAT_PROFILE_HDF5,
        help="rot-180 Isat-channel dead-time line-scan product; the UPSTREAM "
             "probe face at rot-180.  Read by the ES4 upstream bracket only, "
             "and only when --experiment-set 4 is selected",
    )
    parser.add_argument(
        "--area-calibration",
        type=Path,
        default=AREA_CALIBRATION_TOML,
        help="per-probe collecting-electrode area calibration.  Read by the "
             "ES4 upstream bracket only, and only when --experiment-set 4 is "
             "selected",
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
        args.rot180_isat_profiles,
        args.area_calibration,
    )


if __name__ == "__main__":
    main()
