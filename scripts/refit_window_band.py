"""Per-cell fit-window re-fits across the measurement band (the D-i product).

``refit_sweep_windows.py`` asks how far a port's ``T_e`` moves when the
electron-retarding fit window is varied, at ``x = 0`` only.  This script asks
the same question CELL BY CELL across the band that the filled ``T_e`` product
was asked to treat as measurement-authoritative, so that the trust extension is
conditioned on a measured window-convention sensitivity rather than on the
assumption that the core's window stability carries outward.

Protocol (PRE-DECLARED 2026-08-20, before the numbers existed)
--------------------------------------------------------------
* the SAME window family as the ``x = 0`` product: ``p_low`` in
  {3, 8, 15, 25, 35} percent x ``f_high`` in {0.05, 0.10, 0.15, 0.30, 0.50},
  ion branch held fixed at the pipeline defaults, imported from
  ``refit_sweep_windows.refit_sweep`` rather than restated;
* rot-0 runs of experiment sets 1 and 2, every port;
* every plateau cycle in 10--19.5 ms and every shot -- no subsampling;
* cells ``BAND_MIN_CM < |x| <= BAND_MAX_CM``, plus ``x = 0`` retained per port
  as an internal control against the published ``x = 0`` product;
* metric per cell ``dln_te_window = ln(max / min)`` over the 5 x 5 grid of
  nanmedian-over-sweeps ``T_e``, exactly the published product's definition;
* criterion, fixed before the first look: a port passes when its in-band MEDIAN
  ``dln_te_window`` is below ``CRITERION_DLN``; the full per-cell distribution
  is reported either way and the criterion is not adjusted afterwards.

Cell sets and rotations
-----------------------
The pass fits the band above by default.  ``--cells core`` fits the CORE cells
instead (``|x| <= CORE_MAX_CM``, the ``X_CORE_CM`` of ``fit_te_spatial.py``),
which is the cell set a port's own ``T_e`` row is built from, and ``--sets`` /
``--ports`` / ``--rotation-deg`` select which runs are walked.  Every one of
those defaults to the band pass described above, so the default invocation is
the D-i product and nothing else.

A pass that moves ANY of those four flags -- ``--cells``, ``--sets``,
``--ports``, ``--rotation-deg`` -- must be given all three output paths, because
the default ones are the band products ``fit_te_spatial.py`` consumes and
nothing in those files would let it notice the substitution; the pass REFUSES
to start otherwise.  Its checkpoints follow ``--output`` rather than staying in
``processed/``.  So a core pass reads, in full::

  MPLCONFIGDIR=.matplotlib python scripts/refit_window_band.py \
      --cells core --sets 4 --ports 21,29,41 --rotation-deg 0 \
      --output   <outside processed/>/window_refit_band_es4_core_rot0.hdf5 \
      --summary  <outside processed/>/window_refit_band_es4_core_rot0_summary.csv \
      --metadata <outside processed/>/window_refit_band_es4_core_rot0_metadata.json

Outputs
-------
processed/window_refit_band.hdf5
  The full product, including the per-cell 5 x 5 ``T_e`` window grids.  Large
  and regenerable, so it is gitignored like the other derived HDF5 outputs.

processed/window_refit_band_summary.csv
  One row per cell: the per-cell ``dln_te_window``, ``x_cm``, the default-window
  median ``T_e``, and the sweep count.  Small and TRACKED, because the filled
  ``T_e`` product conditions on it -- ``fit_te_spatial.py`` reads this file to
  mark semi-quantitative cells, so it has to be present in a fresh checkout.

processed/window_refit_band_metadata.json
  Protocol, criterion, adjudication reference and the per-port summary.  Also
  tracked, following ``isweep_frontside_arc_shot_exclusions_metadata.json``.
  Its ``numerics_vintage`` block records the interpreter and the numpy / scipy /
  h5py versions the run was produced under, because the product's last digits
  are a function of them; ``scripts/verify_window_band_product.py`` is the
  check to run when regenerating under a different vintage.

Usage
-----
  MPLCONFIGDIR=.matplotlib python scripts/refit_window_band.py

Each ``(set, port)`` is checkpointed to an ``npz`` in ``--work-dir`` as it
completes and an existing checkpoint is reused, so an interrupted pass resumes
and the archival step can be re-run without repeating ~30 minutes of raw-trace
reads.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _datetime
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bapsf_lapd import ChannelKind, LapdDataset  # noqa: E402

from bapsf_lapd.filtering import butterworth_lowpass  # noqa: E402


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: The x = 0 window-refit product.  Its ``refit_sweep`` and its window family
#: are imported rather than restated so the two products cannot drift apart.
_rw = _load_script("refit_sweep_windows", ROOT / "scripts" / "refit_sweep_windows.py")
_pls = _rw.pls

OUTPUT_HDF5 = ROOT / "processed" / "window_refit_band.hdf5"
OUTPUT_SUMMARY = ROOT / "processed" / "window_refit_band_summary.csv"
OUTPUT_METADATA = ROOT / "processed" / "window_refit_band_metadata.json"
#: Checkpoint directory name, resolved beside whichever product is written.
WORK_DIR_NAME = "window_refit_band_work"
WORK_DIR = ROOT / "processed" / WORK_DIR_NAME

SETS = (1, 2)
PLATEAU_MS = (10.0, 19.5)
CLIP_US = 10.0
FILTER_ORDER = 4

#: Inner edge of the band: inside this radius the published x = 0 control and
#: the core-band statistics already speak, so the band starts outside it.
BAND_MIN_CM = 10.0
#: Outer edge: the caliper-measured cathode frame opening (half of the 14.5 in
#: aperture, 2026-08-17), the radius the trust question is asked about.
BAND_MAX_CM = 18.415
#: A cell at or above this window spread is SEMI-QUANTITATIVE; a port passes
#: when its in-band median is below it.  Fixed before the first look.
CRITERION_DLN = 0.50
#: Outer edge of the CORE, the cell set ``--cells core`` fits.  This is the
#: ``X_CORE_CM`` of ``fit_te_spatial.py``, which builds a port's ``T_e`` row
#: from exactly these cells; the two are pinned equal by the unit tests.
CORE_MAX_CM = 10.0

PROTOCOL = (
    "Per-cell fit-window re-fits, PRE-DECLARED 2026-08-20 before the numbers "
    "existed: same window family as processed/sweep_window_refits.hdf5 "
    "(p_low 3/8/15/25/35 percent x f_high 0.05/0.10/0.15/0.30/0.50, ion branch "
    "fixed at pipeline defaults); rot-0 runs of experiment sets 1 and 2, all "
    "ports; all plateau cycles 10-19.5 ms and all shots, no subsampling; cells "
    f"{BAND_MIN_CM:g} < |x| <= {BAND_MAX_CM:g} cm plus x = 0 per port as an "
    "internal control; metric dln_te_window = ln(max/min) over the 5x5 grid of "
    "nanmedian-over-sweeps T_e; criterion = per-port in-band median "
    f"dln_te_window < {CRITERION_DLN:g}, fixed before the first look and not "
    "adjusted afterwards."
)
ADJUDICATION = (
    "Trust-to-aperture is ADOPTED at eight set-ports (ES1 and ES2, p11/p21/"
    "p29/p41) and REFUSED at both p50s and at ES3 entirely.  Any in-band cell "
    f"with dln_te_window >= {CRITERION_DLN:g} is marked semi-quantitative and "
    "keeps its measured value; a port whose x = 0 control fails the criterion "
    "has its whole core marked the same way."
)
LINEAGE = (
    "Grown from scripts/refit_sweep_windows.py (the x = 0 product, 2026-07-22), "
    "which supplies refit_sweep(), the window family and the run selection; the "
    "band pass was first executed 2026-08-20 as the D-i discriminator and "
    "archived here so the filled T_e product can condition on it."
)

#: What the recorded library versions mean for anyone regenerating the product.
REGENERATION_NOTE = (
    "The product's last digits are a function of the numerical library versions "
    "recorded in this block: the same code on the same raw traces reproduces "
    "the fitted quantities to roughly 1e-8 relative, not bit for bit, under a "
    "different numpy / scipy / h5py vintage.  Byte-exact regeneration requires "
    "the recorded versions; under any other vintage the check to run is "
    "scripts/verify_window_band_product.py, which holds the cell sets, run "
    "identities, sweep counts and which cells were fittable to byte identity, "
    "and the fitted floats to a disclosed relative tolerance."
)

#: The eight set-ports at which trust-to-aperture is adopted, recorded per port
#: group so the product carries the verdict it was used to reach.
ADOPTED_SET_PORTS = frozenset(
    {(1, 11), (1, 21), (1, 29), (1, 41), (2, 11), (2, 21), (2, 29), (2, 41)}
)


def band_cell_indices(x_cm: np.ndarray) -> tuple[np.ndarray, int]:
    """Return the fitted cell indices and the index of the ``x = 0`` control.

    The in-band cells come first, in scan order, and the ``x = 0`` control is
    appended last, which is the layout the checkpoints already carry.
    """
    index_x0 = int(np.abs(x_cm).argmin())
    in_band = np.flatnonzero(
        (np.abs(x_cm) > BAND_MIN_CM) & (np.abs(x_cm) <= BAND_MAX_CM)
    )
    return np.concatenate([in_band, [index_x0]]), index_x0


def window_spread_dln(median_grid: np.ndarray) -> float:
    """Return ``ln(max / min)`` over one cell's window-family ``T_e`` grid.

    NaN entries -- windows that held too few samples to fit -- are ignored, and
    a cell whose grid is empty or non-positive returns NaN rather than a
    spread, so an unfittable cell never reads as a stable one.
    """
    grid = np.asarray(median_grid, dtype=float)
    if not np.any(np.isfinite(grid)):
        return float("nan")
    te_min, te_max = np.nanmin(grid), np.nanmax(grid)
    if np.isfinite(te_min) and te_min > 0:
        return float(np.log(te_max / te_min))
    return float("nan")


def window_family_median(median_grid: np.ndarray) -> float:
    """Return the median ``T_e`` over one cell's window family.

    The window-family median is the convention-free summary of a cell: it does
    not privilege the pipeline's default window, and it is the quantity the
    per-port core mean is built from.
    """
    return float(np.nanmedian(np.asarray(median_grid, dtype=float)))


def core_cell_indices(x_cm: np.ndarray) -> tuple[np.ndarray, int]:
    """Return the core cell indices and the index of the ``x = 0`` control.

    The core is ``|x| <= CORE_MAX_CM``, in scan order.  Unlike the band it
    already contains ``x = 0``, so no control cell is appended.
    """
    index_x0 = int(np.abs(x_cm).argmin())
    core = np.flatnonzero(np.abs(x_cm) <= CORE_MAX_CM)
    return core, index_x0


def set_runs(sets, rotation_deg: float = 0.0):
    """Yield ``(set_id, run_id, port)`` for runs at one probe rotation.

    ``refit_sweep_windows.rot0_runs`` is the rot-0 case of this walk and is
    reused verbatim for it, so the default pass cannot drift from the x = 0
    product; ``rotation_deg = 180`` walks the reversed-probe runs of the same
    sets, which no published product consumes and which are reported as a
    labelled second column only.
    """
    if float(rotation_deg) == 0.0:
        yield from _rw.rot0_runs(sets=sets)
        return
    with h5py.File(_rw.SWEEPS_H5, "r") as hdf:
        for set_id in sorted(hdf["experiment_sets"], key=int):
            if int(set_id) not in sets:
                continue
            for run_id in sorted(hdf["experiment_sets"][set_id]):
                group = hdf["experiment_sets"][set_id][run_id]
                if float(group.attrs.get("rotation_deg", 0)) == float(rotation_deg):
                    yield int(set_id), run_id, int(group.attrs["port"])


def refit_port(
    dataset: LapdDataset,
    run_id: str,
    set_id: int,
    port: int,
    x_cm: np.ndarray,
    cells: np.ndarray,
    index_x0: int,
    *,
    plateau_ms: tuple[float, float],
    clip_us: float,
    order: int,
    cells_mode: str = "band",
) -> dict:
    """Re-fit every window in the family, for every sweep, at every *cells* cell."""
    started = time.time()
    run = dataset.run(run_id)
    cycle_start_s = _pls._cycle_start_times(run)
    in_plateau = np.flatnonzero(
        (cycle_start_s * 1e3 >= plateau_ms[0]) & (cycle_start_s * 1e3 <= plateau_ms[1])
    )
    core_cutoff_hz, _ = _pls._cutoff_hz_for_run(run, 100.0e3)
    sample_rate_hz = run.config.acquisition.sample_rate_hz
    i_offset = run.default_zero_offset_v(ChannelKind.I_SWEEP)
    v_offset = run.default_zero_offset_v(ChannelKind.V_SWEEP)
    ramp_slices = run.sweep_ramp_sample_slices(clip_s=clip_us * 1e-6)

    te_default = {cell: [] for cell in cells}
    te_grid = {cell: [] for cell in cells}
    for cycle in in_plateau:
        ramp = ramp_slices[cycle]
        i_all = run.langmuir_traces(ChannelKind.I_SWEEP, ramp, zero_offset_v=i_offset)
        v_all = run.langmuir_traces(ChannelKind.V_SWEEP, ramp, zero_offset_v=v_offset)
        for cell in cells:
            i_cell = butterworth_lowpass(
                i_all[cell],
                sample_rate_hz=sample_rate_hz,
                cutoff_hz=core_cutoff_hz,
                order=order,
            )
            v_cell = butterworth_lowpass(
                v_all[cell],
                sample_rate_hz=sample_rate_hz,
                cutoff_hz=core_cutoff_hz,
                order=order,
            )
            for shot in range(i_cell.shape[0]):
                default, grid = _rw.refit_sweep(v_cell[shot], i_cell[shot])
                te_default[cell].append(default)
                te_grid[cell].append(grid)
        del i_all, v_all

    n_cells = len(cells)
    dln = np.full(n_cells, np.nan)
    te_default_med = np.full(n_cells, np.nan)
    n_sweeps = np.zeros(n_cells, dtype=int)
    med_grids = np.full((n_cells, len(_rw.P_LOW), len(_rw.F_HIGH)), np.nan)
    for index, cell in enumerate(cells):
        defaults = np.asarray(te_default[cell], dtype=float)
        grids = np.asarray(te_grid[cell], dtype=float)
        defaults[(defaults <= 0.05) | (defaults > 30.0)] = np.nan
        grids[(grids <= 0.05) | (grids > 30.0)] = np.nan
        median_grid = np.nanmedian(grids, axis=0)
        med_grids[index] = median_grid
        dln[index] = window_spread_dln(median_grid)
        te_default_med[index] = float(np.nanmedian(defaults))
        n_sweeps[index] = int(np.isfinite(defaults).sum())

    return dict(
        cells=cells,
        x_cm=x_cm[cells],
        is_x0=(cells == index_x0),
        dln_te_window=dln,
        te_default_med=te_default_med,
        n_sweeps=n_sweeps,
        med_grids=med_grids,
        run_id=run_id,
        sid=set_id,
        port=port,
        wall_s=time.time() - started,
        n_plateau=len(in_plateau),
        cells_mode=cells_mode,
    )


def _port_summary(record: dict) -> dict:
    """Return the per-port statistics and the criterion verdict.

    The band pass reports its in-band statistics and the trust-to-aperture
    adjudication; the core pass reports the SAME statistics over its own
    non-``x = 0`` cells, but under ``core_`` names, because a reader who found
    ``in_band_median_dln`` in a core product would reasonably take it for the
    band's.  Neither set of names appears in the other mode.
    """
    is_x0 = np.asarray(record["is_x0"], dtype=bool)
    dln = np.asarray(record["dln_te_window"], dtype=float)
    off_x0 = dln[~is_x0]
    x0 = float(dln[is_x0][0])
    median = float(np.nanmedian(off_x0))
    common = {
        "set_id": int(record["sid"]),
        "port": int(record["port"]),
        "run_id": str(record["run_id"]),
        "n_plateau_cycles": int(record["n_plateau"]),
    }
    statistics = {
        "cells": int(off_x0.size),
        "median_dln": median,
        "upper_quartile_dln": float(np.nanpercentile(off_x0, 75)),
        "max_dln": float(np.nanmax(off_x0)),
        "fraction_at_or_above_criterion": float(np.mean(off_x0 >= CRITERION_DLN)),
    }
    tail = {
        "x0_control_dln": x0,
        "criterion_dln": CRITERION_DLN,
        "x0_control_passes_criterion": bool(x0 < CRITERION_DLN),
    }

    if str(record.get("cells_mode", "band")) != "core":
        return {
            **common,
            **{f"in_band_{key}": value for key, value in statistics.items()},
            "x0_control_dln": tail["x0_control_dln"],
            "criterion_dln": tail["criterion_dln"],
            "passes_criterion": bool(median < CRITERION_DLN),
            "x0_control_passes_criterion": tail["x0_control_passes_criterion"],
            "trust_to_aperture_adopted": bool(
                (int(record["sid"]), int(record["port"])) in ADOPTED_SET_PORTS
            ),
        }

    # The core pass reports the port's own T_e, so it carries the cell count
    # the criterion admits and the mean over exactly those cells.  A port with
    # no admitted cell reports NaN rather than a mean over rejected cells.
    family_median = np.array(
        [window_family_median(grid) for grid in np.asarray(record["med_grids"])]
    )
    passes = np.isfinite(dln) & (dln < CRITERION_DLN)
    n_below = int(passes.sum())
    return {
        **common,
        "cells_mode": "core",
        "core_max_cm": CORE_MAX_CM,
        "core_cells": int(dln.size),
        "core_cells_below_criterion": n_below,
        "core_gate_min_cells": CORE_GATE_MIN_CELLS,
        "core_gate_passes": bool(n_below >= CORE_GATE_MIN_CELLS),
        "core_mean_te_ev": (
            float(np.nanmean(family_median[passes])) if passes.any() else float("nan")
        ),
        "core_mean_te_default_window_ev": (
            float(np.nanmean(np.asarray(record["te_default_med"])[passes]))
            if passes.any()
            else float("nan")
        ),
        **{f"core_off_x0_{key}": value for key, value in statistics.items()},
        **tail,
    }


def _core_metadata(
    summaries: list[dict],
    *,
    created: str,
    core_mode: dict,
    hdf5_path: Path,
    summary_path: Path,
) -> dict:
    """Return the metadata payload describing a CORE pass.

    It states the core protocol and the core gate and carries no band-only
    field, so a consumer that reads the JSON cannot mistake this product for
    the band product whose file name it resembles.  A port the gate refused
    carries ``null`` where its ``T_e`` would be, rather than the ``NaN``
    literal that would make the file invalid JSON to a strict parser.
    """
    ports = [
        {
            key: (
                None
                if isinstance(value, float) and not np.isfinite(value)
                else value
            )
            for key, value in summary.items()
        }
        for summary in summaries
    ]
    return {
        "product": "window_refit_core",
        "created_utc": created,
        "numerics_vintage": _numerics_vintage(),
        "generating_script": "scripts/refit_window_band.py",
        "protocol": CORE_PROTOCOL,
        "lineage": LINEAGE,
        "cells_mode": "core",
        "core_max_cm": core_mode["core_max_cm"],
        "core_gate_min_cells": core_mode["core_gate_min_cells"],
        "rotation_deg": core_mode["rotation_deg"],
        "criterion_dln_te_window": CRITERION_DLN,
        "sets_run": sorted({int(summary["set_id"]) for summary in summaries}),
        "ports_run": [int(summary["port"]) for summary in summaries],
        "plateau_ms": list(PLATEAU_MS),
        "window_p_low_percent": list(_rw.P_LOW),
        "window_f_high_fraction": list(_rw.F_HIGH),
        "full_product_hdf5": _display_path(hdf5_path),
        "per_cell_csv": _display_path(summary_path),
        "ports": ports,
    }


def _numerics_vintage() -> dict:
    """Return the interpreter and library versions this run was produced under.

    Read at run time, never restated, so the block cannot go stale the way a
    written-down version would.  It is written on EVERY pass, band and core
    alike, because the drift it discloses is a property of the numerics rather
    than of which cells were fitted.
    """
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "h5py": h5py.__version__,
        "regeneration_note": REGENERATION_NOTE,
    }


def _display_path(path: Path) -> str:
    """Return ``path`` repo-relative under ``ROOT``, else its resolved absolute form."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def write_product(
    records: list[dict],
    x_cm: np.ndarray,
    *,
    hdf5_path: Path,
    summary_path: Path,
    metadata_path: Path,
    core_mode: dict | None = None,
) -> list[dict]:
    """Write the HDF5 product, the tracked per-cell CSV and the metadata.

    ``core_mode`` carries the core pass's rotation and gate; with the default
    ``None`` all three files are exactly the band products they have always
    been.  When it is given, the metadata JSON describes the CORE protocol and
    drops the band-only fields, so a core product cannot be read as a band one.
    """
    created = _datetime.datetime.now(_datetime.UTC).isoformat(timespec="seconds")
    summaries = [_port_summary(record) for record in records]

    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(hdf5_path, "w") as hdf:
        if core_mode is None:
            hdf.attrs["protocol"] = PROTOCOL
            hdf.attrs["adjudication"] = ADJUDICATION
        else:
            # No band protocol, no band adjudication and no band edges: this
            # file is not a band product and must not read as one.
            hdf.attrs["protocol"] = CORE_PROTOCOL
        hdf.attrs["lineage"] = LINEAGE
        hdf.attrs["generating_script"] = "scripts/refit_window_band.py"
        hdf.attrs["criterion_dln_te_window"] = CRITERION_DLN
        if core_mode is None:
            hdf.attrs["band_min_cm"] = BAND_MIN_CM
            hdf.attrs["band_max_cm"] = BAND_MAX_CM
        hdf.attrs["plateau_ms"] = np.asarray(PLATEAU_MS, dtype=np.float64)
        hdf.attrs["clip_us"] = CLIP_US
        hdf.attrs["filter_order"] = FILTER_ORDER
        hdf.attrs["window_p_low_percent"] = np.asarray(_rw.P_LOW, dtype=np.float64)
        hdf.attrs["window_f_high_fraction"] = np.asarray(_rw.F_HIGH, dtype=np.float64)
        hdf.attrs["source_sweeps_hdf5"] = str(_rw.SWEEPS_H5.name)
        hdf.attrs["created_utc"] = created
        if core_mode is not None:
            hdf.attrs["cells_mode"] = "core"
            hdf.attrs["core_max_cm"] = core_mode["core_max_cm"]
            hdf.attrs["core_gate_min_cells"] = core_mode["core_gate_min_cells"]
            hdf.attrs["rotation_deg"] = core_mode["rotation_deg"]
        hdf.create_dataset("x_cm", data=x_cm)
        for record, summary in zip(records, summaries):
            group = hdf.create_group(f"set{summary['set_id']}/port{summary['port']}")
            for key, value in summary.items():
                group.attrs[key] = value
            group.attrs["wall_s"] = float(record["wall_s"])
            group.create_dataset("cell_index", data=np.asarray(record["cells"]))
            group.create_dataset("x_cm", data=np.asarray(record["x_cm"]))
            group.create_dataset("is_x0", data=np.asarray(record["is_x0"], dtype=bool))
            group.create_dataset(
                "in_band", data=~np.asarray(record["is_x0"], dtype=bool)
            )
            group.create_dataset("dln_te_window", data=record["dln_te_window"])
            group.create_dataset("te_default_med_ev", data=record["te_default_med"])
            group.create_dataset("n_sweeps", data=np.asarray(record["n_sweeps"]))
            group.create_dataset("te_window_ev", data=record["med_grids"])
            if str(record.get("cells_mode", "band")) == "core":
                group.create_dataset(
                    "te_window_family_median_ev",
                    data=np.array(
                        [
                            window_family_median(grid)
                            for grid in np.asarray(record["med_grids"])
                        ]
                    ),
                )
    print(f"wrote {hdf5_path}")

    with summary_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "set_id",
                "port",
                "run_id",
                "x_cm",
                "is_x0_control",
                "in_band",
                "dln_te_window",
                "te_default_med_ev",
                "n_sweeps",
                "at_or_above_criterion",
            ]
        )
        for record, summary in zip(records, summaries):
            is_x0 = np.asarray(record["is_x0"], dtype=bool)
            for index in range(is_x0.size):
                dln = float(record["dln_te_window"][index])
                writer.writerow(
                    [
                        summary["set_id"],
                        summary["port"],
                        summary["run_id"],
                        f"{float(record['x_cm'][index]):.6g}",
                        int(is_x0[index]),
                        int(not is_x0[index]),
                        f"{dln:.9g}",
                        f"{float(record['te_default_med'][index]):.9g}",
                        int(record["n_sweeps"][index]),
                        int(np.isfinite(dln) and dln >= CRITERION_DLN),
                    ]
                )
    print(f"wrote {summary_path}")

    if core_mode is None:
        metadata = {
            "product": "window_refit_band",
            "created_utc": created,
            "numerics_vintage": _numerics_vintage(),
            "generating_script": "scripts/refit_window_band.py",
            "protocol": PROTOCOL,
            "adjudication": ADJUDICATION,
            "lineage": LINEAGE,
            "criterion_dln_te_window": CRITERION_DLN,
            "band_min_cm": BAND_MIN_CM,
            "band_max_cm": BAND_MAX_CM,
            "plateau_ms": list(PLATEAU_MS),
            "window_p_low_percent": list(_rw.P_LOW),
            "window_f_high_fraction": list(_rw.F_HIGH),
            "full_product_hdf5": _display_path(hdf5_path),
            "per_cell_csv": _display_path(summary_path),
            "ports": summaries,
        }
    else:
        metadata = _core_metadata(
            summaries,
            created=created,
            core_mode=core_mode,
            hdf5_path=hdf5_path,
            summary_path=summary_path,
        )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {metadata_path}")
    return summaries


CORE_PROTOCOL = (
    "Core-cell window re-fits: the band pass's window family, plateau window, "
    "filter and retarding-branch fit, run over the CORE cells "
    f"|x| <= {CORE_MAX_CM:g} cm instead of the band, so that the per-cell "
    "window spread is measured on exactly the cells a port's T_e row is built "
    "from.  Gate: dln_te_window < "
    f"{CRITERION_DLN:g} on at least 3 core cells per port.  The port T_e is "
    "the mean over the admitted cells of the per-cell window-family median."
)

#: Cells a port must have below the criterion for its core T_e to be read.
CORE_GATE_MIN_CELLS = 3


def refuse_band_output_paths(
    *,
    cells: str,
    sets,
    ports,
    rotation_deg: float,
    output: Path,
    summary: Path,
    metadata: Path,
) -> None:
    """Refuse to write a non-band pass into the band products' own paths.

    ``processed/window_refit_band_summary.csv`` and its metadata are TRACKED
    and are read by ``fit_te_spatial.py`` to mark semi-quantitative cells; its
    metric-identity guard compares only the window family and the plateau
    window, all of which every pass here shares, so a substituted file would be
    consumed silently.  A pass that moves ANY of the four run-selecting flags
    -- ``--cells``, ``--sets``, ``--ports``, ``--rotation-deg`` -- therefore has
    to name its own output paths, and is refused at argument resolution, before
    any fitting, if it has not.
    """
    requested = []
    if cells != "band":
        requested.append(f"--cells {cells}")
    if float(rotation_deg) != 0.0:
        requested.append(f"--rotation-deg {float(rotation_deg):g}")
    if tuple(sets) != tuple(SETS):
        requested.append("--sets " + ",".join(str(int(s)) for s in sets))
    if ports is not None:
        # A port subset is as much a substitution as a different cell set: the
        # product would carry fewer ports under the same file name.
        requested.append("--ports " + ",".join(str(int(p)) for p in ports))
    if not requested:
        return

    still_default = [
        flag
        for flag, given, default in (
            ("--output", output, OUTPUT_HDF5),
            ("--summary", summary, OUTPUT_SUMMARY),
            ("--metadata", metadata, OUTPUT_METADATA),
        )
        if Path(given) == Path(default)
    ]
    if not still_default:
        return

    raise ValueError(
        f"{', '.join(requested)} asks for a pass that is not the band pass, "
        f"but {', '.join(still_default)} still points at the band product "
        "written by the default invocation.  Those files are the D-i band "
        "products -- scripts/fit_te_spatial.py reads the summary CSV to mark "
        "semi-quantitative cells, and its metric-identity guard compares only "
        "the window family and the plateau window, which this pass shares, so "
        "it could not tell the substitution from the real thing.  Pass "
        f"{' and '.join(still_default)} explicitly, outside processed/."
    )


def resolve_work_dir(work_dir: Path | None, output: Path) -> Path:
    """Return where the per-(set, port) checkpoints live.

    Unset, they sit beside the product they belong to, so a pass sent outside
    ``processed/`` -- as the refusal above requires of every non-band pass --
    leaves nothing behind inside it.  The default band invocation is unaffected:
    its output is already in ``processed/``, so the work directory resolves to
    the one it has always used.
    """
    if work_dir is not None:
        return Path(work_dir)
    return Path(output).parent / WORK_DIR_NAME


def _comma_ints(text: str) -> tuple[int, ...]:
    """Parse a comma-separated integer list for the set and port options."""
    return tuple(int(part) for part in str(text).split(",") if part.strip())


def _refused(summary: dict) -> str:
    """Return the phrase for a port whose gate admitted too few core cells."""
    return (
        f"REFUSED ({summary['core_cells_below_criterion']} of "
        f"{summary['core_cells']} core cells window-stable)"
    )


def _print_core_report(records: list[dict], summaries: list[dict]) -> None:
    """Print the per-cell core tables, the gate and the pre-registered bins."""
    print(f"\ncore cells: |x| <= {CORE_MAX_CM:g} cm; gate: dln_te_window < "
          f"{CRITERION_DLN:g} on >= {CORE_GATE_MIN_CELLS} core cells per port")
    for record, summary in zip(records, summaries):
        print(
            f"\nset{summary['set_id']} port{summary['port']} "
            f"run {summary['run_id']}  ({summary['n_plateau_cycles']} plateau "
            f"cycles)"
        )
        print(f"{'x_cm':>7} {'Te_family_med':>13} {'Te_default':>10} "
              f"{'dln':>7} {'n_sweeps':>8}  {'passes':>6}")
        family = [
            window_family_median(grid) for grid in np.asarray(record["med_grids"])
        ]
        for index in range(len(family)):
            dln = float(record["dln_te_window"][index])
            passes = bool(np.isfinite(dln) and dln < CRITERION_DLN)
            print(
                f"{float(record['x_cm'][index]):7.1f} {family[index]:13.3f} "
                f"{float(record['te_default_med'][index]):10.3f} {dln:7.3f} "
                f"{int(record['n_sweeps'][index]):8d}  {str(passes):>6}"
            )
        if summary["core_cells_below_criterion"] >= CORE_GATE_MIN_CELLS:
            reading = (
                f"gate PASS;  core-mean T_e = {summary['core_mean_te_ev']:.3f} "
                f"eV (default window "
                f"{summary['core_mean_te_default_window_ev']:.3f} eV)"
            )
        else:
            reading = f"gate FAIL;  T_e {_refused(summary)}"
        print(
            f"  cells below criterion: {summary['core_cells_below_criterion']}"
            f" / {summary['core_cells']};  {reading}"
        )
        print(
            f"  off-x0 median dln {summary['core_off_x0_median_dln']:.3f}, "
            f"max {summary['core_off_x0_max_dln']:.3f}, "
            f"fraction >= criterion "
            f"{summary['core_off_x0_fraction_at_or_above_criterion']:.2f}; "
            f"x = 0 control dln {summary['x0_control_dln']:.3f}"
        )

    all_ports = {summary["port"]: summary for summary in summaries}
    readings = {}
    print()
    for port in (29, 41):
        summary = all_ports.get(port)
        if summary is None:
            print(f"bins: T_e(p{port}) = NOT RUN")
            continue
        if summary["core_cells_below_criterion"] >= CORE_GATE_MIN_CELLS:
            readings[port] = float(summary["core_mean_te_ev"])
            print(f"bins: T_e(p{port}) = {readings[port]:.3f} eV")
        else:
            print(f"bins: T_e(p{port}) = {_refused(summary)}")

    te29, te41 = readings.get(29), readings.get(41)
    if te29 is None or te41 is None:
        refused = [
            f"p{port}" for port in (29, 41) if readings.get(port) is None
        ]
        print(
            f"bin : NOT REACHED -- the gate admitted no core T_e at "
            f"{' and '.join(refused)}, so there is no measured value to place "
            "in a bin"
        )
    else:
        if te29 <= 0.8 and te41 <= 0.4:
            verdict = "the T_e-prior explanation is confirmed"
        elif abs(te29 / 3.7 - 1.0) <= 0.25 and abs(te41 / 1.8 - 1.0) <= 0.25:
            verdict = "the collection deficit owns it"
        else:
            verdict = "outside the bins"
        print(f"bin : {verdict}")
    print(
        "  (bin 1 is T_e(p29) <= 0.8 and T_e(p41) <= 0.4; bin 2 is 'at the "
        "prior' read as within 25 percent of p29 = 3.7 and p41 = 1.8, the "
        "tolerance stated here because the registration gave the priors as "
        "approximate values)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_HDF5)
    parser.add_argument("--summary", type=Path, default=OUTPUT_SUMMARY)
    parser.add_argument("--metadata", type=Path, default=OUTPUT_METADATA)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help=(
            "per-(set, port) npz checkpoints; an existing checkpoint is reused. "
            f"Default: {WORK_DIR_NAME}/ beside --output"
        ),
    )
    parser.add_argument("--plateau-ms", nargs=2, type=float, default=PLATEAU_MS)
    parser.add_argument("--clip-us", type=float, default=CLIP_US)
    parser.add_argument("--order", type=int, default=FILTER_ORDER)
    parser.add_argument(
        "--sets",
        type=_comma_ints,
        default=SETS,
        help="experiment sets to walk, comma separated",
    )
    parser.add_argument(
        "--ports",
        type=_comma_ints,
        default=None,
        help="restrict to these ports; default is every port of each set",
    )
    parser.add_argument(
        "--cells",
        choices=("band", "core"),
        default="band",
        help="fit the trust band (default) or the core cells",
    )
    parser.add_argument(
        "--rotation-deg",
        type=float,
        default=0.0,
        help="probe rotation of the runs to walk",
    )
    args = parser.parse_args()
    refuse_band_output_paths(
        cells=args.cells,
        sets=args.sets,
        ports=args.ports,
        rotation_deg=args.rotation_deg,
        output=args.output,
        summary=args.summary,
        metadata=args.metadata,
    )
    args.work_dir = resolve_work_dir(args.work_dir, args.output)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(_rw.SWEEPS_H5, "r") as hdf:
        x_cm = hdf["x_cm"][:]
    if args.cells == "core":
        cells, index_x0 = core_cell_indices(x_cm)
    else:
        cells, index_x0 = band_cell_indices(x_cm)
    dataset = LapdDataset.from_manifest(_rw.MANIFEST)

    records = []
    for set_id, run_id, port in set_runs(args.sets, args.rotation_deg):
        if args.ports is not None and port not in args.ports:
            continue
        if args.cells == "band" and args.rotation_deg == 0.0:
            checkpoint = args.work_dir / f"di_set{set_id}_port{port}.npz"
        else:
            checkpoint = args.work_dir / (
                f"{args.cells}{args.rotation_deg:.0f}_set{set_id}_port{port}.npz"
            )
        if checkpoint.exists():
            with np.load(checkpoint, allow_pickle=False) as loaded:
                record = {key: loaded[key] for key in loaded.files}
            record["run_id"] = str(record["run_id"])
            if "cells_mode" in record:
                record["cells_mode"] = str(record["cells_mode"])
            print(f"[reuse] set{set_id} port{port} <- {checkpoint.name}")
        else:
            record = refit_port(
                dataset,
                run_id,
                set_id,
                port,
                x_cm,
                cells,
                index_x0,
                plateau_ms=tuple(args.plateau_ms),
                clip_us=args.clip_us,
                order=args.order,
                cells_mode=args.cells,
            )
            np.savez(checkpoint, **record)
            print(
                f"[done] set{set_id} port{port} run={run_id} "
                f"wall={record['wall_s']:.0f}s"
            )
        if not np.array_equal(np.asarray(record["cells"]), cells):
            raise ValueError(
                f"{checkpoint} was fitted on a different cell set than the "
                "current band definition"
            )
        records.append(record)

    core_mode = None
    if args.cells == "core":
        core_mode = {
            "core_max_cm": CORE_MAX_CM,
            "core_gate_min_cells": CORE_GATE_MIN_CELLS,
            "rotation_deg": float(args.rotation_deg),
        }
    summaries = write_product(
        records,
        x_cm,
        hdf5_path=args.output,
        summary_path=args.summary,
        metadata_path=args.metadata,
        core_mode=core_mode,
    )

    if args.cells == "core":
        _print_core_report(records, summaries)
        return

    print(
        f"\n{'set':>3} {'port':>5} {'median':>7} {'UQ':>7} {'max':>7} "
        f"{'frac>=c':>8} {'x0':>7}  verdict"
    )
    for summary in summaries:
        verdict = "PASS" if summary["passes_criterion"] else "FAIL"
        if not summary["x0_control_passes_criterion"]:
            verdict += " (x0 control FAILS)"
        print(
            f"{summary['set_id']:>3} {summary['port']:>5} "
            f"{summary['in_band_median_dln']:7.3f} "
            f"{summary['in_band_upper_quartile_dln']:7.3f} "
            f"{summary['in_band_max_dln']:7.3f} "
            f"{summary['in_band_fraction_at_or_above_criterion']:8.2f} "
            f"{summary['x0_control_dln']:7.3f}  {verdict}"
        )


if __name__ == "__main__":
    main()
