"""The digitizer-rail screen, measured once for every product that records it.

Two annotators write the same-named rail attrs onto the two faces of the
rot-180 Mach pair: scripts/annotate_rail_mask.py onto the ISAT product, and
scripts/annotate_saturation_attrs.py -- through the ``screen`` wrapper in
scripts/screen_rot180_saturation.py -- onto the ISWEEP product.  Each one also
records the OPPOSITE face's dead-time fraction, so every one of those numbers
is measured twice, once from each side of the pair, and the pair is only
self-consistent if the two measurements are identical.

``screen_cells`` is that measurement and is its only implementation: both
writers call this function, so the values they write are bitwise equal by
construction rather than by agreement between two transcriptions.  A screen
constant that reaches a product as an attr (the rail codes, the CLIP_S trim,
the plateau window, the ``METHOD`` text) lives here for the same reason.

A sample is railed when its raw uint16 SIS code sits on a converter rail, 0 or
65535; the SIS per-shot header fields Min/Max/Clipped are identically zero in
this dataset and carry no information.  Counts are per averaging CELL -- the
(position, CLIP_S-trimmed inter-sweep dead-time window) cells the profile
products average -- and each run-level scalar is those counts summed over every
cell axis it covers, against a denominator counting the samples of the same
cells.  So a scalar is the sample-weighted mean of the cell fractions it
summarises and cannot exceed one; a denominator short of the position axis
would read n_positions times too large.
"""

from __future__ import annotations

import numpy as np

from bapsf_lapd.config import ChannelKind
from bapsf_lapd.density import inter_sweep_sample_slices

CLIP_S = 10e-6            # same trim as scripts/plot_isat_profiles.py
RAIL_LO, RAIL_HI = 0, 65535
PLATEAU_MS = (14.0, 19.0)
BATCH = 50                # raw rows read per block

METHOD = (
    "raw uint16 SIS codes; a sample is railed at code 0 or 65535. The SIS "
    "per-shot header fields Min/Max/Clipped are identically zero for this "
    "dataset and carry no information. Fractions are over the whole record, "
    "over the CLIP_S-trimmed inter-sweep dead-time windows this product "
    "averages, and over the 14-19 ms plateau cycles."
)


def inter_sweep_times_ms(sw):
    """Mid-time of each inter-sweep dead-time window, in milliseconds."""
    return np.array([
        sw.t0_s + k * sw.tau_cycle_s + 0.5 * (sw.tau_ramp_s + sw.tau_cycle_s)
        for k in range(sw.n_cycles)
    ]) * 1e3


def screen_cells(run, kind: ChannelKind) -> dict:
    """Railed-sample counts for one run and channel, per cell and in total.

    Returns, for the raw record of ``kind``:

    ``rail_counts``, ``n_cell_samples``, ``rail_fraction``, ``rail_mask``
        railed samples per (position, dead-time window) cell, the samples each
        cell holds (one row, broadcasting over positions), their ratio, and the
        per-cell verdict -- a cell is railed when it holds ANY railed sample.
    ``n_all``/``rail_all``, ``n_dead``/``rail_dead``, ``n_plat``/``rail_plat``
        samples and railed samples over the whole record, over the dead-time
        cells, and over the plateau cells; each pair divides to a fraction.
    ``code_lo``/``code_hi``
        the run's own observed code floor and ceiling over the whole record.
    ``dead_idx``/``plateau_dead_idx``, ``plateau_columns``,
    ``plateau_cycles``/``plateau_ms``
        the sample indices the dead-time and plateau counts were taken over,
        the cycles the plateau window selects, and that window's first and last
        cycle index and mid-time -- so a caller measuring something else over
        the same windows measures it over exactly these samples.
    """
    cfg = run.config
    ch = cfg.channel(kind)
    sw, acq = cfg.sweep, cfg.acquisition
    n_pos = acq.n_positions
    n_shots = acq.n_shots_per_position
    n_cycles = sw.n_cycles

    dead = inter_sweep_sample_slices(sw, acq, clip_s=CLIP_S)
    dead_idx = np.concatenate([np.arange(s.start, s.stop) for s in dead])
    seg = np.concatenate([
        np.full(s.stop - s.start, k, dtype=np.int64) for k, s in enumerate(dead)
    ])
    per_window_samples = np.array([s.stop - s.start for s in dead], dtype=np.int64)

    t_ms = inter_sweep_times_ms(sw)
    plateau_columns = (t_ms >= PLATEAU_MS[0]) & (t_ms <= PLATEAU_MS[1])
    plat = np.flatnonzero(plateau_columns)
    plateau_dead_idx = np.concatenate(
        [np.arange(dead[k].start, dead[k].stop) for k in plat]
    )

    rail_counts = np.zeros((n_pos, n_cycles), dtype=np.int64)
    n_all = rail_all = 0
    code_lo, code_hi = RAIL_HI, RAIL_LO

    with run.open() as h5:
        data = h5[ch.hdf5_path]
        for start in range(0, n_pos * n_shots, BATCH):
            block = data[start:start + BATCH, :]
            n_all += block.size
            railed = (block == RAIL_LO) | (block == RAIL_HI)
            rail_all += int(railed.sum())
            code_lo = min(code_lo, int(block.min()))
            code_hi = max(code_hi, int(block.max()))
            sub = railed[:, dead_idx]
            for row in np.flatnonzero(sub.any(axis=1)):
                pos = (start + int(row)) // n_shots
                rail_counts[pos] += np.bincount(seg[sub[row]], minlength=n_cycles)

    n_cell_samples = per_window_samples[None, :] * n_shots
    cell_samples = np.broadcast_to(n_cell_samples, rail_counts.shape)

    return dict(
        rail_fraction=rail_counts / n_cell_samples,
        rail_mask=rail_counts > 0,
        rail_counts=rail_counts,
        n_cell_samples=n_cell_samples,
        n_all=n_all, rail_all=rail_all,
        n_dead=int(cell_samples.sum()), rail_dead=int(rail_counts.sum()),
        n_plat=int(cell_samples[:, plateau_columns].sum()),
        rail_plat=int(rail_counts[:, plateau_columns].sum()),
        code_lo=code_lo, code_hi=code_hi,
        dead_idx=dead_idx, plateau_dead_idx=plateau_dead_idx,
        plateau_columns=plateau_columns,
        plateau_cycles=(int(plat[0]), int(plat[-1])),
        plateau_ms=(t_ms[plat[0]], t_ms[plat[-1]]),
    )
