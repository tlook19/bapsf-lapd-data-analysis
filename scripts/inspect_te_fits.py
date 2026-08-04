"""Interactive T_e fit inspection and manual selection tool — CustomTkinter GUI.

Browse through (position, cycle) cells for a given run.  View individual
shot I-V curves with both log-linear and exponential fit methods, and
record manual selections that override the automated best-of-log-or-exp
choice.

Layout
------
  Left canvas (matplotlib, 4 panels):
    Top-left  : I-V curve — all shots faded, current shot highlighted,
                 ion fit (dashed black), log fit (blue), exp fit (orange),
                 custom exp fit (purple dashed, when active).
    Top-right : log(I_e) vs V — retarding region.  Two draggable vertical
                 lines (green = v_lo, red = v_hi) restrict the exp-fit window.
                 Drag a line, release to refit; purple curve shows result.
    Bottom-left : T_e per shot — log (▲), exp (▼), best (●); current shot
                 highlighted; active override shown as a purple ★.
    Bottom strip: T_e_best vs cycle for the current position (IQR band),
                 flagged cells highlighted in orange.

  Right panel (CTk buttons and labels):
    Review nav : ◀ / ▶ buttons jump to prev/next flagged cell; flag reasons shown.
    Navigation : Prev/Next buttons for Position, Cycle, Shot.
    Status     : live readout of x, t, quality, Vp, Te values, selection.
    Fit window : v_lo / v_hi readout + Reset Bounds button.
    Record     : Log / Exp / Custom / Auto-clear / Bad buttons.
    File       : Save / Quit.

Review flags
------------
  If a langmuir_sweeps.hdf5 containing review_flags datasets is provided
  (via --sweeps-hdf5 or the default path), cells with any flag set are
  highlighted and reachable via the "◀ Prev Review / Next Review ▶" buttons.

  Flags (from flag_te_review.py):
    QUALITY   bit 0 — ≥50 % of shots are bad
    DISAGREE  bit 1 — |Te_log − Te_exp| / Te_best ≥ 30 %
    SPREAD    bit 2 — inter-shot CV ≥ 30 %
    SPATIAL   bit 3 — spatial outlier (3 × MAD)
    TEMPORAL  bit 4 — temporal outlier (3 × MAD)

Fit-window dragging
-------------------
  In the log-space panel, click and drag the green line (v_lo) or the
  red line (v_hi) to restrict the voltage window for the exponential fit.
  Release to refit.  Click "Custom" in the Record section to save the result.
  "Reset Bounds" restores the default retarding region.

Output CSV columns
------------------
  run_id, pos_idx, x_cm, cycle_idx, time_ms, shot_idx, method, te_ev

  method is "log", "exp", or "bad".  Missing rows → downstream treats as
  "auto".

Usage
-----
  MPLCONFIGDIR=.matplotlib ./.venv/bin/python scripts/inspect_te_fits.py \\
      --run-id 32 [--pos 25] [--cycle 0] \\
      [--sweeps-hdf5 processed/langmuir_sweeps.hdf5] \\
      [--output processed/manual_te_selections.csv]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import customtkinter as ctk
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from bapsf_lapd import (
    ChannelKind,
    LapdRun,
    analyze_langmuir_sweep,
    butterworth_lowpass,
    evaluate_langmuir_quality,
    select_best_te_ev,
)
from bapsf_lapd.langmuir import _exponential_fit as _run_exp_fit
from bapsf_lapd.manifest import load_run_manifest


MANIFEST        = Path("config/may2026_run_manifest.toml")
X_CM            = np.linspace(-25.0, 25.0, 51)
EDGE_X_CM       = 10.0
CLIP_US         = 10.0
BASE_CUTOFF_KHZ = 100.0
FILTER_ORDER    = 4

_SEV_COLOR = {"ok": "#2ca02c", "warn": "#ff7f0e", "bad": "#d62728"}

# Review-flag bitmask constants (must match flag_te_review.py)
FLAG_QUALITY  = 0x01
FLAG_DISAGREE = 0x02
FLAG_SPREAD   = 0x04
FLAG_SPATIAL  = 0x08
FLAG_TEMPORAL = 0x10

_FLAG_NAMES = {
    FLAG_QUALITY:  "QUALITY",
    FLAG_DISAGREE: "DISAGREE",
    FLAG_SPREAD:   "SPREAD",
    FLAG_SPATIAL:  "SPATIAL",
    FLAG_TEMPORAL: "TEMPORAL",
}

SWEEPS_HDF5 = Path("processed/langmuir_sweeps.hdf5")


def _decode_flags(flag_byte: int) -> str:
    """Return a short human-readable string of active flag names."""
    active = [name for bit, name in _FLAG_NAMES.items() if flag_byte & bit]
    return "  ".join(active) if active else ""


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cutoffs_hz(run: LapdRun) -> tuple[float, float]:
    sw     = run.config.sweep
    rate   = (sw.voltage_end - sw.voltage_start) / sw.tau_ramp_s
    min_hz = rate / (2.0 * np.pi * 0.30 * (4.45 * 0.5))
    high_hz = max(BASE_CUTOFF_KHZ * 1e3, min_hz)
    cfg = run.config
    if cfg.probe.port == 50 or (cfg.probe.port == 11 and cfg.experiment_set.id == 4):
        return high_hz, high_hz
    return BASE_CUTOFF_KHZ * 1e3, high_hz


def _fit_position(
    raw_i: np.ndarray,
    raw_v: np.ndarray,
    ramp_slices: list[slice],
    pos_idx: int,
    sample_rate_hz: float,
    cutoff_hz: float,
    *,
    progress_cb=None,
) -> dict:
    """Filter and fit every (shot, cycle) for one position.

    Parameters
    ----------
    raw_i, raw_v : (n_pos, n_shots, all_samples) arrays loaded from HDF5.
    progress_cb  : optional callable(done_cycles, total_cycles) for UI updates.

    Returns
    -------
    dict with voltages, currents, analyses, reports, te_best.
    """
    n_cycles = len(ramp_slices)
    n_samp   = ramp_slices[0].stop - ramp_slices[0].start
    n_shots  = raw_i.shape[1]

    voltages = np.full((n_shots, n_cycles, n_samp), np.nan)
    currents = np.full((n_shots, n_cycles, n_samp), np.nan)
    analyses = [[None] * n_cycles for _ in range(n_shots)]
    reports  = [[None] * n_cycles for _ in range(n_shots)]
    te_best  = np.full((n_shots, n_cycles), np.nan)

    for ci, ramp_slice in enumerate(ramp_slices):
        i_pos = raw_i[pos_idx, :, ramp_slice]
        v_pos = raw_v[pos_idx, :, ramp_slice]

        i_filt = butterworth_lowpass(i_pos, sample_rate_hz=sample_rate_hz,
                                     cutoff_hz=cutoff_hz, order=FILTER_ORDER, axis=-1)
        v_filt = butterworth_lowpass(v_pos, sample_rate_hz=sample_rate_hz,
                                     cutoff_hz=cutoff_hz, order=FILTER_ORDER, axis=-1)

        ns = min(n_samp, i_filt.shape[-1])
        voltages[:, ci, :ns] = v_filt[:, :ns]
        currents[:, ci, :ns] = i_filt[:, :ns]

        for si in range(n_shots):
            try:
                a = analyze_langmuir_sweep(v_filt[si], i_filt[si])
                r = evaluate_langmuir_quality(a, current=i_filt[si], peer_current=i_filt)
                analyses[si][ci] = a
                reports[si][ci]  = r
                te_best[si, ci]  = select_best_te_ev(a)
            except Exception:
                pass

        if progress_cb is not None:
            progress_cb(ci + 1, n_cycles)

    return {
        "voltages": voltages,
        "currents": currents,
        "analyses": analyses,
        "reports":  reports,
        "te_best":  te_best,
    }


def _section(parent: ctk.CTkFrame, title: str) -> None:
    ctk.CTkFrame(parent, height=1, fg_color=("gray60", "gray40")).pack(
        fill="x", padx=8, pady=(10, 2))
    ctk.CTkLabel(parent, text=title,
                 font=ctk.CTkFont(size=12, weight="bold")).pack(
        anchor="w", padx=10, pady=(0, 4))


def _info_row(parent: ctk.CTkFrame, label: str, var: ctk.StringVar) -> None:
    row = ctk.CTkFrame(parent, fg_color="transparent")
    row.pack(fill="x", padx=10, pady=1)
    ctk.CTkLabel(row, text=f"{label}:", width=82, anchor="w",
                 font=ctk.CTkFont(size=11)).pack(side="left")
    ctk.CTkLabel(row, textvariable=var, anchor="w",
                 font=ctk.CTkFont(size=11, family="Courier")).pack(side="left")


def _nav_row(
    parent: ctk.CTkFrame,
    label: str,
    var: ctk.StringVar,
    dec_cmd,
    inc_cmd,
) -> None:
    row = ctk.CTkFrame(parent, fg_color="transparent")
    row.pack(fill="x", padx=10, pady=3)
    ctk.CTkLabel(row, text=label, width=68, anchor="w",
                 font=ctk.CTkFont(size=11)).pack(side="left")
    ctk.CTkButton(row, text="◀", width=34, height=28,
                  command=dec_cmd).pack(side="left", padx=2)
    ctk.CTkLabel(row, textvariable=var, width=90, anchor="center",
                 font=ctk.CTkFont(size=11, family="Courier")).pack(side="left")
    ctk.CTkButton(row, text="▶", width=34, height=28,
                  command=inc_cmd).pack(side="left", padx=2)


# ---------------------------------------------------------------------------
# Inspector
# ---------------------------------------------------------------------------

class FitInspector:

    def __init__(self, run: LapdRun, pos_idx: int, cycle_idx: int,
                 output_csv: Path, sweeps_hdf5: Path | None = None) -> None:
        self.run      = run
        self.run_id   = run.config.run_id
        self.n_pos    = run.position_count()
        self.n_shots  = run.shots_per_position()
        self.n_cycles = run.config.sweep.n_cycles
        sw = run.config.sweep
        self.cycle_times_ms = np.array(
            [(sw.t0_s + k * sw.tau_cycle_s) * 1e3 for k in range(self.n_cycles)]
        )
        self.output_csv = output_csv

        self.pos_idx   = max(0, min(pos_idx,   self.n_pos - 1))
        self.cycle_idx = max(0, min(cycle_idx, self.n_cycles - 1))
        self.shot_idx  = 0

        self.selections: dict[tuple[int, int], dict] = {}
        if output_csv.exists():
            self._load_csv()

        # Review flags: (51, n_cycles) uint8 bitmask loaded from HDF5
        self._review_flags: np.ndarray | None = None   # None = no flags available
        self._review_list: list[tuple[int, int]] = []   # sorted [(pos, cycle), ...]
        self._review_idx: int = -1                      # position within _review_list
        self._load_review_flags(sweeps_hdf5)

        # Full raw data (loaded once)
        self._raw_i: np.ndarray | None    = None
        self._raw_v: np.ndarray | None    = None
        self._ramp_slices: list[slice] | None = None

        # Per-position cache
        self._cached_pos: int | None = None
        self._pos_data: dict | None  = None

        # Draggable fit-window state
        self._v_lo: float | None   = None
        self._v_hi: float | None   = None
        self._dragging: str | None = None
        self._vlo_artist           = None
        self._vhi_artist           = None
        self._custom_exp_fit       = None
        self._custom_mask          = None
        self._custom_cell: tuple | None = None

        self._build_gui()

    # ------------------------------------------------------------------
    # GUI construction
    # ------------------------------------------------------------------

    def _build_gui(self) -> None:
        self.root = ctk.CTk()
        self.root.title(f"Te Fit Inspector — run {self.run_id}")
        self.root.geometry("1620x940")
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=0)
        self.root.grid_rowconfigure(0, weight=1)

        # ── Left: matplotlib canvas ───────────────────────────────────
        canvas_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        canvas_frame.grid(row=0, column=0, sticky="nsew", padx=(4, 2), pady=4)

        self.fig = Figure(figsize=(13, 8.8), dpi=100)
        gs = gridspec.GridSpec(
            3, 2, figure=self.fig,
            left=0.07, right=0.97, top=0.94, bottom=0.06,
            wspace=0.32, hspace=0.38,
            height_ratios=[4, 3.5, 1.2],
        )
        self.ax_iv    = self.fig.add_subplot(gs[0, 0])
        self.ax_log   = self.fig.add_subplot(gs[0, 1])
        self.ax_te    = self.fig.add_subplot(gs[1, :])
        self.ax_cycle = self.fig.add_subplot(gs[2, :])

        self.mpl_canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.mpl_canvas.get_tk_widget().pack(fill="both", expand=True)

        self.fig.canvas.mpl_connect("button_press_event",  self._on_mouse_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.fig.canvas.mpl_connect("button_release_event",self._on_mouse_release)

        # ── Right: control panel ─────────────────────────────────────
        right = ctk.CTkScrollableFrame(self.root, width=290)
        right.grid(row=0, column=1, sticky="nsew", padx=(2, 4), pady=4)
        self._build_controls(right)

    def _build_controls(self, parent: ctk.CTkScrollableFrame) -> None:

        # Run header
        ctk.CTkLabel(
            parent,
            text=f"Run {self.run_id}  ·  port {self.run.config.probe.port}",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", padx=10, pady=(8, 0))

        # Status bar for load progress
        self.sv_status = ctk.StringVar(value="Loading…")
        ctk.CTkLabel(parent, textvariable=self.sv_status,
                     font=ctk.CTkFont(size=10),
                     text_color=("gray40", "gray60")).pack(anchor="w", padx=10, pady=(0, 4))

        # ── Review navigation ─────────────────────────────────────────
        _section(parent, "Review Navigation")

        n_flagged = len(self._review_list)
        flag_avail = self._review_flags is not None
        flag_hint = (f"{n_flagged} flagged cells" if flag_avail
                     else "No flags loaded — run flag_te_review.py")
        ctk.CTkLabel(parent, text=flag_hint,
                     font=ctk.CTkFont(size=10),
                     text_color=("#c06010" if flag_avail else ("gray40", "gray60"))
                     ).pack(anchor="w", padx=10, pady=(0, 4))

        self.sv_review_pos = ctk.StringVar(value="—")
        row_rev = ctk.CTkFrame(parent, fg_color="transparent")
        row_rev.pack(fill="x", padx=10, pady=2)
        self._btn_prev_review = ctk.CTkButton(
            row_rev, text="◀ Prev Review", width=118, state="normal" if flag_avail else "disabled",
            command=self._prev_review)
        self._btn_prev_review.pack(side="left", padx=(0, 4))
        self._btn_next_review = ctk.CTkButton(
            row_rev, text="Next Review ▶", width=118, state="normal" if flag_avail else "disabled",
            command=self._next_review)
        self._btn_next_review.pack(side="left")

        # Flag-reason label for the current cell
        self.sv_flag_reason = ctk.StringVar(value="")
        ctk.CTkLabel(parent, textvariable=self.sv_flag_reason,
                     font=ctk.CTkFont(size=10, family="Courier"),
                     text_color="#e08030",
                     wraplength=260, justify="left").pack(anchor="w", padx=12, pady=(2, 0))

        # Review counter label  (e.g. "12 / 84")
        self.sv_review_counter = ctk.StringVar(value="")
        ctk.CTkLabel(parent, textvariable=self.sv_review_counter,
                     font=ctk.CTkFont(size=10),
                     text_color=("gray40", "gray60")).pack(anchor="w", padx=12, pady=(0, 2))

        # ── Navigation ────────────────────────────────────────────────
        _section(parent, "Navigation")

        self.sv_pos  = ctk.StringVar()
        self.sv_cyc  = ctk.StringVar()
        self.sv_shot = ctk.StringVar()

        _nav_row(parent, "Position", self.sv_pos,  self._prev_pos,   self._next_pos)
        _nav_row(parent, "Cycle",    self.sv_cyc,  self._prev_cycle, self._next_cycle)
        _nav_row(parent, "Shot",     self.sv_shot, self._prev_shot,  self._next_shot)

        # ── Status ────────────────────────────────────────────────────
        _section(parent, "Status")

        self.sv_x       = ctk.StringVar()
        self.sv_t       = ctk.StringVar()
        self.sv_qual    = ctk.StringVar()
        self.sv_vp      = ctk.StringVar()
        self.sv_te_log  = ctk.StringVar()
        self.sv_te_exp  = ctk.StringVar()
        self.sv_te_best = ctk.StringVar()
        self.sv_te_cust = ctk.StringVar()
        self.sv_sel     = ctk.StringVar()
        self.sv_nover   = ctk.StringVar()

        _info_row(parent, "x position", self.sv_x)
        _info_row(parent, "time",       self.sv_t)
        _info_row(parent, "quality",    self.sv_qual)
        _info_row(parent, "Vp",         self.sv_vp)
        _info_row(parent, "Te log",     self.sv_te_log)
        _info_row(parent, "Te exp",     self.sv_te_exp)
        _info_row(parent, "Te best",    self.sv_te_best)
        _info_row(parent, "Te custom",  self.sv_te_cust)
        _info_row(parent, "Selection",  self.sv_sel)
        _info_row(parent, "Overrides",  self.sv_nover)

        # ── Fit window ────────────────────────────────────────────────
        _section(parent, "Fit Window")
        ctk.CTkLabel(parent, text="Drag ● green/red lines in log panel",
                     font=ctk.CTkFont(size=10),
                     text_color=("gray40", "gray60")).pack(anchor="w", padx=10, pady=(0, 2))

        self.sv_vlo = ctk.StringVar(value="v_lo = —")
        self.sv_vhi = ctk.StringVar(value="v_hi = —")
        ctk.CTkLabel(parent, textvariable=self.sv_vlo,
                     font=ctk.CTkFont(size=11, family="Courier"),
                     text_color="#2ca02c").pack(anchor="w", padx=14)
        ctk.CTkLabel(parent, textvariable=self.sv_vhi,
                     font=ctk.CTkFont(size=11, family="Courier"),
                     text_color="#e05050").pack(anchor="w", padx=14)
        ctk.CTkButton(parent, text="Reset Bounds", width=240,
                      command=self._reset_bounds).pack(padx=10, pady=(6, 2))

        # ── Record ────────────────────────────────────────────────────
        _section(parent, "Record Selection")

        row1 = ctk.CTkFrame(parent, fg_color="transparent")
        row1.pack(padx=10, pady=3)
        ctk.CTkButton(row1, text="Log",    width=72, fg_color="#1a6ca8", hover_color="#1557856",
                      command=lambda: self._record("log")).pack(side="left", padx=3)
        ctk.CTkButton(row1, text="Exp",    width=72, fg_color="#c06010", hover_color="#9a4c0c",
                      command=lambda: self._record("exp")).pack(side="left", padx=3)
        ctk.CTkButton(row1, text="Custom", width=82, fg_color="#6a3090", hover_color="#522470",
                      command=self._record_custom).pack(side="left", padx=3)

        row2 = ctk.CTkFrame(parent, fg_color="transparent")
        row2.pack(padx=10, pady=3)
        ctk.CTkButton(row2, text="Auto-clear", width=115,
                      command=self._auto_clear).pack(side="left", padx=3)
        ctk.CTkButton(row2, text="Bad / NaN",  width=115, fg_color="#962020", hover_color="#741818",
                      command=lambda: self._record("bad")).pack(side="left", padx=3)

        # ── File ──────────────────────────────────────────────────────
        _section(parent, "File")
        ctk.CTkButton(parent, text="Save CSV", width=240, fg_color="#1a7a1a", hover_color="#125e12",
                      command=self._save).pack(padx=10, pady=3)
        ctk.CTkButton(parent, text="Quit", width=240,
                      command=self._quit).pack(padx=10, pady=(3, 10))

    # ------------------------------------------------------------------
    # Review-flag loading
    # ------------------------------------------------------------------

    def _load_review_flags(self, hdf5_path: Path | None) -> None:
        """Try to load review_flags for this run from the processed HDF5."""
        path = hdf5_path or SWEEPS_HDF5
        if not path.exists():
            return
        try:
            with h5py.File(path, "r") as hf:
                es_root = hf.get("experiment_sets")
                if es_root is None:
                    return
                for es_id in es_root:
                    run_grp = es_root[es_id].get(self.run_id)
                    if run_grp is None:
                        continue
                    if "review_flags" not in run_grp:
                        return
                    self._review_flags = run_grp["review_flags"][:].astype(np.uint8)
                    # Build sorted list of all flagged (pos, cycle) pairs
                    pos_idx_arr, cyc_idx_arr = np.where(self._review_flags != 0)
                    self._review_list = sorted(zip(pos_idx_arr.tolist(), cyc_idx_arr.tolist()))
                    n = len(self._review_list)
                    print(f"Loaded review flags: {n} flagged cells "
                          f"({100*n / self._review_flags.size:.1f}% of {self._review_flags.size})")
                    return
        except Exception as exc:
            print(f"Warning: could not load review flags from {path}: {exc}")

    def _review_flag_at(self, pos_idx: int, cycle_idx: int) -> int:
        """Return the flag byte for (pos_idx, cycle_idx), or 0 if no flags loaded."""
        if self._review_flags is None:
            return 0
        return int(self._review_flags[pos_idx, cycle_idx])

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _ensure_raw_loaded(self) -> None:
        if self._raw_i is not None:
            return
        run   = self.run
        i_off = run.default_zero_offset_v(ChannelKind.I_SWEEP)
        v_off = run.default_zero_offset_v(ChannelKind.V_SWEEP)
        self.sv_status.set("Reading HDF5…")
        self.root.update()
        self._raw_i = run.langmuir_traces(ChannelKind.I_SWEEP, zero_offset_v=i_off)
        self._raw_v = run.langmuir_traces(ChannelKind.V_SWEEP, zero_offset_v=v_off)
        self._ramp_slices = run.sweep_ramp_sample_slices(clip_s=CLIP_US * 1e-6)

    def _refresh_position(self) -> None:
        if self._cached_pos == self.pos_idx:
            return
        n    = self.n_cycles
        core_hz, edge_hz = _cutoffs_hz(self.run)
        cutoff_hz = edge_hz if abs(X_CM[self.pos_idx]) > EDGE_X_CM else core_hz

        def _progress(done, total):
            self.sv_status.set(f"Fitting cycle {done}/{total}…")
            self.root.update_idletasks()

        self._pos_data   = _fit_position(
            self._raw_i, self._raw_v, self._ramp_slices,
            self.pos_idx, self.run.sample_rate_hz(), cutoff_hz,
            progress_cb=_progress,
        )
        self._cached_pos = self.pos_idx
        self.shot_idx    = 0
        self._clear_custom()
        self.sv_status.set("Ready")

    # ------------------------------------------------------------------
    # Figure drawing
    # ------------------------------------------------------------------

    def _update(self) -> None:
        pd      = self._pos_data
        pi      = self.pos_idx
        ci      = self.cycle_idx
        si      = self.shot_idx
        n_shots = self.n_shots

        v_cur = pd["voltages"][si, ci]
        i_cur = pd["currents"][si, ci]
        a_cur = pd["analyses"][si][ci]
        r_cur = pd["reports"][si][ci]

        sev   = r_cur.severity if r_cur is not None else "bad"
        color = _SEV_COLOR.get(sev, _SEV_COLOR["bad"])

        vp = (a_cur.plasma_potential_derivative_v
              if a_cur is not None and np.isfinite(a_cur.plasma_potential_derivative_v)
              else None)

        # ── I-V curve ────────────────────────────────────────────────
        ax = self.ax_iv
        ax.clear()
        for s in range(n_shots):
            v_s = pd["voltages"][s, ci]
            i_s = pd["currents"][s, ci]
            if np.isfinite(v_s).any():
                ax.plot(v_s, i_s, color="0.80", lw=0.5, zorder=1)
        ax.plot(v_cur, i_cur, color=color, lw=1.4, zorder=3)

        if a_cur is not None:
            v_rng = np.linspace(float(np.nanmin(v_cur)), float(np.nanmax(v_cur)), 300)
            ax.plot(v_rng, a_cur.ion_fit.evaluate(v_rng),
                    "k--", lw=1.0, label="ion fit", zorder=4)
            mask_l = a_cur.log_linear_fit.mask
            v_l    = a_cur.voltage[mask_l]
            if v_l.size:
                ax.plot(v_l,
                        a_cur.log_linear_fit.electron_current(v_l)
                        + a_cur.ion_fit.evaluate(v_l),
                        color="#1f77b4", lw=1.8,
                        label=f"log  {a_cur.log_linear_fit.electron_temperature_ev:.2f} eV",
                        zorder=5)
            mask_e  = a_cur.exponential_fit.mask
            v_e_rng = np.linspace(float(a_cur.voltage[mask_e].min()),
                                  float(a_cur.voltage[mask_e].max()), 200)
            ax.plot(v_e_rng, a_cur.exponential_fit.evaluate(v_e_rng),
                    color="#ff7f0e", lw=1.8,
                    label=f"exp  {a_cur.exponential_fit.electron_temperature_ev:.2f} eV",
                    zorder=5)
            if (self._custom_exp_fit is not None and
                    self._custom_cell == (pi, ci, si) and
                    self._custom_mask is not None):
                cef = self._custom_exp_fit
                v_c = np.linspace(float(a_cur.voltage[self._custom_mask].min()),
                                  float(a_cur.voltage[self._custom_mask].max()), 200)
                ax.plot(v_c, cef.evaluate(v_c),
                        color="#9b59b6", lw=2.0, ls="--",
                        label=f"custom  {cef.electron_temperature_ev:.2f} eV", zorder=6)

        if vp is not None:
            ax.axvline(vp, color="0.45", lw=1.0, ls=":", zorder=6,
                       label=f"$V_p$ = {vp:.1f} V")
        ax.set_xlabel("V (V)", fontsize=8)
        ax.set_ylabel("I (A)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, loc="upper left")
        ax.set_title("I-V  (all shots grey, current coloured)", fontsize=8)

        # ── log(I_e) vs V — retarding region ─────────────────────────
        ax = self.ax_log
        ax.clear()
        self._vlo_artist = None
        self._vhi_artist = None

        if a_cur is not None:
            mask  = a_cur.log_linear_fit.mask
            v_ret = a_cur.voltage[mask]
            i_e   = a_cur.current[mask] - a_cur.ion_fit.evaluate(v_ret)
            pos   = i_e > 0
            if pos.any():
                ax.scatter(v_ret[pos], np.log(i_e[pos]),
                           s=10, color=color, zorder=3, label="data")

            v_plt = np.linspace(float(v_ret.min()), float(v_ret.max()), 200)
            lf = a_cur.log_linear_fit
            ax.plot(v_plt, lf.intercept + lf.slope * v_plt,
                    color="#1f77b4", lw=1.8,
                    label=f"log  $T_e$={lf.electron_temperature_ev:.2f} eV")
            ef = a_cur.exponential_fit
            pred = np.clip(ef.log_amplitude + ef.inverse_temperature * v_plt, -700, 700)
            ax.plot(v_plt, pred,
                    color="#ff7f0e", lw=1.8,
                    label=f"exp  $T_e$={ef.electron_temperature_ev:.2f} eV")

            if vp is not None:
                ax.axvline(vp, color="0.45", lw=1.0, ls=":",
                           label=f"$V_p$ = {vp:.1f} V", zorder=6)

            if (self._custom_exp_fit is not None and
                    self._custom_cell == (pi, ci, si)):
                cef   = self._custom_exp_fit
                v_cw  = np.linspace(self._v_lo, self._v_hi, 150)
                cpred = np.clip(cef.log_amplitude + cef.inverse_temperature * v_cw, -700, 700)
                ax.plot(v_cw, cpred, color="#9b59b6", lw=2.5, ls="--",
                        label=f"custom  $T_e$={cef.electron_temperature_ev:.2f} eV", zorder=7)

            # Initialise draggable bounds to retarding region on first draw
            if self._v_lo is None:
                self._v_lo = float(v_ret.min())
            if self._v_hi is None:
                self._v_hi = float(v_ret.max())

            self._vlo_artist = ax.axvline(
                self._v_lo, color="#2ca02c", lw=2.0, ls="-", zorder=8,
                label=f"v_lo = {self._v_lo:.1f} V")
            self._vhi_artist = ax.axvline(
                self._v_hi, color="#d62728", lw=2.0, ls="-", zorder=8,
                label=f"v_hi = {self._v_hi:.1f} V")

        ax.set_xlabel("V (V)", fontsize=8)
        ax.set_ylabel("ln(I$_e$)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.0, loc="upper left")
        ax.set_title("Retarding region — drag green/red lines to set fit window", fontsize=7.5)

        # ── Te per shot ──────────────────────────────────────────────
        ax = self.ax_te
        ax.clear()
        te_log  = np.full(n_shots, np.nan)
        te_exp  = np.full(n_shots, np.nan)
        te_best = np.full(n_shots, np.nan)
        dot_col = []
        for s in range(n_shots):
            a_s = pd["analyses"][s][ci]
            r_s = pd["reports"][s][ci]
            dot_col.append(_SEV_COLOR.get(
                r_s.severity if r_s is not None else "bad", _SEV_COLOR["bad"]))
            if a_s is not None:
                te_log[s]  = a_s.log_linear_fit.electron_temperature_ev
                te_exp[s]  = a_s.exponential_fit.electron_temperature_ev
                te_best[s] = select_best_te_ev(a_s)

        xs = np.arange(n_shots)
        ax.scatter(xs, te_log,  marker="^", s=36, color="#1f77b4", alpha=0.75,
                   label="log ▲",  zorder=3)
        ax.scatter(xs, te_exp,  marker="v", s=36, color="#ff7f0e", alpha=0.75,
                   label="exp ▼",  zorder=3)
        ax.scatter(xs, te_best, marker="o", s=24, color="k", alpha=0.90,
                   label="best ●", zorder=4)
        ax.axvline(si, color="0.45", lw=1.2, ls="--", zorder=2)
        for s, c in enumerate(dot_col):
            ax.axvline(s, color=c, lw=0.4, alpha=0.35, zorder=1)

        sel = self.selections.get((pi, ci))
        if sel is not None:
            if sel["method"] != "bad" and np.isfinite(sel["te_ev"]):
                ax.scatter([sel["shot_idx"]], [sel["te_ev"]],
                           marker="*", s=200, color="#9b59b6", zorder=6,
                           label=f"override ({sel['method']}) ★")
                ax.axhline(sel["te_ev"], color="#9b59b6", lw=0.8, ls="-.", alpha=0.55, zorder=2)
            title_str = ("★ OVERRIDE" if sel["method"] != "bad" else "✗ BAD")
            title_col = "#9b59b6" if sel["method"] != "bad" else "#d62728"
            ax.set_title(f"$T_e$ per shot — {title_str}", fontsize=8, color=title_col)
        else:
            ax.set_title("$T_e$ per shot — this cycle", fontsize=8)

        ax.set_xlabel("shot index", fontsize=8)
        ax.set_ylabel("$T_e$ (eV)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, loc="upper right", ncol=4)

        # ── Cycle overview strip ──────────────────────────────────────
        ax = self.ax_cycle
        ax.clear()
        tb = pd["te_best"]
        te_med = np.nanmedian(tb, axis=0)
        te_p25 = np.nanpercentile(tb, 25, axis=0)
        te_p75 = np.nanpercentile(tb, 75, axis=0)
        cxs    = np.arange(self.n_cycles)

        ax.fill_between(cxs, te_p25, te_p75, alpha=0.25, color="#2ca02c", zorder=1)
        ax.plot(cxs, te_med, color="#2ca02c", lw=1.0, zorder=2)

        finite_med = te_med[np.isfinite(te_med)]
        y_ref = float(np.nanmin(te_p25)) if np.isfinite(te_p25).any() else 0.0
        for (pi2, ci2), sel2 in self.selections.items():
            if pi2 != pi:
                continue
            if sel2["method"] == "bad":
                ax.scatter([ci2], [y_ref], marker="x", s=35,
                           color="#d62728", zorder=4, linewidths=1.5)
            elif np.isfinite(sel2["te_ev"]):
                ax.scatter([ci2], [sel2["te_ev"]], marker="D", s=22,
                           color="#9b59b6", zorder=4)

        # Highlight flagged cycles as translucent orange background bands
        if self._review_flags is not None:
            flags_pos = self._review_flags[pi, :]   # (n_cycles,)
            for cyc_i, fb in enumerate(flags_pos):
                if fb:
                    ax.axvspan(cyc_i - 0.45, cyc_i + 0.45,
                               color="#e08030", alpha=0.18, zorder=0, linewidth=0)

        ax.axvline(ci, color="#ff7f0e", lw=1.5, alpha=0.85, zorder=3)
        ax.set_xlabel("cycle index", fontsize=7)
        ax.set_ylabel("$T_e$ (eV)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_xlim(-0.5, self.n_cycles - 0.5)
        ax.set_title(
            f"$T_e$ best vs cycle — pos {pi}  x={X_CM[pi]:.1f} cm"
            f"  (IQR band;  ◆=override  ×=bad)",
            fontsize=7)

        # Figure title
        self.fig.suptitle(
            f"Te Fit Inspector  ·  run {self.run_id}  ·  "
            f"{len(self.selections)} override(s)",
            fontsize=10)
        self.mpl_canvas.draw_idle()

        # ── Update CTk labels ─────────────────────────────────────────
        te_l  = (a_cur.log_linear_fit.electron_temperature_ev
                 if a_cur is not None else float("nan"))
        te_e  = (a_cur.exponential_fit.electron_temperature_ev
                 if a_cur is not None else float("nan"))
        te_b  = select_best_te_ev(a_cur) if a_cur is not None else float("nan")

        self.sv_pos.set(f"{pi} / {self.n_pos-1}")
        self.sv_cyc.set(f"{ci} / {self.n_cycles-1}")
        self.sv_shot.set(f"{si} / {self.n_shots-1}")

        self.sv_x.set(f"{X_CM[pi]:.1f} cm")
        self.sv_t.set(f"{self.cycle_times_ms[ci]:.1f} ms")
        self.sv_qual.set(sev)
        self.sv_vp.set(f"{vp:.2f} V" if vp is not None else "n/a")
        self.sv_te_log.set(f"{te_l:.3f} eV" if np.isfinite(te_l) else "—")
        self.sv_te_exp.set(f"{te_e:.3f} eV" if np.isfinite(te_e) else "—")
        self.sv_te_best.set(f"{te_b:.3f} eV" if np.isfinite(te_b) else "—")

        if (self._custom_exp_fit is not None and
                self._custom_cell == (pi, ci, si)):
            te_c = self._custom_exp_fit.electron_temperature_ev
            self.sv_te_cust.set(f"{te_c:.3f} eV")
        else:
            self.sv_te_cust.set("— (drag bounds)")

        if sel is not None:
            te_str = (f"{sel['te_ev']:.3f} eV"
                      if np.isfinite(sel["te_ev"]) else "NaN")
            self.sv_sel.set(f"[{sel['method']}] {te_str}")
        else:
            self.sv_sel.set("auto")

        self.sv_nover.set(str(len(self.selections)))

        if self._v_lo is not None:
            self.sv_vlo.set(f"v_lo = {self._v_lo:.2f} V")
            self.sv_vhi.set(f"v_hi = {self._v_hi:.2f} V")
        else:
            self.sv_vlo.set("v_lo = —")
            self.sv_vhi.set("v_hi = —")

        # ── Review flag labels ────────────────────────────────────────
        flag_byte = self._review_flag_at(pi, ci)
        reasons   = _decode_flags(flag_byte)
        self.sv_flag_reason.set(reasons if reasons else "")

        # Keep _review_idx in sync when user navigates manually
        cur = (pi, ci)
        if self._review_list:
            import bisect
            i = bisect.bisect_left(self._review_list, cur)
            if i < len(self._review_list) and self._review_list[i] == cur:
                self._review_idx = i
                n = len(self._review_list)
                self.sv_review_counter.set(f"{i+1} / {n}")
            else:
                self._review_idx = -1
                self.sv_review_counter.set(f"— / {len(self._review_list)}")
        else:
            self.sv_review_counter.set("")

    # ------------------------------------------------------------------
    # Mouse handlers (draggable fit-window bounds in ax_log)
    # ------------------------------------------------------------------

    def _on_mouse_press(self, event) -> None:
        if event.inaxes is not self.ax_log or event.button != 1:
            return
        if event.xdata is None or self._v_lo is None:
            return
        lo_pix = self.ax_log.transData.transform([[self._v_lo, 0]])[0, 0]
        hi_pix = self.ax_log.transData.transform([[self._v_hi, 0]])[0, 0]
        d_lo   = abs(event.x - lo_pix)
        d_hi   = abs(event.x - hi_pix)
        thresh  = 12  # pixels
        if d_lo < d_hi and d_lo < thresh:
            self._dragging = "lo"
        elif d_hi <= d_lo and d_hi < thresh:
            self._dragging = "hi"

    def _on_mouse_move(self, event) -> None:
        if self._dragging is None:
            return
        if event.inaxes is not self.ax_log or event.xdata is None:
            return
        x = float(event.xdata)
        if self._dragging == "lo":
            self._v_lo = min(x, self._v_hi - 0.5)
            if self._vlo_artist is not None:
                self._vlo_artist.set_xdata([self._v_lo, self._v_lo])
                self.sv_vlo.set(f"v_lo = {self._v_lo:.2f} V")
        else:
            self._v_hi = max(x, self._v_lo + 0.5)
            if self._vhi_artist is not None:
                self._vhi_artist.set_xdata([self._v_hi, self._v_hi])
                self.sv_vhi.set(f"v_hi = {self._v_hi:.2f} V")
        self.mpl_canvas.draw_idle()

    def _on_mouse_release(self, event) -> None:
        if self._dragging is None:
            return
        was = self._dragging
        self._dragging = None
        if event.inaxes is self.ax_log and event.xdata is not None:
            x = float(event.xdata)
            if was == "lo":
                self._v_lo = min(x, self._v_hi - 0.5)
            else:
                self._v_hi = max(x, self._v_lo + 0.5)
            self._refit_custom()
            self._update()

    def _refit_custom(self) -> None:
        a = self._pos_data["analyses"][self.shot_idx][self.cycle_idx]
        if a is None or self._v_lo is None:
            return
        v  = a.voltage
        ie = a.current - a.ion_fit.evaluate(v)
        mask = (v >= self._v_lo) & (v <= self._v_hi) & (ie > 0)
        if mask.sum() < 5:
            return
        try:
            cef = _run_exp_fit(v, a.current, a.ion_fit, a.log_linear_fit, mask)
            self._custom_exp_fit = cef
            self._custom_mask    = mask
            self._custom_cell    = (self.pos_idx, self.cycle_idx, self.shot_idx)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Review navigation
    # ------------------------------------------------------------------

    def _go_to_review(self, idx: int) -> None:
        """Navigate to the flagged cell at index *idx* in _review_list."""
        if not self._review_list:
            return
        idx = max(0, min(idx, len(self._review_list) - 1))
        self._review_idx = idx
        new_pos, new_cyc = self._review_list[idx]

        pos_changed = new_pos != self.pos_idx
        self.pos_idx   = new_pos
        self.cycle_idx = new_cyc
        self.shot_idx  = 0
        self._clear_custom()

        if pos_changed:
            self._ensure_raw_loaded()
            self._refresh_position()
        self._update()

    def _prev_review(self) -> None:
        if not self._review_list:
            return
        # Find index of current cell, go one before it
        cur = (self.pos_idx, self.cycle_idx)
        if self._review_idx > 0 and self._review_list[self._review_idx] == cur:
            self._go_to_review(self._review_idx - 1)
        else:
            # Locate insertion point and step back
            import bisect
            i = bisect.bisect_left(self._review_list, cur)
            self._go_to_review(max(0, i - 1))

    def _next_review(self) -> None:
        if not self._review_list:
            return
        cur = (self.pos_idx, self.cycle_idx)
        if self._review_idx >= 0 and self._review_list[self._review_idx] == cur:
            self._go_to_review(self._review_idx + 1)
        else:
            import bisect
            i = bisect.bisect_right(self._review_list, cur)
            self._go_to_review(min(i, len(self._review_list) - 1))

    # ------------------------------------------------------------------
    # Navigation callbacks
    # ------------------------------------------------------------------

    def _prev_pos(self) -> None:
        if self.pos_idx > 0:
            self.pos_idx -= 1
            self._ensure_raw_loaded()
            self._refresh_position()
            self._update()

    def _next_pos(self) -> None:
        if self.pos_idx < self.n_pos - 1:
            self.pos_idx += 1
            self._ensure_raw_loaded()
            self._refresh_position()
            self._update()

    def _prev_cycle(self) -> None:
        if self.cycle_idx > 0:
            self.cycle_idx -= 1
            self._clear_custom()
            self._update()

    def _next_cycle(self) -> None:
        if self.cycle_idx < self.n_cycles - 1:
            self.cycle_idx += 1
            self._clear_custom()
            self._update()

    def _prev_shot(self) -> None:
        if self.shot_idx > 0:
            self.shot_idx -= 1
            self._clear_custom()
            self._update()

    def _next_shot(self) -> None:
        if self.shot_idx < self.n_shots - 1:
            self.shot_idx += 1
            self._clear_custom()
            self._update()

    # ------------------------------------------------------------------
    # Record / selection callbacks
    # ------------------------------------------------------------------

    def _record(self, method: str) -> None:
        a = self._pos_data["analyses"][self.shot_idx][self.cycle_idx]
        if method == "log":
            te = (a.log_linear_fit.electron_temperature_ev
                  if a is not None else float("nan"))
        elif method == "exp":
            te = (a.exponential_fit.electron_temperature_ev
                  if a is not None else float("nan"))
        else:
            te = float("nan")
        self.selections[(self.pos_idx, self.cycle_idx)] = {
            "method":   method,
            "te_ev":    te,
            "shot_idx": self.shot_idx,
        }
        self._update()

    def _record_custom(self) -> None:
        pi, ci, si = self.pos_idx, self.cycle_idx, self.shot_idx
        if (self._custom_exp_fit is None or
                self._custom_cell != (pi, ci, si)):
            self.sv_status.set("No custom fit — drag bounds first.")
            return
        te = self._custom_exp_fit.electron_temperature_ev
        self.selections[(pi, ci)] = {
            "method":   "exp",
            "te_ev":    te,
            "shot_idx": si,
        }
        self._update()

    def _auto_clear(self) -> None:
        self.selections.pop((self.pos_idx, self.cycle_idx), None)
        self._update()

    def _reset_bounds(self) -> None:
        self._v_lo = None
        self._v_hi = None
        self._clear_custom()
        self._update()

    def _clear_custom(self) -> None:
        self._custom_exp_fit = None
        self._custom_mask    = None
        self._custom_cell    = None
        self._v_lo           = None
        self._v_hi           = None

    # ------------------------------------------------------------------
    # CSV I/O
    # ------------------------------------------------------------------

    def _save(self) -> None:
        if not self.selections:
            self.sv_status.set("Nothing to save.")
            return
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["run_id", "pos_idx", "x_cm", "cycle_idx",
                      "time_ms", "shot_idx", "method", "te_ev"]
        rows = []
        for (pi, ci), sel in sorted(self.selections.items()):
            te_str = ("" if not np.isfinite(sel["te_ev"])
                      else f"{sel['te_ev']:.6f}")
            rows.append({
                "run_id":    self.run_id,
                "pos_idx":   pi,
                "x_cm":      f"{X_CM[pi]:.1f}",
                "cycle_idx": ci,
                "time_ms":   f"{self.cycle_times_ms[ci]:.3f}",
                "shot_idx":  sel["shot_idx"],
                "method":    sel["method"],
                "te_ev":     te_str,
            })
        with open(self.output_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        self.sv_status.set(f"Saved {len(rows)} override(s).")
        print(f"Saved {len(rows)} override(s) → {self.output_csv}")

    def _load_csv(self) -> None:
        with open(self.output_csv, newline="") as fh:
            for row in csv.DictReader(fh):
                pi = int(row["pos_idx"])
                ci = int(row["cycle_idx"])
                te_str = row.get("te_ev", "")
                te = float(te_str) if te_str else float("nan")
                self.selections[(pi, ci)] = {
                    "method":   row["method"],
                    "te_ev":    te,
                    "shot_idx": int(row.get("shot_idx", 0)),
                }
        print(f"Loaded {len(self.selections)} existing override(s) from {self.output_csv}")

    def _quit(self) -> None:
        self.root.destroy()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run_event_loop(self) -> None:
        self._ensure_raw_loaded()
        self._refresh_position()
        self._update()
        self.root.mainloop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-id", required=True,
                        help="Two-digit run ID to inspect (e.g. 32).")
    parser.add_argument("--pos", type=int, default=25, metavar="IDX",
                        help="Starting position index 0–50.  Default: 25 (x=0 cm).")
    parser.add_argument("--cycle", type=int, default=0, metavar="IDX",
                        help="Starting cycle index 0–39.  Default: 0.")
    parser.add_argument(
        "--output", type=Path,
        default=Path("processed/manual_te_selections.csv"),
        help="CSV file for manual overrides.  Existing file loaded on startup.",
    )
    parser.add_argument(
        "--sweeps-hdf5", type=Path, default=None, metavar="PATH",
        help=(
            "Path to langmuir_sweeps.hdf5 containing review_flags datasets "
            f"(produced by flag_te_review.py).  Default: {SWEEPS_HDF5} if it exists."
        ),
    )
    args = parser.parse_args()

    manifests = load_run_manifest(MANIFEST)
    if args.run_id not in manifests:
        raise ValueError(
            f"Run ID {args.run_id!r} not in manifest.  Available: {sorted(manifests)}"
        )

    run = LapdRun(manifests[args.run_id])
    if run.config.path is None or not run.config.path.exists():
        raise FileNotFoundError(
            f"HDF5 file for run {args.run_id} not found (path: {run.config.path})"
        )

    inspector = FitInspector(
        run=run,
        pos_idx=args.pos,
        cycle_idx=args.cycle,
        output_csv=args.output,
        sweeps_hdf5=args.sweeps_hdf5,
    )
    inspector.run_event_loop()


if __name__ == "__main__":
    main()
