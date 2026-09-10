"""ES4 rotation-pair half-differences under the sweep rest-bias correction.

What this asks
--------------
``MACH_FACE_ASYMMETRY_M_RMS`` and ``MACH_FACE_ASYMMETRY_M_RANGE`` in
``src/bapsf_lapd/density.py`` are measured from ROTATION PAIRS: the same port is
visited twice, once at rot 0 and once at rot 180, which exchanges which physical
probe face looks upstream.  Over the plateau window the pair MEAN of
``ln(J_up / J_dn)`` is the area-free Mach estimator and HALF the pair difference
is the face-asymmetry systematic, quoted in ``M = ln R / MACH_K``.  The declared
statistic is read at ``x = 0`` over 14-19 ms.

The nine ES1-ES3 pairs (three ports x three sets) are the measured set.  The
three ES4 pairs are excluded from it, on two stated grounds, and this instrument
re-tests ONE of them.

The ground it re-tests
----------------------
Every ion-saturation product is built from the dead-time current the probe
collects while the sweep supply parks it at a fixed negative rest bias.  That
rest bias is not the same in the two sets: on ES1-ES3 the swept face rests near
-74 V, on ES4 near -20 V (the offset frame of this repository's zero-offset
convention).  Ion current grows with the sheath, so the ES4 swept face reads LOW
against the ES1-ES3 convention by a factor F > 1 measured per port and per
rotation by ``scripts/es4_sweep_rest_bias_factor.py``.

That factor does not cancel out of a rotation-pair half-difference, and this is
the whole point: rotating the probe exchanges the faces, so the SWEPT face is
upstream in one rotation and downstream in the other.  Writing ``s = +1`` when
the swept face is upstream and ``s = -1`` when it is downstream,

    ln R_corrected = ln R_raw + s * ln F

so the correction enters the two rotations with OPPOSITE sign and lands on the
half-difference additively::

    half-difference_corrected = half-difference_raw
                                + (ln F_rot0 + ln F_rot180) / 2

An uncorrected ES4 half-difference therefore carries a convention term the
ES1-ES3 pairs do not, and is not a face-asymmetry measurement under the ES1-ES3
convention.  Removing that term is what this instrument does.

Which face is the swept one is NEVER assumed.  It is read per run from the Mach
product's own ``upstream_source_file`` / ``downstream_source_file`` attributes,
and a pair whose two faces cannot be told apart from those attributes is refused
rather than guessed.

What is deliberately NOT decided here
-------------------------------------
Nothing is written into the product chain and no constant in ``density.py``
moves.  Whether a corrected ES4 pair JOINS the face-asymmetry set is a separate
ruling; this instrument only reports where each corrected half-difference falls
against the range the nine ES1-ES3 pairs already span,
``MACH_FACE_ASYMMETRY_M_RANGE``, which it imports rather than restates.

The membership gate is that range's UPPER edge alone.  A magnitude at or below
the largest |M| the nine pairs reach is inside; there is no lower edge.  The
nine happen not to reach zero, but a face asymmetry that the correction leaves
indistinguishable from zero is a SMALLER asymmetry than any of them and not a
different kind of thing, so reading the set's smallest observed value as a floor
the quantity must clear would exclude a pair for agreeing with the convention
too well.

F is a measurement with a spread, so every port is evaluated three times: at F's
central value and at both ends of its spread, with the spread applied to BOTH
rotations together (that is what moves the half-difference furthest, since the
two ln F terms enter with the same sign after the rotation flip).  A verdict
that does not survive all three points is not a verdict.  Half the distance
between the low and high readings is the pair's BAR, printed beside its central
value: the corrected half-difference is a measurement, and F's spread is what
sets its width.

The masked pair
---------------
Run 43 -- ES4 port 21 at rot 180 -- carries a registered channel-state mask over
its ISAT face (``scripts/annotate_state_mask.py``), and the Mach product already
carries it as NaN.  The registration has two entries and between them they cover
the core of the profile and its whole far-positive side, so that pair has no
admitted cell at ``x = 0`` and its statistic cannot be read where the declared
one is read.  It is not silently dropped and it is not silently substituted:
when ``x = 0`` is empty the pair falls back to the cells that survive on BOTH
rotations -- for run 43 the low-state cells that lie outside the mask, which are
the two ends of the profile -- and every line it appears on is LABELLED with the
reduction and the cut actually used, both because that is what was read and
because a number read there sits beside the declared statistic rather than among
it.  A pair with no admitted cell at all is refused outright.

That fallback carries one registered cut, ``EDGE_CUT_CM``: positions at
|x| >= 23 cm are excluded from it.  Both faces are at the 0.1-1 mA floor out
there, and at x = -23 and -24 cm both runs of the pair carry an unexplained
x5-10 feature in ISAT against I_SWEEP, so a ratio taken there is a ratio of two
floor-level currents with a feature neither face explains.  Left in, those 15
cells carry the figure on their own.  The cut is on the EDGE-CELL reduction
alone; the declared ``x = 0`` reduction is untouched by it.

Controls
--------
Two, both printed before the ES4 table.

(i) The nine ES1-ES3 half-differences are recomputed from the same product and
    checked against ``CONTROL_HALF_DIFFERENCES_M``, the values the declared
    statistic already rests on.  If the control does not reproduce, the read is
    not comparable with the declared constants and the run says so.

(ii) The three ES4 half-differences are printed UNCORRECTED beside the corrected
     ones, so the size of the convention term is visible rather than asserted.

Usage
-----
    PYTHONPATH=src python scripts/es4_face_asymmetry_subset.py \
        --rest-bias-csv <es4_sweep_rest_bias_factor.csv> \
        --output es4_face_asymmetry_subset.csv

``--rest-bias-csv`` is required: the factor is a measurement made elsewhere and
this instrument will not invent one.  The output defaults to the working
directory and a path inside ``processed/`` is refused -- this writes
measurements, not products.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bapsf_lapd.density import (  # noqa: E402
    MACH_FACE_ASYMMETRY_M_RANGE,
    MACH_K,
)

#: Plateau window of the declared statistic, seconds (inclusive both ends).
PLATEAU_T_MIN_S = 14.0e-3
PLATEAU_T_MAX_S = 19.0e-3

#: Transverse position the declared statistic is read at, cm.
X_TARGET_CM = 0.0

#: Tolerance on matching ``X_TARGET_CM`` against the product's own ``x_cm``, cm.
X_MATCH_TOL_CM = 1.0e-6

#: Outermost |x| admitted to the edge-cell reduction, cm (strict: a position at
#: or beyond this is excluded).  REGISTERED CUT, and it applies to the edge-cell
#: reduction ONLY -- the declared ``x = 0`` reduction is untouched by it.
#: Positions at |x| >= 23 cm sit at the 0.1-1 mA floor on BOTH faces, and at
#: x = -23 and -24 cm both runs of the pair carry an unexplained x5-10 feature in
#: ISAT against I_SWEEP.  A ratio taken there is a ratio of two floor-level
#: currents with a feature neither face explains, so those positions are excluded
#: from the edge-cell figure by registration rather than left to dominate it:
#: they are 15 of the 75 cells that survive the mask and they carry the figure
#: from about +0.008 M to about -0.17 M on their own.
EDGE_CUT_CM = 23.0

#: Label of the declared reduction, and of the fallback used when it is empty.
#: The fallback's cells are the ones a placed cell mask leaves, inside the
#: registered edge cut: for the one pair that needs it they are the low-state
#: cells at the two ends of the profile, so the label names them, states the cut
#: applied, and says the reading sits beside the declared set.
X0_LABEL = f"x = {X_TARGET_CM:.0f} cm"
EDGE_CELLS_LABEL = f"edge cells |x| < {EDGE_CUT_CM:.0f} cm, beside the set"

#: The membership gate: the largest |M| the nine ES1-ES3 pairs reach.  Inclusive,
#: because it IS an observed pair.  There is deliberately no lower edge; see
#: ``gate_verdict``.
GATE_HIGH_M = max(MACH_FACE_ASYMMETRY_M_RANGE)

#: Experiment set whose pairs are re-tested, and the ports it visits twice.
ES4_SET_ID = "4"
PORTS = (21, 29, 41)

#: Substring identifying the swept (``I_SWEEP``) face in a product source path,
#: and the substring identifying the other face.  Both are required to appear
#: exactly once across a pair's two source attributes, or the pair is refused.
SWEEP_FACE_TOKEN = "isweep"
ISAT_FACE_TOKEN = "isat"

#: A path resolving inside a directory of this name is refused as an output.
FORBIDDEN_OUTPUT_DIR = "processed"

#: The nine ES1-ES3 rotation-pair half-differences in M at ``MACH_K``, keyed by
#: ``(experiment set id, port)``, signed ``(rot0 - rot180) / 2``.  These are the
#: values ``MACH_FACE_ASYMMETRY_M_RMS`` and ``_RANGE`` are computed from; they
#: are carried here as a CONTROL on the read, not as a second home for the
#: statistic.  A read that does not reproduce them to the printed digits is not
#: reading the product the declared constants were measured on.
CONTROL_HALF_DIFFERENCES_M = {
    ("1", 21): +0.112,
    ("1", 29): +0.094,
    ("1", 41): -0.024,
    ("2", 21): +0.067,
    ("2", 29): +0.029,
    ("2", 41): +0.005,
    ("3", 21): +0.052,
    ("3", 29): -0.146,
    ("3", 41): -0.097,
}

#: Digits the control is compared to; the control values are quoted to 3 dp.
CONTROL_TOL_M = 5.0e-4

#: Labels for the three points F is evaluated at.
F_POINTS = ("central", "low", "high")


class PairRefused(ValueError):
    """A rotation pair cannot be read as the declared statistic requires."""


@dataclass(frozen=True)
class FaceFactor:
    """The rest-bias factor for one port and rotation, with its spread.

    ``central`` is the measured factor and ``spread`` its standard deviation
    over the cells it was measured on; both come from the rest-bias CSV.  A
    factor is a current ratio, so it must be positive at every point it is
    evaluated at -- a spread wide enough to reach zero is refused rather than
    clipped.
    """

    port: int
    rotation_deg: int
    central: float
    spread: float

    def at(self, point: str) -> float:
        """Factor at ``central``, ``low`` (central - spread) or ``high``."""
        if point == "central":
            value = self.central
        elif point == "low":
            value = self.central - self.spread
        elif point == "high":
            value = self.central + self.spread
        else:
            raise ValueError(f"unknown bracket point {point!r}; expected one of {F_POINTS}")
        if value <= 0.0:
            raise ValueError(
                f"rest-bias factor at port {self.port} rot {self.rotation_deg} is "
                f"{value:.4f} at the {point} point; a current ratio must be positive"
            )
        return value


@dataclass(frozen=True)
class Reduction:
    """Which cells a pair's half-difference was actually averaged over."""

    label: str
    mask: np.ndarray
    n_admitted: int
    n_candidate: int


def sweep_face_is_upstream(group) -> bool:
    """Whether the swept (``I_SWEEP``) face looks upstream in this run.

    Decided from the Mach product's own ``upstream_source_file`` and
    ``downstream_source_file`` attributes, never from the rotation angle: the
    rotation-to-face mapping is the product's to declare.  Raises
    ``PairRefused`` when the two attributes do not name one swept face and one
    ISAT face.
    """
    upstream = str(group.attrs["upstream_source_file"]).lower()
    downstream = str(group.attrs["downstream_source_file"]).lower()
    up_is_sweep = SWEEP_FACE_TOKEN in upstream
    down_is_sweep = SWEEP_FACE_TOKEN in downstream
    up_is_isat = ISAT_FACE_TOKEN in upstream and not up_is_sweep
    down_is_isat = ISAT_FACE_TOKEN in downstream and not down_is_sweep
    if up_is_sweep and down_is_isat:
        return True
    if down_is_sweep and up_is_isat:
        return False
    raise PairRefused(
        "cannot tell the swept face from the ISAT face for run "
        f"{group.attrs.get('run_id', '?')}: upstream source {upstream!r}, "
        f"downstream source {downstream!r}"
    )


def plateau_ln_ratio(group, t_min_s=PLATEAU_T_MIN_S, t_max_s=PLATEAU_T_MAX_S):
    """``ln(J_upstream / J_downstream)`` over the plateau window, ``(nx, nt)``.

    Cells the product has masked come back as NaN, which is how a masked cell
    stays out of every average taken downstream.
    """
    times = group["inter_sweep_time_s"][()]
    in_window = (times >= t_min_s) & (times <= t_max_s)
    upstream = group["upstream_current_density_a_m2"][()][:, in_window]
    downstream = group["downstream_current_density_a_m2"][()][:, in_window]
    with np.errstate(all="ignore"):
        return np.log(upstream / downstream)


def corrected_ln_ratio(ln_ratio, factor, sweep_is_upstream):
    """Apply the rest-bias factor to the swept face of one run.

    Multiplying the swept face's current by ``factor`` shifts the log ratio by
    ``+ln factor`` when that face is the upstream one and by ``-ln factor``
    when it is the downstream one.
    """
    shift = math.log(factor)
    return ln_ratio + (shift if sweep_is_upstream else -shift)


def choose_reduction(ln_rot0, ln_rot180, x_cm) -> Reduction | None:
    """Cells the pair's half-difference is averaged over, or ``None`` if empty.

    Prefers the declared reduction -- the single row at ``x = 0`` -- and falls
    back to the cells finite on BOTH rotations only when that row is empty,
    which is what a placed cell mask does to a pair.  The fallback is a
    different reduction and carries a different label so it can never be read as
    the declared one.  Both rotations must be finite in a cell: averaging each
    rotation over its own surviving cells would difference two different
    regions of the profile.

    ``EDGE_CUT_CM`` is applied to the fallback and to the fallback ALONE.  The
    declared reduction is read at ``x = 0``, which no cut of this kind can
    reach, so gating it here would be dead code that looked like policy; the
    edge-cell figure, on the other hand, is an average over whatever the mask
    leaves, and without the cut it is carried by positions whose two faces are
    both at the current floor.
    """
    finite_both = np.isfinite(ln_rot0) & np.isfinite(ln_rot180)
    n_candidate = int(finite_both.size)
    at_x0 = np.abs(np.asarray(x_cm) - X_TARGET_CM) <= X_MATCH_TOL_CM
    x0_mask = finite_both & at_x0[:, None]
    if x0_mask.any():
        return Reduction(
            label=X0_LABEL,
            mask=x0_mask,
            n_admitted=int(x0_mask.sum()),
            n_candidate=n_candidate,
        )
    inside_cut = np.abs(np.asarray(x_cm)) < EDGE_CUT_CM
    finite_both = finite_both & inside_cut[:, None]
    if finite_both.any():
        return Reduction(
            label=EDGE_CELLS_LABEL,
            mask=finite_both,
            n_admitted=int(finite_both.sum()),
            n_candidate=n_candidate,
        )
    return None


def half_difference_ln(ln_rot0, ln_rot180, reduction: Reduction) -> float:
    """Half the rot0-minus-rot180 difference, averaged cell by cell in ln R."""
    return float(np.mean((ln_rot0[reduction.mask] - ln_rot180[reduction.mask]) / 2.0))


def gate_verdict(half_difference_m, gate_high=GATE_HIGH_M) -> str:
    """Where a half-difference falls against the ES1-ES3 range, as a sentence.

    The range is a magnitude range, so the sign of the half-difference does not
    enter, and only its UPPER edge gates membership; that edge is inclusive
    because it IS an observed ES1-ES3 pair.  There is no lower edge: a corrected
    value indistinguishable from zero is a smaller face asymmetry than any of
    the nine, which is inside the range and not below it.
    """
    magnitude = abs(half_difference_m)
    if magnitude <= gate_high:
        return "joins as a labelled ES4 member"
    return f"outside the range ({magnitude:.4f} above {gate_high:.3f})"


def f_half_bracket_m(row) -> float:
    """Half the distance between the low-F and high-F readings, in M.

    F's spread is applied to both rotations together, so the low and high points
    are the two ends of what the factor's own uncertainty does to the corrected
    half-difference; half their separation is the bar quoted beside the central
    value.
    """
    low = row["points"]["low"]["half_difference_m"]
    high = row["points"]["high"]["half_difference_m"]
    return (high - low) / 2.0


def refusal_verdict(n_candidate: int) -> str:
    """The verdict for a pair with no admitted cell."""
    return f"REFUSED (0 of {n_candidate})"


def read_rest_bias_factors(path: Path) -> dict[tuple[int, int], FaceFactor]:
    """Per ``(port, rotation)`` direct rest-bias factor and its spread.

    Reads the ``f_direct_up`` column -- the least model-dependent of the routes
    the rest-bias instrument reports, read straight off a measured ion branch --
    and its standard deviation.  The extrapolated routes are deliberately not
    read here: which route the correction should use is settled upstream.
    """
    factors: dict[tuple[int, int], FaceFactor] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["port"]), int(row["rotation_deg"]))
            if key in factors:
                raise ValueError(f"{path} carries port {key[0]} rot {key[1]} twice")
            factors[key] = FaceFactor(
                port=key[0],
                rotation_deg=key[1],
                central=float(row["f_direct_up"]),
                spread=float(row["f_direct_up_std"]),
            )
    return factors


def pair_groups(sets_group, set_id: str, port: int):
    """The rot-0 and rot-180 run groups of one port's rotation pair."""
    by_rotation = {}
    for run_id in sorted(sets_group[set_id]):
        group = sets_group[set_id][run_id]
        if int(group.attrs["port"]) != port:
            continue
        rotation = int(round(float(group.attrs["rotation_deg"])))
        if rotation in by_rotation:
            raise PairRefused(
                f"set {set_id} port {port} has two runs at rotation {rotation}"
            )
        by_rotation[rotation] = (run_id, group)
    if 0 not in by_rotation or 180 not in by_rotation:
        raise PairRefused(f"set {set_id} port {port} is not a rotation pair")
    return by_rotation[0], by_rotation[180]


def run_control(sets_group, x_cm) -> tuple[list[dict], bool]:
    """Recompute the nine ES1-ES3 half-differences and check them."""
    rows = []
    ok = True
    for (set_id, port), expected in sorted(CONTROL_HALF_DIFFERENCES_M.items()):
        (run0, group0), (run180, group180) = pair_groups(sets_group, set_id, port)
        ln0 = plateau_ln_ratio(group0)
        ln180 = plateau_ln_ratio(group180)
        reduction = choose_reduction(ln0, ln180, x_cm)
        if reduction is None:
            raise PairRefused(f"control pair set {set_id} port {port} has no admitted cell")
        measured = half_difference_ln(ln0, ln180, reduction) / MACH_K
        matches = abs(measured - expected) <= CONTROL_TOL_M
        ok = ok and matches and reduction.label == X0_LABEL
        rows.append(
            {
                "experiment_set": set_id,
                "port": port,
                "run_rot0": run0,
                "run_rot180": run180,
                "reduction": reduction.label,
                "measured_m": measured,
                "expected_m": expected,
                "matches": matches,
            }
        )
    return rows, ok


def analyse_es4_pair(sets_group, port: int, factors, x_cm) -> dict:
    """Raw and corrected half-differences for one ES4 port, at all three points."""
    (run0, group0), (run180, group180) = pair_groups(sets_group, ES4_SET_ID, port)
    sweep_up_rot0 = sweep_face_is_upstream(group0)
    sweep_up_rot180 = sweep_face_is_upstream(group180)
    if sweep_up_rot0 == sweep_up_rot180:
        raise PairRefused(
            f"port {port}: the swept face looks the same way at both rotations, so "
            "this is not a rotation pair"
        )
    ln0 = plateau_ln_ratio(group0)
    ln180 = plateau_ln_ratio(group180)
    reduction = choose_reduction(ln0, ln180, x_cm)
    row = {
        "port": port,
        "run_rot0": run0,
        "run_rot180": run180,
        "sweep_face_rot0": "upstream" if sweep_up_rot0 else "downstream",
        "sweep_face_rot180": "upstream" if sweep_up_rot180 else "downstream",
        "reduction": reduction.label if reduction is not None else "none",
        "n_admitted": 0 if reduction is None else reduction.n_admitted,
        "n_candidate": int(ln0.size) if reduction is None else reduction.n_candidate,
        "refused": reduction is None,
        "points": {},
    }
    if reduction is None:
        row["raw_ln"] = float("nan")
        row["raw_m"] = float("nan")
        return row
    row["raw_ln"] = half_difference_ln(ln0, ln180, reduction)
    row["raw_m"] = row["raw_ln"] / MACH_K
    for point in F_POINTS:
        f_rot0 = factors[(port, 0)].at(point)
        f_rot180 = factors[(port, 180)].at(point)
        corrected_ln = half_difference_ln(
            corrected_ln_ratio(ln0, f_rot0, sweep_up_rot0),
            corrected_ln_ratio(ln180, f_rot180, sweep_up_rot180),
            reduction,
        )
        row["points"][point] = {
            "f_rot0": f_rot0,
            "f_rot180": f_rot180,
            "half_difference_ln": corrected_ln,
            "half_difference_m": corrected_ln / MACH_K,
            "verdict": gate_verdict(corrected_ln / MACH_K),
        }
    return row


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


def write_csv(path: Path, rows) -> None:
    """One row per (port, bracket point), plus the raw column beside each."""
    fields = [
        "port",
        "run_rot0",
        "run_rot180",
        "sweep_face_rot0",
        "sweep_face_rot180",
        "reduction",
        "n_admitted_cells",
        "n_candidate_cells",
        "f_point",
        "f_rot0",
        "f_rot180",
        "half_difference_ln_raw",
        "half_difference_m_raw",
        "half_difference_ln_corrected",
        "half_difference_m_corrected",
        "f_half_bracket_m",
        "gate_high_m",
        "verdict",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            base = {
                "port": row["port"],
                "run_rot0": row["run_rot0"],
                "run_rot180": row["run_rot180"],
                "sweep_face_rot0": row["sweep_face_rot0"],
                "sweep_face_rot180": row["sweep_face_rot180"],
                "reduction": row["reduction"],
                "n_admitted_cells": row["n_admitted"],
                "n_candidate_cells": row["n_candidate"],
                "half_difference_ln_raw": row["raw_ln"],
                "half_difference_m_raw": row["raw_m"],
                "gate_high_m": GATE_HIGH_M,
            }
            if row["refused"]:
                writer.writerow(
                    base
                    | {
                        "f_point": "",
                        "f_rot0": "",
                        "f_rot180": "",
                        "half_difference_ln_corrected": "",
                        "half_difference_m_corrected": "",
                        "f_half_bracket_m": "",
                        "verdict": refusal_verdict(row["n_candidate"]),
                    }
                )
                continue
            bar = f_half_bracket_m(row)
            for point in F_POINTS:
                entry = row["points"][point]
                writer.writerow(
                    base
                    | {
                        "f_point": point,
                        "f_rot0": entry["f_rot0"],
                        "f_rot180": entry["f_rot180"],
                        "half_difference_ln_corrected": entry["half_difference_ln"],
                        "half_difference_m_corrected": entry["half_difference_m"],
                        "f_half_bracket_m": bar,
                        "verdict": entry["verdict"],
                    }
                )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mach-hdf5",
        type=Path,
        default=Path("processed/mach_velocity.hdf5"),
        help="rotation-pair Mach product to read (default: processed/mach_velocity.hdf5)",
    )
    parser.add_argument(
        "--rest-bias-csv",
        type=Path,
        required=True,
        help="CSV written by scripts/es4_sweep_rest_bias_factor.py; the "
        "f_direct_up column and its spread are read from it",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("es4_face_asymmetry_subset.csv"),
        help="CSV path; defaults to the working directory, and a path inside "
        f"{FORBIDDEN_OUTPUT_DIR}/ is refused",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    output = checked_output_path(args.output)
    factors = read_rest_bias_factors(args.rest_bias_csv)

    print(f"mach product   {args.mach_hdf5}")
    print(f"rest-bias CSV  {args.rest_bias_csv}  (column f_direct_up +- f_direct_up_std)")
    print(f"plateau        {PLATEAU_T_MIN_S * 1e3:.0f}-{PLATEAU_T_MAX_S * 1e3:.0f} ms")
    print(f"statistic      half of the rot0-rot180 difference of ln(J_up/J_dn), M = ln R / {MACH_K}")
    print(f"gate           |M| <= {GATE_HIGH_M:.3f}, the largest of the nine ES1-ES3 pairs; "
          "NO lower edge --")
    print("               a corrected value indistinguishable from zero is a smaller face")
    print("               asymmetry than any of the nine, which puts it inside the range")
    print(f"edge cut       |x| < {EDGE_CUT_CM:.0f} cm, on the edge-cell reduction ONLY "
          f"(x = 0 reductions untouched):")
    print("               beyond it both faces sit at the 0.1-1 mA floor and both runs")
    print("               carry an unexplained x5-10 ISAT/I_SWEEP feature at x = -23/-24 cm")
    print()
    print("The correction multiplies the SWEPT face's current by F, which is upstream")
    print("at one rotation and downstream at the other, so it does not cancel out of the")
    print("half-difference.  The face-to-rotation mapping is read from the product's own")
    print("source-file attributes.  Nothing is written into processed/ and no constant")
    print("in density.py is changed here.")
    print()

    with h5py.File(args.mach_hdf5, "r") as product:
        x_cm = product["x_cm"][()]
        sets_group = product["experiment_sets"]

        print("=== CONTROL 1: the nine ES1-ES3 half-differences ===")
        control_rows, control_ok = run_control(sets_group, x_cm)
        for row in control_rows:
            print(
                f"  ES{row['experiment_set']} p{row['port']:<2}  runs "
                f"{row['run_rot0']}/{row['run_rot180']}  {row['reduction']:<12}  "
                f"half-difference {row['measured_m']:+.3f} M   expected "
                f"{row['expected_m']:+.3f}   {'match' if row['matches'] else 'MISMATCH'}"
            )
        print(f"  CONTROL 1: {'reproduced' if control_ok else 'DID NOT REPRODUCE'}")
        print()

        es4_rows = [analyse_es4_pair(sets_group, port, factors, x_cm) for port in PORTS]

        print("=== CONTROL 2: the same three ES4 pairs UNCORRECTED ===")
        for row in es4_rows:
            if row["refused"]:
                print(f"  p{row['port']:<2}  {refusal_verdict(row['n_candidate'])}")
                continue
            print(
                f"  p{row['port']:<2}  runs {row['run_rot0']}/{row['run_rot180']}  "
                f"{row['reduction']:<38}  {row['n_admitted']:>3} of "
                f"{row['n_candidate']:<4} cells  half-difference "
                f"{row['raw_ln']:+.4f} ln = {row['raw_m']:+.4f} M"
            )
        print()

        print("=== ES4 pairs, corrected for the rest-bias convention ===")
        for row in es4_rows:
            print(
                f"port {row['port']:>2}  runs {row['run_rot0']}/{row['run_rot180']}  "
                f"swept face: rot0 {row['sweep_face_rot0']}, rot180 {row['sweep_face_rot180']}"
            )
            if row["refused"]:
                print(f"    GATE  {refusal_verdict(row['n_candidate'])}")
                print()
                continue
            print(
                f"    read on   {row['reduction']} ({row['n_admitted']} of "
                f"{row['n_candidate']} plateau cells admitted)"
            )
            print(
                f"    raw       {row['raw_ln']:+.4f} ln = {row['raw_m']:+.4f} M"
            )
            for point in F_POINTS:
                entry = row["points"][point]
                print(
                    f"    F {point:<7} rot0 {entry['f_rot0']:.4f}  rot180 "
                    f"{entry['f_rot180']:.4f}  ->  "
                    f"{entry['half_difference_ln']:+.4f} ln = "
                    f"{entry['half_difference_m']:+.4f} M"
                )
            print(
                f"    bar       {row['points']['central']['half_difference_m']:+.4f} "
                f"+- {f_half_bracket_m(row):.4f} M   (the F half-bracket: half the "
                "low-to-high spread)"
            )
            for point in F_POINTS:
                entry = row["points"][point]
                print(f"    GATE  F {point:<7}  {entry['verdict']}")
            print()

    write_csv(output, es4_rows)
    print(f"wrote {output}")
    return 0 if control_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
