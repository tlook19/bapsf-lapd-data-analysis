"""Regenerate the window-refit band product and check the placed copy against it.

The band product is written by ``scripts/refit_window_band.py`` and consumed by
``scripts/fit_te_spatial.py``, which reads the per-cell CSV to mark
semi-quantitative cells.  Byte-exact regeneration of that product is a function
of the numerical libraries it was captured under: the same code on the same raw
traces reproduces the recorded digits only under the numpy / scipy / h5py
versions the metadata's ``numerics_vintage`` block records, and moves by of
order 1e-8 relative on the fitted quantities under a different vintage.  A
bit-identity check would therefore report a failure for a library upgrade, and
a purely numerical check would hide a changed cell set or a lost run.

So this check is two-sided, and the two outcomes are never conflated:

* every INTEGER, BOOLEAN and STRING dataset, and every float dataset's NaN
  pattern, must be BYTE-IDENTICAL -- those carry the cell sets, the run
  identities, the sweep counts and which cells were fittable, and no library
  version moves them;
* every FLOAT dataset must agree within a DISCLOSED relative tolerance,
  ``--rtol`` (default 1e-6, two orders looser than the measured library-vintage
  drift and many orders tighter than a change of definition);
* the CSV's non-numeric columns -- the identities, counts and criterion flags --
  must be identical as text, and its float columns must agree to the same
  tolerance.

A run that finds every compared dataset byte-identical prints ``byte-exact``; a
run that needed the tolerance prints ``within tolerance`` with the observed
maximum.  Both exit 0.

What is compared is the HDF5 datasets and the CSV.  The HDF5 attributes and the
metadata JSON are NOT compared: they carry ``created_utc``, per-port ``wall_s``
timings and the vintage block, all of which differ between two runs by
construction.

Regeneration source
-------------------
By default the check regenerates from the EXISTING per-(set, port) checkpoints
beside the placed product, which is fast and exercises the assembly step alone.
``--from-raw`` regenerates from the raw digitizer traces instead, which is the
whole pass and takes roughly half an hour.  Either way the regeneration is
written into a temporary directory and the placed product is never touched.

Usage
-----
  python scripts/verify_window_band_product.py
  python scripts/verify_window_band_product.py --from-raw
  python scripts/verify_window_band_product.py --rtol 1e-8

Exit status
-----------
  0  the placed product matches the regeneration (byte-exact, or within --rtol)
  1  a dataset or CSV column is outside tolerance; the first one is named
  2  the placed product, or the checkpoints the check needs, are missing
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

PLACED_HDF5 = ROOT / "processed" / "window_refit_band.hdf5"
PLACED_SUMMARY = ROOT / "processed" / "window_refit_band_summary.csv"
PLACED_METADATA = ROOT / "processed" / "window_refit_band_metadata.json"
#: Checkpoint directory name, as ``refit_window_band.py`` resolves it.
WORK_DIR_NAME = "window_refit_band_work"

#: Default relative tolerance on the float datasets and the CSV's float columns.
#: The measured library-vintage drift on this product is ~2.5e-8 relative.
DEFAULT_RTOL = 1.0e-6

#: The CSV columns that carry fitted floating-point values.  Every other column
#: in the header is an identity, a count or a criterion flag and is compared as
#: text, so a renumbered run or a moved cell set cannot pass on a tolerance.
CSV_FLOAT_COLUMNS = ("x_cm", "dln_te_window", "te_default_med_ev")


@dataclass
class Row:
    """One compared dataset or CSV column."""

    name: str
    kind: str
    byte_identical: bool
    max_rel: float
    note: str = ""


@dataclass
class Comparison:
    """The outcome of comparing a placed product against a regeneration."""

    rows: list[Row] = field(default_factory=list)
    failure: str | None = None

    @property
    def byte_exact(self) -> bool:
        return self.failure is None and all(row.byte_identical for row in self.rows)

    @property
    def max_rel(self) -> float:
        finite = [row.max_rel for row in self.rows if np.isfinite(row.max_rel)]
        return max(finite) if finite else 0.0

    def fail(self, message: str) -> "Comparison":
        """Record the first failure; later ones are not looked for."""
        if self.failure is None:
            self.failure = message
        return self


def _max_rel_diff(placed: np.ndarray, new: np.ndarray) -> float:
    """Return max |new - placed| / |placed| over cells finite in both.

    The placed product is the reference, so it is the denominator; a zero
    reference falls back to the absolute difference, which is what a relative
    measure of a zero can honestly mean.
    """
    both = np.isfinite(placed) & np.isfinite(new)
    if not both.any():
        return 0.0
    a, b = np.asarray(new)[both], np.asarray(placed)[both]
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(b != 0, np.abs((a - b) / b), np.abs(a - b))
    return float(np.max(rel)) if rel.size else 0.0


def _dataset_paths(handle: h5py.File) -> list[str]:
    """Return every dataset path in the file, depth first and sorted."""
    found: list[str] = []
    handle.visititems(
        lambda name, item: found.append(name)
        if isinstance(item, h5py.Dataset)
        else None
    )
    return sorted(found)


def _compare_dataset(
    name: str, placed: np.ndarray, new: np.ndarray, *, rtol: float
) -> Row:
    """Compare one dataset; float data is tolerated, everything else is not."""
    if placed.dtype.kind == "O":
        # Variable-length strings arrive as an object array, whose buffer holds
        # pointers rather than characters; comparing its bytes would compare
        # addresses.  The values themselves are what identity means here.
        identical = bool(np.array_equal(placed, new))
        if not identical:
            return Row(name, "text", False, float("nan"), "NOT identical")
        return Row(name, "text", True, 0.0)

    byte_identical = placed.tobytes() == new.tobytes()
    if placed.dtype.kind != "f":
        kind = {"i": "int", "u": "int", "b": "bool"}.get(placed.dtype.kind, "text")
        if not byte_identical:
            return Row(name, kind, False, float("nan"), "NOT byte-identical")
        return Row(name, kind, True, 0.0)

    nan_placed, nan_new = np.isnan(placed), np.isnan(new)
    if not np.array_equal(nan_placed, nan_new):
        moved = int(np.sum(nan_placed != nan_new))
        return Row(name, "float", False, float("nan"), f"NaN mask moved in {moved} cell(s)")

    max_rel = _max_rel_diff(placed, new)
    note = "" if max_rel <= rtol else f"max rel {max_rel:.3e} > rtol {rtol:.3e}"
    return Row(name, "float", byte_identical, max_rel, note)


def compare_hdf5(placed_path: Path, new_path: Path, *, rtol: float) -> Comparison:
    """Compare every dataset of two band products."""
    result = Comparison()
    with h5py.File(placed_path, "r") as placed, h5py.File(new_path, "r") as new:
        placed_names, new_names = _dataset_paths(placed), _dataset_paths(new)
        for missing in sorted(set(placed_names) - set(new_names)):
            result.fail(f"{missing}: present in the placed product, absent from the regeneration")
        for extra in sorted(set(new_names) - set(placed_names)):
            result.fail(f"{extra}: present in the regeneration, absent from the placed product")

        for name in placed_names:
            if name not in new_names:
                continue
            a, b = placed[name][()], new[name][()]
            a, b = np.asarray(a), np.asarray(b)
            if a.shape != b.shape:
                result.rows.append(
                    Row(name, "shape", False, float("nan"), f"{a.shape} vs {b.shape}")
                )
                result.fail(f"{name}: shape {a.shape} became {b.shape}")
                continue
            if a.dtype != b.dtype:
                result.rows.append(
                    Row(name, "dtype", False, float("nan"), f"{a.dtype} vs {b.dtype}")
                )
                result.fail(f"{name}: dtype {a.dtype} became {b.dtype}")
                continue
            row = _compare_dataset(name, a, b, rtol=rtol)
            result.rows.append(row)
            if row.note:
                result.fail(f"{name}: {row.note}")
    return result


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    return rows[0], rows[1:]


def compare_csv(placed_path: Path, new_path: Path, *, rtol: float) -> Comparison:
    """Compare the per-cell CSV column by column."""
    result = Comparison()
    placed_header, placed_rows = _read_csv(placed_path)
    new_header, new_rows = _read_csv(new_path)
    if placed_header != new_header:
        return result.fail(
            f"{placed_path.name}: header {placed_header} became {new_header}"
        )
    if len(placed_rows) != len(new_rows):
        return result.fail(
            f"{placed_path.name}: {len(placed_rows)} cell rows became {len(new_rows)}"
        )

    for index, column in enumerate(placed_header):
        name = f"{placed_path.name}:{column}"
        placed_column = [row[index] for row in placed_rows]
        new_column = [row[index] for row in new_rows]
        byte_identical = placed_column == new_column
        if column not in CSV_FLOAT_COLUMNS:
            if not byte_identical:
                differing = sum(a != b for a, b in zip(placed_column, new_column))
                result.rows.append(
                    Row(name, "text", False, float("nan"), f"{differing} row(s) differ")
                )
                result.fail(f"{name}: {differing} row(s) differ")
            else:
                result.rows.append(Row(name, "text", True, 0.0))
            continue

        a = np.array([float(value) for value in placed_column])
        b = np.array([float(value) for value in new_column])
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            moved = int(np.sum(np.isnan(a) != np.isnan(b)))
            result.rows.append(
                Row(name, "float", False, float("nan"), f"NaN mask moved in {moved} row(s)")
            )
            result.fail(f"{name}: NaN mask moved in {moved} row(s)")
            continue
        max_rel = _max_rel_diff(a, b)
        note = "" if max_rel <= rtol else f"max rel {max_rel:.3e} > rtol {rtol:.3e}"
        result.rows.append(Row(name, "float", byte_identical, max_rel, note))
        if note:
            result.fail(f"{name}: {note}")
    return result


def compare_band_products(
    *,
    placed_hdf5: Path,
    placed_summary: Path,
    new_hdf5: Path,
    new_summary: Path,
    rtol: float,
) -> Comparison:
    """Compare a placed band product against a regeneration of it."""
    hdf5 = compare_hdf5(placed_hdf5, new_hdf5, rtol=rtol)
    summary = compare_csv(placed_summary, new_summary, rtol=rtol)
    merged = Comparison(rows=hdf5.rows + summary.rows)
    for failure in (hdf5.failure, summary.failure):
        if failure is not None:
            merged.fail(failure)
    return merged


def exit_code(result: Comparison) -> int:
    """Return the process exit status a completed comparison earns.

    A comparison that reached its end without a failure exits 0 whether it was
    byte-exact or needed the tolerance; the distinction is in what is PRINTED,
    never in the status, so a caller cannot silently read one as the other.
    """
    return 1 if result.failure is not None else 0


def regenerate(
    *, out_dir: Path, work_dir: Path, from_raw: bool, checkpoints: Path
) -> tuple[Path, Path]:
    """Regenerate the band product into ``out_dir``; return its two paths.

    The regeneration runs ``refit_window_band.py`` itself rather than a copy of
    its assembly, so what is checked is the product the pass would write today.
    In the default (not ``--from-raw``) mode the existing checkpoints are COPIED
    into a temporary work directory first, so the placed product's own
    checkpoints are never written to.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    if not from_raw:
        found = sorted(checkpoints.glob("di_set*_port*.npz"))
        if not found:
            raise SystemExit(
                f"no per-(set, port) checkpoints under {checkpoints}; regenerate "
                "them, or pass --from-raw to refit from the raw traces"
            )
        for path in found:
            shutil.copy2(path, work_dir / path.name)

    new_hdf5 = out_dir / "window_refit_band.hdf5"
    new_summary = out_dir / "window_refit_band_summary.csv"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "refit_window_band.py"),
        "--output", str(new_hdf5),
        "--summary", str(new_summary),
        "--metadata", str(out_dir / "window_refit_band_metadata.json"),
        "--work-dir", str(work_dir),
    ]
    print("regenerating: " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    return new_hdf5, new_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output", type=Path, default=PLACED_HDF5,
                        help="the placed band HDF5 product to check")
    parser.add_argument("--summary", type=Path, default=PLACED_SUMMARY,
                        help="the placed per-cell CSV to check")
    parser.add_argument("--metadata", type=Path, default=PLACED_METADATA,
                        help=(
                            "the placed metadata JSON.  Its presence is required "
                            "-- it is part of the product -- but it is not "
                            "compared: it carries created_utc and the vintage "
                            "block, which differ between two runs by construction"
                        ))
    parser.add_argument("--work-dir", type=Path, default=None,
                        help=(
                            "per-(set, port) checkpoints to regenerate from. "
                            f"Default: {WORK_DIR_NAME}/ beside --output"
                        ))
    parser.add_argument("--from-raw", action="store_true",
                        help="refit from the raw traces instead of the checkpoints (slow)")
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL,
                        help=f"relative tolerance on the float data (default {DEFAULT_RTOL:g})")
    args = parser.parse_args()

    missing = [
        path
        for path in (args.output, args.summary, args.metadata)
        if not path.exists()
    ]
    if missing:
        print("missing placed product: " + ", ".join(str(path) for path in missing))
        raise SystemExit(2)

    checkpoints = (
        Path(args.work_dir)
        if args.work_dir is not None
        else Path(args.output).parent / WORK_DIR_NAME
    )
    with tempfile.TemporaryDirectory(prefix="window_band_verify_") as temp:
        temp_dir = Path(temp)
        new_hdf5, new_summary = regenerate(
            out_dir=temp_dir,
            work_dir=temp_dir / "work",
            from_raw=args.from_raw,
            checkpoints=checkpoints,
        )
        result = compare_band_products(
            placed_hdf5=args.output,
            placed_summary=args.summary,
            new_hdf5=new_hdf5,
            new_summary=new_summary,
            rtol=args.rtol,
        )

        print(f"\nplaced:       {args.output}")
        print(f"              {args.summary}")
        print(f"regenerated:  {'raw traces' if args.from_raw else f'checkpoints in {checkpoints}'}")
        print(f"rtol:         {args.rtol:.3e}\n")
        print(f"{'dataset / column':>44} {'kind':>6} {'byte-id':>8} "
              f"{'max rel':>11} {'rtol':>11}  note")
        for row in result.rows:
            print(
                f"{row.name:>44} {row.kind:>6} {str(row.byte_identical):>8} "
                f"{row.max_rel:11.3e} {args.rtol:11.3e}  {row.note}"
            )

    if result.failure is not None:
        print(f"\nFAIL: {result.failure}")
        raise SystemExit(exit_code(result))
    if result.byte_exact:
        print("\nPASS: byte-exact")
    else:
        print(f"\nPASS: within tolerance (max rel {result.max_rel:.3e})")


if __name__ == "__main__":
    main()
