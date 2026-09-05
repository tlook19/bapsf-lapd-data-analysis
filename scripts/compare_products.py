"""Bit-identity comparison of two processed products, per run and per dataset.

Works on any product laid out as ``experiment_sets/<set>/<run>/<datasets>`` --
the dead-time profile products and the Mach/velocity product alike.  Reports,
for every shared dataset: whether it is bit-identical, the largest absolute
difference over cells finite in both, and how the NaN pattern moved.  Root and
per-run attribute differences are listed separately, since a rebuild routinely
adds attrs while leaving every value untouched.

NaN-AWARENESS IS THE POINT.  Several of these datasets legitimately carry NaN
(non-positive current density, rail-excluded cells), and ``numpy.array_equal``
without ``equal_nan=True`` reports False for ANY array containing one, because
NaN != NaN.  The 2026-09-03 copy of this script omitted that flag and was
therefore blind on the Mach product: it printed False with a maximum absolute
difference of 0.000e+00 for every Mach row, in both directions, including a
comparison of a file with itself in all but name.  This version supersedes it;
a gate run with the older copy on any NaN-carrying product should be re-run.

Usage:  python scripts/compare_products.py OLD.hdf5 NEW.hdf5
"""

import sys

import h5py
import numpy as np


def _run_keys(handle: h5py.File) -> set[tuple[str, str]]:
    return {
        (set_id, run_id)
        for set_id in handle["experiment_sets"]
        for run_id in handle[f"experiment_sets/{set_id}"]
    }


def _attr_equal(a, b) -> bool:
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    return np.array_equal(a, b)


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__.strip().splitlines()[-1])
    old_path, new_path = sys.argv[1], sys.argv[2]

    with h5py.File(old_path, "r") as old, h5py.File(new_path, "r") as new:
        print(f"OLD: {old_path}\nNEW: {new_path}\n")
        if "x_cm" in old and "x_cm" in new:
            print("x_cm bit-identical:", np.array_equal(old["x_cm"][()], new["x_cm"][()]))
        old_keys, new_keys = _run_keys(old), _run_keys(new)
        print("only in OLD:", sorted(old_keys - new_keys))
        print("only in NEW:", sorted(new_keys - old_keys))

        moved = []
        print(f"\n{'set':>3} {'run':>4} {'dataset':>30} {'identical':>10} "
              f"{'maxabsdiff':>12} {'newNaN':>7} {'lostNaN':>8}")
        for set_id, run_id in sorted(old_keys & new_keys):
            g_old = old[f"experiment_sets/{set_id}/{run_id}"]
            g_new = new[f"experiment_sets/{set_id}/{run_id}"]
            for name in sorted(set(g_old) & set(g_new)):
                a, b = g_old[name][()], g_new[name][()]
                new_nan = lost_nan = 0
                if a.shape != b.shape:
                    identical, diff = False, float("nan")
                elif a.dtype.kind == "f" and b.dtype.kind == "f":
                    identical = np.array_equal(a, b, equal_nan=True)
                    nan_a, nan_b = np.isnan(a), np.isnan(b)
                    new_nan = int((nan_b & ~nan_a).sum())
                    lost_nan = int((nan_a & ~nan_b).sum())
                    both = ~nan_a & ~nan_b
                    diff = float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0
                else:
                    identical = np.array_equal(a, b)
                    diff = float(np.max(np.abs(a.astype(float) - b.astype(float))))
                if not identical:
                    moved.append((set_id, run_id, name, diff, new_nan, lost_nan))
                print(f"{set_id:>3} {run_id:>4} {name:>30} {str(identical):>10} "
                      f"{diff:>12.3e} {new_nan:>7} {lost_nan:>8}")
            only_old = sorted(set(g_old) - set(g_new))
            only_new = sorted(set(g_new) - set(g_old))
            if only_old or only_new:
                print(f"{set_id:>3} {run_id:>4}   datasets "
                      f"only-old={only_old} only-new={only_new}")

        print("\n--- root attrs diff ---")
        for key in sorted(set(old.attrs) | set(new.attrs)):
            v_old = old.attrs.get(key, "<absent>")
            v_new = new.attrs.get(key, "<absent>")
            if not _attr_equal(v_old, v_new):
                print(f"  {key}: old={v_old!r} new={v_new!r}")

        print("\n--- per-run attrs diff (shared runs) ---")
        for set_id, run_id in sorted(old_keys & new_keys):
            g_old = old[f"experiment_sets/{set_id}/{run_id}"]
            g_new = new[f"experiment_sets/{set_id}/{run_id}"]
            for key in sorted(set(g_old.attrs) | set(g_new.attrs)):
                v_old = g_old.attrs.get(key, "<absent>")
                v_new = g_new.attrs.get(key, "<absent>")
                if not _attr_equal(v_old, v_new):
                    print(f"  ES{set_id} run {run_id}  {key}: "
                          f"old={v_old!r} new={v_new!r}")

        print("\n" + "=" * 70)
        if not moved:
            print("GATE: every shared dataset is bit-identical (NaN-aware).")
        else:
            print(f"GATE: {len(moved)} dataset(s) moved, in these (set, run) groups:")
            for set_id, run_id in sorted({(m[0], m[1]) for m in moved}):
                rows = [m for m in moved if m[0] == set_id and m[1] == run_id]
                print(f"  ES{set_id} run {run_id}: " + ", ".join(
                    f"{m[2]}(maxabsdiff={m[3]:.3e}, newNaN={m[4]}, lostNaN={m[5]})"
                    for m in rows))


if __name__ == "__main__":
    main()
