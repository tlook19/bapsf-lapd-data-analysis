"""Key-by-key additivity check for an annotation pass.

An annotation is ADDITIVE when it adds keys and changes none.  This compares a
BEFORE and an AFTER copy of the same artefact key by key and reports, for every
shared key, whether it is byte-identical -- then lists the keys only one side
has.  It exits non-zero if any shared key moved, so it can be used as a gate.

Two artefact kinds are understood, chosen by suffix:

``.toml``
    Flattened to dotted key paths, so ``runs.05.artifacts[0].sample_index`` is
    one key.  Comparison is on the parsed value, and additionally on the raw
    TEXT of every line outside the blocks the AFTER file added, which is the
    stronger claim: the annotator carries foreign bytes through untouched
    rather than round-tripping them.

``.npz``
    Compared on each array's raw ``.tobytes()`` plus dtype and shape, so NaN
    compares equal to NaN and a bit-level change cannot hide behind a
    tolerance.  Object arrays (the overlay's string fields) fall back to
    ``repr`` equality.

Usage:  python scripts/compare_run_annotations.py BEFORE AFTER
"""
import sys
import tomllib
from pathlib import Path

import numpy as np


def _flatten(value, prefix=""):
    """Flatten nested TOML tables and arrays to dotted key paths."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return out
    if isinstance(value, list) and any(isinstance(item, dict) for item in value):
        out = {}
        for index, item in enumerate(value):
            out.update(_flatten(item, f"{prefix}[{index}]"))
        return out
    return {prefix: value}


def _toml_keys(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "rb") as stream:
        return _flatten(tomllib.load(stream))


def _npz_keys(path: Path) -> dict:
    archive = np.load(path, allow_pickle=True)
    out = {}
    for name in archive.files:
        array = archive[name]
        if array.dtype == object:
            out[name] = ("object", array.shape, repr(array.tolist()))
        else:
            out[name] = (str(array.dtype), array.shape, array.tobytes())
    return out


def _foreign_lines(path: Path, added_runs: set[str]) -> list[str]:
    """Non-blank registry lines outside the blocks of ``added_runs``.

    Blank lines are dropped because appending a block adds the blank that
    separates it from the one before, which is formatting rather than content;
    every other byte of every foreign line has to match exactly.
    """
    if not path.exists():
        return []
    keep, skipping = [], False
    for line in path.read_text().splitlines(keepends=True):
        if line.startswith("[["):
            skipping = any(
                line.rstrip("\n") == f"[[runs.{run}.artifacts]]" for run in added_runs
            )
        if skipping:
            if line.strip():
                continue
            skipping = False
        if line.strip():
            keep.append(line)
    return keep


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    before, after = Path(argv[0]), Path(argv[1])
    suffix = after.suffix

    if suffix == ".toml":
        before_keys, after_keys = _toml_keys(before), _toml_keys(after)
    elif suffix == ".npz":
        before_keys, after_keys = _npz_keys(before), _npz_keys(after)
    else:
        print(f"unsupported artefact kind: {suffix}", file=sys.stderr)
        return 2

    print(f"BEFORE: {before}\nAFTER:  {after}\n")
    shared = sorted(set(before_keys) & set(after_keys))
    moved = [key for key in shared if before_keys[key] != after_keys[key]]
    added = sorted(set(after_keys) - set(before_keys))
    removed = sorted(set(before_keys) - set(after_keys))

    print(f"shared keys:      {len(shared)}")
    print(f"  byte-identical: {len(shared) - len(moved)}")
    print(f"  MOVED:          {len(moved)}")
    for key in moved:
        print(f"    {key}")
    print(f"added keys:       {len(added)}")
    for key in added:
        print(f"    {key}")
    print(f"removed keys:     {len(removed)}")
    for key in removed:
        print(f"    {key}")

    text_ok = True
    if suffix == ".toml":
        added_runs = {
            key.split(".")[1] for key in added if key.startswith("runs.")
        }
        before_lines = _foreign_lines(before, added_runs)
        after_lines = _foreign_lines(after, added_runs)
        text_ok = before_lines == after_lines
        print(
            f"\nnon-blank raw text outside the added run blocks "
            f"{sorted(added_runs)}: "
            f"{'byte-identical' if text_ok else 'CHANGED'} "
            f"({len(before_lines)} vs {len(after_lines)} lines)"
        )

    ok = not moved and not removed and text_ok
    print(f"\nVERDICT: {'ADDITIVE' if ok else 'NOT ADDITIVE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
