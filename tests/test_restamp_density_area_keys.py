"""The re-stamp pass moves ``density_area_key`` and nothing else.

``scripts/restamp_density_area_keys.py`` exists because the two rot-180
dead-time products cannot be rebuilt to pick up a moved area-key rule: they
carry annotator-written datasets and run-level attrs
(``rail_mask``/``rail_fraction`` from ``scripts/annotate_rail_mask.py``,
``state_mask`` from ``scripts/annotate_state_mask.py``) that the writer,
``scripts/plot_isat_profiles.py``, knows nothing about and a rebuild drops.
So the whole value of the pass is that everything it does not target survives
it BYTE for byte -- which is what this file measures, on a synthetic product
carrying an annotator-style dataset and attr set, by comparing raw bytes
before and after rather than values (a NaN sentinel compares unequal to
itself under both ``==`` and ``np.array_equal``).

Also checked: the channel a group is asked about is the group's OWN
``deadtime_source_channel``, so a crossed-cable run whose source was
overridden is answered on the channel it actually averaged; a group missing
either identifying attr is an error rather than a group answered nominally;
and the ``processed/`` output guard refuses before any file is opened.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from restamp_density_area_keys import (  # noqa: E402
    refuse_processed_paths,
    restamp_product,
)

# One crossed-cable run and one nominal run, on both channels.
ROWS = (
    # (set_id, run_id, source_channel, stale key, key the rule gives)
    ("1", "02", "isat", "ap_L_cm2", "ap_R_cm2"),
    ("3", "31", "i_sweep", "ap_L_cm2", "ap_R_cm2"),
    ("3", "31b", "isat", "ap_L_cm2", "ap_L_cm2"),
)


def _annotator_datasets(rng):
    """Datasets and attrs of the shape the two rot-180 annotators write."""
    rail_mask = rng.integers(0, 2, size=(6, 4)).astype(bool)
    rail_fraction = rng.random((6, 4))
    rail_fraction[0, 0] = np.nan  # a NaN sentinel must survive byte-for-byte
    state_mask = rng.integers(0, 2, size=(6, 4)).astype(bool)
    isat_a = rng.random((6, 4)) * 1e-3
    isat_a[2, 1] = np.nan
    return {
        "rail_mask": rail_mask,
        "rail_fraction": rail_fraction,
        "state_mask": state_mask,
        "isat_a": isat_a,
        "n_shots_used": rng.integers(10, 21, size=(6, 4)).astype(np.int32),
    }


def _write_product(path: Path, rows=ROWS) -> None:
    rng = np.random.default_rng(20260910)
    with h5py.File(path, "w") as hdf:
        hdf.attrs["rotation_filter_deg"] = 180.0
        hdf.attrs["rail_mask_rule"] = "any railed sample in the cell"
        hdf.attrs["state_mask_runs"] = np.array(["43"], dtype=object)
        sets = hdf.create_group("experiment_sets")
        for set_id, run_id, channel, stale, _ in rows:
            group = sets.require_group(set_id).create_group(run_id)
            group.attrs["run_id"] = run_id[:2]
            group.attrs["deadtime_source_channel"] = channel
            group.attrs["requested_deadtime_source_channel"] = "isat"
            group.attrs["deadtime_source_overridden"] = channel != "isat"
            group.attrs["density_area_key"] = stale
            group.attrs["rail_fraction_deadtime"] = 0.125
            group.attrs["rail_mask_rule"] = "any railed sample in the cell"
            group.attrs["state_factor_applied"] = False
            group.attrs["z_cm"] = 470.05
            for name, data in _annotator_datasets(rng).items():
                dataset = group.create_dataset(name, data=data)
                dataset.attrs["units"] = "A" if name == "isat_a" else "1"


def _snapshot(path: Path) -> dict:
    """Raw bytes of every dataset and every attr, keyed by full path."""
    state: dict[str, object] = {}

    def visit(name, obj):
        name = name if name.startswith("/") else f"/{name}"
        for key, value in obj.attrs.items():
            array = np.asarray(value)
            state[f"{name}@{key}"] = (
                tuple(np.atleast_1d(array).astype(str).ravel().tolist())
                if array.dtype.kind in "USO"
                else (str(array.dtype), array.shape, array.tobytes())
            )
        if isinstance(obj, h5py.Dataset):
            data = np.asarray(obj[()])
            state[name] = (str(data.dtype), data.shape, data.tobytes())

    with h5py.File(path, "r") as hdf:
        visit("/", hdf)
        hdf.visititems(visit)
    return state


def test_only_the_area_key_attrs_move(tmp_path):
    """Every dataset and every other attr is byte-identical after the pass."""
    product = tmp_path / "isat_rot180_deadtime_profiles.hdf5"
    _write_product(product)
    before = _snapshot(product)

    rows, moved, table = restamp_product(product)
    after = _snapshot(product)

    assert rows == len(ROWS)
    assert moved == sum(1 for *_, stale, want in ROWS if stale != want) == 2

    changed = {key for key in before if before[key] != after[key]}
    assert changed == {
        "/experiment_sets/1/02@density_area_key",
        "/experiment_sets/3/31@density_area_key",
    }
    assert set(before) == set(after), "the pass added or removed an object"

    # The annotator-written datasets and their NaN sentinels survive intact.
    for path in (
        "/experiment_sets/3/31/rail_mask",
        "/experiment_sets/3/31/rail_fraction",
        "/experiment_sets/3/31/state_mask",
        "/experiment_sets/3/31/isat_a",
    ):
        assert before[path] == after[path]

    assert [(row[0], row[1], row[3], row[4]) for row in table] == [
        (set_id, run_id[:2], stale, want) for set_id, run_id, _, stale, want in ROWS
    ]


def test_the_overridden_channel_is_read_from_the_group(tmp_path):
    """Run 31's row is answered on ``i_sweep``, the channel it averaged."""
    product = tmp_path / "isat_profiles.hdf5"
    _write_product(product)
    restamp_product(product)
    with h5py.File(product, "r") as hdf:
        # ISAT on a crossed-cable run reads the LEFT electrode; the row that
        # sourced I_SWEEP therefore takes the RIGHT one, and the two differ.
        assert hdf["experiment_sets/3/31"].attrs["density_area_key"] == "ap_R_cm2"
        assert hdf["experiment_sets/3/31b"].attrs["density_area_key"] == "ap_L_cm2"
        assert hdf["experiment_sets/1/02"].attrs["density_area_key"] == "ap_R_cm2"


def test_a_second_pass_changes_nothing(tmp_path):
    """The pass is idempotent: a re-stamped product is left byte-identical."""
    product = tmp_path / "isat_profiles.hdf5"
    _write_product(product)
    restamp_product(product)
    before = _snapshot(product)
    _, moved, _ = restamp_product(product)
    assert moved == 0
    assert _snapshot(product) == before


@pytest.mark.parametrize("missing", ("run_id", "deadtime_source_channel"))
def test_a_group_missing_an_identifying_attr_is_refused(tmp_path, missing):
    """A group is never answered nominally because an attr is absent."""
    product = tmp_path / "isat_profiles.hdf5"
    _write_product(product)
    with h5py.File(product, "r+") as hdf:
        del hdf["experiment_sets/3/31"].attrs[missing]
    with pytest.raises(ValueError, match=missing):
        restamp_product(product)


def test_processed_paths_are_refused_without_the_flag(tmp_path):
    """The guard fires at argument resolution, before any file is opened."""
    placed = tmp_path / "processed" / "isat_profiles.hdf5"
    placed.parent.mkdir()
    with pytest.raises(ValueError, match="--allow-processed"):
        refuse_processed_paths([placed], allow_processed=False)
    assert not placed.exists(), "the guard must not create or open its input"

    refuse_processed_paths([placed], allow_processed=True)
    refuse_processed_paths([tmp_path / "regen" / "isat_profiles.hdf5"], allow_processed=False)


def test_a_file_named_processed_outside_a_processed_directory_is_allowed(tmp_path):
    """The guard reads directory components, not the file name."""
    refuse_processed_paths([tmp_path / "processed.hdf5"], allow_processed=False)
