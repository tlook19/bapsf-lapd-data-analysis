"""Coverage of the band pass's window statistics and its run resolver.

The pass itself reads raw digitizer traces; what is pinned here is the two
per-cell statistics the product reports and which runs each cell set and
rotation walks, none of which depends on the measured values.
"""
from pathlib import Path

import h5py
import numpy as np

from scripts.refit_window_band import (
    CORE_MAX_CM,
    core_cell_indices,
    set_runs,
    window_family_median,
    window_spread_dln,
)


def _write_sweeps(path, runs, *, x_cm):
    """Write a minimal ``langmuir_sweeps``-shaped source for the resolver."""
    with h5py.File(path, "w") as hdf:
        hdf.create_dataset("x_cm", data=np.asarray(x_cm, dtype=np.float64))
        for set_id, run_id, port, rotation_deg in runs:
            group = hdf.create_group(f"experiment_sets/{set_id}/{run_id}")
            group.attrs["port"] = np.int64(port)
            group.attrs["rotation_deg"] = np.float64(rotation_deg)
    return path


# --------------------------------------------------------------------------
# window-spread statistic
# --------------------------------------------------------------------------


def test_window_spread_is_the_log_ratio_of_the_family_extremes():
    grid = np.array([[1.0, 2.0], [4.0, 3.0]])
    assert window_spread_dln(grid) == np.log(4.0)


def test_window_spread_ignores_unfittable_windows():
    grid = np.array([[np.nan, 2.0], [4.0, np.nan]])
    assert window_spread_dln(grid) == np.log(2.0)


def test_window_spread_of_a_flat_family_is_zero():
    assert window_spread_dln(np.full((5, 5), 1.7)) == 0.0


def test_window_spread_is_nan_when_no_window_fitted():
    assert np.isnan(window_spread_dln(np.full((5, 5), np.nan)))


def test_window_spread_is_nan_rather_than_a_spread_at_a_non_positive_minimum():
    # A non-positive T_e is unphysical, so the cell reports no spread rather
    # than a ratio that would read as stability or as a finite excursion.
    assert np.isnan(window_spread_dln(np.array([[0.0, 2.0], [4.0, 3.0]])))
    assert np.isnan(window_spread_dln(np.array([[-1.0, 2.0], [4.0, 3.0]])))


def test_window_family_median_is_the_median_over_every_window():
    grid = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, np.nan]])
    assert window_family_median(grid) == 3.0


# --------------------------------------------------------------------------
# core cell set
# --------------------------------------------------------------------------


def test_core_cells_are_the_cells_within_the_core_radius():
    x_cm = np.arange(-25.0, 25.5, 1.0)
    cells, index_x0 = core_cell_indices(x_cm)
    assert np.all(np.abs(x_cm[cells]) <= CORE_MAX_CM)
    assert cells.size == int((np.abs(x_cm) <= CORE_MAX_CM).sum())
    # x = 0 belongs to the core, so it is not appended as a separate control
    # the way the band pass appends it.
    assert x_cm[index_x0] == 0.0
    assert index_x0 in cells.tolist()
    assert cells.tolist() == sorted(set(cells.tolist()))


def test_core_radius_is_the_radius_the_te_row_is_built_from():
    from scripts.fit_te_spatial import X_CORE_CM

    assert CORE_MAX_CM == X_CORE_CM


# --------------------------------------------------------------------------
# set-4 port/run resolver
# --------------------------------------------------------------------------


ES4_RUNS = [
    (4, "41", 11, 0.0),
    (4, "42", 21, 0.0),
    (4, "43", 21, 180.0),
    (4, "44", 29, 0.0),
    (4, "45", 29, 180.0),
    (4, "46", 41, 0.0),
    (4, "47", 41, 180.0),
    (4, "48", 50, 0.0),
    (1, "02", 21, 0.0),
    (1, "03", 21, 180.0),
]


def _resolver_source(monkeypatch, tmp_path):
    import scripts.refit_window_band as module

    sweeps = _write_sweeps(
        tmp_path / "langmuir_sweeps.hdf5",
        ES4_RUNS,
        x_cm=np.arange(-25.0, 25.5, 1.0),
    )
    monkeypatch.setattr(module._rw, "SWEEPS_H5", sweeps)
    return module


def test_set_four_rot0_resolves_to_the_rot0_runs(monkeypatch, tmp_path):
    _resolver_source(monkeypatch, tmp_path)
    walked = list(set_runs((4,), 0.0))
    assert walked == [
        (4, "41", 11),
        (4, "42", 21),
        (4, "44", 29),
        (4, "46", 41),
        (4, "48", 50),
    ]


def test_set_four_rot180_resolves_to_the_reversed_probe_runs(monkeypatch, tmp_path):
    _resolver_source(monkeypatch, tmp_path)
    walked = list(set_runs((4,), 180.0))
    assert walked == [(4, "43", 21), (4, "45", 29), (4, "47", 41)]


def test_the_resolver_keeps_the_sets_apart(monkeypatch, tmp_path):
    _resolver_source(monkeypatch, tmp_path)
    assert all(set_id == 1 for set_id, _, _ in set_runs((1,), 0.0))
    assert list(set_runs((4,), 90.0)) == []


def test_rot0_resolution_is_the_published_walk(monkeypatch, tmp_path):
    # The rot-0 case must stay the x = 0 product's own walk, not a re-derived
    # one, so the band product cannot drift away from it.
    module = _resolver_source(monkeypatch, tmp_path)
    assert list(set_runs((1, 2, 4), 0.0)) == list(
        module._rw.rot0_runs(sets=(1, 2, 4))
    )


# --------------------------------------------------------------------------
# blast radius: a non-band pass may not write the band products
# --------------------------------------------------------------------------


import pytest  # noqa: E402

from scripts.refit_window_band import (  # noqa: E402
    OUTPUT_HDF5,
    OUTPUT_METADATA,
    OUTPUT_SUMMARY,
    SETS,
    refuse_band_output_paths,
)

BAND_DEFAULTS = dict(
    output=OUTPUT_HDF5, summary=OUTPUT_SUMMARY, metadata=OUTPUT_METADATA
)


def test_the_band_pass_resolves_against_its_own_defaults():
    # The default invocation is the one that MAY write the band products.
    refuse_band_output_paths(
        cells="band", sets=SETS, rotation_deg=0.0, **BAND_DEFAULTS
    )


def test_a_core_pass_is_refused_at_the_band_output_paths():
    with pytest.raises(ValueError) as excinfo:
        refuse_band_output_paths(
            cells="core", sets=SETS, rotation_deg=0.0, **BAND_DEFAULTS
        )
    message = str(excinfo.value)
    assert "--cells core" in message
    # The refusal names every flag the caller has to pass, and why.
    for flag in ("--output", "--summary", "--metadata"):
        assert flag in message
    assert "fit_te_spatial.py" in message


def test_a_rot180_pass_and_a_non_band_set_list_are_refused_too():
    with pytest.raises(ValueError, match="--rotation-deg 180"):
        refuse_band_output_paths(
            cells="band", sets=SETS, rotation_deg=180.0, **BAND_DEFAULTS
        )
    with pytest.raises(ValueError, match="--sets 4"):
        refuse_band_output_paths(
            cells="band", sets=(4,), rotation_deg=0.0, **BAND_DEFAULTS
        )


def test_the_refusal_names_only_the_paths_still_at_their_band_default(tmp_path):
    with pytest.raises(ValueError) as excinfo:
        refuse_band_output_paths(
            cells="core",
            sets=(4,),
            rotation_deg=0.0,
            output=tmp_path / "core.hdf5",
            summary=OUTPUT_SUMMARY,
            metadata=tmp_path / "core.json",
        )
    message = str(excinfo.value)
    assert "--summary" in message
    assert "--output" not in message
    assert "--metadata" not in message


def test_a_fully_redirected_core_pass_resolves(tmp_path):
    refuse_band_output_paths(
        cells="core",
        sets=(4,),
        rotation_deg=180.0,
        output=tmp_path / "core.hdf5",
        summary=tmp_path / "core.csv",
        metadata=tmp_path / "core.json",
    )


# --------------------------------------------------------------------------
# provenance: the core metadata describes the core pass
# --------------------------------------------------------------------------


from scripts.refit_window_band import (  # noqa: E402
    ADJUDICATION,
    CORE_GATE_MIN_CELLS,
    CORE_PROTOCOL,
    PROTOCOL,
    _core_metadata,
)


def _core_summary(port, below):
    return {
        "set_id": 4,
        "port": port,
        "run_id": "42",
        "n_plateau_cycles": 10,
        "in_band_cells": 20,
        "in_band_median_dln": 0.4,
        "in_band_upper_quartile_dln": 0.5,
        "in_band_max_dln": 1.0,
        "in_band_fraction_at_or_above_criterion": 0.25,
        "x0_control_dln": 0.44,
        "criterion_dln": 0.5,
        "passes_criterion": True,
        "x0_control_passes_criterion": True,
        "trust_to_aperture_adopted": False,
        "core_cells": 21,
        "core_cells_below_criterion": below,
        "core_mean_te_ev": 0.66,
        "core_mean_te_default_window_ev": 0.70,
        "core_max_cm": 10.0,
    }


def _built_core_metadata(**overrides):
    summaries = overrides.pop(
        "summaries", [_core_summary(21, 16), _core_summary(29, 0)]
    )
    return _core_metadata(
        summaries,
        created="2026-01-01T00:00:00+00:00",
        core_mode={
            "core_max_cm": 10.0,
            "core_gate_min_cells": CORE_GATE_MIN_CELLS,
            "rotation_deg": 180.0,
        },
        hdf5_path=Path("/tmp/core.hdf5"),
        summary_path=Path("/tmp/core.csv"),
        **overrides,
    )


def test_core_metadata_states_the_core_protocol_not_the_band_one():
    metadata = _built_core_metadata()
    assert metadata["product"] == "window_refit_core"
    assert metadata["protocol"] == CORE_PROTOCOL
    assert metadata["protocol"] != PROTOCOL
    assert ADJUDICATION not in metadata.values()
    assert "adjudication" not in metadata


def test_core_metadata_carries_the_core_pass_parameters():
    metadata = _built_core_metadata()
    assert metadata["cells_mode"] == "core"
    assert metadata["core_max_cm"] == 10.0
    assert metadata["core_gate_min_cells"] == CORE_GATE_MIN_CELLS
    assert metadata["rotation_deg"] == 180.0
    assert metadata["sets_run"] == [4]
    assert metadata["ports_run"] == [21, 29]


def test_core_metadata_drops_every_band_only_field():
    metadata = _built_core_metadata()
    assert "band_min_cm" not in metadata
    assert "band_max_cm" not in metadata
    for port in metadata["ports"]:
        assert not any(key.startswith("in_band") for key in port)
        assert "trust_to_aperture_adopted" not in port
        assert "passes_criterion" not in port


def test_core_metadata_records_the_gate_per_port():
    ports = {port["port"]: port for port in _built_core_metadata()["ports"]}
    assert ports[21]["core_gate_passes"] is True
    assert ports[29]["core_gate_passes"] is False
    assert ports[29]["core_cells_below_criterion"] == 0


def _one_record(cells_mode):
    """A minimal two-cell record the writer accepts, in either cell mode."""
    record = dict(
        cells=np.array([0, 1]),
        x_cm=np.array([-1.0, 0.0]),
        is_x0=np.array([False, True]),
        dln_te_window=np.array([0.30, 0.44]),
        te_default_med=np.array([0.70, 0.74]),
        n_sweeps=np.array([200, 200]),
        med_grids=np.full((2, 5, 5), 0.66),
        run_id="42",
        sid=4,
        port=21,
        wall_s=1.0,
        n_plateau=10,
    )
    if cells_mode == "core":
        record["cells_mode"] = "core"
    return record


def _write_both_modes(tmp_path):
    """Write a band-mode and a core-mode product; return the two JSONs."""
    import json

    from scripts.refit_window_band import write_product

    written = {}
    for mode, core_mode in (
        ("band", None),
        (
            "core",
            {
                "core_max_cm": 10.0,
                "core_gate_min_cells": CORE_GATE_MIN_CELLS,
                "rotation_deg": 0.0,
            },
        ),
    ):
        metadata_path = tmp_path / f"{mode}.json"
        write_product(
            [_one_record(mode)],
            np.array([-1.0, 0.0]),
            hdf5_path=tmp_path / f"{mode}.hdf5",
            summary_path=tmp_path / f"{mode}.csv",
            metadata_path=metadata_path,
            core_mode=core_mode,
        )
        written[mode] = json.loads(metadata_path.read_text())
    return written


def test_the_two_modes_write_distinguishable_metadata(tmp_path):
    written = _write_both_modes(tmp_path)
    band, core = written["band"], written["core"]

    # The band metadata is unchanged: it still states the band protocol, the
    # band adjudication and the band edges.
    assert band["product"] == "window_refit_band"
    assert band["protocol"] == PROTOCOL
    assert band["adjudication"] == ADJUDICATION
    assert "band_min_cm" in band and "band_max_cm" in band
    assert "in_band_median_dln" in band["ports"][0]
    assert "cells_mode" not in band

    # The core metadata states the core protocol and no band-only field.
    assert core["product"] == "window_refit_core"
    assert core["protocol"] == CORE_PROTOCOL
    assert core["cells_mode"] == "core"
    assert "band_min_cm" not in core and "band_max_cm" not in core
    assert "adjudication" not in core
    assert not any(key.startswith("in_band") for key in core["ports"][0])


def test_the_core_product_marks_its_cell_mode_in_the_hdf5_root(tmp_path):
    _write_both_modes(tmp_path)
    with h5py.File(tmp_path / "core.hdf5", "r") as core:
        assert core.attrs["cells_mode"] == "core"
        assert core.attrs["core_protocol"] == CORE_PROTOCOL
        assert core.attrs["core_max_cm"] == 10.0
        assert core.attrs["rotation_deg"] == 0.0
    with h5py.File(tmp_path / "band.hdf5", "r") as band:
        assert "cells_mode" not in band.attrs
        assert "core_protocol" not in band.attrs


def test_a_refused_port_carries_null_rather_than_a_nan_literal(tmp_path):
    import json

    metadata_path = tmp_path / "core.json"
    write_product_metadata = _built_core_metadata(
        summaries=[_core_summary(29, 0) | {"core_mean_te_ev": float("nan")}]
    )
    metadata_path.write_text(json.dumps(write_product_metadata, indent=2))
    # json.loads accepts the NaN literal; a strict parser does not, so the
    # refused reading has to be null in the file itself.
    assert "NaN" not in metadata_path.read_text()
    reloaded = json.loads(metadata_path.read_text(), parse_constant=_no_constants)
    assert reloaded["ports"][0]["core_mean_te_ev"] is None


def _no_constants(name):
    raise AssertionError(f"non-standard JSON constant in the product: {name}")
