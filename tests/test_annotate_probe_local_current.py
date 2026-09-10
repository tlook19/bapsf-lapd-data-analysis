"""The probe-local-current caveat annotator writes the registry onto exactly
the registered run's group and touches nothing else.
"""

import h5py
import pytest

from scripts.annotate_probe_local_current import (
    CHANNEL,
    main,
    refuse_processed_output,
)
from scripts.export_es1_sim1d_overlay import LATE_AFTERGLOW_PROBE_LOCAL_CURRENT


def _write_minimal_isat_profiles(path):
    """A product just structured enough for the annotator: two runs of one set."""
    with h5py.File(path, "w") as hf:
        for run_id in ("21", "22"):
            hf.create_group(f"experiment_sets/2/{run_id}")


def test_only_the_registered_run_gets_the_caveat_attrs(tmp_path):
    product = tmp_path / "isat_profiles.hdf5"
    _write_minimal_isat_profiles(product)
    summary = tmp_path / "screen.json"

    main([str(tmp_path), str(product), str(summary)])

    entry = LATE_AFTERGLOW_PROBE_LOCAL_CURRENT[("22", CHANNEL)]
    with h5py.File(product, "r") as hf:
        g22 = hf["experiment_sets/2/22"]
        assert bool(g22.attrs["probe_local_current_suspected"]) is True
        assert g22.attrs["probe_local_current_reason"] == entry["reason"]
        assert g22.attrs["probe_local_current_tau_ms"] == entry["tau_ms"]
        assert (
            g22.attrs["probe_local_current_photodiode_tau_ms"]
            == entry["photodiode_tau_ms"]
        )
        assert list(g22.attrs["probe_local_current_neighbor_tau_range_ms"]) == list(
            entry["neighbor_tau_range_ms"]
        )
        assert bool(g22.attrs["probe_local_current_afterglow_only"]) is True
        assert g22.attrs["probe_local_current_source"] == entry["source"]

        g21 = hf["experiment_sets/2/21"]
        assert "probe_local_current_suspected" not in g21.attrs

    assert summary.exists()


def test_summary_json_names_exactly_the_annotated_runs(tmp_path):
    product = tmp_path / "isat_profiles.hdf5"
    _write_minimal_isat_profiles(product)
    summary = tmp_path / "screen.json"

    main([str(tmp_path), str(product), str(summary)])

    import json

    payload = json.loads(summary.read_text())
    assert payload["channel"] == CHANNEL
    assert payload["annotated_runs"] == ["22"]


def test_a_processed_product_path_is_refused_without_the_flag(tmp_path):
    """The guard fires at argument resolution, before any file is opened."""
    placed = tmp_path / "processed" / "isat_profiles.hdf5"
    placed.parent.mkdir()
    placed.write_bytes(b"not really hdf5, the guard never opens it")
    with pytest.raises(ValueError, match="--allow-processed"):
        refuse_processed_output(placed, allow_processed=False)
    assert placed.read_bytes() == b"not really hdf5, the guard never opens it"

    refuse_processed_output(placed, allow_processed=True)
    refuse_processed_output(tmp_path / "regen" / "isat_profiles.hdf5", allow_processed=False)


def test_main_proceeds_on_a_processed_path_with_the_flag(tmp_path):
    placed = tmp_path / "processed" / "isat_profiles.hdf5"
    placed.parent.mkdir()
    _write_minimal_isat_profiles(placed)
    summary = tmp_path / "processed" / "screen.json"

    with pytest.raises(ValueError, match="--allow-processed"):
        main([str(tmp_path), str(placed), str(summary)])

    main([str(tmp_path), str(placed), str(summary), "--allow-processed"])

    with h5py.File(placed, "r") as hf:
        assert bool(
            hf["experiment_sets/2/22"].attrs["probe_local_current_suspected"]
        ) is True
