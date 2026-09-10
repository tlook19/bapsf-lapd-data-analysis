"""The probe-local-current caveat annotator writes the registry onto exactly
the registered run's group and touches nothing else.
"""

import h5py

from scripts.annotate_probe_local_current import CHANNEL, main
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
