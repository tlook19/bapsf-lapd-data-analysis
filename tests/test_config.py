from pathlib import Path

import pytest

from bapsf_lapd import ChannelKind, LapdDataset, default_run_config, load_run_manifest


def test_default_run_config_maps_experiment_set_and_sweep_resistor():
    config = default_run_config("42", path=Path("dummy.hdf5"), port=21, rotation_deg=0)

    assert config.experiment_set.v_bank == 100.0
    assert config.experiment_set.v_puff == 110.0
    assert config.probe.z_cm == pytest.approx(789.55)
    assert config.channel(ChannelKind.I_SWEEP).resistor_ohm == 3.0
    assert config.sweep.ramp_voltage == 20.0
    assert config.sweep.voltage_start == -20.0
    assert config.sweep.voltage_end == 20.0

    aliased = default_run_config("01", path=Path("dummy.hdf5"), port=11, rotation_deg=0)
    assert aliased.experiment_set.id == 1


def test_default_run_config_rejects_missing_experiment_set():
    with pytest.raises(ValueError, match="No experiment set"):
        default_run_config("52")


def test_dataset_discovers_may2026_files():
    dataset = LapdDataset.from_directory("data/may2026")
    if len(dataset) == 0:
        pytest.skip("local HDF5 data files are not present")

    assert len(dataset) == 32
    assert dataset.run_ids()[0] == "01"
    assert dataset.config("01").probe.port == 11
    assert dataset.config("47").probe.rotation_deg == 180.0


def test_manifest_loads_run_and_trace_ports():
    configs = load_run_manifest("config/may2026_run_manifest.toml")

    run01 = configs["01"]
    assert run01.path.name.startswith("01_xline")
    assert run01.probe.port == 11
    assert run01.channel("reference_photodiode").port == 16
    assert run01.channel("moving_photodiode").port == 21
    assert run01.channel("isat").port == 11
    assert run01.channel("i_sweep").port == 11
    assert run01.channel("v_sweep").port == 11

    run42 = configs["42"]
    assert run01.experiment_set.id == 1
    assert run42.experiment_set.v_puff == 110.0
    assert run42.channel("i_sweep").resistor_ohm == 3.0


def test_manifest_loads_current_attenuation_rules():
    dataset = LapdDataset.from_manifest("config/may2026_run_manifest.toml")

    for run_id in dataset.run_ids():
        config = dataset.config(run_id)
        expected = 2.0 if config.experiment_set.id in {1, 2, 3} and 2 <= int(run_id[1]) <= 7 else 1.0
        assert config.channel("isat").attenuation == expected
        assert config.channel("i_sweep").attenuation == expected


def test_manifest_loads_sweep_families():
    dataset = LapdDataset.from_manifest("config/may2026_run_manifest.toml")

    for run_id in [f"0{i}" for i in range(1, 9)] + [f"2{i}" for i in range(1, 9)]:
        sweep = dataset.config(run_id).sweep
        assert sweep.n_cycles == 40
        assert sweep.tau_ramp_s == pytest.approx(250e-6)
        assert sweep.tau_cycle_s == pytest.approx(500e-6)
        assert sweep.voltage_start == -75.0
        assert sweep.voltage_end == 75.0

    for run_id in [f"3{i}" for i in range(1, 9)] + ["41"]:
        sweep = dataset.config(run_id).sweep
        assert sweep.n_cycles == 20
        assert sweep.tau_ramp_s == pytest.approx(500e-6)
        assert sweep.tau_cycle_s == pytest.approx(1e-3)
        assert sweep.voltage_start == -75.0
        assert sweep.voltage_end == 75.0

    for run_id in [f"4{i}" for i in range(2, 9)]:
        sweep = dataset.config(run_id).sweep
        assert sweep.n_cycles == 20
        assert sweep.tau_ramp_s == pytest.approx(500e-6)
        assert sweep.tau_cycle_s == pytest.approx(1e-3)
        assert sweep.ramp_voltage == 20.0
        assert sweep.voltage_start == -20.0
        assert sweep.voltage_end == 20.0


def test_dataset_can_load_from_manifest():
    dataset = LapdDataset.from_manifest("config/may2026_run_manifest.toml")

    assert len(dataset) == 32
    assert dataset.config("08").channel("moving_photodiode").port == 16
    assert dataset.experiment_set_ids() == [1, 2, 3, 4]
    assert dataset.config("08").acquisition.hardware_average_samples == 16
    assert dataset.config("08").acquisition.n_positions == 51
    assert dataset.config("08").acquisition.n_shots_per_position == 20
    assert dataset.config("08").acquisition.sample_rate_hz == 6_250_000.0
    assert dataset.config("08").acquisition.sample_dt_s == pytest.approx(16 / 100e6)


def test_manifest_moving_photodiode_port_patterns():
    dataset = LapdDataset.from_manifest("config/may2026_run_manifest.toml")
    set0_pattern = {
        "1": 21,
        "2": 26,
        "3": 26,
        "4": 32,
        "5": 32,
        "6": 44,
        "7": 44,
        "8": 16,
    }
    other_set_pattern = {
        "1": 16,
        "2": 21,
        "3": 21,
        "4": 26,
        "5": 26,
        "6": 32,
        "7": 32,
        "8": 44,
    }

    for run_id in dataset.run_ids():
        pattern = set0_pattern if run_id.startswith("0") else other_set_pattern
        assert dataset.config(run_id).channel("moving_photodiode").port == pattern[run_id[1]]
