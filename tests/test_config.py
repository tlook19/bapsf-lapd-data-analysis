from pathlib import Path

import pytest

from bapsf_lapd import (
    ChannelKind,
    LapdDataset,
    density_area_key_for_deadtime_source,
    electrical_connections_swapped,
    default_run_config,
    effective_deadtime_source,
    effective_rotation_deg,
    load_run_manifest,
)
from bapsf_lapd.config import PORT_MAP_DEFAULT, z_from_port


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


def test_effective_rotation_overrides_es3_p21_swap():
    dataset = LapdDataset.from_manifest("config/may2026_run_manifest.toml")

    assert dataset.config("32").probe.rotation_deg == 0.0
    assert dataset.config("33").probe.rotation_deg == 180.0
    assert effective_rotation_deg("32", dataset.config("32").probe.rotation_deg) == 180.0
    assert effective_rotation_deg("33", dataset.config("33").probe.rotation_deg) == 0.0
    assert effective_rotation_deg("35", dataset.config("35").probe.rotation_deg) == 180.0


def test_effective_deadtime_source_uses_isat_for_known_wiring_swap():
    source, invert, overridden = effective_deadtime_source("31", 11, ChannelKind.I_SWEEP, True)

    assert electrical_connections_swapped("31") is True
    assert source == ChannelKind.ISAT
    assert invert is False
    assert overridden is True
    # Crossed cables: run 31's ISAT channel sat on the LEFT electrode, so its
    # upstream row takes the left electrode's area.
    assert density_area_key_for_deadtime_source("31", source) == "ap_L_cm2"
    assert density_area_key_for_deadtime_source("31", ChannelKind.I_SWEEP) == "ap_R_cm2"

    source, invert, overridden = effective_deadtime_source("01", 11, ChannelKind.I_SWEEP, True)
    assert electrical_connections_swapped("01") is False
    assert source == ChannelKind.I_SWEEP
    assert invert is True
    assert overridden is False

    source, invert, overridden = effective_deadtime_source("33", 21, ChannelKind.I_SWEEP, True)
    assert source == ChannelKind.I_SWEEP
    assert invert is True
    assert overridden is False
    assert density_area_key_for_deadtime_source("33", source) == "ap_L_cm2"


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


def test_port_maps_carry_both_ladders_and_the_default_is_the_nominal_one():
    nominal = {11: 470.05, 21: 789.55, 29: 1045.15, 41: 1428.55, 50: 1716.10}
    cad = {11: 470.67, 21: 790.67, 29: 1046.67, 41: 1430.67, 50: 1718.67}
    interferometer_nominal = {20: 757.60, 29: 1045.15, 40: 1396.60}
    interferometer_cad = {20: 758.67, 29: 1046.67, 40: 1398.67}

    for port, z_cm in (nominal | interferometer_nominal).items():
        assert z_from_port(port) == pytest.approx(z_cm)
        assert z_from_port(port, "nominal") == pytest.approx(z_cm)
        assert z_from_port(port) == z_from_port(port, PORT_MAP_DEFAULT)

    for port, z_cm in (cad | interferometer_cad).items():
        assert z_from_port(port, "cad") == pytest.approx(z_cm)

    # The CAD ladder is anchored one port lower and is the wider of the two, so
    # it moves every measurement port downstream by an offset that grows with
    # port number.  Nothing reads it unless it is asked for by name.
    offsets = [z_from_port(port, "cad") - z_from_port(port) for port in nominal]
    assert offsets == sorted(offsets)
    assert offsets[0] == pytest.approx(0.62)
    assert offsets[-1] == pytest.approx(2.57)


def test_unknown_port_map_is_refused():
    with pytest.raises(ValueError, match="unknown port map"):
        z_from_port(21, "CAD")
