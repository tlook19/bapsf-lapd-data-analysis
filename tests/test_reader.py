import pytest

from bapsf_lapd import LapdDataset


DATASET = LapdDataset.from_directory("data/may2026")


@pytest.mark.skipif(len(DATASET) == 0, reason="local HDF5 data files are not present")
def test_reader_derives_discharge_summary():
    run = DATASET.run("01")
    summary = run.discharge_summary()

    assert summary.peak_current_a > 0
    assert summary.voltage_at_peak_v > 0
    assert summary.raw_voltage_at_peak_v < 0
    assert summary.peak_power_w > 0
    assert summary.sample_index >= 0


@pytest.mark.skipif(len(DATASET) == 0, reason="local HDF5 data files are not present")
def test_reader_converts_trace_slice():
    run = DATASET.run("01")
    trace = run.trace("moving_photodiode", shot=0, sample=slice(0, 10))

    assert trace.shape == (10,)


@pytest.mark.skipif(len(DATASET) == 0, reason="local HDF5 data files are not present")
def test_reader_averages_trace_slice():
    run = DATASET.run("01")
    stats = run.trace_stats("moving_photodiode", sample=slice(0, 10))

    assert stats.mean.shape == (10,)
    assert stats.std.shape == (10,)
    assert stats.stderr.shape == (10,)
    assert stats.n == run.shot_count("moving_photodiode")
    assert stats.time_s[1] == pytest.approx(16 / 100e6)


@pytest.mark.skipif(len(DATASET) == 0, reason="local HDF5 data files are not present")
def test_reader_reshapes_langmuir_trace_slice():
    run = DATASET.run("01")
    traces = run.langmuir_traces("isat", sample=slice(0, 10))
    stats = run.langmuir_position_stats("isat", sample=slice(0, 10))

    assert traces.shape == (51, 20, 10)
    assert stats.mean.shape == (51, 10)
    assert stats.stderr.shape == (51, 10)
    assert stats.n == 20


@pytest.mark.skipif(len(DATASET) == 0, reason="local HDF5 data files are not present")
def test_reader_slices_configured_sweep_ramps():
    run = DATASET.run("46")
    ramp_slices = run.sweep_ramp_sample_slices()
    clipped_slices = run.sweep_ramp_sample_slices(clip_s=10e-6)
    clip_samples = round(10e-6 / run.sample_dt_s())

    assert len(ramp_slices) == 20
    assert ramp_slices[0].start == 0
    assert ramp_slices[0].stop - ramp_slices[0].start == round(500e-6 / run.sample_dt_s())
    assert ramp_slices[1].start - ramp_slices[0].start == round(1e-3 / run.sample_dt_s())
    assert clipped_slices[0].start == ramp_slices[0].start + clip_samples
    assert clipped_slices[0].stop == ramp_slices[0].stop - clip_samples


@pytest.mark.skipif(len(DATASET) == 0, reason="local HDF5 data files are not present")
def test_reader_estimates_current_channel_zero_offset():
    run = DATASET.run("01")
    offset = run.zero_offset_stats("isat", tail_duration_s=200e-6)

    assert offset.channel == "isat"
    assert offset.target_end_v == 0.0
    assert offset.offset_v == pytest.approx(offset.measured_mean_v)
    assert offset.n_shots == run.shot_count("isat")
    assert offset.n_samples == round(200e-6 / run.sample_dt_s())
    assert offset.sample_stop > offset.sample_start
    assert offset.std_v >= 0
    assert offset.stderr_v >= 0


@pytest.mark.skipif(len(DATASET) == 0, reason="local HDF5 data files are not present")
def test_reader_estimates_sweep_voltage_offset_to_ramp_low_side():
    run = DATASET.run("01")
    offset = run.zero_offset_stats("v_sweep", tail_duration_s=200e-6)

    assert offset.channel == "v_sweep"
    assert offset.target_end_v == pytest.approx(run.config.sweep.voltage_start / 100.0)
    assert offset.offset_v == pytest.approx(offset.measured_mean_v - offset.target_end_v)


@pytest.mark.skipif(len(DATASET) == 0, reason="local HDF5 data files are not present")
def test_dataset_averages_reference_slice_across_experiment_set():
    stats = DATASET.reference_stats_for_experiment_set(1, sample=slice(0, 10))

    assert stats.mean.shape == (10,)
    assert stats.n > DATASET.run("01").shot_count("reference_photodiode")
