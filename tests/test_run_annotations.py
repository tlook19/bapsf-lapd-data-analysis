"""The declared run-artifact registry, and the window guard that honours it."""
from pathlib import Path

import numpy as np
import pytest

from bapsf_lapd.annotations import (
    RUN_ARTIFACTS_TOML,
    RunArtifact,
    artifacts_in_window,
    load_run_artifacts,
    refuse_artifacts_in_window,
)
from scripts.export_es1_sim1d_overlay import (
    RAW_PLATEAU_WINDOW_MS,
    _discharge_stats,
)

# ES1 run 05's MSI discharge grid: 4096 samples at 0.04 ms starting 20.48 ms
# before the trigger, so sample 1012 lands on t = 20.000 ms.  The annotation is
# measured off the real record by scripts/annotate_discharge_artifacts.py;
# these are the coordinates it wrote.
RUN05_SAMPLE_INDEX = 1012
RUN05_TIME_MS = 19.999999494757503
DISCHARGE_START_MS = -20.479999482631683
DISCHARGE_DT_MS = 0.03999999898951501
DISCHARGE_N_SAMPLES = 4096
DISCHARGE_TIME_MS = (
    DISCHARGE_START_MS + np.arange(DISCHARGE_N_SAMPLES) * DISCHARGE_DT_MS
)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
def test_the_tracked_registry_declares_run_05s_termination_spike():
    registry = load_run_artifacts(RUN_ARTIFACTS_TOML)

    assert "05" in registry
    (artifact,) = registry["05"]
    assert artifact.kind == "termination_spike"
    assert artifact.channel == "discharge_current"
    assert artifact.sample_index == RUN05_SAMPLE_INDEX
    assert artifact.time_ms == pytest.approx(RUN05_TIME_MS)
    # One amplitude per stored shot; both shots carry the excursion.
    assert len(artifact.peak_amplitude_a) == 2
    assert all(4000.0 < peak < 5000.0 for peak in artifact.peak_amplitude_a)
    assert "FLAGGED not excluded" in artifact.description


def test_the_annotated_sample_index_and_time_name_the_same_sample():
    (artifact,) = load_run_artifacts(RUN_ARTIFACTS_TOML)["05"]

    assert DISCHARGE_TIME_MS[artifact.sample_index] == pytest.approx(
        artifact.time_ms, abs=1e-9
    )


def test_the_registry_carries_the_rule_that_found_each_artifact():
    (artifact,) = load_run_artifacts(RUN_ARTIFACTS_TOML)["05"]

    # The criterion rides the record, so the entry can be re-measured and
    # argued with without reading the annotator's source.
    assert "one-sample" in artifact.detection_rule
    assert "same sample index in every stored shot" in artifact.detection_rule


def test_a_missing_registry_is_an_error_not_an_empty_result():
    # A consumer that read "no annotations" off a missing file would compute
    # over an artifact silently, which is the failure the registry prevents.
    with pytest.raises(FileNotFoundError):
        load_run_artifacts(Path("config/there_is_no_such_registry.toml"))


REQUIRED_ENTRY_FIELDS = (
    "kind",
    "channel",
    "sample_index",
    "time_ms",
    "peak_amplitude_a",
    "description",
    "detection_rule",
)


@pytest.mark.parametrize("dropped", REQUIRED_ENTRY_FIELDS)
def test_the_loader_refuses_an_entry_missing_any_required_field(tmp_path, dropped):
    # An entry short of a field would declare an artifact nobody can
    # re-measure; detection_rule is in this list, so a block written without a
    # rule cannot load clean.
    lines = [
        "[[runs.05.artifacts]]",
        'kind = "termination_spike"',
        'channel = "discharge_current"',
        "sample_index = 1012",
        "time_ms = 20.0",
        "peak_amplitude_a = [4950.7, 4065.9]",
        'description = "a one-sample excursion"',
        'detection_rule = "a stated rule"',
    ]
    kept = [line for line in lines if not line.startswith(f"{dropped} =")]
    assert len(kept) == len(lines) - 1
    registry = tmp_path / "run_artifacts.toml"
    registry.write_text("\n".join(kept) + "\n")

    with pytest.raises(KeyError) as excinfo:
        load_run_artifacts(registry)

    message = str(excinfo.value)
    assert dropped in message
    assert "05" in message


# ---------------------------------------------------------------------------
# The window guard
# ---------------------------------------------------------------------------
def _artifact(time_ms=RUN05_TIME_MS, channel="discharge_current"):
    return {
        "05": (
            RunArtifact(
                run_id="05",
                kind="termination_spike",
                channel=channel,
                sample_index=RUN05_SAMPLE_INDEX,
                time_ms=time_ms,
                peak_amplitude_a=(4950.7, 4065.9),
                description="a one-sample excursion",
                detection_rule="a stated rule",
            ),
        )
    }


def test_the_guard_sees_an_artifact_only_for_a_run_in_the_ensemble():
    window = (15.0, 20.5)

    assert artifacts_in_window(["05"], window, artifacts=_artifact())
    assert not artifacts_in_window(["01", "02"], window, artifacts=_artifact())


def test_the_guard_is_restricted_to_the_channel_the_caller_reads():
    window = (15.0, 20.5)

    assert not artifacts_in_window(
        ["05"], window, channel="isat", artifacts=_artifact()
    )
    assert artifacts_in_window(
        ["05"], window, channel="discharge_current", artifacts=_artifact()
    )


def test_the_guard_refusal_names_the_annotation():
    with pytest.raises(ValueError) as excinfo:
        refuse_artifacts_in_window(
            ["05"], (15.0, 20.5), "a plateau", artifacts=_artifact()
        )

    message = str(excinfo.value)
    assert "run 05" in message
    assert "termination_spike" in message
    assert f"sample {RUN05_SAMPLE_INDEX}" in message
    assert "may2026_run_artifacts.toml" in message
    assert "a plateau" in message


# ---------------------------------------------------------------------------
# The consumer: the raw discharge ensemble's plateau
# ---------------------------------------------------------------------------
def _run05_ensemble():
    """Two synthetic run-05 shots on the real discharge grid, spike included."""
    frac = np.clip((DISCHARGE_TIME_MS - 2.0) / 2.0, 0.0, 1.0)
    shots = []
    for plateau, peak in ((3000.0, 4950.7), (2960.0, 4065.9)):
        shot = np.where(DISCHARGE_TIME_MS >= 20.1, 0.0, plateau * frac)
        shot[RUN05_SAMPLE_INDEX] = peak
        shots.append(shot)
    return np.array(shots)


class _StubRun:
    def __init__(self, current):
        self._current = current

    def discharge_traces(self):
        return (
            self._current,
            -np.full_like(self._current, 100.0),
            DISCHARGE_TIME_MS / 1000.0,
        )

    def discharge_zero_offset_stats(self):
        return type("_Offsets", (), {"offset_a": 0.0})()


class _StubDataset:
    def __init__(self, runs):
        self._runs = runs

    def experiment_set_run_ids(self, experiment_set_id):
        return list(self._runs)

    def run(self, run_id):
        return _StubRun(self._runs[run_id])


def test_the_plateau_refuses_a_window_that_reaches_the_annotated_sample():
    dataset = _StubDataset({"05": _run05_ensemble()})
    # A window moved past 20.0 ms -- the case the annotation exists to catch.
    window = (15.0, 20.5)
    assert window[0] <= RUN05_TIME_MS <= window[1]

    with pytest.raises(ValueError) as excinfo:
        _discharge_stats(
            dataset,
            1,
            raw_ensemble=True,
            plateau_window_ms=window,
            run_artifacts=_artifact(),
        )

    message = str(excinfo.value)
    assert "termination_spike" in message
    assert f"sample {RUN05_SAMPLE_INDEX}" in message
    assert "plateau current" in message


def test_the_plateau_passes_on_the_standard_window():
    dataset = _StubDataset({"05": _run05_ensemble()})
    assert RAW_PLATEAU_WINDOW_MS[1] < RUN05_TIME_MS

    stats = _discharge_stats(
        dataset,
        1,
        raw_ensemble=True,
        plateau_window_ms=RAW_PLATEAU_WINDOW_MS,
        run_artifacts=_artifact(),
    )

    # The spike is in the ensemble and reaches the record maximum, and the
    # plateau still reads the plasma current rather than the transient.
    assert stats["raw"]["current_mean_a"].max() > 4000.0
    assert stats["raw"]["t_half_level_a"] == pytest.approx(0.5 * 2980.0, rel=1e-3)


def test_the_guard_does_not_run_when_no_raw_plateau_is_taken():
    # Flag off: no plateau, no window, nothing to refuse -- even with a
    # registry entry whose sample sits inside the default window.
    dataset = _StubDataset({"05": _run05_ensemble()})

    stats = _discharge_stats(
        dataset, 1, raw_ensemble=False, run_artifacts=_artifact(time_ms=17.0)
    )

    assert stats["raw"] == {}
