"""Record declared discharge-current artifacts in the tracked run registry.

ANNOTATES
    config/may2026_run_artifacts.toml
    -- the tracked, per-run artifact registry read by
    ``bapsf_lapd.annotations.load_run_artifacts``.

Companions: scripts/annotate_saturation_attrs.py and
scripts/annotate_rail_mask.py, which record digitizer-saturation screens ON a
processed product.  This one writes to a TRACKED CONFIG instead, because the
thing it records is a property of the RUN's raw discharge channel rather than
of any one product: every product built from that run inherits it, and none of
them owns it.  A processed product is also regenerable and gitignored, so an
annotation written into one is lost at the next rebuild.

WHAT IT MEASURES
Both stored shots of ES1 run 05 carry a one-sample discharge-current excursion
to ~4-5 kA at the same sample, immediately followed by the discharge's own
fall-off, against a plateau within 2 % of the ES1 pack median.  Two shots
putting it at the SAME sample index is what makes it a deterministic
termination-switching transient rather than shot noise, and that agreement is
part of the detection rule rather than an observation about the result:

    a sample is a one-sample excursion when it is the maximum of the
    offset-corrected record, lies after the scoring plateau window, has both
    immediate neighbours below SPIKE_NEIGHBOUR_MAX_FRACTION of it, and every
    stored shot of the run puts it at the same sample index.

The numbers written are the ones this pass measured, not a transcribed copy;
run it again and it re-measures.

THE RULING IS FLAG, NOT EXCLUDE.  Nothing here drops a sample or a shot.  What
the annotation buys is the guard in ``bapsf_lapd.annotations``: a consumer
computing a plateau or a timing statistic over a window refuses when its window
covers an annotated sample.  Run 05's spike sits 0.5 ms past the end of the
scoring plateau window, so every consumer passes today; the guard is what makes
a future window moved past it fail loudly instead of averaging a switching
transient into a plasma measurement.

THE WRITE IS ADDITIVE.  The registry is edited as TEXT: the header and every
block belonging to another run come through byte-identical, and only this run's
block is replaced or appended.  Re-running on an unchanged dataset therefore
reproduces the file byte-for-byte.  Prove it with
scripts/compare_run_annotations.py.

Needs h5py and numpy and the raw run files -- the environment.yml env, not the
dead ./.venv.  Reads one run's MSI discharge traces, so it is quick.

Usage:  python scripts/annotate_discharge_artifacts.py <repo-root> [--run 05]
                [--registry config/may2026_run_artifacts.toml]
                [--summary-json processed/discharge_run_artifacts.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bapsf_lapd import LapdDataset  # noqa: E402

#: The scoring plateau window, in ms.  The excursion must lie AFTER it for the
#: run to be summarised as "plateau normal, artifact past the window"; the same
#: window is what the exporter's raw discharge ensemble averages over.
PLATEAU_MS = (15.0, 19.5)

#: Both immediate neighbours of a one-sample excursion must sit below this
#: fraction of it.  Run 05's neighbours read ~0.61 and ~0.49 of the peak; a
#: genuine plasma current maximum has neighbours within a percent or two.
SPIKE_NEIGHBOUR_MAX_FRACTION = 0.85

KIND = "termination_spike"
CHANNEL = "discharge_current"

RULE = (
    "one-sample discharge-current excursion: the maximum sample of the "
    "offset-corrected record, lying after the scoring plateau window, with "
    "both immediate neighbours below "
    f"{SPIKE_NEIGHBOUR_MAX_FRACTION:g} of it, at the same sample index in "
    "every stored shot of the run"
)

DESCRIPTION = (
    "one-sample discharge-current excursion to ~4-5 kA immediately followed "
    "by the discharge fall-off; a deterministic discharge-termination "
    "switching transient, FLAGGED not excluded"
)


def measure_spike(run) -> dict:
    """Measure the run's one-sample discharge-current excursion.

    Raises ``ValueError`` when the record does not satisfy the detection rule,
    so a run without the artifact cannot be annotated as having one.
    """
    current, _voltage, time_s = run.discharge_traces()
    current = current - run.discharge_zero_offset_stats().offset_a
    time_ms = time_s * 1000.0

    peak_index = [int(np.argmax(shot)) for shot in current]
    if len(set(peak_index)) != 1:
        raise ValueError(
            f"run {run.config.run_id}: stored shots peak at different samples "
            f"{peak_index}, so the excursion is not deterministic"
        )
    index = peak_index[0]
    if index <= 0 or index >= current.shape[1] - 1:
        raise ValueError(
            f"run {run.config.run_id}: the record maximum is at an endpoint "
            f"(sample {index}) and has no neighbours to judge"
        )
    if time_ms[index] <= PLATEAU_MS[1]:
        raise ValueError(
            f"run {run.config.run_id}: the record maximum at t = "
            f"{time_ms[index]:.6g} ms is inside or before the scoring plateau "
            f"window, so it is not a post-plateau excursion"
        )

    peaks = current[:, index]
    neighbours = np.maximum(current[:, index - 1], current[:, index + 1])
    ratio = neighbours / peaks
    if not np.all(ratio < SPIKE_NEIGHBOUR_MAX_FRACTION):
        raise ValueError(
            f"run {run.config.run_id}: sample {index} is not a ONE-sample "
            f"excursion; neighbour/peak ratios {ratio.tolist()} are not all "
            f"below {SPIKE_NEIGHBOUR_MAX_FRACTION:g}"
        )

    window = (time_ms >= PLATEAU_MS[0]) & (time_ms <= PLATEAU_MS[1])
    return {
        "run_id": run.config.run_id,
        "experiment_set_id": int(run.config.experiment_set.id),
        "sample_index": index,
        "time_ms": float(time_ms[index]),
        "peak_amplitude_a": [float(value) for value in peaks],
        "neighbour_peak_ratio": [float(value) for value in ratio],
        "plateau_current_a": [float(value) for value in current[:, window].mean(axis=1)],
        "zero_offset_a": float(run.discharge_zero_offset_stats().offset_a),
        "n_shots": int(current.shape[0]),
        "rule": RULE,
    }


def _toml_float(value: float) -> str:
    return repr(float(value))


def _block_lines(measurement: dict) -> list[str]:
    """The registry block for one measured artifact, as lines."""
    amplitudes = ", ".join(_toml_float(v) for v in measurement["peak_amplitude_a"])
    return [
        f'[[runs.{measurement["run_id"]}.artifacts]]\n',
        f'kind = "{KIND}"\n',
        f'channel = "{CHANNEL}"\n',
        f'sample_index = {measurement["sample_index"]}\n',
        f'time_ms = {_toml_float(measurement["time_ms"])}\n',
        f"peak_amplitude_a = [{amplitudes}]\n",
        f'description = "{DESCRIPTION}"\n',
        f'detection_rule = "{RULE}"\n',
    ]


HEADER = [
    "# Declared per-run signal artifacts for the May 2026 LAPD dataset.\n",
    "#\n",
    "# An artifact is a stretch of a run's record that the instrument produced\n",
    "# and the plasma did not.  Declaring one FLAGS it; nothing here excludes a\n",
    "# sample, a shot or a run.  What the declaration buys is the window guard in\n",
    "# bapsf_lapd.annotations: a consumer computing a plateau or a timing\n",
    "# statistic over a window refuses when its window covers an annotated\n",
    "# sample, instead of averaging across it.\n",
    "#\n",
    "# sample_index and time_ms name the same sample on that channel's own\n",
    "# acquisition grid; time_ms is what the guard compares a window against.\n",
    "# peak_amplitude_a carries one entry per stored shot, offset-corrected the\n",
    "# way the analysis reads the channel.\n",
    "#\n",
    "# Written by scripts/annotate_discharge_artifacts.py, which MEASURES each\n",
    "# entry from the raw record; the write is additive, so re-running it leaves\n",
    "# every other run's block byte-identical.\n",
    "\n",
]


def merge_block(existing_text: str | None, run_id: str, lines: list[str]) -> str:
    """Return the registry text with ``run_id``'s block replaced or appended.

    Every byte outside ``run_id``'s own block is carried through unchanged,
    which is what makes the write additive without depending on a TOML
    round-trip preserving foreign formatting.  A block runs from its
    ``[[runs.NN.artifacts]]`` header to the next header or to the blank line
    that separates it from what follows, whichever comes first: blocks in this
    file carry no internal blank line, and treating the separator as the end is
    what keeps the blank between two blocks from being eaten on a replace.
    """
    if existing_text is None:
        return "".join(HEADER + lines)

    marker = f"[[runs.{run_id}.artifacts]]"
    out: list[str] = []
    skipping = False
    replaced = False
    for line in existing_text.splitlines(keepends=True):
        if line.startswith("[["):
            skipping = line.rstrip("\n") == marker
            if skipping:
                if not replaced:
                    out.extend(lines)
                    replaced = True
                continue
        if skipping:
            if line.strip():
                continue
            skipping = False
        out.append(line)

    if not replaced:
        if out and not out[-1].endswith("\n"):
            out.append("\n")
        if out and out[-1].strip():
            out.append("\n")
        out.extend(lines)
    return "".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "repo_root",
        type=Path,
        help="repository root holding config/may2026_run_manifest.toml "
             "and the raw run directory it names",
    )
    parser.add_argument(
        "--run",
        default="05",
        help="run id to measure and annotate",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="artifact registry to edit in place "
             "(default: <repo-root>/config/may2026_run_artifacts.toml)",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="path for the JSON measurement sidecar; omit to write none",
    )
    args = parser.parse_args(argv)

    registry = args.registry or args.repo_root / "config/may2026_run_artifacts.toml"
    dataset = LapdDataset.from_manifest(
        args.repo_root / "config/may2026_run_manifest.toml"
    )
    measurement = measure_spike(dataset.run(args.run))

    existing = registry.read_text() if registry.exists() else None
    registry.write_text(merge_block(existing, args.run, _block_lines(measurement)))

    print(f"annotated {registry}")
    for key in (
        "run_id",
        "sample_index",
        "time_ms",
        "peak_amplitude_a",
        "neighbour_peak_ratio",
        "plateau_current_a",
    ):
        print(f"  {key}: {measurement[key]}")

    if args.summary_json is not None:
        args.summary_json.write_text(json.dumps(measurement, indent=2))
        print(f"wrote {args.summary_json}")


if __name__ == "__main__":
    main()
