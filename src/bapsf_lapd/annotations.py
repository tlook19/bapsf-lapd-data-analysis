"""Declared per-run signal artifacts, and the window guard that honours them.

An ARTIFACT is a stretch of a run's record that the instrument produced and the
plasma did not: a switching transient, a converter rail, a pickup burst.  The
declaration says the samples are real readings of something other than the
quantity the analysis is after; it is NOT an exclusion, and nothing here drops
a sample.  A consumer that averages or times over a window decides for itself
what to do, and the one thing it must not do is average across an artifact
without noticing.

The declaration lives in the tracked registry ``config/may2026_run_artifacts.toml``
rather than in a processed product, for two reasons: the products are
regenerable and gitignored, so an annotation written into one is lost at the
next rebuild; and the artifact is a property of the RUN, so every product built
from that run inherits it and none of them owns it.  The registry is written by
``scripts/annotate_discharge_artifacts.py``, which measures the artifact from
the raw record rather than transcribing it.

The guard is ``refuse_artifacts_in_window``.  A consumer computing a plateau or
a timing statistic over a window passes the window and the run ids in its
ensemble; if a declared artifact sample falls inside, the consumer RAISES and
the message names the annotation.  That is deliberately loud: the alternative
is a window that is silently correct today and silently wrong the day someone
moves it, which is the whole reason the artifact is declared at all.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

#: The tracked registry of declared per-run artifacts, relative to the repo root.
RUN_ARTIFACTS_TOML = Path("config/may2026_run_artifacts.toml")


@dataclass(frozen=True)
class RunArtifact:
    """One declared artifact on one run's channel.

    ``sample_index`` and ``time_ms`` name the SAME sample on that channel's own
    acquisition grid; ``time_ms`` is what the window guard compares against,
    because a window is stated in time and a sample index is meaningless across
    two clocks.  ``peak_amplitude_a`` carries the artifact's amplitude in one
    entry per stored shot, in the channel's own units, offset-corrected the way
    the analysis reads that channel.

    ``detection_rule`` is the criterion the annotator applied to find this
    artifact, carried on the record rather than left in the annotator's source.
    It is required, not optional: an entry that says an artifact is present
    without saying what test found it cannot be re-measured or argued with, and
    the loader refuses one.
    """

    run_id: str
    kind: str
    channel: str
    sample_index: int
    time_ms: float
    peak_amplitude_a: tuple[float, ...]
    description: str
    detection_rule: str

    @property
    def label(self) -> str:
        """A one-line name for this annotation, for use in error messages."""
        return (
            f"run {self.run_id} {self.channel} {self.kind} at sample "
            f"{self.sample_index} (t = {self.time_ms:.6g} ms)"
        )


#: Every field a registry entry must carry; ``run_id`` comes from the block's
#: own key rather than from the entry, so it is not among them.
_REQUIRED_FIELDS = (
    "kind",
    "channel",
    "sample_index",
    "time_ms",
    "peak_amplitude_a",
    "description",
    "detection_rule",
)


def load_run_artifacts(
    path: str | Path = RUN_ARTIFACTS_TOML,
) -> dict[str, tuple[RunArtifact, ...]]:
    """Load the artifact registry, keyed by run id.

    Raises ``FileNotFoundError`` if the registry is missing.  A consumer that
    fell back to "no annotations" on a missing file would compute over an
    artifact silently, which is exactly the failure the registry exists to
    prevent, so the absence is an error rather than an empty result.

    Raises ``KeyError`` naming the run and the missing field if an entry is
    short of any field ``RunArtifact`` declares, ``detection_rule`` included.
    Every field is load-bearing -- a partially-filled entry would flag a window
    with an artifact nobody can re-measure -- so none of them is optional.
    """
    path = Path(path)
    with open(path, "rb") as stream:
        data = tomllib.load(stream)

    registry: dict[str, tuple[RunArtifact, ...]] = {}
    for run_id, block in sorted(data.get("runs", {}).items()):
        records = []
        for position, entry in enumerate(block.get("artifacts", ())):
            missing = [name for name in _REQUIRED_FIELDS if name not in entry]
            if missing:
                raise KeyError(
                    f"run {run_id} artifact {position} in {path} is missing "
                    f"{', '.join(missing)}; every field of a declared artifact "
                    f"is required"
                )
            records.append(
                RunArtifact(
                    run_id=str(run_id),
                    kind=str(entry["kind"]),
                    channel=str(entry["channel"]),
                    sample_index=int(entry["sample_index"]),
                    time_ms=float(entry["time_ms"]),
                    peak_amplitude_a=tuple(
                        float(value) for value in entry["peak_amplitude_a"]
                    ),
                    description=str(entry["description"]),
                    detection_rule=str(entry["detection_rule"]),
                )
            )
        if records:
            registry[str(run_id)] = tuple(records)
    return registry


def artifacts_in_window(
    run_ids: Iterable[str],
    window_ms: Sequence[float],
    channel: str | None = None,
    artifacts: Mapping[str, Sequence[RunArtifact]] | None = None,
) -> tuple[RunArtifact, ...]:
    """Declared artifacts of ``run_ids`` whose sample falls inside ``window_ms``.

    ``window_ms`` is an inclusive ``(low, high)`` pair on the same clock as the
    annotations' ``time_ms``.  ``channel`` restricts the answer to one channel;
    ``None`` asks about every channel.  ``artifacts`` overrides the registry,
    which is what tests and offline callers use; the default reads the tracked
    registry.
    """
    if artifacts is None:
        artifacts = load_run_artifacts()
    low, high = float(window_ms[0]), float(window_ms[1])
    if low > high:
        raise ValueError(f"window {window_ms} runs backwards")

    hits: list[RunArtifact] = []
    seen: set[str] = set()
    for run_id in run_ids:
        run_id = str(run_id)
        if run_id in seen:
            continue
        seen.add(run_id)
        for artifact in artifacts.get(run_id, ()):
            if channel is not None and artifact.channel != channel:
                continue
            if low <= artifact.time_ms <= high:
                hits.append(artifact)
    return tuple(hits)


def refuse_artifacts_in_window(
    run_ids: Iterable[str],
    window_ms: Sequence[float],
    statistic: str,
    channel: str | None = None,
    artifacts: Mapping[str, Sequence[RunArtifact]] | None = None,
) -> None:
    """Raise if ``window_ms`` covers a declared artifact of any of ``run_ids``.

    ``statistic`` names what the caller was about to compute, so the message
    says which quantity is refused as well as which annotation refused it.
    Returns ``None`` when the window is clear, which is the ordinary case.
    """
    hits = artifacts_in_window(run_ids, window_ms, channel, artifacts)
    if not hits:
        return
    named = "; ".join(f"{hit.label}: {hit.description}" for hit in hits)
    raise ValueError(
        f"{statistic} would be taken over "
        f"{float(window_ms[0]):g}-{float(window_ms[1]):g} ms, which covers a "
        f"declared run artifact -- {named}. The annotation is in "
        f"{RUN_ARTIFACTS_TOML}; move the window off the artifact, or decide "
        f"explicitly how the statistic treats it, rather than averaging "
        f"across it."
    )
