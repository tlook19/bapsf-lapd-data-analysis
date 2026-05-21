"""Shared helpers for exporting Matplotlib figures."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


RASTER_FORMATS = {"png"}
VECTOR_FORMATS = {"pdf", "svg"}
SUPPORTED_FORMATS = RASTER_FORMATS | VECTOR_FORMATS


def configure_latex_friendly_matplotlib() -> None:
    """Use editable text in vector exports where Matplotlib supports it."""
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
        }
    )


def parse_formats(value: str | Iterable[str]) -> tuple[str, ...]:
    """Parse a CLI format list into normalized output formats."""
    if isinstance(value, str):
        requested = [item.strip().lower() for item in value.split(",")]
    else:
        requested = [item.strip().lower() for item in value]

    formats: list[str] = []
    for item in requested:
        if not item:
            continue
        if item == "latex":
            formats.extend(["pdf", "svg"])
        elif item == "vector":
            formats.extend(["pdf", "svg"])
        elif item == "all":
            formats.extend(["png", "pdf", "svg"])
        elif item in SUPPORTED_FORMATS:
            formats.append(item)
        else:
            supported = ", ".join(sorted(SUPPORTED_FORMATS | {"all", "latex", "vector"}))
            raise ValueError(f"Unsupported plot format {item!r}; choose from {supported}")

    if not formats:
        raise ValueError("At least one plot format is required")

    return tuple(dict.fromkeys(formats))


def save_figure(fig: plt.Figure, output: Path, formats: Iterable[str], *, dpi: int = 180) -> list[Path]:
    """Save a figure next to output using each requested format."""
    output.parent.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for fmt in parse_formats(formats):
        path = output.with_suffix(f".{fmt}")
        kwargs = {"format": fmt}
        if fmt in RASTER_FORMATS:
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        saved_paths.append(path)
        print(path)
    return saved_paths
