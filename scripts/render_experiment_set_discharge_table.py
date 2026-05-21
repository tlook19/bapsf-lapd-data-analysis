"""Render the experiment-set discharge summary as an image."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from plot_labels import I_DIS, P_PEAK, V_BANK, V_DIS, V_PUFF


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "processed" / "experiment_set_discharge_table.csv"
OUTPUT = ROOT / "figures" / "experiment_set_discharge_table.png"


def fmt_pm(mean: float, std: float, digits: int = 1) -> str:
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def main() -> None:
    with CSV_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))

    headers = [
        "Set",
        rf"{V_BANK} (V)",
        rf"{V_PUFF} (V)",
        rf"{I_DIS} (A)",
        rf"{V_DIS} (V)",
        rf"{P_PEAK} (kW)",
    ]
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row["experiment_set"],
                f"{float(row['v_bank_v']):.0f}",
                f"{float(row['v_puff_v']):.1f}",
                fmt_pm(
                    float(row["discharge_current_mean_a"]),
                    float(row["discharge_current_std_a"]),
                    digits=0,
                ),
                fmt_pm(
                    float(row["discharge_voltage_mean_v"]),
                    float(row["discharge_voltage_std_v"]),
                ),
                fmt_pm(
                    float(row["peak_power_mean_kw"]),
                    float(row["peak_power_std_kw"]),
                ),
            ]
        )

    fig, ax = plt.subplots(figsize=(11.5, 3.2), constrained_layout=True)
    ax.axis("off")
    ax.set_title("Experiment-Set Discharge Conditions", fontsize=18, pad=16, weight="bold")

    table = ax.table(
        cellText=table_rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.08, 0.14, 0.14, 0.22, 0.19, 0.22],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.8)

    for (row_index, _), cell in table.get_celld().items():
        cell.set_edgecolor("#4b5563")
        cell.set_linewidth(0.8)
        if row_index == 0:
            cell.set_facecolor("#dbeafe")
            cell.set_text_props(weight="bold", color="#111827")
        else:
            cell.set_facecolor("#f9fafb" if row_index % 2 else "#ffffff")
            cell.set_text_props(color="#111827")

    fig.text(
        0.5,
        0.035,
        "Values are mean +/- one standard deviation across runs; peak metrics use transient-rejected discharge-power peaks.",
        ha="center",
        fontsize=9.5,
        color="#4b5563",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    print(OUTPUT)


if __name__ == "__main__":
    main()
