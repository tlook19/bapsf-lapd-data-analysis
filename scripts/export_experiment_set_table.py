"""Export experiment-set discharge stats as CSV and LaTeX table rows."""

from __future__ import annotations

import csv
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "processed" / "discharge_experiment_set_summary.toml"
CSV_PATH = ROOT / "processed" / "experiment_set_discharge_table.csv"
TEX_PATH = ROOT / "processed" / "experiment_set_discharge_table.tex"


def fmt(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    with SUMMARY_PATH.open("rb") as f:
        summary = tomllib.load(f)

    experiment_sets = summary["experiment_sets"]
    rows = []
    for set_id in sorted(experiment_sets, key=int):
        item = experiment_sets[set_id]
        rows.append(
            {
                "experiment_set": int(set_id),
                "v_bank_v": item["v_bank"],
                "v_puff_v": item["v_puff"],
                "discharge_current_mean_a": item["peak_current_mean_a"],
                "discharge_current_std_a": item["peak_current_std_a"],
                "discharge_voltage_mean_v": item["discharge_voltage_mean_v"],
                "discharge_voltage_std_v": item["discharge_voltage_std_v"],
                "peak_power_mean_kw": item["peak_power_mean_kw"],
                "peak_power_std_kw": item["peak_power_std_kw"],
            }
        )

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    tex_lines = [
        r"\begin{tabular}{rrrrrrr}",
        r"\hline",
        (
            r"Set & $V_\mathrm{bank}$ (V) & $V_\mathrm{puff}$ (V) & "
            r"$I_\mathrm{dis}$ (A) & $V_\mathrm{dis}$ (V) & "
            r"$P_\mathrm{peak}$ (kW) \\"
        ),
        r"\hline",
    ]
    for row in rows:
        tex_lines.append(
            " & ".join(
                [
                    str(row["experiment_set"]),
                    fmt(row["v_bank_v"]),
                    fmt(row["v_puff_v"]),
                    (
                        rf"{fmt(row['discharge_current_mean_a'], 0)} "
                        rf"$\pm$ {fmt(row['discharge_current_std_a'], 0)}"
                    ),
                    (
                        rf"{fmt(row['discharge_voltage_mean_v'])} "
                        rf"$\pm$ {fmt(row['discharge_voltage_std_v'])}"
                    ),
                    (
                        rf"{fmt(row['peak_power_mean_kw'])} "
                        rf"$\pm$ {fmt(row['peak_power_std_kw'])}"
                    ),
                ]
            )
            + r" \\"
        )
    tex_lines.extend([r"\hline", r"\end{tabular}", ""])
    TEX_PATH.write_text("\n".join(tex_lines))

    sys.stdout.write(f"Wrote {CSV_PATH}\nWrote {TEX_PATH}\n")


if __name__ == "__main__":
    main()
