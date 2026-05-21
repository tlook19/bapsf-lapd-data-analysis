"""Shared Matplotlib label strings."""

from __future__ import annotations

from bapsf_lapd.config import ExperimentSet


V_BANK = r"$V_{\mathrm{bank}}$"
V_PUFF = r"$V_{\mathrm{puff}}$"
I_DIS = r"$I_{\mathrm{dis}}$"
V_DIS = r"$V_{\mathrm{dis}}$"
P_PEAK = r"$P_{\mathrm{peak}}$"


def experiment_title(experiment_set_id: int, experiment: ExperimentSet) -> str:
    return (
        f"Set {experiment_set_id}: "
        rf"{V_BANK}={experiment.v_bank:g} V, {V_PUFF}={experiment.v_puff:g} V"
    )
