"""The two ES3 density variants carry the calibration record their sibling does.

``scripts/plot_density_profiles.py`` writes one product per T_e input, and the
probe-A factor record it stamps at the root -- the factor, its bracket bounds,
the uncertainty and how the uncertainty propagates -- is what makes a density
in the file readable as a measurement rather than a number.  The ES3 variants
``processed/density_profiles_isweep_es3_edge_suppressed.hdf5`` and
``processed/density_profiles_isweep_es3_p11_p29_te.hdf5`` are written by the
same code path as ``processed/density_profiles_isweep.hdf5`` and so carry the
same root record; a variant carrying fewer root attrs than its sibling was
built before that record existed and is a stale product, not a variant.

Per run, the same holds for the area record: ``density_area_key`` names which
calibrated electrode area divided that run's current, and ``ap_cm2`` is the
value it resolved to.  Without them the file cannot say which face it read.

Reads the products as placed; skips the ones that are not in the checkout, and
refuses to pass vacuously.
"""

from pathlib import Path

import h5py
import pytest

REFERENCE = Path("processed/density_profiles_isweep.hdf5")
VARIANTS = (
    Path("processed/density_profiles_isweep_es3_edge_suppressed.hdf5"),
    Path("processed/density_profiles_isweep_es3_p11_p29_te.hdf5"),
)
REQUIRED_ROOT_ATTRS = (
    "m_i_amu",
    "source_current_hdf5",
    "source_te_hdf5",
    "source_calibration_toml",
    "probe_a_factor",
    "probe_a_factor_lower_bound",
    "probe_a_factor_upper_bound",
    "probe_a_factor_uncertainty",
    "probe_a_factor_relative_uncertainty",
    "probe_a_factor_uncertainty_definition",
    "probe_a_area_uncertainty_propagation",
)
REQUIRED_RUN_ATTRS = ("density_area_key", "ap_cm2", "ap_L_cm2", "ap_R_cm2", "probe_id")


@pytest.mark.parametrize("path", VARIANTS, ids=lambda p: p.name)
def test_variant_carries_the_full_calibration_record(path):
    """Every root attr the sibling product stamps is on the variant too."""
    if not path.exists():
        pytest.skip(f"{path} is not in this checkout")

    with h5py.File(path, "r") as hdf:
        missing = [name for name in REQUIRED_ROOT_ATTRS if name not in hdf.attrs]
        assert not missing, (
            f"{path} is missing root attrs {missing}; it predates the probe-A "
            "factor record and needs scripts/plot_density_profiles.py re-run"
        )
        runs = hdf["experiment_sets/3/runs"]
        assert len(runs) == 5, f"{path} carries {len(runs)} ES3 runs, expected 5"
        for run_id in runs:
            attrs = runs[run_id].attrs
            missing = [name for name in REQUIRED_RUN_ATTRS if name not in attrs]
            assert not missing, f"{path} run {run_id} is missing {missing}"
            assert str(attrs["density_area_key"]) in ("ap_L_cm2", "ap_R_cm2")
            assert float(attrs["ap_cm2"]) == pytest.approx(
                float(attrs[str(attrs["density_area_key"])])
            )


@pytest.mark.parametrize("path", VARIANTS, ids=lambda p: p.name)
def test_variant_agrees_with_its_sibling_on_the_calibration(path):
    """A variant differs only in its T_e input, never in the area record."""
    if not (path.exists() and REFERENCE.exists()):
        pytest.skip("the ES3 variant or its sibling product is not in this checkout")

    with h5py.File(path, "r") as variant, h5py.File(REFERENCE, "r") as reference:
        if "probe_a_factor" not in variant.attrs:
            pytest.skip(f"{path} predates the probe-A factor record")
        assert variant.attrs["probe_a_factor"] == reference.attrs["probe_a_factor"]
        assert (
            str(variant.attrs["source_calibration_toml"])
            == str(reference.attrs["source_calibration_toml"])
        )
        assert str(variant.attrs["source_te_hdf5"]) != str(
            reference.attrs["source_te_hdf5"]
        ), "the variant must name its own T_e product"

        variant_runs = variant["experiment_sets/3/runs"]
        reference_runs = reference["experiment_sets/3/runs"]
        assert sorted(variant_runs) == sorted(reference_runs)
        for run_id in variant_runs:
            for name in ("density_area_key", "ap_cm2", "probe_id"):
                assert str(variant_runs[run_id].attrs[name]) == str(
                    reference_runs[run_id].attrs[name]
                ), f"{path} run {run_id}: {name} disagrees with the sibling product"
