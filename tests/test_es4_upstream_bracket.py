"""The ES4 upstream-face bracket the overlay exporter adds at experiment set 4.

The rows themselves are checked against SYNTHETIC dead-time products written
here, so the arithmetic is closed-form: two faces carrying the same current
must differ by exactly their area ratio, and a row moved to a different T_e
must differ by exactly sqrt(T_e_old / T_e_new).  Nothing reads a placed
product, so these run in a checkout with nothing in ``processed/``.

Three things beyond the arithmetic are pinned here:

* the SCHEMA the exporter writes and the augmenter's acceptance of it, which
  is what tells a consumer which families a file carries;
* that the family is ADDITIVE -- present at set 4 and absent everywhere else,
  and never overwriting a name the product already uses;
* that every run-selecting command-line flag reaches its own parameter of
  ``export_overlay``, alongside ``--output``.  That last one is not
  ceremony: the exporter is called positionally, so a flag filed into the
  neighbouring slot would read the wrong product and still produce a file.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.augment_sim1d_overlay_isat_drive import AUGMENTED_SCHEMA
from scripts.export_es1_sim1d_overlay import (
    AREA_CALIBRATION_TOML,
    ES4_UPSTREAM_BRACKET_PORTS,
    ES4_UPSTREAM_F_APPLIED,
    ES4_UPSTREAM_P21_TE_MEASURED_EV,
    ES4_UPSTREAM_P21_TE_MEASURED_WINDOW_MS,
    ES4_UPSTREAM_SET_ID,
    ES4_UPSTREAM_TE_REDERIVED_PORT,
    PORTS,
    RAW_PLATEAU_WINDOW_MS,
    ROT180_ISAT_PROFILE_HDF5,
    _es4_upstream_definitions,
    _es4_upstream_rows,
)
from scripts.es4_upstream_rows_rot180_isat import PORT_RUNS

X_CM = np.linspace(-25.0, 25.0, 51)
T_MS = np.arange(0.75, 20.0, 1.0)
PLATEAU = (T_MS >= RAW_PLATEAU_WINDOW_MS[0]) & (T_MS <= RAW_PLATEAU_WINDOW_MS[1])
#: probe id -> (ap_L_cm2, ap_R_cm2); deliberately unequal so an area swap shows.
AREAS = {"B": (0.05, 0.055), "C": (0.04, 0.052), "D": (0.06, 0.045)}
PROBE_OF_PORT = {21: "B", 29: "C", 41: "D"}


def _write_area_toml(path: Path) -> Path:
    lines = []
    for probe, (left, right) in AREAS.items():
        lines += [f"[probe_{probe}]", f"ap_L_cm2 = {left}", f"ap_R_cm2 = {right}", ""]
    path.write_text("\n".join(lines))
    return path


def _write_face(path: Path, currents: dict[str, np.ndarray], *, masked=()) -> Path:
    """A minimal dead-time line-scan product: one group per run, set 4."""
    with h5py.File(path, "w") as hdf:
        hdf.create_dataset("x_cm", data=X_CM)
        for run_id, current in currents.items():
            group = hdf.create_group(f"experiment_sets/4/{run_id}")
            group.create_dataset("inter_sweep_time_s", data=T_MS * 1e-3)
            group.create_dataset("isat_a", data=current)
            group.create_dataset("isat_a_std", data=np.full_like(current, 0.01))
            group.create_dataset("n_shots_used", data=np.full_like(current, 20.0))
            group.create_dataset(
                "rail_mask", data=np.zeros(current.shape, dtype=bool)
            )
            state = np.zeros(current.shape, dtype=bool)
            if run_id in masked:
                state[:] = True
            group.create_dataset("state_mask", data=state)
            group.attrs["probe_a_area_factor_applied"] = 1.0
            group.attrs["density_area_key"] = "ap_R_cm2"
    return path


def _flat_current(level: float) -> np.ndarray:
    return np.full((X_CM.size, T_MS.size), level, dtype=np.float64)


def _build(tmp_path, *, isat_level=0.02, isweep_level=0.02, te_ev=2.0, masked=()):
    """Build the family from synthetic products, both faces flat in x and t."""
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    isweep = _write_face(
        tmp_path / "isweep.hdf5",
        {PORT_RUNS[port][0]: _flat_current(isweep_level) for port in PORT_RUNS},
    )
    isat = _write_face(
        tmp_path / "isat_rot180.hdf5",
        {PORT_RUNS[port][1]: _flat_current(isat_level) for port in PORT_RUNS},
        masked=masked,
    )
    areas = _write_area_toml(tmp_path / "areas.toml")
    n_ports, n_time = len(PORTS), T_MS.size
    return _es4_upstream_rows(
        te_core_mean_ev=np.full((n_ports, n_time), te_ev),
        te_time_ms=T_MS,
        density_mean_cm3=np.full((n_ports, n_time), 1.0e13),
        density_ftavg_cm3=np.full((n_ports, n_time), 8.0e12),
        density_core_count=np.full((n_ports, n_time), 21, dtype=np.int64),
        density_time_ms=T_MS,
        isweep_path=isweep,
        isat_rot180_path=isat,
        area_toml_path=areas,
    )


def _row(fields, key):
    return int(np.flatnonzero(fields["es4_upstream_row_key"] == key)[0])


# ---------------------------------------------------------------- schema ---

def test_the_exported_schema_is_odd_and_the_augmenter_accepts_it():
    """Odd is an exporter version, even is an augmented one; both must hold."""
    from scripts.export_es1_sim1d_overlay import export_overlay  # noqa: F401
    import inspect

    source = inspect.getsource(
        __import__("scripts.export_es1_sim1d_overlay", fromlist=["x"])
    )
    schema = int(
        source.split("schema_version=np.array(")[1].split(",")[0]
    )
    assert schema % 2 == 1, "an exporter schema is odd; even means augmented"
    assert schema in AUGMENTED_SCHEMA, (
        f"the augmenter does not accept schema v{schema}, so an export at this "
        "version cannot be augmented"
    )
    assert AUGMENTED_SCHEMA[schema] == schema + 1
    assert AUGMENTED_SCHEMA[schema] not in AUGMENTED_SCHEMA, (
        "an augmented version must not itself be an accepted input"
    )


def test_no_augmented_schema_is_ever_an_accepted_input():
    assert not set(AUGMENTED_SCHEMA.values()) & set(AUGMENTED_SCHEMA)


# ------------------------------------------------------------ the bracket ---

def test_the_two_faces_differ_only_by_their_electrode_areas(tmp_path):
    """Equal currents on the two faces give exactly the inverse area ratio."""
    fields = _build(tmp_path, isat_level=0.02, isweep_level=0.02)
    for port in ES4_UPSTREAM_BRACKET_PORTS:
        left, right = AREAS[PROBE_OF_PORT[port]]
        primary = fields["es4_upstream_density_mean_cm3"][
            _row(fields, f"p{port}_rot180_isat")
        ]
        partner = fields["es4_upstream_density_mean_cm3"][
            _row(fields, f"p{port}_rot0_isweep")
        ]
        assert np.allclose(primary / partner, left / right)


def test_the_isat_row_takes_the_right_electrode_and_its_partner_the_left(tmp_path):
    fields = _build(tmp_path)
    for port in ES4_UPSTREAM_BRACKET_PORTS:
        left, right = AREAS[PROBE_OF_PORT[port]]
        isat = _row(fields, f"p{port}_rot180_isat")
        isweep = _row(fields, f"p{port}_rot0_isweep")
        assert fields["es4_upstream_row_area_key"][isat] == "ap_R_cm2"
        assert fields["es4_upstream_row_area_key"][isweep] == "ap_L_cm2"
        assert fields["es4_upstream_row_area_cm2"][isat] == pytest.approx(right)
        assert fields["es4_upstream_row_area_cm2"][isweep] == pytest.approx(left)


def test_each_bracket_row_names_its_partner_and_they_name_each_other(tmp_path):
    fields = _build(tmp_path)
    keys = list(fields["es4_upstream_row_key"])
    partners = list(fields["es4_upstream_row_bracket_partner"])
    for key, partner in zip(keys, partners):
        assert partner in keys
        assert partners[keys.index(partner)] == key


def test_the_rot180_row_is_the_primary_and_the_rot0_row_the_partner(tmp_path):
    fields = _build(tmp_path)
    for port in ES4_UPSTREAM_BRACKET_PORTS:
        isat = _row(fields, f"p{port}_rot180_isat")
        isweep = _row(fields, f"p{port}_rot0_isweep")
        assert fields["es4_upstream_row_role"][isat] == "primary"
        assert fields["es4_upstream_row_face"][isat] == "rot180_isat"
        assert fields["es4_upstream_row_run_id"][isat] == PORT_RUNS[port][1]
        assert fields["es4_upstream_row_role"][isweep] == "bracket_partner"
        assert fields["es4_upstream_row_face"][isweep] == "rot0_isweep"
        assert fields["es4_upstream_row_run_id"][isweep] == PORT_RUNS[port][0]


def test_a_masked_rot180_run_removes_its_cells_from_both_faces(tmp_path):
    """The admission mask is the ISAT product's, and it is shared by the pair."""
    masked = tuple(PORT_RUNS[port][1] for port in ES4_UPSTREAM_BRACKET_PORTS)
    fields = _build(tmp_path, masked=masked)
    for port in ES4_UPSTREAM_BRACKET_PORTS:
        for key in (f"p{port}_rot180_isat", f"p{port}_rot0_isweep"):
            index = _row(fields, key)
            assert np.all(fields["es4_upstream_density_core_count"][index] == 0)
            assert np.all(
                np.isnan(fields["es4_upstream_density_mean_cm3"][index])
            )


def test_no_bracket_row_claims_a_measured_temperature(tmp_path):
    """The face bracket moves the FACE; the T_e stays the port's prior."""
    fields = _build(tmp_path)
    for port in ES4_UPSTREAM_BRACKET_PORTS:
        for key in (f"p{port}_rot180_isat", f"p{port}_rot0_isweep"):
            index = _row(fields, key)
            assert not fields["es4_upstream_row_te_measured"][index]
            assert np.isnan(fields["es4_upstream_row_te_measured_ev"][index])
            assert fields["es4_upstream_row_te_face"][index] == ""


def test_the_bracket_covers_only_the_ports_with_a_rot180_partner(tmp_path):
    fields = _build(tmp_path)
    bracket = {
        int(port)
        for port, chain in zip(
            fields["es4_upstream_row_port"], fields["es4_upstream_row_chain"]
        )
        if chain == "deadtime_face_pair"
    }
    assert bracket == set(ES4_UPSTREAM_BRACKET_PORTS)
    assert 11 not in bracket and 50 not in bracket
    assert ES4_UPSTREAM_TE_REDERIVED_PORT not in bracket


# ------------------------------------------------- the p21 re-derivation ---

def test_the_p21_primary_is_the_prior_row_rescaled_by_the_sqrt_of_the_te_ratio(
    tmp_path,
):
    te_prior = 2.15
    fields = _build(tmp_path, te_ev=te_prior)
    measured = fields["es4_upstream_density_mean_cm3"][
        _row(fields, f"p{ES4_UPSTREAM_TE_REDERIVED_PORT}_rot0_isweep_te_measured")
    ]
    prior = fields["es4_upstream_density_mean_cm3"][
        _row(fields, f"p{ES4_UPSTREAM_TE_REDERIVED_PORT}_rot0_isweep_te_prior")
    ]
    assert np.allclose(
        measured / prior, np.sqrt(te_prior / ES4_UPSTREAM_P21_TE_MEASURED_EV)
    )


def test_the_retained_p21_row_is_the_untouched_export_row(tmp_path):
    fields = _build(tmp_path)
    index = _row(
        fields, f"p{ES4_UPSTREAM_TE_REDERIVED_PORT}_rot0_isweep_te_prior"
    )
    assert np.allclose(fields["es4_upstream_density_mean_cm3"][index], 1.0e13)
    assert np.allclose(fields["es4_upstream_density_ftavg_cm3"][index], 8.0e12)
    assert fields["es4_upstream_row_role"][index] == "retained_prior_derived"
    assert not fields["es4_upstream_row_te_measured"][index]


def test_the_p21_primary_stamps_the_value_face_window_and_estimator(tmp_path):
    fields = _build(tmp_path)
    index = _row(
        fields, f"p{ES4_UPSTREAM_TE_REDERIVED_PORT}_rot0_isweep_te_measured"
    )
    assert fields["es4_upstream_row_te_measured"][index]
    assert fields["es4_upstream_row_te_measured_ev"][index] == pytest.approx(
        ES4_UPSTREAM_P21_TE_MEASURED_EV
    )
    assert np.allclose(
        fields["es4_upstream_row_te_window_ms"][index],
        ES4_UPSTREAM_P21_TE_MEASURED_WINDOW_MS,
    )
    assert "run 42" in str(fields["es4_upstream_row_te_face"][index])
    assert str(fields["es4_upstream_row_te_basis"][index]).strip() != ""
    assert np.allclose(
        fields["es4_upstream_te_ev"][index], ES4_UPSTREAM_P21_TE_MEASURED_EV
    )


def test_the_measured_te_is_quoted_at_the_scoring_plateau_window():
    """A prior and the measurement replacing it must share their window."""
    assert tuple(ES4_UPSTREAM_P21_TE_MEASURED_WINDOW_MS) == tuple(
        RAW_PLATEAU_WINDOW_MS
    )


def test_the_p21_rows_come_from_the_export_density_chain_not_the_face_pair(
    tmp_path,
):
    fields = _build(tmp_path)
    for suffix in ("te_measured", "te_prior"):
        index = _row(
            fields, f"p{ES4_UPSTREAM_TE_REDERIVED_PORT}_rot0_isweep_{suffix}"
        )
        assert fields["es4_upstream_row_chain"][index] == "overlay_density_isweep"
        assert fields["es4_upstream_row_face"][index] == "rot0_isweep"


# ------------------------------------------------------------ provenance ---

def test_every_row_stamps_the_rest_bias_factor_it_was_built_with(tmp_path):
    fields = _build(tmp_path)
    assert np.allclose(
        fields["es4_upstream_row_f_applied"], ES4_UPSTREAM_F_APPLIED
    )
    assert ES4_UPSTREAM_F_APPLIED == 1.0


def test_every_row_array_shares_the_row_axis_and_the_time_axis(tmp_path):
    fields = _build(tmp_path)
    n_rows = fields["es4_upstream_row_key"].size
    for name, value in fields.items():
        if name.startswith("es4_upstream_row_"):
            assert value.shape[0] == n_rows, name
        if name.startswith("es4_upstream_density_") or name == "es4_upstream_te_ev":
            assert value.shape == (n_rows, T_MS.size), name
    assert np.allclose(fields["es4_upstream_time_ms"], T_MS)


def test_the_family_names_the_three_products_it_read(tmp_path):
    fields = _build(tmp_path)
    for name in (
        "es4_upstream_source_file_rot180_isat",
        "es4_upstream_source_file_rot0_isweep",
        "es4_upstream_source_file_area_calibration",
    ):
        assert str(fields[name]) != ""


def test_the_definitions_name_every_row_field_they_describe():
    definitions = " ".join(str(v) for v in _es4_upstream_definitions().values())
    for name in (
        "es4_upstream_row_key",
        "es4_upstream_row_port",
        "es4_upstream_row_role",
        "es4_upstream_row_face",
        "es4_upstream_row_chain",
        "es4_upstream_row_area_key",
        "es4_upstream_row_bracket_partner",
        "es4_upstream_row_te_measured",
        "es4_upstream_row_f_applied",
        "es4_upstream_density_mean_cm3",
        "es4_upstream_density_ftavg_cm3",
        "es4_upstream_time_ms",
    ):
        assert name in definitions, name


def test_the_family_never_shadows_a_name_the_product_already_uses(tmp_path):
    """Additive means additive: no key of this family may be an existing one."""
    fields = _build(tmp_path)
    fields.update(_es4_upstream_definitions())
    existing = {
        "density_mean_cm3",
        "density_ftavg_cm3",
        "density_total_sem_cm3",
        "density_core_count",
        "density_time_ms",
        "te_mean_ev",
        "te_row_measured",
        "te_row_measured_cells",
        "schema_version",
        "port",
        "z_cm",
    }
    assert not (set(fields) & existing)
    assert all(name.startswith("es4_upstream_") for name in fields)


# ------------------------------------------------------- the CLI plumbing ---

RUN_SELECTING_FLAGS = {
    "--density": "density_path",
    "--te-filled": "te_path",
    "--isat-profiles": "isat_profile_path",
    "--rot0-isat-profiles": "rot0_isat_profile_path",
    "--rot180-isat-profiles": "rot180_isat_profile_path",
    "--area-calibration": "area_calibration_path",
    "--zero-offsets": "zero_offsets_path",
    "--window-refits": "window_refits_path",
    "--manifest": "manifest_path",
}


def _captured_call(monkeypatch, argv):
    import inspect

    import scripts.export_es1_sim1d_overlay as exporter

    captured = {}
    signature = inspect.signature(exporter.export_overlay)

    def fake_export(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        captured.update(bound.arguments)
        return bound.arguments["output_path"]

    monkeypatch.setattr(exporter, "export_overlay", fake_export)
    monkeypatch.setattr("sys.argv", ["export_es1_sim1d_overlay.py", *argv])
    exporter.main()
    return captured


@pytest.mark.parametrize("flag,parameter", sorted(RUN_SELECTING_FLAGS.items()))
def test_every_run_selecting_flag_reaches_its_own_parameter(
    monkeypatch, tmp_path, flag, parameter
):
    """A flag filed into the neighbouring positional slot reads the wrong product."""
    sentinel = tmp_path / f"{parameter}.marker"
    argv = [
        "--experiment-set", "4",
        "--output", str(tmp_path / "out.npz"),
        flag, str(sentinel),
    ]
    captured = _captured_call(monkeypatch, argv)
    assert captured[parameter] == sentinel
    others = {
        other
        for other in RUN_SELECTING_FLAGS.values()
        if other != parameter
    }
    assert all(captured[other] != sentinel for other in others)


def test_the_output_flag_is_where_the_product_is_written(monkeypatch, tmp_path):
    target = tmp_path / "somewhere" / "es4.npz"
    captured = _captured_call(
        monkeypatch, ["--experiment-set", "4", "--output", str(target)]
    )
    assert captured["output_path"] == target
    assert captured["experiment_set_id"] == ES4_UPSTREAM_SET_ID


def test_without_output_the_product_goes_to_the_selected_sets_own_name(
    monkeypatch, tmp_path
):
    captured = _captured_call(monkeypatch, ["--experiment-set", "4"])
    assert captured["output_path"] == Path("processed/es4_sim1d_overlay.npz")


def test_the_run_selecting_defaults_are_the_placed_products(monkeypatch, tmp_path):
    captured = _captured_call(
        monkeypatch, ["--experiment-set", "4", "--output", str(tmp_path / "o.npz")]
    )
    assert captured["rot180_isat_profile_path"] == ROT180_ISAT_PROFILE_HDF5
    assert captured["area_calibration_path"] == AREA_CALIBRATION_TOML


def test_the_es4_route_reads_the_rot180_product_it_was_given(tmp_path):
    """Point the flag at a different product and the primary row must move."""
    low = _build(tmp_path / "low", isat_level=0.02)
    high = _build(tmp_path / "high", isat_level=0.04)
    port = ES4_UPSTREAM_BRACKET_PORTS[0]
    index = _row(low, f"p{port}_rot180_isat")
    assert np.allclose(
        high["es4_upstream_density_mean_cm3"][index]
        / low["es4_upstream_density_mean_cm3"][index],
        2.0,
    )
    partner = _row(low, f"p{port}_rot0_isweep")
    assert np.allclose(
        high["es4_upstream_density_mean_cm3"][partner],
        low["es4_upstream_density_mean_cm3"][partner],
    )


def test_the_es4_route_reads_the_area_calibration_it_was_given(tmp_path):
    fields = _build(tmp_path)
    port = ES4_UPSTREAM_BRACKET_PORTS[0]
    index = _row(fields, f"p{port}_rot180_isat")
    assert fields["es4_upstream_row_area_cm2"][index] == pytest.approx(
        AREAS[PROBE_OF_PORT[port]][1]
    )
