"""Rail-mask and channel-state-mask exclusion in the Mach/velocity computation."""

import h5py
import numpy as np
import pytest

from compute_mach_velocity import (
    MACH_K,
    RAIL_MASK_DATASET,
    STATE_MASK_DATASET,
    X_CM,
    _compute_run,
    _face_rail_mask,
    _face_state_mask,
    _processed_source_attr,
    _rail_rule,
    _state_rule,
)


N_X = X_CM.size
N_WINDOWS = 4
Z_CM = 789.55
RULE_TEXT = "cell excluded when it contains any railed raw sample"
STATE_RULE_TEXT = "cell excluded when its position lies in the state shot range"


def _write_face(group, current_a, *, rail_mask=None, rule=None,
                state_mask=None, state_rule=None, state_factor=None):
    """One probe face of a dead-time profile product."""
    group.create_dataset("isat_a_raw", data=current_a)
    group.create_dataset("isat_a_raw_std", data=np.full_like(current_a, 1e-4))
    group.create_dataset(
        "inter_sweep_time_s", data=np.linspace(1e-3, 4e-3, N_WINDOWS)
    )
    group.attrs["run_id"] = "43"
    group.attrs["z_cm"] = Z_CM
    if rail_mask is not None:
        group.create_dataset(RAIL_MASK_DATASET, data=rail_mask)
    if rule is not None:
        group.file.attrs["rail_mask_rule"] = rule
    if state_mask is not None:
        group.create_dataset(STATE_MASK_DATASET, data=state_mask)
    if state_rule is not None:
        group.file.attrs["state_mask_rule"] = state_rule
    if state_factor is not None:
        group.attrs["state_factor"] = state_factor


def _make_inputs(tmp_path, *, upstream_mask=None, downstream_mask=None, rule=None,
                 upstream_state=None, downstream_state=None, state_rule=None,
                 state_factor=None):
    """A minimal upstream/downstream pair plus the filled-T_e grid they need."""
    rng = np.random.default_rng(20260904)
    upstream_a = 0.05 + 0.01 * rng.random((N_X, N_WINDOWS))
    downstream_a = 0.02 + 0.01 * rng.random((N_X, N_WINDOWS))

    up_path = tmp_path / "upstream.hdf5"
    down_path = tmp_path / "downstream.hdf5"
    te_path = tmp_path / "te.hdf5"

    with h5py.File(up_path, "w") as f:
        _write_face(f.create_group("run"), upstream_a, rail_mask=upstream_mask,
                    rule=rule, state_mask=upstream_state, state_rule=state_rule,
                    state_factor=state_factor)
    with h5py.File(down_path, "w") as f:
        _write_face(f.create_group("run"), downstream_a, rail_mask=downstream_mask,
                    state_mask=downstream_state)
    with h5py.File(te_path, "w") as f:
        grp = f.create_group("experiment_sets/1")
        grp.create_dataset("x_cm", data=X_CM)
        grp.create_dataset("z_cm", data=np.array([Z_CM]))
        grp.create_dataset("cycle_time_ms", data=np.linspace(0.0, 20.0, 8))
        grp.create_dataset("te_filled", data=np.full((1, N_X, 8), 4.0))

    return up_path, down_path, te_path


def _run(tmp_path, **kwargs):
    up_path, down_path, te_path = _make_inputs(tmp_path, **kwargs)
    with (
        h5py.File(up_path, "r") as up,
        h5py.File(down_path, "r") as down,
        h5py.File(te_path, "r") as te,
    ):
        return _compute_run(
            up["run"],
            down["run"],
            te,
            "1",
            upstream_area_m2=2.0e-6,
            downstream_area_m2=1.0e-6,
            current_factor=1.0,
        )


MASKED_DATASETS = (
    "mach",
    "mach_std",
    "velocity_km_s",
    "velocity_km_s_std",
    "upstream_current_density_a_m2",
    "downstream_current_density_a_m2",
)


def test_source_file_attr_is_the_repo_relative_processed_basename(tmp_path):
    """A source opened from OUTSIDE processed/ still stamps processed/<name>.

    A source product regenerated in a throwaway worktree or an artifacts
    directory opens under some transient absolute path; the stamped attr must
    name the placed product's own convention, not wherever this invocation
    happened to read it from.
    """
    outside = tmp_path / "some" / "other" / "place" / "isweep_deadtime_profiles.hdf5"
    outside.parent.mkdir(parents=True)
    with h5py.File(outside, "w") as f:
        f.create_dataset("x", data=[1])
    with h5py.File(outside, "r") as f:
        assert _processed_source_attr(f) == "processed/isweep_deadtime_profiles.hdf5"

    # Opened by its ordinary repo-relative path, the result is unchanged.
    import os
    old_cwd = os.getcwd()
    try:
        os.chdir(outside.parent)
        with h5py.File("isweep_deadtime_profiles.hdf5", "r") as f:
            assert _processed_source_attr(f) == "processed/isweep_deadtime_profiles.hdf5"
    finally:
        os.chdir(old_cwd)


def test_no_mask_excludes_nothing_and_is_recorded(tmp_path):
    results, prov = _run(tmp_path)

    assert prov["rail_cells_excluded"] == 0
    assert prov["rail_cells_kept"] == N_X * N_WINDOWS
    assert prov["rail_mask_upstream_present"] is False
    assert prov["rail_mask_downstream_present"] is False
    # A product predating the marking is recorded as unmasked, not as clean.
    assert "predates the rail-mask marking" in prov["rail_exclusion_rule"]
    assert not results["rail_excluded_cells"].any()
    for name in MASKED_DATASETS:
        assert np.isfinite(results[name]).all()


def test_upstream_mask_drops_those_cells_on_both_faces(tmp_path):
    mask = np.zeros((N_X, N_WINDOWS), dtype=bool)
    mask[10, 1] = True
    mask[25, :2] = True

    masked = _run(tmp_path, upstream_mask=mask, rule=RULE_TEXT)[0]
    unmasked = _run(tmp_path)[0]

    for name in MASKED_DATASETS:
        assert np.isnan(masked[name][mask]).all(), name
        # Both faces lose the cell, so a survivor is still a true pair; and the
        # surviving cells are untouched, bit for bit.
        assert np.array_equal(masked[name][~mask], unmasked[name][~mask]), name

    assert np.array_equal(masked["rail_excluded_cells"], mask)


def test_downstream_mask_also_excludes(tmp_path):
    mask = np.zeros((N_X, N_WINDOWS), dtype=bool)
    mask[3, 2] = True

    results, prov = _run(tmp_path, downstream_mask=mask)

    assert prov["rail_mask_upstream_present"] is False
    assert prov["rail_mask_downstream_present"] is True
    assert prov["rail_cells_excluded"] == 1
    assert np.isnan(results["mach"][3, 2])


def test_masks_from_both_faces_are_unioned(tmp_path):
    up_mask = np.zeros((N_X, N_WINDOWS), dtype=bool)
    up_mask[5, 0] = True
    down_mask = np.zeros((N_X, N_WINDOWS), dtype=bool)
    down_mask[7, 3] = True

    results, prov = _run(tmp_path, upstream_mask=up_mask, downstream_mask=down_mask)

    assert prov["rail_cells_excluded"] == 2
    assert prov["rail_cells_excluded_upstream"] == 1
    assert prov["rail_cells_excluded_downstream"] == 1
    assert np.isnan(results["mach"][5, 0])
    assert np.isnan(results["mach"][7, 3])
    assert np.isfinite(results["mach"][5, 3])


def test_rule_is_read_from_the_product_that_carries_the_mask(tmp_path):
    mask = np.zeros((N_X, N_WINDOWS), dtype=bool)
    mask[0, 0] = True
    _, prov = _run(tmp_path, upstream_mask=mask, rule=RULE_TEXT)
    assert prov["rail_exclusion_rule"] == RULE_TEXT


def test_unmasked_mach_still_matches_the_closed_form(tmp_path):
    results, _ = _run(tmp_path)
    expected = np.log(
        results["upstream_current_density_a_m2"]
        / results["downstream_current_density_a_m2"]
    ) / MACH_K
    assert np.allclose(results["mach"], expected, rtol=0, atol=0)


def test_wrong_shaped_mask_is_refused(tmp_path):
    up_path, _, _ = _make_inputs(
        tmp_path, upstream_mask=np.zeros((N_X, N_WINDOWS), dtype=bool)
    )
    with h5py.File(up_path, "r") as f:
        with pytest.raises(ValueError, match="per \\(position, dead-time window\\)"):
            _face_rail_mask(f["run"], (N_X, N_WINDOWS + 1))


def test_rule_falls_back_when_no_product_states_one(tmp_path):
    up_path, down_path, _ = _make_inputs(tmp_path)
    with h5py.File(up_path, "r") as up, h5py.File(down_path, "r") as down:
        assert "predates the rail-mask marking" in _rail_rule(up["run"], down["run"])


def test_no_state_mask_excludes_nothing_and_is_recorded(tmp_path):
    results, prov = _run(tmp_path)

    assert prov["state_cells_excluded"] == 0
    assert prov["state_mask_upstream_present"] is False
    assert prov["state_mask_downstream_present"] is False
    # Absence records that no state was registered, not that the run is clean.
    assert "no channel state is registered" in prov["state_exclusion_rule"]
    assert prov["state_factors_disclosed_not_applied"] == ""
    assert not results["state_excluded_cells"].any()


def test_state_mask_drops_those_cells_on_both_faces(tmp_path):
    mask = np.zeros((N_X, N_WINDOWS), dtype=bool)
    mask[12:35, :] = True

    masked, prov = _run(tmp_path, upstream_state=mask, state_rule=STATE_RULE_TEXT)
    unmasked = _run(tmp_path)[0]

    for name in MASKED_DATASETS:
        assert np.isnan(masked[name][mask]).all(), name
        assert np.array_equal(masked[name][~mask], unmasked[name][~mask]), name

    assert np.array_equal(masked["state_excluded_cells"], mask)
    assert prov["state_cells_excluded"] == int(mask.sum())
    assert prov["state_cells_excluded_upstream"] == int(mask.sum())
    assert prov["state_cells_excluded_downstream"] == 0
    assert prov["state_exclusion_rule"] == STATE_RULE_TEXT
    # The rail mask is a separate book and stays empty here.
    assert prov["rail_cells_excluded"] == 0


def test_rail_and_state_masks_stack(tmp_path):
    rail = np.zeros((N_X, N_WINDOWS), dtype=bool)
    rail[20, 1] = True
    state = np.zeros((N_X, N_WINDOWS), dtype=bool)
    state[12:35, :] = True
    assert state[rail].all(), "the rail cell must lie inside the state range"

    results, prov = _run(tmp_path, upstream_mask=rail, upstream_state=state)

    assert prov["rail_cells_excluded"] == 1
    assert prov["state_cells_excluded"] == int(state.sum())
    assert prov["cells_excluded_any_mask"] == int((rail | state).sum())
    assert prov["cells_kept_any_mask"] == int((~(rail | state)).sum())
    assert np.array_equal(results["excluded_cells"], rail | state)
    assert np.isnan(results["mach"][rail | state]).all()


def test_state_mask_on_the_downstream_face_also_excludes(tmp_path):
    mask = np.zeros((N_X, N_WINDOWS), dtype=bool)
    mask[4, 2] = True

    results, prov = _run(tmp_path, downstream_state=mask)

    assert prov["state_mask_upstream_present"] is False
    assert prov["state_mask_downstream_present"] is True
    assert prov["state_cells_excluded"] == 1
    assert np.isnan(results["mach"][4, 2])


def test_state_factor_is_disclosed_and_not_applied(tmp_path):
    mask = np.zeros((N_X, N_WINDOWS), dtype=bool)
    mask[12:35, :] = True

    masked, prov = _run(tmp_path, upstream_state=mask, state_factor=1.76)
    unmasked = _run(tmp_path)[0]

    assert prov["state_factors_disclosed_not_applied"] == "43:1.76"
    # Disclosure only: no surviving value is scaled by the factor.
    assert np.array_equal(
        masked["isat_upstream_a"], unmasked["isat_upstream_a"]
    )
    assert np.array_equal(
        masked["mach"][~mask], unmasked["mach"][~mask]
    )


def test_wrong_shaped_state_mask_is_refused(tmp_path):
    up_path, _, _ = _make_inputs(
        tmp_path, upstream_state=np.zeros((N_X, N_WINDOWS), dtype=bool)
    )
    with h5py.File(up_path, "r") as f:
        with pytest.raises(ValueError, match="per \\(position, dead-time window\\)"):
            _face_state_mask(f["run"], (N_X, N_WINDOWS + 1))


def test_state_rule_falls_back_when_no_product_states_one(tmp_path):
    up_path, down_path, _ = _make_inputs(tmp_path)
    with h5py.File(up_path, "r") as up, h5py.File(down_path, "r") as down:
        assert "no channel state is registered" in _state_rule(up["run"], down["run"])


def test_state_rule_is_the_absence_note_when_neither_face_carries_a_mask(tmp_path):
    # The product states a rule because ANOTHER run in it has a registered
    # state; a pair with no mask of its own is not subject to that rule.
    up_path, down_path, _ = _make_inputs(tmp_path, state_rule=STATE_RULE_TEXT)
    with h5py.File(up_path, "r") as up, h5py.File(down_path, "r") as down:
        assert "no channel state is registered" in _state_rule(up["run"], down["run"])

    _, prov = _run(tmp_path, state_rule=STATE_RULE_TEXT)
    assert prov["state_mask_upstream_present"] is False
    assert prov["state_mask_downstream_present"] is False
    assert "no channel state is registered" in prov["state_exclusion_rule"]


@pytest.mark.parametrize("face", ["upstream", "downstream"])
def test_the_registered_run_43_mask_is_honoured_on_either_face(tmp_path, face):
    """Run 43's own registration, projected onto cells, on each face in turn.

    The registration is two entries of one state, so the cells it excludes have
    a gap: positions 35-37 (x = +10, +11 and +12 cm) lie between the entries and
    must survive whichever face carries the mask.  The mask is built from the
    registry rather than from a hand-written slice, so what is checked here is
    the shipped shot ranges and not a restatement of them.
    """
    from scripts.annotate_state_mask import STATE_REGISTRY, state_mask_for_run

    entry = STATE_REGISTRY[("43", "isat")]
    mask, covered = state_mask_for_run(N_X, N_WINDOWS, 20, entry["shot_ranges"])
    gap = np.flatnonzero(~covered)

    assert covered[12:35].all() and covered[38:].all()
    assert not covered[35:38].any()

    results, prov = _run(tmp_path, **{f"{face}_state": mask})
    unmasked = _run(tmp_path)[0]

    assert prov["state_cells_excluded"] == int(mask.sum())
    assert prov[f"state_cells_excluded_{face}"] == int(mask.sum())
    assert np.array_equal(results["state_excluded_cells"], mask)
    for name in MASKED_DATASETS:
        assert np.isnan(results[name][mask]).all(), name
        # Untouched means equal to the unmasked run cell for cell, not merely
        # finite: the mask must not move a value it does not exclude.
        assert np.array_equal(results[name][gap], unmasked[name][gap]), name
