"""Importing an annotator script must not run it or read ``sys.argv``.

These three modules used to build their paths from bare ``sys.argv`` indices at
module scope, so importing one -- from a test, a sibling script, or a REPL --
raised ``IndexError`` before any of its functions became reachable.  The
argument parsing now lives in ``main(argv=None)``, and this test pins that: the
import runs with an argv that carries no arguments at all.
"""
import importlib
import sys

import pytest

SCRIPT_MODULES = [
    "scripts.annotate_discharge_artifacts",
    "scripts.annotate_rail_mask",
    "scripts.annotate_saturation_attrs",
    "scripts.annotate_state_mask",
    "scripts.compare_run_annotations",
    "scripts.screen_consecutive_shot_steps",
    "scripts.screen_rot180_saturation",
]


@pytest.mark.parametrize("module_name", SCRIPT_MODULES)
def test_script_module_imports_without_arguments(module_name, monkeypatch):
    monkeypatch.setattr(sys, "argv", [module_name])
    module = importlib.reload(importlib.import_module(module_name))
    assert callable(module.main)
