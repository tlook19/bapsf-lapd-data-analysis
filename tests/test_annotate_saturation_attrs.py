"""The processed/ output guard on the ISWEEP rot-180 saturation annotator."""

import pytest

from scripts.annotate_saturation_attrs import refuse_processed_output


def test_a_processed_product_path_is_refused_without_the_flag(tmp_path):
    """The guard fires at argument resolution, before any file is opened."""
    placed = tmp_path / "processed" / "isweep_rot180_deadtime_profiles.hdf5"
    placed.parent.mkdir()
    placed.write_bytes(b"not really hdf5, the guard never opens it")
    with pytest.raises(ValueError, match="--allow-processed"):
        refuse_processed_output(placed, allow_processed=False)
    assert placed.read_bytes() == b"not really hdf5, the guard never opens it"

    refuse_processed_output(placed, allow_processed=True)
    refuse_processed_output(tmp_path / "regen" / "isweep_rot180.hdf5", allow_processed=False)


def test_a_file_named_processed_outside_a_processed_directory_is_allowed(tmp_path):
    """The guard reads directory components, not the file name."""
    refuse_processed_output(tmp_path / "processed.hdf5", allow_processed=False)
