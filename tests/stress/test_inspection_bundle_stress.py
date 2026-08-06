from pathlib import Path
import numpy as np

from core.inspection import load_inspection_bundle
from tests.inspection_helpers import write_inspection_bundle


def test_repeated_bundle_loads_are_deterministic_and_do_not_share_mutable_masks(tmp_path: Path):
    directory = write_inspection_bundle(tmp_path / "bundle")
    first_summary, first_arrays = load_inspection_bundle(directory)
    expected = first_arrays[first_summary.objects[0].mask_key].copy()
    for _ in range(100):
        summary, arrays = load_inspection_bundle(directory)
        key = summary.objects[0].mask_key
        assert np.array_equal(arrays[key], expected)
        arrays[key][:] = False
    _, final_arrays = load_inspection_bundle(directory)
    assert np.array_equal(final_arrays[first_summary.objects[0].mask_key], expected)
