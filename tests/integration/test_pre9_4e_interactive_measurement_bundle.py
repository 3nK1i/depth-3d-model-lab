from __future__ import annotations

from pathlib import Path

import numpy as np

from core.inspection import load_inspection_bundle
from tests.inspection_helpers import write_inspection_bundle


def test_additive_numeric_depth_arrays_survive_bundle_reload(tmp_path: Path) -> None:
    bundle = write_inspection_bundle(tmp_path / "bundle")
    summary, arrays = load_inspection_bundle(bundle)
    arrays = dict(arrays)
    arrays.update(
        aligned_depth=np.full((summary.image_height, summary.image_width), 1.75, dtype=np.float32),
        aligned_depth_valid_mask=np.ones((summary.image_height, summary.image_width), dtype=bool),
        aligned_intrinsics_matrix=np.asarray(
            [[75.0, 0.0, 32.0], [0.0, 75.0, 24.0], [0.0, 0.0, 1.0]], dtype=np.float64
        ),
    )
    np.savez_compressed(summary.arrays_path, **arrays)

    reloaded_summary, reloaded_arrays = load_inspection_bundle(bundle)
    assert {"aligned_depth", "aligned_depth_valid_mask", "aligned_intrinsics_matrix"}.issubset(reloaded_arrays)
    assert reloaded_arrays["aligned_depth"].shape == (reloaded_summary.image_height, reloaded_summary.image_width)
    assert reloaded_arrays["aligned_intrinsics_matrix"].shape == (3, 3)


def test_legacy_bundle_remains_loadable_without_interactive_depth_arrays(tmp_path: Path) -> None:
    summary, arrays = load_inspection_bundle(write_inspection_bundle(tmp_path / "legacy"))
    assert summary.depth_scale == "metric"
    assert "aligned_depth" not in arrays
    assert "aligned_intrinsics_matrix" not in arrays
