from __future__ import annotations

import numpy as np

from core.reconstruction.mvs_da3_matched_view import derive_shared_visual_frame


def test_shared_frame_is_stable_with_large_sparse_cloud_and_outliers():
    rng = np.random.default_rng(42)
    core = rng.normal(size=(100_000, 3))
    outliers = np.array([[1e6, 1e6, 1e6], [-1e6, -1e6, -1e6]])
    points = np.vstack([core, outliers])
    sfm = {"landmarks": [{"position_reconstruction": p.tolist()} for p in points]}
    frame = derive_shared_visual_frame(sfm)
    assert frame.robust_diagonal < 20.0
    assert frame.robust_diagonal > 1.0
    assert frame.camera_distance("context") > frame.camera_distance("detail")
