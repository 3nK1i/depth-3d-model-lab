from __future__ import annotations

import numpy as np

from core.object_geometry.sampling import deterministic_spatial_sample_indices


def test_sampling_is_deterministic_and_budgeted():
    rng = np.random.default_rng(7)
    points = rng.normal(size=(1000, 3))
    first = deterministic_spatial_sample_indices(points, 100)
    second = deterministic_spatial_sample_indices(points, 100)
    assert len(first) == 100
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 100
