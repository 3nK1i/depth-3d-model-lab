from __future__ import annotations

import numpy as np
import pytest

from core.contracts.validation import ContractValidationError
from core.object_geometry.projection import pixel_indices_from_mask, project_selected_pixels
from tests.object_geometry_helpers import make_bundle


def test_direct_projection_uses_camera_axis_convention():
    bundle = make_bundle(width=10, height=8, depth=np.full((8, 10), 2.0))
    pixels = np.array([[5, 4], [6, 4], [5, 5]], dtype=np.int64)
    points = project_selected_pixels(bundle.depth, pixels, bundle.intrinsics)
    assert np.allclose(points[0], [0.0, 0.0, 2.0])
    assert points[1, 0] > 0
    assert points[2, 1] > 0


def test_pixel_indices_are_explicit_xy_and_bounds_checked():
    mask = np.zeros((3, 4), bool); mask[1, 2] = True
    assert pixel_indices_from_mask(mask).tolist() == [[2, 1]]
    bundle = make_bundle(width=4, height=3, depth=np.ones((3, 4)))
    with pytest.raises(ContractValidationError):
        project_selected_pixels(bundle.depth, np.array([[4, 1]]), bundle.intrinsics)
