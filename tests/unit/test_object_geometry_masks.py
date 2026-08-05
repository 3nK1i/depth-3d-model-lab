from __future__ import annotations

import numpy as np

from core.object_geometry.masks import (
    erode_mask_conservatively,
    exterior_ring,
    largest_connected_component,
    mask_centroid_xy,
)


def test_largest_component_uses_requested_connectivity():
    mask = np.zeros((8, 8), bool)
    mask[1:4, 1:4] = True
    mask[6, 6] = True
    kept = largest_connected_component(mask)
    assert np.count_nonzero(kept) == 9


def test_conservative_erosion_skips_destructive_result():
    mask = np.zeros((7, 7), bool)
    mask[2:5, 2:5] = True
    output, skipped = erode_mask_conservatively(mask, pixels=2, minimum_pixels=8)
    assert skipped
    assert np.array_equal(output, mask)


def test_exterior_ring_excludes_original_and_centroid_is_xy():
    mask = np.zeros((9, 9), bool)
    mask[3:6, 2:5] = True
    ring = exterior_ring(mask, pixels=1)
    assert not np.any(ring & mask)
    assert mask_centroid_xy(mask) == (3.0, 4.0)
