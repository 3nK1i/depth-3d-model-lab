from __future__ import annotations

import numpy as np

from core.object_geometry.filters import (
    background_plane_keep_mask,
    depth_discontinuity_keep_mask,
    robust_xyz_outlier_keep_mask,
)
from core.object_geometry.policy import (
    BackgroundPlanePolicy,
    DepthDiscontinuityPolicy,
    StatisticalOutlierPolicy,
)


def test_depth_discontinuity_rejects_injected_spike():
    depth = np.ones((9, 9), float)
    depth[4, 4] = 8.0
    mask = np.ones((9, 9), bool)
    result = depth_discontinuity_keep_mask(
        depth, mask, DepthDiscontinuityPolicy(minimum_neighbor_samples=4)
    )
    assert not result.keep[4, 4]
    assert result.threshold is not None


def test_robust_xyz_filter_removes_far_point_and_handles_degenerate_axes():
    cloud = np.column_stack((np.linspace(0, 1, 40), np.zeros(40), np.ones(40)))
    cloud[-1] = (100, 100, 100)
    result = robust_xyz_outlier_keep_mask(cloud, StatisticalOutlierPolicy(minimum_points=10))
    assert not result.keep[-1]
    assert np.count_nonzero(result.keep) == 39


def test_background_plane_removal_is_conservative():
    object_points = np.array([[x, 0.0, 1.0] for x in np.linspace(-0.2, 0.2, 20)] + [[x, 0.0, 2.0] for x in np.linspace(-0.2, 0.2, 4)])
    yy, xx = np.mgrid[-4:5, -4:5]
    ring = np.column_stack((xx.ravel() * 0.03, yy.ravel() * 0.03, np.full(xx.size, 2.0)))
    policy = BackgroundPlanePolicy(
        enabled=True, minimum_ring_points=20, distance_threshold=0.03,
        minimum_behind_separation=0.2, maximum_removal_fraction=0.3,
    )
    result = background_plane_keep_mask(object_points, ring, policy)
    assert np.count_nonzero(result.keep) == 20


def test_discontinuity_filter_handles_single_pixel_without_shape_errors():
    depth = np.array([[1.0]])
    mask = np.array([[True]])
    result = depth_discontinuity_keep_mask(depth, mask, DepthDiscontinuityPolicy())
    assert result.keep.tolist() == [[True]]


def test_background_plane_filter_skips_when_plane_is_not_behind_object():
    object_points = np.column_stack((np.linspace(-0.1, 0.1, 20), np.zeros(20), np.full(20, 2.0)))
    yy, xx = np.mgrid[-4:5, -4:5]
    ring = np.column_stack((xx.ravel() * 0.03, yy.ravel() * 0.03, np.full(xx.size, 1.0)))
    result = background_plane_keep_mask(
        object_points,
        ring,
        BackgroundPlanePolicy(enabled=True, minimum_ring_points=20, minimum_behind_separation=0.1),
    )
    assert result.keep.all()
    assert result.warning == "background_plane_not_behind_object"
