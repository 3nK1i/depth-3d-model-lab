from __future__ import annotations

import numpy as np

from core.object_geometry.clustering import select_voxel_cluster
from core.object_geometry.policy import ClusteringPolicy


def test_cluster_containing_mask_centroid_is_preferred():
    first = np.column_stack((np.linspace(0, 0.05, 20), np.zeros(20), np.ones(20)))
    second = np.column_stack((np.linspace(1, 1.05, 30), np.zeros(30), np.ones(30)))
    points = np.vstack((first, second))
    pixels = np.vstack((np.tile([5, 5], (20, 1)), np.tile([50, 50], (30, 1))))
    selection = select_voxel_cluster(
        points, pixels, mask_centroid_xy=(5, 5),
        policy=ClusteringPolicy(voxel_size=0.1, minimum_cluster_points=5),
    )
    assert selection.method == "mask_centroid_cluster"
    assert np.count_nonzero(selection.keep) == 20


def test_weak_cluster_dominance_is_reported():
    a = np.column_stack((np.linspace(0, 0.03, 10), np.zeros(10), np.ones(10)))
    b = np.column_stack((np.linspace(1, 1.03, 9), np.zeros(9), np.ones(9)))
    points = np.vstack((a, b))
    pixels = np.arange(len(points) * 2).reshape(-1, 2)
    selection = select_voxel_cluster(
        points, pixels, mask_centroid_xy=None,
        policy=ClusteringPolicy(voxel_size=0.1, minimum_cluster_points=3, weak_dominance_ratio=1.5),
    )
    assert selection.ambiguous
