from __future__ import annotations

import numpy as np

from core.object_geometry import (
    ClusteringPolicy,
    DepthDiscontinuityPolicy,
    ObjectPointExtractionRequest,
    StatisticalOutlierPolicy,
    SupportRetentionPolicy,
    extract_object_point_clouds,
)
from core.object_geometry.clustering import select_voxel_cluster
from core.object_geometry.filters import (
    depth_discontinuity_keep_mask,
    robust_xyz_outlier_keep_mask,
)
from tests.object_geometry_helpers import make_bundle


def test_depth_consensus_rejects_spike_but_preserves_thin_depth_ramp() -> None:
    height = width = 40
    depth = np.full((height, width), 5.0, dtype=np.float64)
    mask = np.zeros((height, width), dtype=bool)
    mask[4:18, 10:30] = True
    depth[4:18, 10:30] = 3.0
    mask[18:25, 15:25] = True
    depth[18:25, 15:25] = 2.9
    mask[25:38, 16:18] = True
    depth[25:38, 16:18] = 2.8 + np.arange(13)[:, None] * 0.005
    mask[25:38, 22:24] = True
    depth[25:38, 22:24] = 2.82 + np.arange(13)[:, None] * 0.005
    depth[10, 20] = 8.0

    result = depth_discontinuity_keep_mask(
        depth,
        mask,
        DepthDiscontinuityPolicy(minimum_neighbor_samples=4),
    )

    assert not result.keep[10, 20]
    assert result.keep[35, 16]
    assert np.count_nonzero(result.keep) >= np.count_nonzero(mask) - 2
    assert result.method == "robust_eight_neighbour_consensus_mad"


def test_outlier_support_guard_preserves_lower_structure_and_rejects_extreme_point() -> None:
    rng = np.random.default_rng(7)
    backrest = np.column_stack(
        (
            rng.normal(0.0, 0.10, 800),
            rng.normal(0.0, 0.08, 800),
            rng.normal(3.0, 0.02, 800),
        )
    )
    legs = np.column_stack(
        (
            rng.normal(0.0, 0.08, 120),
            rng.normal(0.9, 0.06, 120),
            rng.normal(2.8, 0.03, 120),
        )
    )
    points = np.vstack((backrest, legs, np.array([[4.0, 4.0, 9.0]])))
    pixels = np.vstack(
        (
            np.column_stack((rng.integers(10, 31, 800), rng.integers(10, 31, 800))),
            np.column_stack((rng.integers(14, 27, 120), rng.integers(31, 80, 120))),
            np.array([[20, 40]]),
        )
    )

    result = robust_xyz_outlier_keep_mask(
        points,
        StatisticalOutlierPolicy(minimum_points=32),
        source_pixel_indices_xy=pixels,
        support_policy=SupportRetentionPolicy(),
    )

    assert result.support_guard_applied
    assert result.method == "support_guard_quantile_fallback"
    assert not result.keep[-1]
    assert np.count_nonzero(result.keep[800:920]) >= 110
    assert result.support_diagnostics is not None
    assert result.support_diagnostics.vertical_span_retention == 1.0


def test_cluster_guard_merges_mask_adjacent_object_support_but_rejects_distant_background() -> None:
    primary = np.column_stack(
        (np.linspace(0.0, 0.1, 40), np.zeros(40), np.full(40, 3.0))
    )
    thin_support = np.column_stack(
        (np.linspace(0.0, 0.05, 20), np.full(20, 0.5), np.full(20, 2.8))
    )
    background = np.column_stack(
        (np.linspace(0.0, 0.05, 25), np.full(25, 0.8), np.full(25, 5.0))
    )
    points = np.vstack((primary, thin_support, background))
    pixels = np.vstack(
        (
            np.column_stack((np.arange(40) % 10 + 10, np.arange(40) // 10 + 10)),
            np.column_stack((np.arange(20) % 4 + 12, np.arange(20) // 4 + 14)),
            np.column_stack((np.arange(25) % 5 + 13, np.arange(25) // 5 + 18)),
        )
    )

    selection = select_voxel_cluster(
        points,
        pixels,
        mask_centroid_xy=(12.0, 12.0),
        policy=ClusteringPolicy(
            voxel_size=0.03,
            minimum_cluster_points=5,
            minimum_preserved_cluster_points=5,
            maximum_pixel_gap=4,
        ),
        support_policy=SupportRetentionPolicy(
            minimum_point_retention=0.70,
            minimum_axis_span_retention=0.80,
            minimum_grid_coverage_retention=0.70,
        ),
    )

    assert selection.support_guard_applied
    assert np.count_nonzero(selection.keep[:40]) == 40
    assert np.count_nonzero(selection.keep[40:60]) == 20
    assert np.count_nonzero(selection.keep[60:]) == 0
    assert selection.merged_cluster_count == 2


def test_pipeline_preserves_chair_like_lower_support_without_background_patch() -> None:
    height = width = 100
    mask = np.zeros((height, width), dtype=bool)
    mask[10:45, 25:75] = True
    mask[45:58, 30:70] = True
    mask[58:92, 31:37] = True
    mask[58:92, 63:69] = True
    mask[35:65, 20:27] = True
    mask[35:65, 73:80] = True

    depth = np.full((height, width), 5.0, dtype=np.float32)
    depth[10:45, 25:75] = 3.0
    depth[45:58, 30:70] = 2.75
    depth[58:92, 31:37] = 2.80
    depth[58:92, 63:69] = 2.85
    depth[35:65, 20:27] = 2.90
    depth[35:65, 73:80] = 2.92
    depth[48:54, 48:54] = 4.5

    item = extract_object_point_clouds(
        ObjectPointExtractionRequest(
            make_bundle(width=width, height=height, masks=(mask,), depth=depth)
        )
    ).object_results[0]

    assert item.filtered_cloud is not None
    pixels = item.filtered_cloud.source_pixel_indices_xy
    assert int(np.max(pixels[:, 1])) >= 88
    assert np.count_nonzero(pixels[:, 1] > 70) >= 100
    contamination = (
        (pixels[:, 0] >= 48)
        & (pixels[:, 0] < 54)
        & (pixels[:, 1] >= 48)
        & (pixels[:, 1] < 54)
    )
    assert not np.any(contamination)
    assert "thin_structure_support_preserved_after_cluster_guard" in item.warnings
