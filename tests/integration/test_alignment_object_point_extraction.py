from __future__ import annotations

from core.object_geometry import (
    ClusteringPolicy,
    DepthDiscontinuityPolicy,
    MaskPreprocessingPolicy,
    ObjectPointExtractionPolicy,
    ObjectPointExtractionRequest,
    StatisticalOutlierPolicy,
    extract_object_point_clouds,
)
from tests.object_geometry_helpers import make_bundle


def test_aligned_bundle_round_trip_to_object_point_cloud():
    bundle = make_bundle()
    policy = ObjectPointExtractionPolicy(
        mask=MaskPreprocessingPolicy(erosion_pixels=1, minimum_mask_pixels=16),
        discontinuity=DepthDiscontinuityPolicy(enabled=True, minimum_neighbor_samples=4),
        outliers=StatisticalOutlierPolicy(enabled=True, minimum_points=16),
        clustering=ClusteringPolicy(enabled=False),
        minimum_retained_points=16,
    )
    result = extract_object_point_clouds(ObjectPointExtractionRequest(bundle, policy=policy))
    cloud = result.object_results[0].filtered_cloud
    assert cloud is not None
    assert cloud.metric_ready
    assert cloud.points_xyz_m.shape[1] == 3
    assert cloud.source_pixel_indices_xy.shape[1] == 2
