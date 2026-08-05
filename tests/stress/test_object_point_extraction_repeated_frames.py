from __future__ import annotations

from core.object_geometry import (
    ClusteringPolicy, DepthDiscontinuityPolicy, MaskPreprocessingPolicy,
    ObjectPointExtractionPolicy, ObjectPointExtractionRequest,
    StatisticalOutlierPolicy, extract_object_point_clouds,
)
from tests.object_geometry_helpers import make_bundle


def test_repeated_extraction_is_deterministic():
    bundle = make_bundle(width=80, height=60)
    policy = ObjectPointExtractionPolicy(
        mask=MaskPreprocessingPolicy(erosion_pixels=1, minimum_mask_pixels=16),
        discontinuity=DepthDiscontinuityPolicy(enabled=False),
        outliers=StatisticalOutlierPolicy(enabled=False),
        clustering=ClusteringPolicy(enabled=False),
        minimum_retained_points=16,
    )
    baseline = extract_object_point_clouds(ObjectPointExtractionRequest(bundle, policy=policy))
    expected = baseline.object_results[0].filtered_cloud.points_xyz_m
    for _ in range(50):
        current = extract_object_point_clouds(ObjectPointExtractionRequest(bundle, policy=policy))
        assert (current.object_results[0].filtered_cloud.points_xyz_m == expected).all()
