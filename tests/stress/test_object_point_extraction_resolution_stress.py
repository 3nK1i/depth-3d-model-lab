from __future__ import annotations

from core.object_geometry import (
    ClusteringPolicy, DepthDiscontinuityPolicy, MaskPreprocessingPolicy,
    ObjectPointExtractionPolicy, ObjectPointExtractionRequest,
    StatisticalOutlierPolicy, extract_object_point_clouds,
)
from tests.object_geometry_helpers import make_bundle


def test_landscape_portrait_and_square_resolutions():
    policy = ObjectPointExtractionPolicy(
        mask=MaskPreprocessingPolicy(erosion_pixels=0, minimum_mask_pixels=4),
        discontinuity=DepthDiscontinuityPolicy(enabled=False),
        outliers=StatisticalOutlierPolicy(enabled=False),
        clustering=ClusteringPolicy(enabled=False),
        minimum_retained_points=4,
    )
    for width, height in ((96, 54), (54, 96), (64, 64)):
        bundle = make_bundle(width=width, height=height)
        result = extract_object_point_clouds(ObjectPointExtractionRequest(bundle, policy=policy))
        assert result.object_results[0].filtered_cloud is not None
