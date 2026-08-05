from __future__ import annotations

import numpy as np

from core.object_geometry import (
    ClusteringPolicy, DepthDiscontinuityPolicy, MaskPreprocessingPolicy,
    ObjectPointExtractionPolicy, ObjectPointExtractionRequest,
    StatisticalOutlierPolicy, extract_object_point_clouds,
)
from tests.object_geometry_helpers import make_bundle


def test_many_overlapping_objects_remain_independent():
    masks = []
    for index in range(20):
        mask = np.zeros((100, 120), bool)
        x0 = 5 + index * 3
        mask[20:80, x0:x0 + 30] = True
        masks.append(mask)
    bundle = make_bundle(width=120, height=100, masks=tuple(masks))
    policy = ObjectPointExtractionPolicy(
        mask=MaskPreprocessingPolicy(erosion_pixels=0, minimum_mask_pixels=16),
        discontinuity=DepthDiscontinuityPolicy(enabled=False),
        outliers=StatisticalOutlierPolicy(enabled=False),
        clustering=ClusteringPolicy(enabled=False),
        minimum_retained_points=16,
    )
    result = extract_object_point_clouds(ObjectPointExtractionRequest(bundle, policy=policy))
    assert len(result.object_results) == 20
    assert all(item.filtered_cloud is not None for item in result.object_results)
