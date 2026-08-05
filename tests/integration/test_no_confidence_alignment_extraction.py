from __future__ import annotations

import numpy as np

from core.alignment import (
    AlignmentRequest,
    align_frame,
    declare_full_frame_resize,
    depth_input_from_inference_result,
)
from core.contracts.common import FrameReference, ImageSize
from core.contracts.spatial import PixelCoordinateSpace, PixelSpaceRole
from core.object_geometry import (
    ClusteringPolicy,
    DepthDiscontinuityPolicy,
    MaskPreprocessingPolicy,
    ObjectPointExtractionPolicy,
    ObjectPointExtractionRequest,
    StatisticalOutlierPolicy,
    extract_object_point_clouds,
)
from core.perception.adapters.mock_adapter import DevelopmentPerceptionMockAdapter
from core.perception.messages import FramePerceptionRequest
from core.perception.metadata import PerceptionTask
from core.perception.pipeline import PerceptionPipeline
from core.perception.registry import PERCEPTION_MODEL_REGISTRY
from core.runtime.messages import InferenceResult


def test_no_confidence_depth_result_does_not_reject_all_object_pixels():
    frame = FrameReference(21, 4, 0.0, ImageSize(64, 48))
    rgb_space = PixelCoordinateSpace("rgb", frame, frame.source_size, PixelSpaceRole.RGB)
    rgb = np.zeros((48, 64, 3), np.uint8)

    adapter = DevelopmentPerceptionMockAdapter()
    adapter.load({"emit_native_masks": True})
    adapter.warmup()
    try:
        perception = PerceptionPipeline(
            adapter,
            PERCEPTION_MODEL_REGISTRY.get("development-perception-mock"),
        ).process_frame(
            FramePerceptionRequest(
                frame,
                rgb,
                rgb_space,
                (PerceptionTask.DETECTION, PerceptionTask.INSTANCE_SEGMENTATION),
            )
        )
    finally:
        adapter.unload()

    depth_result = InferenceResult(
        21,
        4,
        "depth-pro-like",
        np.full((24, 32), 2.0, np.float32),
        1.0,
        intrinsics=np.array(
            [[30.0, 0.0, 16.0], [0.0, 30.0, 12.0], [0.0, 0.0, 1.0]]
        ),
        valid_mask=np.ones((24, 32), bool),
        point_stride=1,
        is_metric=True,
        model_name="Depth Pro Like",
        precision="fp16",
        depth_scale="metric",
        intrinsics_source="model_predicted",
        confidence=None,
        confidence_kind="none",
    )
    depth_input = depth_input_from_inference_result(depth_result, frame)
    aligned = align_frame(
        AlignmentRequest(
            frame,
            rgb,
            rgb_space,
            depth_input,
            perception,
            (declare_full_frame_resize(rgb_space, depth_input.depth_space),),
        )
    )

    assert aligned.confidence is None
    assert aligned.confidence_valid_mask.all()

    policy = ObjectPointExtractionPolicy(
        mask=MaskPreprocessingPolicy(erosion_pixels=0, minimum_mask_pixels=8),
        discontinuity=DepthDiscontinuityPolicy(enabled=False),
        outliers=StatisticalOutlierPolicy(enabled=False),
        clustering=ClusteringPolicy(enabled=False),
        minimum_retained_points=8,
    )
    extraction = extract_object_point_clouds(
        ObjectPointExtractionRequest(aligned, policy=policy)
    )

    assert any(result.filtered_cloud is not None for result in extraction.object_results)
    assert any(result.stage_counts.confidence_valid_pixels > 0 for result in extraction.object_results)
