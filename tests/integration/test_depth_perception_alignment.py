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
from core.perception.adapters.mock_adapter import DevelopmentPerceptionMockAdapter
from core.perception.messages import FramePerceptionRequest
from core.perception.metadata import PerceptionTask
from core.perception.pipeline import PerceptionPipeline
from core.perception.registry import PERCEPTION_MODEL_REGISTRY
from core.runtime.messages import InferenceResult


def test_mock_perception_and_depth_ipc_results_align_without_runtime_coupling():
    frame = FrameReference(11, 12, 1.5, ImageSize(20, 16))
    rgb_space = PixelCoordinateSpace("rgb", frame, frame.source_size, PixelSpaceRole.RGB)
    rgb = np.zeros((16, 20, 3), np.uint8)

    adapter = DevelopmentPerceptionMockAdapter()
    adapter.load({"emit_native_masks": True})
    adapter.warmup()
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

    depth_result = InferenceResult(
        11,
        12,
        "development-mock",
        np.full((8, 10), 2.0, np.float32),
        1.0,
        intrinsics=np.array([[10.0, 0.0, 5.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]]),
        valid_mask=np.ones((8, 10), bool),
        point_stride=1,
        is_metric=True,
        model_name="Development Mock",
        precision="fp32",
        depth_scale="synthetic",
        intrinsics_source="estimated",
        confidence_kind="none",
    )
    depth_input = depth_input_from_inference_result(depth_result, frame)
    transform = declare_full_frame_resize(rgb_space, depth_input.depth_space)
    aligned = align_frame(
        AlignmentRequest(
            frame,
            rgb,
            rgb_space,
            depth_input,
            perception,
            (transform,),
        )
    )

    assert aligned.depth.shape == (8, 10)
    assert aligned.aligned_rgb.shape == (8, 10, 3)
    assert len(aligned.aligned_observations) == 2
    assert all(value.aligned_mask is not None for value in aligned.aligned_observations)
    assert all(value.target_space.space_id == "depth" for value in aligned.aligned_observations)
    adapter.unload()
