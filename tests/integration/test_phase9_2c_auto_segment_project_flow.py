from __future__ import annotations

import numpy as np

from core.contracts import FrameReference, ImageSize, PixelCoordinateSpace, PixelSpaceRole
from core.perception import (
    AutoSegmentEngine,
    AutoSegmentPolicy,
    PerceptionTask,
    SemanticReviewStatus,
    accepted_observations,
    add_project_class,
    assign_workspace_class,
    create_segmentation_project,
    set_workspace_review_status,
)
from core.perception.messages import FramePerceptionRequest
from core.perception.registry import PERCEPTION_MODEL_REGISTRY


def _request() -> FramePerceptionRequest:
    frame = FrameReference(12, 4, 0.5, ImageSize(64, 48))
    space = PixelCoordinateSpace("rgb", frame, frame.source_size, PixelSpaceRole.RGB)
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    return FramePerceptionRequest(
        frame,
        rgb,
        space,
        (PerceptionTask.AUTOMATIC_SEGMENTATION,),
    )


def test_2c_registry_mock_generates_unlabeled_proposals_then_project_class_round_trips() -> None:
    descriptor = PERCEPTION_MODEL_REGISTRY.get("development-perception-mock")
    adapter = PERCEPTION_MODEL_REGISTRY.build(descriptor.model_id)
    adapter.load({})
    adapter.warmup()
    try:
        project = create_segmentation_project("Factory", project_id="factory")
        project, conveyor = add_project_class(project, "conveyor")
        engine = AutoSegmentEngine(
            adapter,
            descriptor,
            policy=AutoSegmentPolicy(minimum_mask_pixels=32, max_proposals=16),
        )
        result = engine.generate(_request(), workspace_id="ws", project=project)
        assert result.kept_proposals == 2
        assert all(item.class_id is None for item in result.workspace.objects)

        first = result.workspace.objects[0]
        workspace = assign_workspace_class(
            result.workspace,
            first.object_id,
            conveyor,
            instance_name=project.suggest_instance_name(conveyor.class_id),
        )
        workspace = set_workspace_review_status(
            workspace,
            first.object_id,
            SemanticReviewStatus.ACCEPTED,
        )
        observations = accepted_observations(workspace)
        assert len(observations) == 1
        assert observations[0].class_name == "conveyor"
        assert observations[0].selected_mask is not None
        assert workspace.objects[0].instance_name == "conveyor_001"
    finally:
        adapter.unload()


def test_2c_automatic_and_prompted_sam_capabilities_coexist_without_changing_model_id() -> None:
    descriptor = PERCEPTION_MODEL_REGISTRY.get("ultralytics-sam2-prompted")
    assert descriptor.model_id == "ultralytics-sam2-prompted"
    assert PerceptionTask.PROMPTED_SEGMENTATION in descriptor.capabilities.tasks
    assert PerceptionTask.AUTOMATIC_SEGMENTATION in descriptor.capabilities.tasks
