from __future__ import annotations

import numpy as np

from core.contracts import FrameReference, ImageSize, ModelProvenance, PixelCoordinateSpace, PixelSpaceRole
from core.models.metadata import ConfidenceKind
from core.perception import (
    DenseMaskPayload,
    InstanceMask,
    MaskSource,
    SemanticReviewStatus,
    add_project_class,
    accepted_observations,
    assign_workspace_class,
    create_segmentation_project,
    merge_project_catalog_into_workspace,
    set_workspace_review_status,
    workspace_from_mask_proposals,
)


def test_2b_project_class_flows_from_unlabeled_auto_segment_to_existing_observation_contract() -> None:
    frame = FrameReference(5, 2, 0.25, ImageSize(32, 24))
    space = PixelCoordinateSpace("rgb", frame, frame.source_size, PixelSpaceRole.RGB)
    data = np.zeros((24, 32), dtype=bool)
    data[4:18, 7:25] = True
    mask = InstanceMask(
        mask_id="mask-auto-1",
        frame_reference=frame,
        pixel_space=space,
        payload=DenseMaskPayload(data),
        source=MaskSource.NATIVE_INSTANCE,
        model_provenance=ModelProvenance("auto-seg", "Auto Segment", "segmentation"),
        confidence=0.91,
        confidence_kind=ConfidenceKind.CONFIDENCE,
    )

    project = create_segmentation_project("Factory", project_id="factory")
    project, robot = add_project_class(project, "robot_arm")
    workspace = workspace_from_mask_proposals((mask,), workspace_id="ws-auto")
    workspace = merge_project_catalog_into_workspace(workspace, project)
    instance_name = project.suggest_instance_name(robot.class_id)
    object_id = workspace.objects[0].object_id
    workspace = assign_workspace_class(workspace, object_id, robot, instance_name=instance_name)
    workspace = set_workspace_review_status(workspace, object_id, SemanticReviewStatus.ACCEPTED)

    observations = accepted_observations(workspace)
    assert len(observations) == 1
    assert observations[0].class_id == robot.class_id
    assert observations[0].class_name == "robot_arm"
    assert observations[0].selected_mask_id == "mask-auto-1"
    assert workspace.objects[0].instance_name == "robot_arm_001"
