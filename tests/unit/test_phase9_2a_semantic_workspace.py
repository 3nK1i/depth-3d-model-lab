from __future__ import annotations

import numpy as np
import pytest

from core.contracts import FrameReference, ImageSize, ModelProvenance, PixelCoordinateSpace, PixelSpaceRole
from core.contracts.validation import ContractValidationError
from core.models.metadata import ConfidenceKind
from core.perception import (
    DenseMaskPayload,
    InstanceMask,
    MaskSource,
    ObjectObservation,
    PerceptionFrameResult,
    SemanticClassCatalog,
    SemanticClassDefinition,
    SemanticClassOrigin,
    SemanticLabelSource,
    SemanticReviewStatus,
    SemanticWorkspace,
    SemanticWorkspaceObject,
    accepted_observations,
    assign_workspace_class,
    set_workspace_review_status,
    workspace_from_mask_proposals,
    workspace_from_perception_result,
)


def _frame() -> FrameReference:
    return FrameReference(92, 7, 1.5, ImageSize(16, 12))


def _provenance() -> ModelProvenance:
    return ModelProvenance("ultralytics-yolo-seg", "YOLO Seg", "instance_segmentation")


def _observation(index: int, class_id: int = 0, class_name: str = "person") -> ObjectObservation:
    frame = _frame()
    space = PixelCoordinateSpace("rgb", frame, frame.source_size, PixelSpaceRole.RGB)
    data = np.zeros((12, 16), dtype=bool)
    data[2:8, 2 + index : 7 + index] = True
    mask = InstanceMask(
        mask_id=f"mask-{index}",
        frame_reference=frame,
        pixel_space=space,
        payload=DenseMaskPayload(data),
        source=MaskSource.NATIVE_INSTANCE,
        model_provenance=_provenance(),
        confidence=0.9,
        confidence_kind=ConfidenceKind.CONFIDENCE,
    )
    return ObjectObservation(
        observation_id=f"obs-{index}",
        frame_reference=frame,
        class_id=class_id,
        class_name=class_name,
        masks=(mask,),
    )


def _workspace() -> SemanticWorkspace:
    result = PerceptionFrameResult(
        _frame(),
        (_observation(0), _observation(1)),
        (_provenance(),),
    )
    return workspace_from_perception_result(result, workspace_id="ws-1")


def test_2a_yolo_result_import_preserves_class_mask_and_pending_review() -> None:
    workspace = _workspace()
    assert [(c.class_id, c.class_name) for c in workspace.class_catalog.classes] == [(0, "person")]
    assert [obj.class_name for obj in workspace.objects] == ["person", "person"]
    assert all(obj.label_source is SemanticLabelSource.MODEL_PREDICTION for obj in workspace.objects)
    assert all(obj.review_status is SemanticReviewStatus.PENDING for obj in workspace.objects)
    assert [obj.selected_mask_id for obj in workspace.objects] == ["mask-0", "mask-1"]


def test_2a_user_can_create_class_and_reclassify_one_instance_without_mutating_mask() -> None:
    workspace = _workspace()
    catalog, robot = workspace.class_catalog.with_custom_class("robot_arm")
    workspace = SemanticWorkspace(
        workspace.workspace_id,
        workspace.frame_reference,
        catalog,
        workspace.objects,
        workspace.revision,
    )
    original_mask = workspace.objects[0].observation.selected_mask
    updated = assign_workspace_class(
        workspace,
        "obs-0",
        robot,
        instance_name="robot_arm_001",
    )
    obj = updated.object_by_id("obs-0")
    assert obj is not None
    assert obj.class_id == robot.class_id
    assert obj.class_name == "robot_arm"
    assert obj.instance_name == "robot_arm_001"
    assert obj.label_source is SemanticLabelSource.USER_ASSIGNED
    assert obj.observation.selected_mask is original_mask
    assert updated.object_by_id("obs-1").class_name == "person"


def test_2a_multiple_instances_can_share_class_but_keep_distinct_instance_identity() -> None:
    workspace = _workspace()
    catalog, pallet = workspace.class_catalog.with_custom_class("pallet")
    workspace = SemanticWorkspace(workspace.workspace_id, workspace.frame_reference, catalog, workspace.objects)
    workspace = assign_workspace_class(workspace, "obs-0", pallet, instance_name="pallet_001")
    workspace = assign_workspace_class(workspace, "obs-1", pallet, instance_name="pallet_002")
    assert [obj.class_name for obj in workspace.objects] == ["pallet", "pallet"]
    assert [obj.instance_name for obj in workspace.objects] == ["pallet_001", "pallet_002"]


def test_2a_accept_requires_label_and_rejected_objects_do_not_flow_downstream() -> None:
    source_observation = _observation(0)
    proposal_mask = source_observation.selected_mask
    assert proposal_mask is not None
    workspace = SemanticWorkspace(
        "auto-seg",
        _frame(),
        SemanticClassCatalog(),
        (
            SemanticWorkspaceObject(
                object_id="proposal-0",
                proposal_mask=proposal_mask,
            ),
        ),
    )
    with pytest.raises(ContractValidationError, match="Accepted objects require"):
        set_workspace_review_status(workspace, "proposal-0", SemanticReviewStatus.ACCEPTED)

    catalog, conveyor = workspace.class_catalog.with_custom_class("conveyor")
    workspace = SemanticWorkspace(workspace.workspace_id, workspace.frame_reference, catalog, workspace.objects)
    workspace = assign_workspace_class(workspace, "proposal-0", conveyor)
    rejected = set_workspace_review_status(workspace, "proposal-0", SemanticReviewStatus.REJECTED)
    assert accepted_observations(rejected) == ()

    accepted = set_workspace_review_status(workspace, "proposal-0", SemanticReviewStatus.ACCEPTED)
    observations = accepted_observations(accepted)
    assert len(observations) == 1
    assert observations[0].class_id == conveyor.class_id
    assert observations[0].class_name == "conveyor"
    assert observations[0].selected_mask_id == proposal_mask.mask_id


def test_2a_semantic_workspace_has_no_geometry_metric_or_measurement_authority() -> None:
    workspace = _workspace()
    for obj in workspace.objects:
        assert obj.is_geometry_authority is False
        assert obj.is_metric_authority is False
        assert obj.is_measurement_authority is False


def test_2a_catalog_rejects_id_and_name_collisions() -> None:
    catalog = SemanticClassCatalog.from_model_names({0: "person"}, source_model_id="yolo")
    with pytest.raises(ContractValidationError, match="already assigned"):
        catalog.with_class(SemanticClassDefinition(0, "robot", SemanticClassOrigin.CUSTOM))
    with pytest.raises(ContractValidationError, match="already assigned"):
        catalog.with_class(SemanticClassDefinition(2, "PERSON", SemanticClassOrigin.CUSTOM))


def test_2a_mask_only_auto_segment_proposals_remain_unlabeled_until_user_assignment() -> None:
    source = _observation(0)
    mask = source.selected_mask
    assert mask is not None
    workspace = workspace_from_mask_proposals((mask,), workspace_id="auto-proposals")
    proposal = workspace.objects[0]
    assert proposal.class_id is None
    assert proposal.class_name == ""
    assert proposal.label_source is SemanticLabelSource.UNASSIGNED
    assert proposal.review_status is SemanticReviewStatus.PENDING
    assert proposal.selected_mask_id == mask.mask_id

    catalog, cabinet = workspace.class_catalog.with_custom_class("control_cabinet")
    workspace = SemanticWorkspace(workspace.workspace_id, workspace.frame_reference, catalog, workspace.objects)
    workspace = assign_workspace_class(workspace, proposal.object_id, cabinet, instance_name="control_cabinet_001")
    workspace = set_workspace_review_status(workspace, proposal.object_id, SemanticReviewStatus.ACCEPTED)
    reviewed = accepted_observations(workspace)
    assert len(reviewed) == 1
    assert reviewed[0].class_name == "control_cabinet"
    assert reviewed[0].selected_mask_id == mask.mask_id
