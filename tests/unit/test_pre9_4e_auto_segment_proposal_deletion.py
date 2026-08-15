from __future__ import annotations

import numpy as np
import pytest

from core.contracts import FrameReference, ImageSize, ModelProvenance, PixelCoordinateSpace, PixelSpaceRole
from core.contracts.validation import ContractValidationError
from core.perception import (
    DenseMaskPayload,
    InstanceMask,
    MaskSource,
    SemanticReviewStatus,
    SemanticWorkspace,
    delete_workspace_object,
    merge_workspace_objects,
    set_workspace_review_status,
    workspace_from_mask_proposals,
)


def _workspace(count: int = 2) -> SemanticWorkspace:
    frame = FrameReference(0, 0, 0.0, ImageSize(80, 60))
    space = PixelCoordinateSpace("rgb", frame, frame.source_size, PixelSpaceRole.RGB)
    masks = []
    for index in range(count):
        data = np.zeros((60, 80), dtype=bool)
        data[5 + index * 10 : 20 + index * 10, 8 + index * 12 : 28 + index * 12] = True
        masks.append(
            InstanceMask(
                f"mask-{index}",
                frame,
                space,
                DenseMaskPayload(data),
                MaskSource.NATIVE_INSTANCE,
                ModelProvenance("sam2", "SAM2", "automatic_segmentation"),
            )
        )
    return workspace_from_mask_proposals(masks, workspace_id="delete-test")


def test_pre9_4e_pending_proposal_delete_is_immutable_revision_without_authority() -> None:
    workspace = _workspace(2)
    target = workspace.objects[0].object_id
    retained = workspace.objects[1].object_id

    deleted = delete_workspace_object(workspace, target)

    assert workspace.object_by_id(target) is not None
    assert deleted.object_by_id(target) is None
    assert deleted.object_by_id(retained) is not None
    assert deleted.revision == workspace.revision + 1
    assert deleted.objects[0].is_geometry_authority is False
    assert deleted.objects[0].is_metric_authority is False
    assert deleted.objects[0].is_measurement_authority is False


def test_pre9_4e_rejected_proposal_can_be_deleted() -> None:
    workspace = _workspace(1)
    target = workspace.objects[0].object_id
    rejected = set_workspace_review_status(workspace, target, SemanticReviewStatus.REJECTED)

    deleted = delete_workspace_object(rejected, target)

    assert deleted.objects == ()
    assert deleted.revision == rejected.revision + 1


def test_pre9_4e_accepted_object_is_protected_from_delete() -> None:
    workspace = _workspace(1)
    target = workspace.objects[0].object_id
    catalog, definition = workspace.class_catalog.with_custom_class("plant")
    workspace = SemanticWorkspace(
        workspace.workspace_id,
        workspace.frame_reference,
        catalog,
        workspace.objects,
        revision=workspace.revision,
    )
    from core.perception import assign_workspace_class

    workspace = assign_workspace_class(workspace, target, definition, instance_name="plant_001")
    accepted = set_workspace_review_status(workspace, target, SemanticReviewStatus.ACCEPTED)

    with pytest.raises(ContractValidationError, match="Accepted semantic objects are protected"):
        delete_workspace_object(accepted, target)


def test_pre9_4e_merged_semantic_result_is_protected_from_delete() -> None:
    workspace = _workspace(2)
    merged, merged_id = merge_workspace_objects(
        workspace,
        tuple(item.object_id for item in workspace.objects),
    )

    with pytest.raises(ContractValidationError, match="Merged semantic results are protected"):
        delete_workspace_object(merged, merged_id)


def test_pre9_4e_delete_rejects_unknown_object() -> None:
    with pytest.raises(ContractValidationError, match="Unknown semantic workspace object"):
        delete_workspace_object(_workspace(1), "proposal:missing")
