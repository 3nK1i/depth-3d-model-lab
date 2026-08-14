from __future__ import annotations

import numpy as np
import pytest

from core.contracts import FrameReference, ImageSize, ModelProvenance, PixelCoordinateSpace, PixelSpaceRole
from core.contracts.validation import ContractValidationError
from core.perception import (
    DenseMaskPayload,
    InstanceMask,
    MaskSource,
    SemanticClassCatalog,
    SemanticReviewStatus,
    SemanticWorkspace,
    SemanticWorkspaceObject,
    accepted_observations,
    assign_workspace_class,
    merge_workspace_objects,
    set_workspace_review_status,
)


def _frame() -> FrameReference:
    return FrameReference(501, 3, 1.0, ImageSize(12, 10))


def _mask(mask_id: str, y0: int, y1: int, x0: int, x1: int) -> InstanceMask:
    frame = _frame()
    space = PixelCoordinateSpace("rgb", frame, frame.source_size, PixelSpaceRole.RGB)
    data = np.zeros((10, 12), dtype=bool)
    data[y0:y1, x0:x1] = True
    return InstanceMask(
        mask_id=mask_id,
        frame_reference=frame,
        pixel_space=space,
        payload=DenseMaskPayload(data),
        source=MaskSource.NATIVE_INSTANCE,
        model_provenance=ModelProvenance("sam2", "SAM2", "automatic_segmentation"),
    )


def _workspace() -> SemanticWorkspace:
    frame = _frame()
    catalog, plant = SemanticClassCatalog().with_custom_class("plant")
    ws = SemanticWorkspace(
        "merge-ws",
        frame,
        catalog,
        (
            SemanticWorkspaceObject("proposal:plant", proposal_mask=_mask("plant-mask", 1, 7, 1, 5)),
            SemanticWorkspaceObject("proposal:pot", proposal_mask=_mask("pot-mask", 6, 10, 2, 6)),
            SemanticWorkspaceObject("proposal:table", proposal_mask=_mask("table-mask", 3, 8, 7, 11)),
        ),
    )
    ws = assign_workspace_class(ws, "proposal:plant", plant, instance_name="plant_001")
    return set_workspace_review_status(ws, "proposal:plant", SemanticReviewStatus.ACCEPTED)


def test_9_2f_merge_unions_masks_replaces_sources_and_preserves_primary_identity() -> None:
    ws = _workspace()
    merged, merged_id = merge_workspace_objects(
        ws,
        ("proposal:plant", "proposal:pot"),
        primary_object_id="proposal:plant",
    )
    assert merged.object_by_id("proposal:plant") is None
    assert merged.object_by_id("proposal:pot") is None
    assert merged.object_by_id("proposal:table") is not None
    obj = merged.object_by_id(merged_id)
    assert obj is not None
    assert obj.class_name == "plant"
    assert obj.instance_name == "plant_001"
    assert obj.review_status is SemanticReviewStatus.PENDING
    assert obj.merged_from_object_ids == ("proposal:plant", "proposal:pot")
    assert obj.merged_from_mask_ids == ("plant-mask", "pot-mask")
    assert obj.proposal_mask is not None
    assert obj.proposal_mask.source is MaskSource.SEMANTIC_MERGE
    expected = np.logical_or(
        ws.object_by_id("proposal:plant").proposal_mask.payload.data,
        ws.object_by_id("proposal:pot").proposal_mask.payload.data,
    )
    assert np.array_equal(obj.proposal_mask.payload.data, expected)
    assert obj.is_geometry_authority is False
    assert obj.is_metric_authority is False
    assert obj.is_measurement_authority is False


def test_9_2f_merge_of_unlabeled_primary_remains_unlabeled_and_pending() -> None:
    ws = _workspace()
    merged, merged_id = merge_workspace_objects(
        ws,
        ("proposal:pot", "proposal:table"),
        primary_object_id="proposal:pot",
    )
    obj = merged.object_by_id(merged_id)
    assert obj is not None
    assert obj.class_id is None
    assert obj.class_name == ""
    assert obj.instance_name == ""
    assert obj.review_status is SemanticReviewStatus.PENDING


def test_9_2f_merge_provenance_flattens_when_merged_again() -> None:
    ws = _workspace()
    first, first_id = merge_workspace_objects(ws, ("proposal:plant", "proposal:pot"), primary_object_id="proposal:plant")
    second, second_id = merge_workspace_objects(first, (first_id, "proposal:table"), primary_object_id=first_id)
    obj = second.object_by_id(second_id)
    assert obj is not None
    assert obj.merged_from_object_ids == ("proposal:plant", "proposal:pot", "proposal:table")
    assert obj.merged_from_mask_ids == ("plant-mask", "pot-mask", "table-mask")
    assert obj.class_name == "plant"
    assert obj.instance_name == "plant_001"
    assert obj.review_status is SemanticReviewStatus.PENDING


def test_9_2f_accepted_merge_normalizes_to_existing_object_observation_contract() -> None:
    ws = _workspace()
    merged, merged_id = merge_workspace_objects(ws, ("proposal:plant", "proposal:pot"), primary_object_id="proposal:plant")
    merged = set_workspace_review_status(merged, merged_id, SemanticReviewStatus.ACCEPTED)
    observations = accepted_observations(merged)
    assert len(observations) == 1
    obs = observations[0]
    assert obs.class_name == "plant"
    assert obs.selected_mask is not None
    assert obs.selected_mask.source is MaskSource.SEMANTIC_MERGE


def test_9_2f_merge_rejects_less_than_two_or_unknown_objects() -> None:
    ws = _workspace()
    with pytest.raises(ContractValidationError, match="at least two"):
        merge_workspace_objects(ws, ("proposal:plant",))
    with pytest.raises(ContractValidationError, match="Unknown"):
        merge_workspace_objects(ws, ("proposal:plant", "missing"))
