from __future__ import annotations

import numpy as np
import pytest

from core.contracts import FrameReference, ImageSize, ModelProvenance, PixelCoordinateSpace, PixelSpaceRole
from core.contracts.validation import ContractValidationError
from core.models.metadata import ConfidenceKind
from core.perception.messages import (
    FramePerceptionRequest,
    PromptedSegmentationRequest,
    SegmentationPrompt,
    SegmentationPromptKind,
)
from core.perception.metadata import PerceptionTask
from core.perception.raw import RawPromptedMask
from core.perception.result import DenseMaskPayload, InstanceMask, MaskSource
from core.perception.semantic_workspace import (
    SemanticReviewStatus,
    SemanticWorkspace,
    append_workspace_mask_proposal,
    replace_workspace_object_mask,
)


def _frame() -> FrameReference:
    return FrameReference(0, 0, 0.0, ImageSize(80, 60))


def _space(frame: FrameReference) -> PixelCoordinateSpace:
    return PixelCoordinateSpace("rgb", frame, frame.source_size, PixelSpaceRole.RGB)


def _request() -> FramePerceptionRequest:
    frame = _frame()
    space = _space(frame)
    return FramePerceptionRequest(
        frame,
        np.zeros((60, 80, 3), dtype=np.uint8),
        space,
        (PerceptionTask.PROMPTED_SEGMENTATION,),
    )


def _prompt(request: FramePerceptionRequest) -> SegmentationPrompt:
    return SegmentationPrompt(
        "point",
        SegmentationPromptKind.POINTS,
        request.rgb_space,
        positive_points_xy=((30.0, 20.0),),
    )


def _mask(mask_id: str, *, revision: int = 0, parent: str = "") -> InstanceMask:
    frame = _frame()
    space = _space(frame)
    data = np.zeros((60, 80), dtype=bool)
    data[10:40, 15:55] = True
    return InstanceMask(
        mask_id,
        frame,
        space,
        DenseMaskPayload(data),
        MaskSource.PROMPTED_REFINEMENT,
        ModelProvenance("sam2", "SAM2", "prompted_segmentation"),
        revision=revision,
        parent_mask_id=parent,
        foreground_valid=True,
    )


def test_9_2e_new_instance_prompt_is_explicit_and_parentless() -> None:
    request = _request()
    value = PromptedSegmentationRequest(
        request,
        _prompt(request),
        creates_new_instance=True,
    )
    assert value.creates_new_instance is True
    assert value.parent_detection_id == ""
    assert value.parent_mask_id == ""

    raw = RawPromptedMask(
        request.frame_reference,
        "",
        "",
        np.ones((60, 80), dtype=bool),
        ModelProvenance("sam2", "SAM2", "prompted_segmentation"),
        None,
        ConfidenceKind.NONE,
        creates_new_instance=True,
    )
    assert raw.creates_new_instance is True


def test_9_2e_refinement_still_requires_real_parent() -> None:
    request = _request()
    with pytest.raises(ContractValidationError, match="parent"):
        PromptedSegmentationRequest(request, _prompt(request))
    with pytest.raises(ContractValidationError, match="must not claim"):
        PromptedSegmentationRequest(
            request,
            _prompt(request),
            parent_mask_id="mask-0",
            creates_new_instance=True,
        )


def test_9_2e_append_interactive_proposal_adds_no_semantic_authority() -> None:
    frame = _frame()
    from core.perception.semantic_workspace import SemanticClassCatalog

    workspace = SemanticWorkspace("ws", frame, SemanticClassCatalog(), ())
    updated = append_workspace_mask_proposal(workspace, _mask("interactive:r0"), object_id="interactive:1")
    obj = updated.object_by_id("interactive:1")
    assert obj is not None
    assert obj.class_id is None
    assert obj.review_status is SemanticReviewStatus.PENDING
    assert obj.is_geometry_authority is False
    assert obj.is_metric_authority is False
    assert obj.is_measurement_authority is False


def test_9_2e_refined_mask_preserves_identity_and_returns_to_pending() -> None:
    frame = _frame()
    from core.perception.semantic_workspace import (
        SemanticClassCatalog,
        SemanticClassDefinition,
        SemanticClassOrigin,
        SemanticLabelSource,
        SemanticWorkspaceObject,
    )

    catalog = SemanticClassCatalog().with_class(
        SemanticClassDefinition(4, "robot_arm", origin=SemanticClassOrigin.CUSTOM)
    )
    original = _mask("mask-r0")
    obj = SemanticWorkspaceObject(
        object_id="proposal:0",
        proposal_mask=original,
        class_id=4,
        class_name="robot_arm",
        instance_name="robot_arm_001",
        label_source=SemanticLabelSource.USER_ASSIGNED,
        review_status=SemanticReviewStatus.ACCEPTED,
    )
    workspace = SemanticWorkspace("ws", frame, catalog, (obj,))
    refined = _mask("mask-r1", revision=1, parent="mask-r0")
    updated = replace_workspace_object_mask(workspace, "proposal:0", refined)
    value = updated.object_by_id("proposal:0")
    assert value is not None
    assert value.class_id == 4
    assert value.class_name == "robot_arm"
    assert value.instance_name == "robot_arm_001"
    assert value.proposal_mask is refined
    assert value.review_status is SemanticReviewStatus.PENDING
