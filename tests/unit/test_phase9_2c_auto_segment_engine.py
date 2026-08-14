from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from core.contracts import FrameReference, ImageSize, PixelCoordinateSpace, PixelSpaceRole
from core.contracts.common import ModelProvenance
from core.contracts.validation import ContractValidationError
from core.models.metadata import ConfidenceKind
from core.perception import (
    AdapterCapability,
    AutoSegmentEngine,
    AutoSegmentGenerationStatus,
    AutoSegmentPolicy,
    AutomaticSegmentationAdapter,
    PerceptionCapabilities,
    PerceptionDescriptor,
    PerceptionTask,
    RawAutomaticMaskBatch,
    RawAutomaticMaskProposal,
    SegmentationProject,
    add_project_class,
    assign_workspace_class,
    create_segmentation_project,
    set_workspace_review_status,
    accepted_observations,
    SemanticReviewStatus,
)
from core.perception.messages import FramePerceptionRequest
from core.perception.registry import PERCEPTION_MODEL_REGISTRY


def _request(width: int = 20, height: int = 16) -> FramePerceptionRequest:
    frame = FrameReference(9, 3, 0.2, ImageSize(width, height))
    space = PixelCoordinateSpace("rgb", frame, frame.source_size, PixelSpaceRole.RGB)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    return FramePerceptionRequest(
        frame,
        rgb,
        space,
        (PerceptionTask.AUTOMATIC_SEGMENTATION,),
    )


class StaticAutoAdapter:
    def __init__(self, batch: RawAutomaticMaskBatch):
        self.batch = batch

    def propose_masks(self, request: FramePerceptionRequest) -> RawAutomaticMaskBatch:
        return self.batch


def _descriptor() -> PerceptionDescriptor:
    return PerceptionDescriptor(
        "auto-test",
        "Auto Test",
        "tests.unit.test_phase9_2c_auto_segment_engine:StaticAutoAdapter",
        PerceptionCapabilities(
            (PerceptionTask.AUTOMATIC_SEGMENTATION,),
            (AdapterCapability.AUTOMATIC_SEGMENTATION,),
        ),
    )


def _batch(request: FramePerceptionRequest, masks: tuple[np.ndarray, ...]) -> RawAutomaticMaskBatch:
    provenance = ModelProvenance("auto-test", "Auto Test", "automatic_segmentation")
    return RawAutomaticMaskBatch(
        request.frame_reference,
        request.rgb_space.space_id,
        tuple(RawAutomaticMaskProposal(f"p-{index}", mask) for index, mask in enumerate(masks)),
        provenance,
    )


def test_2c_builtin_sam_and_mock_declare_automatic_segmentation_capability() -> None:
    sam = PERCEPTION_MODEL_REGISTRY.get("ultralytics-sam2-prompted")
    mock = PERCEPTION_MODEL_REGISTRY.get("development-perception-mock")
    assert PerceptionTask.AUTOMATIC_SEGMENTATION in sam.capabilities.tasks
    assert AdapterCapability.AUTOMATIC_SEGMENTATION in sam.capabilities.adapter_capabilities
    assert PerceptionTask.AUTOMATIC_SEGMENTATION in mock.capabilities.tasks
    assert isinstance(PERCEPTION_MODEL_REGISTRY.build("development-perception-mock"), AutomaticSegmentationAdapter)


def test_2c_auto_segment_filters_small_exact_duplicates_and_truncates_deterministically() -> None:
    request = _request()
    large = np.zeros((16, 20), dtype=bool)
    large[1:14, 1:12] = True
    second = np.zeros((16, 20), dtype=bool)
    second[4:15, 12:19] = True
    third = np.zeros((16, 20), dtype=bool)
    third[1:8, 13:19] = True
    tiny = np.zeros((16, 20), dtype=bool)
    tiny[0:2, 0:2] = True
    batch = _batch(request, (second, large, large.copy(), tiny, third))
    engine = AutoSegmentEngine(
        StaticAutoAdapter(batch),
        _descriptor(),
        policy=AutoSegmentPolicy(minimum_mask_pixels=8, max_proposals=2),
    )
    result = engine.generate(request, workspace_id="ws")

    assert result.status is AutoSegmentGenerationStatus.PARTIAL
    assert result.generated_proposals == 5
    assert result.kept_proposals == 2
    assert result.dropped_small == 1
    assert result.dropped_duplicates == 1
    assert result.dropped_over_limit == 1
    assert len(result.workspace.objects) == 2
    assert all(item.class_id is None for item in result.workspace.objects)
    # Largest proposal is deterministically first after post-processing.
    assert np.count_nonzero(result.workspace.objects[0].proposal_mask.payload.data) == np.count_nonzero(large)


def test_2c_project_catalog_is_available_but_auto_proposals_remain_unlabeled() -> None:
    request = _request()
    mask = np.zeros((16, 20), dtype=bool)
    mask[2:14, 4:16] = True
    project = create_segmentation_project("Factory", project_id="factory")
    project, robot = add_project_class(project, "robot_arm")
    engine = AutoSegmentEngine(
        StaticAutoAdapter(_batch(request, (mask,))),
        _descriptor(),
        policy=AutoSegmentPolicy(minimum_mask_pixels=1),
    )
    result = engine.generate(request, workspace_id="ws", project=project)

    assert result.workspace.class_catalog.by_id(robot.class_id) == robot
    assert result.workspace.objects[0].class_id is None
    assert result.workspace.objects[0].class_name == ""
    assert result.is_geometry_authority is False
    assert result.is_metric_authority is False
    assert result.is_measurement_authority is False


def test_2c_user_can_assign_project_class_accept_and_return_existing_observation_contract() -> None:
    request = _request()
    mask = np.zeros((16, 20), dtype=bool)
    mask[2:14, 4:16] = True
    project = create_segmentation_project("Factory", project_id="factory")
    project, robot = add_project_class(project, "robot_arm")
    engine = AutoSegmentEngine(
        StaticAutoAdapter(_batch(request, (mask,))),
        _descriptor(),
        policy=AutoSegmentPolicy(minimum_mask_pixels=1),
    )
    result = engine.generate(request, workspace_id="ws", project=project)
    obj = result.workspace.objects[0]
    workspace = assign_workspace_class(
        result.workspace,
        obj.object_id,
        robot,
        instance_name=project.suggest_instance_name(robot.class_id),
    )
    workspace = set_workspace_review_status(workspace, obj.object_id, SemanticReviewStatus.ACCEPTED)
    observations = accepted_observations(workspace)

    assert len(observations) == 1
    assert observations[0].class_name == "robot_arm"
    assert observations[0].selected_mask is not None
    assert observations[0].selected_mask.source.value == "native_instance_segmentation"
    assert workspace.objects[0].instance_name == "robot_arm_001"


def test_2c_empty_batch_is_a_valid_empty_workspace_not_a_fake_object() -> None:
    request = _request()
    engine = AutoSegmentEngine(
        StaticAutoAdapter(_batch(request, ())),
        _descriptor(),
        policy=AutoSegmentPolicy(minimum_mask_pixels=1),
    )
    result = engine.generate(request, workspace_id="empty")
    assert result.status is AutoSegmentGenerationStatus.EMPTY
    assert result.workspace.objects == ()
    assert result.kept_proposals == 0


def test_2c_wrong_task_or_pixel_space_fails_closed() -> None:
    request = _request()
    mask = np.ones((16, 20), dtype=bool)
    engine = AutoSegmentEngine(
        StaticAutoAdapter(_batch(request, (mask,))),
        _descriptor(),
        policy=AutoSegmentPolicy(minimum_mask_pixels=1),
    )
    wrong_request = replace(request, requested_tasks=(PerceptionTask.INSTANCE_SEGMENTATION,))
    with pytest.raises(ContractValidationError, match="AUTOMATIC_SEGMENTATION"):
        engine.generate(wrong_request, workspace_id="wrong")

    wrong_batch = replace(_batch(request, (mask,)), pixel_space_id="other")
    wrong_engine = AutoSegmentEngine(
        StaticAutoAdapter(wrong_batch),
        _descriptor(),
        policy=AutoSegmentPolicy(minimum_mask_pixels=1),
    )
    with pytest.raises(ContractValidationError, match="pixel space"):
        wrong_engine.generate(request, workspace_id="wrong-space")


def test_2c_raw_proposal_confidence_contract_remains_explicit() -> None:
    mask = np.ones((4, 4), dtype=bool)
    with pytest.raises(ContractValidationError, match="confidence must be None"):
        RawAutomaticMaskProposal("p", mask, 0.9, ConfidenceKind.NONE)
