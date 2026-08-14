from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from core.alignment.calibration import (
    rectify_bgr_with_camera_calibration,
    rectify_boolean_mask_with_applied_calibration,
)
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
    assign_workspace_class,
    merge_workspace_objects,
    set_workspace_review_status,
)
from core.perception.reviewed_analysis import (
    load_reviewed_semantic_selection,
    perception_from_reviewed_selection,
    reviewed_selection_from_workspace,
    save_reviewed_semantic_selection,
    source_image_sha256,
)
from core.reconstruction.calibration_profile import (
    CameraCalibrationMethod,
    CameraCalibrationProfile,
    CameraCalibrationValidationState,
)
from core.reconstruction.camera import CameraDistortionModel, ReconstructionCamera
from core.contracts import AxisConvention, CalibrationStatus, CameraIntrinsics, CoordinateFrame, CoordinateFrameKind
from core.models.metadata import IntrinsicsSource


def _workspace(image: np.ndarray) -> SemanticWorkspace:
    h, w = image.shape[:2]
    frame = FrameReference(91, 7, 0.0, ImageSize(w, h))
    space = PixelCoordinateSpace("rgb", frame, frame.source_size, PixelSpaceRole.RGB)
    catalog, plant = SemanticClassCatalog().with_custom_class("plant")
    catalog, table = catalog.with_custom_class("table")

    def mask(mask_id: str, y0: int, y1: int, x0: int, x1: int) -> InstanceMask:
        dense = np.zeros((h, w), dtype=bool)
        dense[y0:y1, x0:x1] = True
        return InstanceMask(
            mask_id=mask_id,
            frame_reference=frame,
            pixel_space=space,
            payload=DenseMaskPayload(dense),
            source=MaskSource.NATIVE_INSTANCE,
            model_provenance=ModelProvenance("sam2", "SAM2", "automatic_segmentation"),
        )

    workspace = SemanticWorkspace(
        "reviewed-ws",
        frame,
        catalog,
        (
            SemanticWorkspaceObject("proposal:plant", proposal_mask=mask("plant-mask", 4, 18, 3, 13)),
            SemanticWorkspaceObject("proposal:pot", proposal_mask=mask("pot-mask", 16, 26, 5, 15)),
            SemanticWorkspaceObject("proposal:table", proposal_mask=mask("table-mask", 8, 25, 22, 38)),
        ),
    )
    workspace = assign_workspace_class(workspace, "proposal:plant", plant, instance_name="plant_001")
    workspace, merged_id = merge_workspace_objects(
        workspace,
        ("proposal:plant", "proposal:pot"),
        primary_object_id="proposal:plant",
    )
    workspace = set_workspace_review_status(workspace, merged_id, SemanticReviewStatus.ACCEPTED)
    workspace = assign_workspace_class(workspace, "proposal:table", table, instance_name="table_001")
    return set_workspace_review_status(workspace, "proposal:table", SemanticReviewStatus.ACCEPTED)


def test_9_2g_reviewed_selection_round_trip_preserves_instances_and_merge_provenance(tmp_path: Path) -> None:
    image = np.zeros((30, 40, 3), dtype=np.uint8)
    selection = reviewed_selection_from_workspace(_workspace(image), image)
    assert len(selection.objects) == 2
    assert selection.source_image_sha256 == source_image_sha256(image)
    plant = next(item for item in selection.objects if item.class_name == "plant")
    assert plant.instance_name == "plant_001"
    assert plant.mask_source is MaskSource.SEMANTIC_MERGE
    assert plant.merged_from_mask_ids == ("plant-mask", "pot-mask")
    metadata_path, arrays_path = save_reviewed_semantic_selection(selection, tmp_path)
    assert metadata_path.is_file() and arrays_path.is_file()
    loaded = load_reviewed_semantic_selection(metadata_path)
    assert loaded.workspace_id == selection.workspace_id
    assert loaded.workspace_revision == selection.workspace_revision
    assert [item.instance_name for item in loaded.objects] == [item.instance_name for item in selection.objects]
    assert np.array_equal(loaded.objects[0].mask, selection.objects[0].mask)
    assert loaded.is_geometry_authority is False
    assert loaded.is_metric_authority is False
    assert loaded.is_measurement_authority is False


def test_9_2g_selection_fails_closed_for_different_source_pixels() -> None:
    image = np.zeros((30, 40, 3), dtype=np.uint8)
    selection = reviewed_selection_from_workspace(_workspace(image), image)
    changed = image.copy()
    changed[0, 0] = (1, 2, 3)
    assert selection.source_image_sha256 != source_image_sha256(changed)


def test_9_2g_reviewed_selection_normalizes_into_existing_perception_contract() -> None:
    image = np.zeros((30, 40, 3), dtype=np.uint8)
    selection = reviewed_selection_from_workspace(_workspace(image), image)
    frame = FrameReference(1, 1, 0.0, ImageSize(40, 30))
    space = PixelCoordinateSpace("rgb", frame, frame.source_size, PixelSpaceRole.RGB)
    result, metadata = perception_from_reviewed_selection(selection, frame, space)
    assert len(result.observations) == 2
    assert result.model_provenance[0].model_id == "reviewed-auto-segment"
    assert {value.class_name for value in result.observations} == {"plant", "table"}
    assert metadata[result.observations[0].observation_id]["semantic_review_status"] == "accepted"
    assert all(value.selected_mask is not None for value in result.observations)


def _distorted_profile() -> CameraCalibrationProfile:
    size = ImageSize(40, 30)
    return CameraCalibrationProfile(
        profile_id="phase9-2g-calibration",
        camera=ReconstructionCamera(
            camera_id="camera",
            image_size=size,
            intrinsics=CameraIntrinsics(
                np.array([[35.0, 0.0, 20.0], [0.0, 35.0, 15.0], [0.0, 0.0, 1.0]]),
                "rgb-rectified",
                IntrinsicsSource.CALIBRATED,
                CalibrationStatus.CALIBRATED,
                calibration_record_id="phase9-2g-record",
            ),
            camera_frame=CoordinateFrame(
                "camera-frame",
                CoordinateFrameKind.CAMERA_GENERIC,
                AxisConvention.X_RIGHT_Y_DOWN_Z_FORWARD,
                parent_frame_id="scene",
            ),
            distortion_model=CameraDistortionModel.BROWN_CONRADY,
            distortion_coefficients=(-0.08, 0.01, 0.0, 0.0, 0.0),
        ),
        method=CameraCalibrationMethod.MANUAL,
        validation_state=CameraCalibrationValidationState.UNVALIDATED,
        provenance="phase9-2g-test",
        created_at_utc="2026-08-14T12:00:00+00:00",
    )


def test_9_2g_reviewed_mask_uses_same_calibration_rectification_as_rgb() -> None:
    image = np.zeros((30, 40, 3), dtype=np.uint8)
    image[5:25, 8:32] = (10, 60, 180)
    dense = np.zeros((30, 40), dtype=bool)
    dense[5:25, 8:32] = True
    prepared = rectify_bgr_with_camera_calibration(image, _distorted_profile())
    rectified = rectify_boolean_mask_with_applied_calibration(dense, prepared.application)
    assert rectified.shape == dense.shape
    assert rectified.dtype == np.bool_
    assert np.count_nonzero(rectified) > 0
    assert np.all(np.logical_not(rectified) | prepared.support_mask)


def test_9_2g_reviewed_selection_rejects_empty_acceptance_set() -> None:
    image = np.zeros((30, 40, 3), dtype=np.uint8)
    workspace = _workspace(image)
    pending = replace(
        workspace,
        objects=tuple(replace(item, review_status=SemanticReviewStatus.PENDING) for item in workspace.objects),
    )
    with pytest.raises(ContractValidationError, match="at least one accepted"):
        reviewed_selection_from_workspace(pending, image)
