from __future__ import annotations

import numpy as np
import pytest

from core.alignment.contracts import AlignmentRequest, AlignmentStatus, DepthAlignmentInput
from core.alignment.factories import declare_full_frame_resize
from core.alignment.pipeline import align_frame
from core.contracts.common import FrameReference, ImageSize, ModelProvenance
from core.contracts.spatial import (
    CalibrationStatus,
    CameraIntrinsics,
    PixelCoordinateSpace,
    PixelSpaceRole,
)
from core.contracts.validation import ContractValidationError
from core.models.metadata import ConfidenceKind, DepthScale, IntrinsicsSource
from core.perception.result import (
    ApprovalStatus,
    DenseMaskPayload,
    Detection2D,
    InstanceMask,
    MaskSource,
    ObjectObservation,
    PerceptionFrameResult,
    normalize_detection_box,
)


def _request(*, destination_size=(4, 3), observations=None, confidence=False, source_size=(8, 6)):
    frame = FrameReference(4, 8, 1.25, ImageSize(*source_size))
    rgb_space = PixelCoordinateSpace("rgb", frame, ImageSize(*source_size), PixelSpaceRole.RGB)
    depth_space = PixelCoordinateSpace("depth", frame, ImageSize(*destination_size), PixelSpaceRole.DEPTH)
    depth = np.arange(destination_size[0] * destination_size[1], dtype=np.float32).reshape(destination_size[1], destination_size[0]) + 1
    depth[0, 0] = np.nan
    confidence_values = np.full(depth.shape, 0.8, np.float32) if confidence else None
    confidence_kind = ConfidenceKind.CONFIDENCE if confidence else ConfidenceKind.NONE
    intrinsics = CameraIntrinsics(
        np.array([[4.0, 0.0, destination_size[0] / 2], [0.0, 4.0, destination_size[1] / 2], [0.0, 0.0, 1.0]]),
        "depth",
        IntrinsicsSource.ESTIMATED,
        CalibrationStatus.NOT_CALIBRATED,
    )
    depth_provenance = ModelProvenance("depth", "Depth", "depth")
    depth_input = DepthAlignmentInput(
        frame,
        depth_space,
        depth,
        np.ones(depth.shape, bool),
        confidence_values,
        confidence_kind,
        intrinsics,
        DepthScale.METRIC,
        depth_provenance,
    )
    perception_provenance = ModelProvenance("seg", "Segmentation", "instance_segmentation")
    perception = PerceptionFrameResult(frame, tuple(observations or ()), (perception_provenance,))
    transform = declare_full_frame_resize(rgb_space, depth_space)
    return AlignmentRequest(
        frame,
        np.full((source_size[1], source_size[0], 3), 100, np.uint8),
        rgb_space,
        depth_input,
        perception,
        (transform,),
    ), rgb_space, perception_provenance


def _observation(frame, rgb_space, provenance, observation_id, mask, *, box=(1.0, 1.0, 7.0, 5.0), masks=None):
    detection = Detection2D(
        f"{observation_id}:det",
        frame,
        0,
        "object",
        0.9,
        normalize_detection_box(box, rgb_space),
        provenance,
    )
    native = InstanceMask(
        f"{observation_id}:mask",
        frame,
        rgb_space,
        DenseMaskPayload(mask),
        MaskSource.NATIVE_INSTANCE,
        provenance,
        confidence=0.9,
        confidence_kind=ConfidenceKind.CONFIDENCE,
        foreground_valid=bool(np.any(mask)),
    )
    all_masks = tuple(masks) if masks is not None else (native,)
    return ObjectObservation(observation_id, frame, 0, "object", detection, all_masks)


def test_pipeline_keeps_native_depth_and_intrinsics_and_aligns_rgb_mask():
    base, rgb_space, provenance = _request()
    mask = np.zeros((6, 8), bool)
    mask[1:5, 2:7] = True
    observation = _observation(base.frame_reference, rgb_space, provenance, "obj", mask)
    request, _, _ = _request(observations=(observation,), confidence=True)
    result = align_frame(request)
    assert result.status is AlignmentStatus.SUCCESS
    assert np.array_equal(result.depth, request.depth_input.depth, equal_nan=True)
    assert result.intrinsics.matrix.tolist() == request.depth_input.intrinsics.matrix.tolist()
    assert result.aligned_rgb.shape == (3, 4, 3)
    assert result.aligned_observations[0].aligned_mask.shape == (3, 4)
    assert result.depth_valid_mask[0, 0] == 0
    assert result.confidence_valid_mask.all()


def test_pipeline_preserves_overlapping_instance_masks():
    base, rgb_space, provenance = _request()
    first = np.zeros((6, 8), bool); first[1:5, 1:5] = True
    second = np.zeros((6, 8), bool); second[2:6, 3:7] = True
    observations = (
        _observation(base.frame_reference, rgb_space, provenance, "a", first),
        _observation(base.frame_reference, rgb_space, provenance, "b", second),
    )
    request, _, _ = _request(observations=observations)
    result = align_frame(request)
    assert np.any(
        result.aligned_observations[0].aligned_mask
        & result.aligned_observations[1].aligned_mask
    )


def test_approved_manual_mask_remains_authoritative_after_alignment():
    base, rgb_space, provenance = _request()
    native_data = np.zeros((6, 8), bool); native_data[1:3, 1:3] = True
    manual_data = np.zeros((6, 8), bool); manual_data[3:6, 4:8] = True
    native = InstanceMask(
        "native", base.frame_reference, rgb_space, DenseMaskPayload(native_data),
        MaskSource.NATIVE_INSTANCE, provenance, confidence=0.7,
        confidence_kind=ConfidenceKind.CONFIDENCE, foreground_valid=True,
    )
    manual = InstanceMask(
        "manual", base.frame_reference, rgb_space, DenseMaskPayload(manual_data),
        MaskSource.MANUAL_CORRECTION, ModelProvenance("user", "Manual", "manual"),
        revision=1, parent_mask_id="native", approval_status=ApprovalStatus.APPROVED,
        approved_by="reviewer", approved_timestamp=1.0, foreground_valid=True,
    )
    observation = _observation(
        base.frame_reference, rgb_space, provenance, "obj", native_data, masks=(native, manual)
    )
    request, _, _ = _request(observations=(observation,))
    result = align_frame(request)
    aligned = result.aligned_observations[0]
    assert aligned.source_mask_id == "manual"
    assert aligned.mask_source is MaskSource.MANUAL_CORRECTION
    assert aligned.approval_status is ApprovalStatus.APPROVED


def test_missing_selected_mask_becomes_partial_not_fabricated():
    base, rgb_space, provenance = _request()
    detection = Detection2D(
        "det", base.frame_reference, 0, "object", 0.8,
        normalize_detection_box((1, 1, 4, 4), rgb_space), provenance,
    )
    observation = ObjectObservation("obj", base.frame_reference, 0, "object", detection, ())
    request, _, _ = _request(observations=(observation,))
    result = align_frame(request)
    aligned = result.aligned_observations[0]
    assert result.status is AlignmentStatus.PARTIAL
    assert aligned.aligned_mask is None
    assert "selected_mask_unavailable" in aligned.warnings
    assert aligned.aligned_detection_box is not None


def test_empty_transformed_mask_is_retained_as_partial_warning():
    frame = FrameReference(5, 1, 0.0, ImageSize(8, 6))
    rgb_space = PixelCoordinateSpace("rgb", frame, ImageSize(8, 6), PixelSpaceRole.RGB)
    depth_space = PixelCoordinateSpace("depth", frame, ImageSize(4, 3), PixelSpaceRole.DEPTH)
    provenance = ModelProvenance("seg", "Seg", "seg")
    mask = np.zeros((6, 8), bool); mask[0, 0] = True
    observation = _observation(frame, rgb_space, provenance, "obj", mask, box=(0, 0, 1, 1))
    intrinsics = CameraIntrinsics(np.array([[4,0,2],[0,4,1.5],[0,0,1]], float), "depth", IntrinsicsSource.ESTIMATED, CalibrationStatus.NOT_CALIBRATED)
    depth_input = DepthAlignmentInput(frame, depth_space, np.ones((3,4), np.float32), None, None, ConfidenceKind.NONE, intrinsics, DepthScale.METRIC, ModelProvenance("d","D","depth"))
    # Crop away the source top-left pixel while still producing the depth grid.
    from core.alignment.factories import declare_crop_resize_padding
    transform = declare_crop_resize_padding(rgb_space, depth_space, crop_xyxy=(2, 2, 8, 6), resized_size=ImageSize(4, 3), padding_ltrb=(0,0,0,0))
    request = AlignmentRequest(frame, np.zeros((6,8,3), np.uint8), rgb_space, depth_input, PerceptionFrameResult(frame,(observation,),(provenance,)), (transform,))
    result = align_frame(request)
    assert result.status is AlignmentStatus.PARTIAL
    assert "aligned_mask_empty" in result.aligned_observations[0].warnings


def test_nonuniform_resize_is_explicitly_reported():
    request, rgb_space, provenance = _request(destination_size=(4, 4), source_size=(8, 6))
    mask = np.ones((6, 8), bool)
    observation = _observation(request.frame_reference, rgb_space, provenance, "obj", mask)
    request, _, _ = _request(destination_size=(4, 4), observations=(observation,), source_size=(8, 6))
    result = align_frame(request)
    assert "nonuniform_rgb_resize" in result.warnings
    assert "nonuniform_mask_resize" in result.aligned_observations[0].warnings


def test_pipeline_rejects_missing_explicit_rgb_transform():
    request, _, _ = _request()
    request = AlignmentRequest(
        request.frame_reference,
        request.rgb,
        request.rgb_space,
        request.depth_input,
        request.perception_result,
        (),
    )
    with pytest.raises(ContractValidationError):
        align_frame(request)


def test_schema_ready_non_dense_mask_is_reported_as_partial():
    from core.perception.result import RLEMaskPayload

    base, rgb_space, provenance = _request()
    rle = InstanceMask(
        "rle",
        base.frame_reference,
        rgb_space,
        RLEMaskPayload(8, 6, (10, 5, 33)),
        MaskSource.NATIVE_INSTANCE,
        provenance,
        confidence=0.8,
        confidence_kind=ConfidenceKind.CONFIDENCE,
        foreground_valid=True,
    )
    detection = Detection2D(
        "det-rle",
        base.frame_reference,
        0,
        "object",
        0.8,
        normalize_detection_box((1, 1, 7, 5), rgb_space),
        provenance,
    )
    observation = ObjectObservation(
        "obj-rle", base.frame_reference, 0, "object", detection, (rle,)
    )
    request, _, _ = _request(observations=(observation,))
    result = align_frame(request)
    aligned = result.aligned_observations[0]
    assert result.status is AlignmentStatus.PARTIAL
    assert aligned.aligned_mask is None
    assert any(value.startswith("mask_representation_not_normalized") for value in aligned.warnings)
