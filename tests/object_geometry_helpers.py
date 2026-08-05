from __future__ import annotations

import numpy as np

from core.alignment.contracts import (
    AlignedFrameBundle,
    AlignedObjectObservation,
    AlignmentStatus,
)
from core.contracts.common import FrameReference, ImageSize, ModelProvenance
from core.contracts.spatial import (
    CalibrationStatus,
    CameraIntrinsics,
    PixelCoordinateSpace,
    PixelSpaceRole,
)
from core.models.metadata import ConfidenceKind, DepthScale, IntrinsicsSource
from core.perception.result import ApprovalStatus, MaskSource


def make_bundle(
    *,
    width: int = 48,
    height: int = 36,
    depth_scale: DepthScale = DepthScale.METRIC,
    confidence_kind: ConfidenceKind = ConfidenceKind.NONE,
    confidence: np.ndarray | None = None,
    masks: tuple[np.ndarray, ...] | None = None,
    depth: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    rgb_support: np.ndarray | None = None,
    border_contacts: tuple[tuple[str, ...], ...] | None = None,
    manual: bool = False,
) -> AlignedFrameBundle:
    frame = FrameReference(3, 7, 1.25, ImageSize(width, height))
    space = PixelCoordinateSpace("depth", frame, frame.source_size, PixelSpaceRole.DEPTH)
    if depth is None:
        yy, xx = np.mgrid[0:height, 0:width]
        depth = (1.0 + xx * 0.001 + yy * 0.001).astype(np.float32)
    else:
        depth = np.asarray(depth, dtype=np.float32)
    valid = np.ones((height, width), bool) if valid_mask is None else np.asarray(valid_mask, bool)
    support = np.ones((height, width), bool) if rgb_support is None else np.asarray(rgb_support, bool)
    if confidence_kind is ConfidenceKind.NONE:
        confidence = None
    elif confidence is None:
        confidence = np.full((height, width), 0.9 if confidence_kind is ConfidenceKind.CONFIDENCE else 0.1, np.float32)
    confidence_valid = np.ones((height, width), bool)
    if confidence is not None:
        confidence_valid = np.isfinite(confidence)
    if masks is None:
        mask = np.zeros((height, width), bool)
        mask[8:28, 12:36] = True
        masks = (mask,)
    contacts = border_contacts or tuple(() for _ in masks)
    observations = []
    for index, (mask, contact) in enumerate(zip(masks, contacts, strict=True)):
        observations.append(
            AlignedObjectObservation(
                observation_id=f"object-{index}",
                class_id=index,
                class_name=f"object-{index}",
                target_space=space,
                source_mask_id=f"mask-{index}",
                source_mask_space_id="rgb",
                aligned_mask=np.asarray(mask, bool),
                source_foreground_pixels=int(np.count_nonzero(mask)),
                aligned_foreground_pixels=int(np.count_nonzero(mask)),
                foreground_retention_ratio=1.0 if np.any(mask) else 0.0,
                border_contacts=contact,
                mask_source=MaskSource.MANUAL_CORRECTION if manual else MaskSource.NATIVE_INSTANCE,
                mask_revision=1 if manual else 0,
                approval_status=ApprovalStatus.APPROVED if manual else ApprovalStatus.NOT_REQUIRED,
                mask_model_provenance=ModelProvenance("mask", "Mask", "segmentation"),
            )
        )
    intrinsics = CameraIntrinsics(
        np.array([[50.0, 0.0, width / 2.0], [0.0, 50.0, height / 2.0], [0.0, 0.0, 1.0]]),
        space.space_id,
        IntrinsicsSource.ESTIMATED,
        CalibrationStatus.NOT_CALIBRATED,
    )
    return AlignedFrameBundle(
        frame_reference=frame,
        target_depth_space=space,
        aligned_rgb=np.full((height, width, 3), (20, 80, 160), np.uint8),
        depth=depth,
        depth_valid_mask=valid,
        rgb_support_mask=support,
        confidence=confidence,
        confidence_valid_mask=confidence_valid,
        confidence_kind=confidence_kind,
        intrinsics=intrinsics,
        depth_scale=depth_scale,
        point_stride=1,
        depth_model_provenance=ModelProvenance("depth", "Depth", "depth"),
        perception_model_provenance=(ModelProvenance("seg", "Seg", "segmentation"),),
        aligned_observations=tuple(observations),
        transforms=(),
        status=AlignmentStatus.SUCCESS,
    )
