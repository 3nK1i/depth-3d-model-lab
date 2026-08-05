"""Factories for Step 4 requests, camera frames and coordinate semantics."""

from __future__ import annotations

from core.alignment.contracts import AlignedFrameBundle
from core.contracts.spatial import AxisConvention, CoordinateFrame, CoordinateFrameKind
from core.models.metadata import DepthScale
from core.object_geometry.contracts import PointCoordinateSemantics
from core.object_geometry.policy import ObjectPointExtractionPolicy
from core.object_geometry.pipeline import ObjectPointExtractionRequest


def make_depth_camera_frame(bundle: AlignedFrameBundle) -> CoordinateFrame:
    return CoordinateFrame(
        frame_id=f"camera-depth:s{bundle.frame_reference.session_id}:f{bundle.frame_reference.frame_id}",
        kind=CoordinateFrameKind.CAMERA_DEPTH,
        axis_convention=AxisConvention.X_RIGHT_Y_DOWN_Z_FORWARD,
        source_frame=bundle.frame_reference,
    )


def coordinate_semantics_for_depth_scale(depth_scale: DepthScale) -> PointCoordinateSemantics:
    if depth_scale is DepthScale.METRIC:
        return PointCoordinateSemantics.METRIC_METRES
    if depth_scale is DepthScale.RELATIVE:
        return PointCoordinateSemantics.RELATIVE_MODEL_UNITS
    if depth_scale is DepthScale.SYNTHETIC:
        return PointCoordinateSemantics.SYNTHETIC_UNITS
    return PointCoordinateSemantics.UNKNOWN


def make_extraction_request(
    bundle: AlignedFrameBundle,
    *,
    observation_ids: tuple[str, ...] = (),
    policy: ObjectPointExtractionPolicy | None = None,
) -> ObjectPointExtractionRequest:
    return ObjectPointExtractionRequest(
        aligned_bundle=bundle,
        observation_ids=observation_ids,
        policy=policy or ObjectPointExtractionPolicy(),
    )
