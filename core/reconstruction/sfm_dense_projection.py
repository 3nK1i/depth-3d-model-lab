"""Project one SfM-aligned dense RGB-D keyframe into reconstruction coordinates.

Phase C is deliberately a single-frame bridge. It converts an offline-SfM
camera model into the existing :class:`ReconstructionCamera` contract, binds an
accepted B1/B2 sparse-correspondence depth alignment, and reuses the dense
projection kernel to produce XYZRGB points in the global SfM reconstruction
frame.

No multi-view fusion, persistence, physical metric scale, measurement authority,
or certified metrology is introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.contracts import (
    AxisConvention,
    CalibrationStatus,
    CameraIntrinsics,
    CoordinateFrame,
    CoordinateFrameKind,
)
from core.contracts.validation import require_nonempty
from core.models.metadata import IntrinsicsSource
from core.reconstruction.camera import CameraDistortionModel, ReconstructionCamera
from core.reconstruction.dense_fusion import (
    DENSE_FUSION_POSE_CONVENTION,
    DenseDepthUnit,
    DenseFrameProjection,
    DenseFusionFrame,
    project_dense_frame_to_reconstruction,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.sfm_contracts import SfmCameraEstimate, SfmCameraModelEstimate
from core.reconstruction.sfm_dense_alignment import SfmDenseDepthAlignmentEvidence
from core.reconstruction.validation import require_nonnegative_int

SFM_DENSE_PROJECTION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class SfmDenseKeyframeProjection:
    """One canonical XYZRGB cloud derived from one registered SfM keyframe."""

    session_id: str
    keyframe_id: str
    frame_id: str
    camera_id: str
    camera_model_name: str
    reconstruction_frame_id: str
    depth_provenance: str
    timestamp_ns: int
    pixel_stride: int
    camera: ReconstructionCamera
    projection: DenseFrameProjection
    sfm_scale_status: str
    sfm_world_alignment_status: str
    pose_convention: str = DENSE_FUSION_POSE_CONVENTION
    metric_scale_claimed: bool = False
    canonical_geometry_authoritative: bool = False
    measurement_authoritative: bool = False
    certified_metrology: bool = False
    schema_version: int = SFM_DENSE_PROJECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "session_id",
            "keyframe_id",
            "frame_id",
            "camera_id",
            "camera_model_name",
            "reconstruction_frame_id",
            "depth_provenance",
            "sfm_scale_status",
            "sfm_world_alignment_status",
            "pose_convention",
        ):
            object.__setattr__(
                self, name, require_nonempty(getattr(self, name), field_name=name)
            )
        object.__setattr__(
            self,
            "timestamp_ns",
            require_nonnegative_int(self.timestamp_ns, field_name="timestamp_ns"),
        )
        if isinstance(self.pixel_stride, bool) or int(self.pixel_stride) <= 0:
            raise ReconstructionValidationError("pixel_stride must be a positive integer.")
        object.__setattr__(self, "pixel_stride", int(self.pixel_stride))
        if not isinstance(self.camera, ReconstructionCamera):
            raise TypeError("camera must be ReconstructionCamera.")
        if not isinstance(self.projection, DenseFrameProjection):
            raise TypeError("projection must be DenseFrameProjection.")
        for name in (
            "metric_scale_claimed",
            "canonical_geometry_authoritative",
            "measurement_authoritative",
            "certified_metrology",
        ):
            if bool(getattr(self, name)):
                raise ReconstructionValidationError(
                    f"Phase C validation projection cannot set {name}=True."
                )
            object.__setattr__(self, name, False)
        if (
            isinstance(self.schema_version, bool)
            or int(self.schema_version) != SFM_DENSE_PROJECTION_SCHEMA_VERSION
        ):
            raise ReconstructionValidationError(
                f"Unsupported SfM/dense projection schema version: {self.schema_version}."
            )
        object.__setattr__(self, "schema_version", SFM_DENSE_PROJECTION_SCHEMA_VERSION)

    @property
    def points_reconstruction(self) -> np.ndarray:
        return self.projection.points_reconstruction

    @property
    def colors_rgb(self) -> np.ndarray:
        return self.projection.colors_rgb

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "keyframe_id": self.keyframe_id,
            "frame_id": self.frame_id,
            "camera_id": self.camera_id,
            "camera_model_name": self.camera_model_name,
            "reconstruction_frame_id": self.reconstruction_frame_id,
            "timestamp_ns": self.timestamp_ns,
            "pixel_stride": self.pixel_stride,
            "pose_convention": self.pose_convention,
            "sfm_scale_status": self.sfm_scale_status,
            "sfm_world_alignment_status": self.sfm_world_alignment_status,
            "depth_provenance": self.depth_provenance,
            "camera": self.camera.as_dict(),
            "projection": self.projection.as_dict(),
            "metric_scale_claimed": False,
            "canonical_geometry_authoritative": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
        }


def reconstruction_camera_from_sfm_model(
    camera_model: SfmCameraModelEstimate,
    *,
    keyframe_id: str,
    reconstruction_frame_id: str = "reconstruction",
) -> ReconstructionCamera:
    """Translate supported COLMAP camera models without upgrading calibration authority."""

    if not isinstance(camera_model, SfmCameraModelEstimate):
        raise TypeError("camera_model must be SfmCameraModelEstimate.")
    keyframe = require_nonempty(keyframe_id, field_name="keyframe_id")
    reconstruction_frame = require_nonempty(
        reconstruction_frame_id, field_name="reconstruction_frame_id"
    )
    name = camera_model.backend_model_name.strip().upper()
    params = tuple(float(value) for value in camera_model.parameters)

    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model = CameraDistortionModel.NONE
    distortion_coefficients: tuple[float, ...] = ()

    if name == "SIMPLE_PINHOLE" and len(params) == 3:
        f, cx, cy = params
        fx = fy = f
    elif name == "PINHOLE" and len(params) == 4:
        fx, fy, cx, cy = params
    elif name == "SIMPLE_RADIAL" and len(params) == 4:
        f, cx, cy, k1 = params
        fx = fy = f
        distortion_model = CameraDistortionModel.BROWN_CONRADY
        distortion_coefficients = (k1, 0.0, 0.0, 0.0, 0.0)
    elif name == "RADIAL" and len(params) == 5:
        f, cx, cy, k1, k2 = params
        fx = fy = f
        distortion_model = CameraDistortionModel.BROWN_CONRADY
        distortion_coefficients = (k1, k2, 0.0, 0.0, 0.0)
    elif name == "OPENCV" and len(params) == 8:
        fx, fy, cx, cy, k1, k2, p1, p2 = params
        distortion_model = CameraDistortionModel.BROWN_CONRADY
        distortion_coefficients = (k1, k2, p1, p2)
    elif name == "FULL_OPENCV" and len(params) == 12:
        fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 = params
        distortion_model = CameraDistortionModel.BROWN_CONRADY
        distortion_coefficients = (k1, k2, p1, p2, k3, k4, k5, k6)
    elif name == "OPENCV_FISHEYE" and len(params) == 8:
        fx, fy, cx, cy, k1, k2, k3, k4 = params
        distortion_model = CameraDistortionModel.OPENCV_FISHEYE
        distortion_coefficients = (k1, k2, k3, k4)
    else:
        raise ReconstructionValidationError(
            "Phase C does not silently reinterpret unsupported SfM camera model "
            f"{name!r} with {len(params)} parameters."
        )

    matrix = np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    intrinsics = CameraIntrinsics(
        matrix=matrix,
        pixel_space_id=(
            f"sfm-{camera_model.camera_id}-{camera_model.image_size.width}x"
            f"{camera_model.image_size.height}"
        ),
        source=IntrinsicsSource.ESTIMATED,
        calibration_status=CalibrationStatus.NOT_CALIBRATED,
    )
    frame = CoordinateFrame(
        frame_id=f"{camera_model.camera_id}:{keyframe}:sfm-camera",
        kind=CoordinateFrameKind.CAMERA_RGB,
        axis_convention=AxisConvention.X_RIGHT_Y_DOWN_Z_FORWARD,
        parent_frame_id=reconstruction_frame,
    )
    return ReconstructionCamera(
        camera_id=camera_model.camera_id,
        image_size=camera_model.image_size,
        intrinsics=intrinsics,
        camera_frame=frame,
        distortion_model=distortion_model,
        distortion_coefficients=distortion_coefficients,
    )


def project_sfm_keyframe_dense_rgb(
    camera_estimate: SfmCameraEstimate,
    camera_model: SfmCameraModelEstimate,
    *,
    timestamp_ns: int,
    rgb: np.ndarray,
    dense_depth_source_units: np.ndarray,
    alignment_evidence: SfmDenseDepthAlignmentEvidence,
    pixel_stride: int = 1,
    reconstruction_frame_id: str = "reconstruction",
    confidence: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> SfmDenseKeyframeProjection:
    """Create one canonical XYZRGB cloud from an accepted SfM/dense-depth keyframe."""

    if not isinstance(camera_estimate, SfmCameraEstimate):
        raise TypeError("camera_estimate must be SfmCameraEstimate.")
    if not isinstance(camera_model, SfmCameraModelEstimate):
        raise TypeError("camera_model must be SfmCameraModelEstimate.")
    if not isinstance(alignment_evidence, SfmDenseDepthAlignmentEvidence):
        raise TypeError("alignment_evidence must be SfmDenseDepthAlignmentEvidence.")
    if not alignment_evidence.is_accepted or alignment_evidence.alignment is None:
        raise ReconstructionValidationError(
            "Phase C projection requires accepted SfM/dense-depth alignment evidence."
        )
    if camera_estimate.keyframe_id != alignment_evidence.keyframe_id:
        raise ReconstructionValidationError(
            "Camera estimate and alignment evidence must reference the same keyframe."
        )
    if camera_estimate.frame_id != alignment_evidence.frame_id:
        raise ReconstructionValidationError(
            "Camera estimate and alignment evidence must reference the same frame."
        )
    if camera_estimate.camera_id != camera_model.camera_id:
        raise ReconstructionValidationError(
            "SfM camera pose must reference the supplied camera model."
        )
    if alignment_evidence.camera_id and alignment_evidence.camera_id != camera_model.camera_id:
        raise ReconstructionValidationError(
            "Alignment evidence camera_id does not match the supplied camera model."
        )
    if (
        alignment_evidence.camera_model_name
        and alignment_evidence.camera_model_name.strip().upper()
        != camera_model.backend_model_name.strip().upper()
    ):
        raise ReconstructionValidationError(
            "Alignment evidence camera model does not match the supplied SfM camera model."
        )

    rgb_array = np.asarray(rgb)
    depth_array = np.asarray(dense_depth_source_units, dtype=np.float32)
    expected = (camera_model.image_size.height, camera_model.image_size.width)
    if rgb_array.shape != (expected[0], expected[1], 3) or rgb_array.dtype != np.uint8:
        raise ReconstructionValidationError(
            "Phase C RGB must be uint8 HxWx3 on the SfM camera-model pixel grid."
        )
    if depth_array.shape != expected:
        raise ReconstructionValidationError(
            "Phase C dense depth must match the SfM camera-model pixel grid."
        )
    if alignment_evidence.depth_image_size != camera_model.image_size:
        raise ReconstructionValidationError(
            "Alignment evidence pixel grid must match the SfM camera model."
        )

    camera = reconstruction_camera_from_sfm_model(
        camera_model,
        keyframe_id=camera_estimate.keyframe_id,
        reconstruction_frame_id=reconstruction_frame_id,
    )
    frame = DenseFusionFrame(
        frame_id=camera_estimate.frame_id,
        timestamp_ns=timestamp_ns,
        rgb=rgb_array,
        depth=depth_array,
        T_reconstruction_from_camera=camera_estimate.T_reconstruction_from_camera,
        depth_unit=DenseDepthUnit.SOURCE_MODEL_UNITS,
        depth_alignment=alignment_evidence.alignment,
        confidence=confidence,
        valid_mask=valid_mask,
        pose_layer="offline_sfm_bundle_adjusted",
    )
    projection = project_dense_frame_to_reconstruction(
        frame,
        camera,
        pixel_stride=pixel_stride,
        minimum_confidence=0.0,
    )
    if projection.valid_depth_samples <= 0:
        raise ReconstructionValidationError(
            "Phase C projection produced no finite positive canonical points."
        )

    return SfmDenseKeyframeProjection(
        session_id=alignment_evidence.session_id,
        keyframe_id=camera_estimate.keyframe_id,
        frame_id=camera_estimate.frame_id,
        camera_id=camera_estimate.camera_id,
        camera_model_name=camera_model.backend_model_name,
        reconstruction_frame_id=reconstruction_frame_id,
        depth_provenance=alignment_evidence.depth_provenance,
        timestamp_ns=timestamp_ns,
        pixel_stride=pixel_stride,
        camera=camera,
        projection=projection,
        sfm_scale_status=alignment_evidence.sfm_scale_status.value,
        sfm_world_alignment_status=alignment_evidence.sfm_world_alignment_status.value,
    )
