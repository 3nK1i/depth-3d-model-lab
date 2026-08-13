"""Safe application of camera-calibration profiles across image resolutions.

Phase 9.1B keeps calibration records immutable at their calibration resolution.
Consumers receive an explicit application result containing a camera model in
one target pixel space plus auditable evidence describing how it was derived.

Only exact-resolution reuse and same-aspect-ratio full-frame resize are
qualified here. Crops, padding, digital zoom, stabilisation crops, changed
field-of-view camera modes, and other unknown transforms remain rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isclose
from typing import Any

from core.alignment.factories import declare_full_frame_resize, declare_identity_transform
from core.alignment.intrinsics import transform_intrinsics
from core.alignment.transforms import is_full_frame_resize, is_nonuniform_scale
from core.contracts import FrameReference, ImageSize, PixelCoordinateSpace, PixelSpaceRole
from core.contracts.spatial import ImageTransform2D
from core.contracts.validation import require_nonempty
from core.reconstruction.calibration_profile import CameraCalibrationProfile
from core.reconstruction.camera import ReconstructionCamera
from core.reconstruction.errors import ReconstructionValidationError


class CalibrationApplicationMode(str, Enum):
    """Qualified way a calibration profile was mapped to a target image grid."""

    EXACT_RESOLUTION = "exact_resolution"
    SAME_ASPECT_FULL_FRAME_RESIZE = "same_aspect_full_frame_resize"


@dataclass(frozen=True, slots=True)
class AppliedCameraCalibration:
    """One explicit, auditable application of an immutable calibration profile."""

    profile_id: str
    calibration_record_id: str
    calibration_image_size: ImageSize
    target_image_size: ImageSize
    camera: ReconstructionCamera
    mode: CalibrationApplicationMode
    pixel_transform: ImageTransform2D
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_id", require_nonempty(self.profile_id, field_name="profile_id"))
        object.__setattr__(
            self,
            "calibration_record_id",
            require_nonempty(self.calibration_record_id, field_name="calibration_record_id"),
        )
        if not isinstance(self.calibration_image_size, ImageSize):
            raise TypeError("calibration_image_size must be an ImageSize.")
        if not isinstance(self.target_image_size, ImageSize):
            raise TypeError("target_image_size must be an ImageSize.")
        if not isinstance(self.camera, ReconstructionCamera):
            raise TypeError("camera must be a ReconstructionCamera.")
        if not isinstance(self.mode, CalibrationApplicationMode):
            raise TypeError("mode must be a CalibrationApplicationMode.")
        if not isinstance(self.pixel_transform, ImageTransform2D):
            raise TypeError("pixel_transform must be an ImageTransform2D.")
        if self.camera.image_size != self.target_image_size:
            raise ReconstructionValidationError(
                "Applied calibration camera size must match target_image_size."
            )
        if self.camera.intrinsics.calibration_record_id != self.calibration_record_id:
            raise ReconstructionValidationError(
                "Applied calibration must preserve the calibration record ID."
            )
        object.__setattr__(
            self,
            "warnings",
            tuple(require_nonempty(value, field_name="warning") for value in self.warnings),
        )

    @property
    def was_rescaled(self) -> bool:
        return self.mode is CalibrationApplicationMode.SAME_ASPECT_FULL_FRAME_RESIZE

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "calibration_record_id": self.calibration_record_id,
            "calibration_image_size": self.calibration_image_size.as_dict(),
            "target_image_size": self.target_image_size.as_dict(),
            "mode": self.mode.value,
            "was_rescaled": self.was_rescaled,
            "pixel_transform": {
                "source_space": {
                    "space_id": self.pixel_transform.source_space.space_id,
                    "size": self.pixel_transform.source_space.size.as_dict(),
                },
                "destination_space": {
                    "space_id": self.pixel_transform.destination_space.space_id,
                    "size": self.pixel_transform.destination_space.size.as_dict(),
                },
                "matrix": self.pixel_transform.matrix.tolist(),
                "crop_xyxy": (
                    list(self.pixel_transform.crop_xyxy)
                    if self.pixel_transform.crop_xyxy is not None
                    else None
                ),
                "resized_size": (
                    self.pixel_transform.resized_size.as_dict()
                    if self.pixel_transform.resized_size is not None
                    else None
                ),
                "padding_ltrb": list(self.pixel_transform.padding_ltrb),
                "interpolation": self.pixel_transform.interpolation,
            },
            "warnings": list(self.warnings),
            "camera": self.camera.as_dict(),
        }


def _same_aspect_ratio(source: ImageSize, target: ImageSize, *, tolerance: float = 1e-9) -> bool:
    return isclose(
        source.width / source.height,
        target.width / target.height,
        rel_tol=tolerance,
        abs_tol=tolerance,
    )


def _application_spaces(
    profile: CameraCalibrationProfile,
    target_image_size: ImageSize,
    *,
    target_pixel_space_id: str,
) -> tuple[PixelCoordinateSpace, PixelCoordinateSpace]:
    source_size = profile.calibration_image_size
    frame = FrameReference(
        session_id=0,
        frame_id=0,
        source_timestamp=0.0,
        source_size=source_size,
    )
    source = PixelCoordinateSpace(
        space_id=profile.camera.intrinsics.pixel_space_id,
        frame_reference=frame,
        size=source_size,
        role=PixelSpaceRole.RGB,
    )
    destination = PixelCoordinateSpace(
        space_id=target_pixel_space_id,
        frame_reference=frame,
        size=target_image_size,
        role=PixelSpaceRole.RGB,
    )
    return source, destination


def apply_camera_calibration_profile(
    profile: CameraCalibrationProfile,
    target_image_size: ImageSize,
    *,
    allow_same_aspect_full_frame_resize: bool = False,
    target_pixel_space_id: str | None = None,
) -> AppliedCameraCalibration:
    """Apply ``profile`` to exactly one target image grid.

    Resolution changes are fail-closed. A different target size is accepted only
    when the caller explicitly enables same-aspect full-frame resize. This API
    does not infer crop, padding, digital zoom, stabilisation, sensor ROI/binning,
    or field-of-view changes from dimensions alone.
    """

    if not isinstance(profile, CameraCalibrationProfile):
        raise TypeError("profile must be a CameraCalibrationProfile.")
    if not isinstance(target_image_size, ImageSize):
        raise TypeError("target_image_size must be an ImageSize.")

    source_size = profile.calibration_image_size
    target_space_id = (
        require_nonempty(target_pixel_space_id, field_name="target_pixel_space_id")
        if target_pixel_space_id is not None
        else (
            profile.camera.intrinsics.pixel_space_id
            if target_image_size == source_size
            else f"{profile.camera.intrinsics.pixel_space_id}-full-frame-"
            f"{target_image_size.width}x{target_image_size.height}"
        )
    )
    source_space, destination_space = _application_spaces(
        profile,
        target_image_size,
        target_pixel_space_id=target_space_id,
    )

    if target_image_size == source_size:
        transform = declare_identity_transform(source_space, destination_space)
        intrinsics = transform_intrinsics(profile.camera.intrinsics, transform)
        camera = ReconstructionCamera(
            camera_id=profile.camera.camera_id,
            image_size=target_image_size,
            intrinsics=intrinsics,
            camera_frame=profile.camera.camera_frame,
            distortion_model=profile.camera.distortion_model,
            distortion_coefficients=profile.camera.distortion_coefficients,
            backend_distortion_name=profile.camera.backend_distortion_name,
        )
        return AppliedCameraCalibration(
            profile_id=profile.profile_id,
            calibration_record_id=profile.calibration_record_id,
            calibration_image_size=source_size,
            target_image_size=target_image_size,
            camera=camera,
            mode=CalibrationApplicationMode.EXACT_RESOLUTION,
            pixel_transform=transform,
        )

    if not allow_same_aspect_full_frame_resize:
        raise ReconstructionValidationError(
            "Calibration resolution differs from the target resolution. "
            "An explicit qualified resize policy is required."
        )
    if not _same_aspect_ratio(source_size, target_image_size):
        raise ReconstructionValidationError(
            "Calibration rescaling requires the same aspect ratio. "
            "Aspect-ratio changes may indicate crop, padding, zoom, sensor-mode, "
            "stabilisation, or field-of-view changes and are not inferred from dimensions."
        )

    transform = declare_full_frame_resize(source_space, destination_space)
    if not is_full_frame_resize(transform):
        raise ReconstructionValidationError(
            "Calibration rescaling requires an explicit full-frame resize transform."
        )
    if is_nonuniform_scale(transform):
        raise ReconstructionValidationError(
            "Calibration rescaling rejected a non-uniform pixel scale."
        )

    intrinsics = transform_intrinsics(profile.camera.intrinsics, transform)
    camera = ReconstructionCamera(
        camera_id=profile.camera.camera_id,
        image_size=target_image_size,
        intrinsics=intrinsics,
        camera_frame=profile.camera.camera_frame,
        distortion_model=profile.camera.distortion_model,
        distortion_coefficients=profile.camera.distortion_coefficients,
        backend_distortion_name=profile.camera.backend_distortion_name,
    )
    return AppliedCameraCalibration(
        profile_id=profile.profile_id,
        calibration_record_id=profile.calibration_record_id,
        calibration_image_size=source_size,
        target_image_size=target_image_size,
        camera=camera,
        mode=CalibrationApplicationMode.SAME_ASPECT_FULL_FRAME_RESIZE,
        pixel_transform=transform,
        warnings=("calibration_intrinsics_rescaled_from_full_frame_profile",),
    )
