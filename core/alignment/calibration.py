"""Calibration-aware image rectification for pinhole measurement geometry.

Phase 9.1G keeps object projection on the existing pinhole contract.  A real
camera calibration may include lens distortion, so captured RGB is rectified
onto an ideal pinhole grid *before* perception/depth and the calibrated
intrinsics are then transformed into the native depth grid.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.contracts import ImageSize
from core.contracts.validation import ContractValidationError, readonly_array
from core.reconstruction.calibration_application import (
    AppliedCameraCalibration,
    apply_camera_calibration_profile,
)
from core.reconstruction.calibration_profile import CameraCalibrationProfile
from core.reconstruction.camera import CameraDistortionModel


@dataclass(frozen=True, slots=True)
class RectifiedCalibrationFrame:
    """One calibrated image represented on an ideal pinhole pixel grid."""

    image_bgr: np.ndarray
    support_mask: np.ndarray
    application: AppliedCameraCalibration
    rectified: bool
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        image = readonly_array(self.image_bgr, field_name="image_bgr", dtype=np.uint8, ndim=3)
        if image.shape[2] != 3:
            raise ContractValidationError("image_bgr must have shape HxWx3.")
        expected = (self.application.target_image_size.height, self.application.target_image_size.width)
        if image.shape[:2] != expected:
            raise ContractValidationError("Rectified image size must match the applied calibration target.")
        support = readonly_array(
            self.support_mask,
            field_name="support_mask",
            dtype=np.bool_,
            shape=expected,
        )
        object.__setattr__(self, "image_bgr", image)
        object.__setattr__(self, "support_mask", support)
        object.__setattr__(self, "rectified", bool(self.rectified))
        object.__setattr__(self, "warnings", tuple(str(value).strip() for value in self.warnings if str(value).strip()))



def _rectification_maps(application: AppliedCameraCalibration) -> tuple[np.ndarray, np.ndarray] | None:
    """Return OpenCV maps for the applied calibration, or None for ideal pinhole input."""

    camera = application.camera
    model = camera.distortion_model
    if model is CameraDistortionModel.NONE:
        return None
    K = np.asarray(camera.intrinsics.matrix, dtype=np.float64)
    D = np.asarray(camera.distortion_coefficients, dtype=np.float64)
    size = (camera.image_size.width, camera.image_size.height)
    if model is CameraDistortionModel.BROWN_CONRADY:
        return cv2.initUndistortRectifyMap(K, D, None, K, size, cv2.CV_32FC1)
    if model is CameraDistortionModel.OPENCV_FISHEYE:
        return cv2.fisheye.initUndistortRectifyMap(
            K,
            D.reshape(-1, 1),
            np.eye(3, dtype=np.float64),
            K,
            size,
            cv2.CV_32FC1,
        )
    raise ContractValidationError(
        f"Unsupported calibration distortion model for measurement rectification: {model.value}."
    )


def rectify_boolean_mask_with_applied_calibration(
    mask: np.ndarray,
    application: AppliedCameraCalibration,
) -> np.ndarray:
    """Rectify one source-grid semantic mask onto the same ideal grid as calibrated RGB."""

    expected = (application.target_image_size.height, application.target_image_size.width)
    dense = np.asarray(mask, dtype=np.bool_)
    if dense.shape != expected:
        raise ContractValidationError(
            f"Calibration mask shape {dense.shape} does not match applied target {expected}."
        )
    maps = _rectification_maps(application)
    if maps is None:
        output = np.array(dense, dtype=np.bool_, copy=True)
        output.setflags(write=False)
        return output
    map_x, map_y = maps
    rectified = cv2.remap(
        dense.astype(np.uint8),
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.bool_)
    rectified.setflags(write=False)
    return rectified

def rectify_bgr_with_camera_calibration(
    image_bgr: np.ndarray,
    profile: CameraCalibrationProfile,
    *,
    allow_same_aspect_full_frame_resize: bool = True,
    target_pixel_space_id: str = "rgb-rectified",
) -> RectifiedCalibrationFrame:
    """Safely apply ``profile`` and rectify BGR onto its ideal pinhole grid.

    The output keeps the target size and calibrated K. Distortion is removed
    from image content, not from the stored immutable calibration evidence.
    Unsupported/backend-specific distortion fails closed in the profile layer.
    """

    image = np.ascontiguousarray(image_bgr, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ContractValidationError("Calibration rectification requires HxWx3 BGR input.")
    target_size = ImageSize(width=int(image.shape[1]), height=int(image.shape[0]))
    application = apply_camera_calibration_profile(
        profile,
        target_size,
        allow_same_aspect_full_frame_resize=allow_same_aspect_full_frame_resize,
        target_pixel_space_id=target_pixel_space_id,
    )
    camera = application.camera
    model = camera.distortion_model
    if model is CameraDistortionModel.NONE:
        support = np.ones(image.shape[:2], dtype=np.bool_)
        return RectifiedCalibrationFrame(
            image_bgr=image,
            support_mask=support,
            application=application,
            rectified=False,
            warnings=application.warnings,
        )

    maps = _rectification_maps(application)
    assert maps is not None
    map_x, map_y = maps

    rectified = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    support_u8 = cv2.remap(
        np.ones(image.shape[:2], dtype=np.uint8),
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warnings = (*application.warnings, "camera_distortion_rectified_for_measurement")
    return RectifiedCalibrationFrame(
        image_bgr=rectified,
        support_mask=support_u8.astype(np.bool_),
        application=application,
        rectified=True,
        warnings=warnings,
    )
