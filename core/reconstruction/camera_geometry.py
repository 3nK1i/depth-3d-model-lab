"""Camera-geometry profiles for provisional and calibrated reconstruction.

Phase 8.6A introduces a stable camera-model abstraction without introducing
metric scale.  Estimated intrinsics are explicitly provisional and remain
``NOT_CALIBRATED``; calibrated profiles retain their calibration record.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from math import isfinite, pi, tan
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from core.contracts import (
    AxisConvention,
    CalibrationStatus,
    CameraIntrinsics,
    CoordinateFrame,
    CoordinateFrameKind,
    ImageSize,
)
from core.contracts.validation import require_nonempty
from core.models.metadata import IntrinsicsSource
from core.reconstruction.camera import CameraDistortionModel, ReconstructionCamera
from core.reconstruction.errors import ReconstructionValidationError

CAMERA_GEOMETRY_SCHEMA_VERSION = 1


class CameraGeometryMode(str, Enum):
    ESTIMATED = "estimated"
    CALIBRATED = "calibrated"


class CameraGeometryMethod(str, Enum):
    ASSUMED_HORIZONTAL_FOV = "assumed_horizontal_fov"
    PROVIDED_INTRINSICS = "provided_intrinsics"
    CALIBRATION_RECORD = "calibration_record"


class CameraGeometryPurpose(str, Enum):
    VISUAL_TRACKING = "visual_tracking"
    RELATIVE_RECONSTRUCTION = "relative_reconstruction"
    CALIBRATED_RAY_GEOMETRY = "calibrated_ray_geometry"


class CameraIntrinsicsPolicy(str, Enum):
    """Explicit policy for consumers that may accept provisional intrinsics.

    The default remains calibrated-only.  ``ALLOW_ESTIMATED`` is intentionally
    opt-in so existing production live-SLAM callers preserve Phase 8.4/8.5
    behavior unless they explicitly request an experimental/provisional path.
    """

    CALIBRATED_ONLY = "calibrated_only"
    ALLOW_ESTIMATED = "allow_estimated"


@dataclass(frozen=True, slots=True)
class CameraGeometryAssessment:
    purpose: CameraGeometryPurpose
    allowed: bool
    provisional: bool
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.purpose, CameraGeometryPurpose):
            raise TypeError("purpose must be CameraGeometryPurpose.")
        if not isinstance(self.allowed, bool) or not isinstance(self.provisional, bool):
            raise TypeError("allowed and provisional must be bool values.")
        object.__setattr__(self, "message", str(self.message).strip())

    def as_dict(self) -> dict[str, Any]:
        return {
            "purpose": self.purpose.value,
            "allowed": self.allowed,
            "provisional": self.provisional,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class CameraGeometryProfile:
    profile_id: str
    camera: ReconstructionCamera
    mode: CameraGeometryMode
    method: CameraGeometryMethod
    horizontal_fov_degrees: float | None = None
    focal_length_relative_uncertainty: float | None = None
    distortion_assumed_zero: bool = False
    notes: str = ""
    schema_version: int = CAMERA_GEOMETRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "profile_id", require_nonempty(self.profile_id, field_name="profile_id")
        )
        if not isinstance(self.camera, ReconstructionCamera):
            raise TypeError("camera must be ReconstructionCamera.")
        if not isinstance(self.mode, CameraGeometryMode):
            raise TypeError("mode must be CameraGeometryMode.")
        if not isinstance(self.method, CameraGeometryMethod):
            raise TypeError("method must be CameraGeometryMethod.")
        if int(self.schema_version) != CAMERA_GEOMETRY_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported camera-geometry schema version.")

        fov = self.horizontal_fov_degrees
        if fov is not None:
            fov = float(fov)
            if not isfinite(fov) or not (1.0 <= fov < 179.0):
                raise ReconstructionValidationError(
                    "horizontal_fov_degrees must be finite and in [1, 179)."
                )
            object.__setattr__(self, "horizontal_fov_degrees", fov)

        uncertainty = self.focal_length_relative_uncertainty
        if uncertainty is not None:
            uncertainty = float(uncertainty)
            if not isfinite(uncertainty) or uncertainty < 0.0 or uncertainty >= 1.0:
                raise ReconstructionValidationError(
                    "focal_length_relative_uncertainty must be finite and in [0, 1)."
                )
            object.__setattr__(self, "focal_length_relative_uncertainty", uncertainty)

        intrinsics = self.camera.intrinsics
        if self.mode is CameraGeometryMode.ESTIMATED:
            if intrinsics.source is not IntrinsicsSource.ESTIMATED:
                raise ReconstructionValidationError(
                    "Estimated camera profiles require IntrinsicsSource.ESTIMATED."
                )
            if intrinsics.calibration_status is not CalibrationStatus.NOT_CALIBRATED:
                raise ReconstructionValidationError(
                    "Estimated camera profiles must remain NOT_CALIBRATED."
                )
            if self.method not in {
                CameraGeometryMethod.ASSUMED_HORIZONTAL_FOV,
                CameraGeometryMethod.PROVIDED_INTRINSICS,
            }:
                raise ReconstructionValidationError(
                    "Estimated camera profiles require an estimated-intrinsics method."
                )
            if self.method is CameraGeometryMethod.ASSUMED_HORIZONTAL_FOV and fov is None:
                raise ReconstructionValidationError(
                    "Assumed-FOV camera profiles require horizontal_fov_degrees."
                )
            if uncertainty is None:
                raise ReconstructionValidationError(
                    "Estimated camera profiles require explicit focal-length uncertainty."
                )
        else:
            if intrinsics.source is not IntrinsicsSource.CALIBRATED:
                raise ReconstructionValidationError(
                    "Calibrated camera profiles require IntrinsicsSource.CALIBRATED."
                )
            if intrinsics.calibration_status is not CalibrationStatus.CALIBRATED:
                raise ReconstructionValidationError(
                    "Calibrated camera profiles require calibrated intrinsics."
                )
            if self.method is not CameraGeometryMethod.CALIBRATION_RECORD:
                raise ReconstructionValidationError(
                    "Calibrated camera profiles require CALIBRATION_RECORD method."
                )
            if not intrinsics.calibration_record_id:
                raise ReconstructionValidationError(
                    "Calibrated camera profiles require a calibration record ID."
                )
            if self.distortion_assumed_zero:
                raise ReconstructionValidationError(
                    "Calibrated camera profiles cannot mark distortion as assumed zero."
                )

        object.__setattr__(self, "distortion_assumed_zero", bool(self.distortion_assumed_zero))
        object.__setattr__(self, "notes", str(self.notes).strip())
        object.__setattr__(self, "schema_version", CAMERA_GEOMETRY_SCHEMA_VERSION)

    @property
    def is_provisional(self) -> bool:
        return self.mode is CameraGeometryMode.ESTIMATED

    @property
    def is_calibrated(self) -> bool:
        return self.mode is CameraGeometryMode.CALIBRATED

    def assess(self, purpose: CameraGeometryPurpose) -> CameraGeometryAssessment:
        if not isinstance(purpose, CameraGeometryPurpose):
            raise TypeError("purpose must be CameraGeometryPurpose.")
        if self.is_calibrated:
            return CameraGeometryAssessment(
                purpose=purpose,
                allowed=True,
                provisional=False,
                message=(
                    "Camera intrinsics are calibrated. Metric scale still requires a separate "
                    "scale/reference source."
                ),
            )
        if purpose in {
            CameraGeometryPurpose.VISUAL_TRACKING,
            CameraGeometryPurpose.RELATIVE_RECONSTRUCTION,
        }:
            return CameraGeometryAssessment(
                purpose=purpose,
                allowed=True,
                provisional=True,
                message=(
                    "Estimated intrinsics are allowed only for provisional tracking/relative "
                    "reconstruction; metric accuracy is not established."
                ),
            )
        return CameraGeometryAssessment(
            purpose=purpose,
            allowed=False,
            provisional=True,
            message="Calibrated ray geometry requires an explicitly calibrated camera profile.",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "mode": self.mode.value,
            "method": self.method.value,
            "camera": self.camera.as_dict(),
            "horizontal_fov_degrees": self.horizontal_fov_degrees,
            "focal_length_relative_uncertainty": self.focal_length_relative_uncertainty,
            "distortion_assumed_zero": self.distortion_assumed_zero,
            "notes": self.notes,
        }


def estimate_camera_geometry_from_horizontal_fov(
    *,
    profile_id: str,
    camera_id: str,
    image_size: ImageSize,
    horizontal_fov_degrees: float = 70.0,
    focal_length_relative_uncertainty: float = 0.15,
    camera_frame_id: str | None = None,
    reconstruction_frame_id: str = "live-reconstruction",
    notes: str = "",
) -> CameraGeometryProfile:
    """Build a provisional pinhole camera from resolution and assumed horizontal FOV.

    The estimate assumes square pixels, centered principal point, and zero lens
    distortion.  These assumptions are recorded explicitly and never labelled
    as calibrated.
    """

    if not isinstance(image_size, ImageSize):
        raise TypeError("image_size must be ImageSize.")
    fov = float(horizontal_fov_degrees)
    if not isfinite(fov) or not (1.0 <= fov < 179.0):
        raise ReconstructionValidationError(
            "horizontal_fov_degrees must be finite and in [1, 179)."
        )
    fx = image_size.width / (2.0 * tan(fov * pi / 360.0))
    fy = fx
    cx = image_size.width / 2.0
    cy = image_size.height / 2.0
    matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    intrinsics = CameraIntrinsics(
        matrix,
        f"estimated-{camera_id}-{image_size.width}x{image_size.height}",
        IntrinsicsSource.ESTIMATED,
        CalibrationStatus.NOT_CALIBRATED,
    )
    frame = CoordinateFrame(
        camera_frame_id or f"{camera_id}-frame",
        CoordinateFrameKind.CAMERA_GENERIC,
        AxisConvention.X_RIGHT_Y_DOWN_Z_FORWARD,
        parent_frame_id=require_nonempty(
            reconstruction_frame_id, field_name="reconstruction_frame_id"
        ),
    )
    camera = ReconstructionCamera(
        camera_id=camera_id,
        image_size=image_size,
        intrinsics=intrinsics,
        camera_frame=frame,
        distortion_model=CameraDistortionModel.NONE,
        distortion_coefficients=(),
    )
    return CameraGeometryProfile(
        profile_id=profile_id,
        camera=camera,
        mode=CameraGeometryMode.ESTIMATED,
        method=CameraGeometryMethod.ASSUMED_HORIZONTAL_FOV,
        horizontal_fov_degrees=fov,
        focal_length_relative_uncertainty=focal_length_relative_uncertainty,
        distortion_assumed_zero=True,
        notes=notes,
    )


def camera_geometry_from_estimated_camera(
    *,
    profile_id: str,
    camera: ReconstructionCamera,
    focal_length_relative_uncertainty: float,
    notes: str = "",
) -> CameraGeometryProfile:
    return CameraGeometryProfile(
        profile_id=profile_id,
        camera=camera,
        mode=CameraGeometryMode.ESTIMATED,
        method=CameraGeometryMethod.PROVIDED_INTRINSICS,
        focal_length_relative_uncertainty=focal_length_relative_uncertainty,
        distortion_assumed_zero=(camera.distortion_model is CameraDistortionModel.NONE),
        notes=notes,
    )


def camera_geometry_from_calibrated_camera(
    *,
    profile_id: str,
    camera: ReconstructionCamera,
    notes: str = "",
) -> CameraGeometryProfile:
    return CameraGeometryProfile(
        profile_id=profile_id,
        camera=camera,
        mode=CameraGeometryMode.CALIBRATED,
        method=CameraGeometryMethod.CALIBRATION_RECORD,
        focal_length_relative_uncertainty=None,
        distortion_assumed_zero=False,
        notes=notes,
    )


def camera_geometry_profile_to_json(
    profile: CameraGeometryProfile,
    *,
    indent: int | None = 2,
) -> str:
    if not isinstance(profile, CameraGeometryProfile):
        raise TypeError("profile must be CameraGeometryProfile.")
    return json.dumps(
        profile.as_dict(),
        indent=indent,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        allow_nan=False,
    )


def camera_geometry_profile_from_json(payload: str | bytes | bytearray) -> CameraGeometryProfile:
    try:
        raw = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        raise ReconstructionValidationError("Camera-geometry profile must be valid JSON.") from exc
    if not isinstance(raw, Mapping):
        raise ReconstructionValidationError("Camera-geometry profile root must be an object.")
    return camera_geometry_profile_from_dict(raw)


def camera_geometry_profile_from_dict(raw: Mapping[str, Any]) -> CameraGeometryProfile:
    try:
        if int(raw["schema_version"]) != CAMERA_GEOMETRY_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported camera-geometry schema version.")
        camera = _camera_from_dict(_mapping(raw["camera"], "camera"))
        return CameraGeometryProfile(
            profile_id=str(raw["profile_id"]),
            camera=camera,
            mode=CameraGeometryMode(str(raw["mode"])),
            method=CameraGeometryMethod(str(raw["method"])),
            horizontal_fov_degrees=(
                None
                if raw.get("horizontal_fov_degrees") is None
                else float(raw["horizontal_fov_degrees"])
            ),
            focal_length_relative_uncertainty=(
                None
                if raw.get("focal_length_relative_uncertainty") is None
                else float(raw["focal_length_relative_uncertainty"])
            ),
            distortion_assumed_zero=bool(raw.get("distortion_assumed_zero", False)),
            notes=str(raw.get("notes", "")),
        )
    except KeyError as exc:
        raise ReconstructionValidationError(
            f"Camera-geometry profile is missing required field: {exc.args[0]}."
        ) from exc
    except ReconstructionValidationError:
        raise
    except (TypeError, ValueError) as exc:
        raise ReconstructionValidationError(
            f"Camera-geometry profile contains an invalid value: {exc}."
        ) from exc


def load_camera_geometry_profile(path: str | Path) -> CameraGeometryProfile:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ReconstructionValidationError(f"Camera-geometry profile does not exist: {source}")
    try:
        return camera_geometry_profile_from_json(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ReconstructionValidationError(
            f"Could not read camera-geometry profile: {source}"
        ) from exc


def save_camera_geometry_profile(profile: CameraGeometryProfile, path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.write_text(camera_geometry_profile_to_json(profile) + "\n", encoding="utf-8")
    except OSError as exc:
        raise ReconstructionValidationError(
            f"Could not write camera-geometry profile: {destination}"
        ) from exc
    return destination


def _camera_from_dict(raw: Mapping[str, Any]) -> ReconstructionCamera:
    image_raw = _mapping(raw["image_size"], "image_size")
    intrinsics_raw = _mapping(raw["intrinsics"], "intrinsics")
    frame_raw = _mapping(raw["camera_frame"], "camera_frame")
    matrix = np.asarray(intrinsics_raw["matrix"], dtype=np.float64)
    intrinsics = CameraIntrinsics(
        matrix,
        str(intrinsics_raw["pixel_space_id"]),
        IntrinsicsSource(str(intrinsics_raw["source"])),
        CalibrationStatus(str(intrinsics_raw["calibration_status"])),
        calibration_record_id=str(intrinsics_raw.get("calibration_record_id", "")),
    )
    frame = CoordinateFrame(
        str(frame_raw["frame_id"]),
        CoordinateFrameKind(str(frame_raw["kind"])),
        AxisConvention(str(frame_raw["axis_convention"])),
        parent_frame_id=str(frame_raw.get("parent_frame_id", "")),
    )
    return ReconstructionCamera(
        camera_id=str(raw["camera_id"]),
        image_size=ImageSize(int(image_raw["width"]), int(image_raw["height"])),
        intrinsics=intrinsics,
        camera_frame=frame,
        distortion_model=CameraDistortionModel(str(raw.get("distortion_model", "none"))),
        distortion_coefficients=tuple(float(value) for value in raw.get("distortion_coefficients", ())),
        backend_distortion_name=str(raw.get("backend_distortion_name", "")),
    )


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"{field_name} must be an object.")
    return value
