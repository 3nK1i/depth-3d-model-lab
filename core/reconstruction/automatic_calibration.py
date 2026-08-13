"""Phase 9.1D automatic checkerboard and ChArUco camera calibration.

This module is deliberately dependency-light and authority-safe. It derives
camera intrinsics/distortion from calibration image sets, attaches explicit
reprojection evidence to ``CameraCalibrationProfile``, and never changes metric
scene authority. ChArUco support is feature-detected from the active OpenCV
build; the production environment is not modified implicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Iterable, Sequence
from uuid import uuid4

import cv2
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
from core.reconstruction.calibration_profile import (
    CalibrationQualityMetrics,
    CameraCalibrationMethod,
    CameraCalibrationProfile,
    CameraCalibrationValidationState,
    calibration_source_binding_from_metadata,
)
from core.reconstruction.camera import CameraDistortionModel, ReconstructionCamera
from core.reconstruction.contracts import InputSourceMetadata
from core.reconstruction.errors import ReconstructionValidationError


class CalibrationFrameDisposition(str, Enum):
    ACCEPTED = "accepted"
    TARGET_NOT_DETECTED = "target_not_detected"
    INSUFFICIENT_TARGET_POINTS = "insufficient_target_points"


@dataclass(frozen=True, slots=True)
class OpenCvCalibrationCapabilities:
    opencv_version: str
    checkerboard: bool
    checkerboard_sb: bool
    charuco: bool
    charuco_detector: bool
    charuco_legacy_fallback: bool
    charuco_missing_features: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "opencv_version": self.opencv_version,
            "checkerboard": self.checkerboard,
            "checkerboard_sb": self.checkerboard_sb,
            "charuco": self.charuco,
            "charuco_detector": self.charuco_detector,
            "charuco_legacy_fallback": self.charuco_legacy_fallback,
            "charuco_missing_features": list(self.charuco_missing_features),
        }


def inspect_opencv_calibration_capabilities() -> OpenCvCalibrationCapabilities:
    """Report only capabilities actually present in the imported ``cv2`` build."""

    checkerboard_required = ("calibrateCamera", "findChessboardCorners", "cornerSubPix", "projectPoints")
    checkerboard = all(hasattr(cv2, name) for name in checkerboard_required)
    aruco = getattr(cv2, "aruco", None)
    missing: list[str] = []
    detector = False
    legacy = False
    charuco = False
    if aruco is None:
        missing.append("cv2.aruco")
    else:
        board_factory = hasattr(aruco, "CharucoBoard") or hasattr(aruco, "CharucoBoard_create")
        dictionary_factory = hasattr(aruco, "getPredefinedDictionary")
        detector = hasattr(aruco, "CharucoDetector")
        legacy = hasattr(aruco, "detectMarkers") and hasattr(aruco, "interpolateCornersCharuco")
        point_matching = board_factory  # verified again on the created board at runtime
        if not board_factory:
            missing.append("cv2.aruco.CharucoBoard")
        if not dictionary_factory:
            missing.append("cv2.aruco.getPredefinedDictionary")
        if not (detector or legacy):
            missing.append("ChArUco detector API")
        charuco = board_factory and dictionary_factory and (detector or legacy) and point_matching
    return OpenCvCalibrationCapabilities(
        opencv_version=str(getattr(cv2, "__version__", "unknown")),
        checkerboard=checkerboard,
        checkerboard_sb=hasattr(cv2, "findChessboardCornersSB"),
        charuco=charuco,
        charuco_detector=detector,
        charuco_legacy_fallback=legacy,
        charuco_missing_features=tuple(missing),
    )


@dataclass(frozen=True, slots=True)
class CheckerboardCalibrationTarget:
    inner_corners_x: int
    inner_corners_y: int
    square_length: float

    def __post_init__(self) -> None:
        for name in ("inner_corners_x", "inner_corners_y"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) < 2:
                raise ReconstructionValidationError(f"{name} must be an integer >= 2.")
            object.__setattr__(self, name, int(value))
        length = float(self.square_length)
        if not isfinite(length) or length <= 0.0:
            raise ReconstructionValidationError("square_length must be finite and positive.")
        object.__setattr__(self, "square_length", length)

    @property
    def point_count(self) -> int:
        return self.inner_corners_x * self.inner_corners_y

    def object_points(self) -> np.ndarray:
        grid = np.zeros((self.point_count, 3), dtype=np.float32)
        coordinates = np.mgrid[0 : self.inner_corners_x, 0 : self.inner_corners_y].T.reshape(-1, 2)
        grid[:, :2] = coordinates.astype(np.float32) * np.float32(self.square_length)
        return grid

    def as_dict(self) -> dict[str, object]:
        return {
            "inner_corners_x": self.inner_corners_x,
            "inner_corners_y": self.inner_corners_y,
            "square_length": self.square_length,
        }


@dataclass(frozen=True, slots=True)
class CharucoCalibrationTarget:
    squares_x: int
    squares_y: int
    square_length: float
    marker_length: float
    dictionary_name: str = "DICT_4X4_50"
    legacy_pattern: bool = False

    def __post_init__(self) -> None:
        for name in ("squares_x", "squares_y"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) < 3:
                raise ReconstructionValidationError(f"{name} must be an integer >= 3.")
            object.__setattr__(self, name, int(value))
        square = float(self.square_length)
        marker = float(self.marker_length)
        if not isfinite(square) or square <= 0.0:
            raise ReconstructionValidationError("square_length must be finite and positive.")
        if not isfinite(marker) or marker <= 0.0 or marker >= square:
            raise ReconstructionValidationError(
                "marker_length must be finite, positive, and smaller than square_length."
            )
        object.__setattr__(self, "square_length", square)
        object.__setattr__(self, "marker_length", marker)
        object.__setattr__(
            self,
            "dictionary_name",
            require_nonempty(self.dictionary_name, field_name="dictionary_name"),
        )
        if not isinstance(self.legacy_pattern, bool):
            raise TypeError("legacy_pattern must be a bool.")

    def as_dict(self) -> dict[str, object]:
        return {
            "squares_x": self.squares_x,
            "squares_y": self.squares_y,
            "square_length": self.square_length,
            "marker_length": self.marker_length,
            "dictionary_name": self.dictionary_name,
            "legacy_pattern": self.legacy_pattern,
        }


@dataclass(frozen=True, slots=True)
class AutomaticCalibrationQualityPolicy:
    """Explicit caller-owned acceptance thresholds; no hidden product constants."""

    minimum_accepted_views: int
    maximum_rms_reprojection_error_px: float
    maximum_mean_reprojection_error_px: float
    maximum_view_reprojection_error_px: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_accepted_views, bool)
            or int(self.minimum_accepted_views) != self.minimum_accepted_views
            or int(self.minimum_accepted_views) < 3
        ):
            raise ReconstructionValidationError("minimum_accepted_views must be an integer >= 3.")
        object.__setattr__(self, "minimum_accepted_views", int(self.minimum_accepted_views))
        for name in (
            "maximum_rms_reprojection_error_px",
            "maximum_mean_reprojection_error_px",
            "maximum_view_reprojection_error_px",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ReconstructionValidationError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)

    def as_dict(self) -> dict[str, object]:
        return {
            "minimum_accepted_views": self.minimum_accepted_views,
            "maximum_rms_reprojection_error_px": self.maximum_rms_reprojection_error_px,
            "maximum_mean_reprojection_error_px": self.maximum_mean_reprojection_error_px,
            "maximum_view_reprojection_error_px": self.maximum_view_reprojection_error_px,
        }


@dataclass(frozen=True, slots=True)
class CalibrationFrameDiagnostic:
    source_name: str
    image_size: ImageSize
    disposition: CalibrationFrameDisposition
    detected_point_count: int
    reprojection_error_px: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_name",
            require_nonempty(self.source_name, field_name="source_name"),
        )
        if not isinstance(self.image_size, ImageSize):
            raise TypeError("image_size must be an ImageSize.")
        if not isinstance(self.disposition, CalibrationFrameDisposition):
            raise TypeError("disposition must be a CalibrationFrameDisposition.")
        count = self.detected_point_count
        if isinstance(count, bool) or int(count) != count or int(count) < 0:
            raise ReconstructionValidationError("detected_point_count must be a non-negative integer.")
        object.__setattr__(self, "detected_point_count", int(count))
        error = self.reprojection_error_px
        if error is not None:
            value = float(error)
            if not isfinite(value) or value < 0.0:
                raise ReconstructionValidationError(
                    "reprojection_error_px must be finite and non-negative when provided."
                )
            object.__setattr__(self, "reprojection_error_px", value)
        if (
            self.disposition is CalibrationFrameDisposition.ACCEPTED
            and self.detected_point_count == 0
        ):
            raise ReconstructionValidationError("Accepted calibration views require detected target points.")
        if (
            self.disposition is not CalibrationFrameDisposition.ACCEPTED
            and self.reprojection_error_px is not None
        ):
            raise ReconstructionValidationError(
                "Rejected calibration views cannot carry reprojection error evidence."
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "source_name": self.source_name,
            "image_size": self.image_size.as_dict(),
            "disposition": self.disposition.value,
            "detected_point_count": self.detected_point_count,
            "reprojection_error_px": self.reprojection_error_px,
        }


@dataclass(frozen=True, slots=True)
class AutomaticCalibrationResult:
    profile: CameraCalibrationProfile
    diagnostics: tuple[CalibrationFrameDiagnostic, ...]
    quality_policy: AutomaticCalibrationQualityPolicy | None
    validation_reasons: tuple[str, ...]
    capabilities: OpenCvCalibrationCapabilities

    def __post_init__(self) -> None:
        if not isinstance(self.profile, CameraCalibrationProfile):
            raise TypeError("profile must be a CameraCalibrationProfile.")
        diagnostics = tuple(self.diagnostics)
        if not diagnostics or any(
            not isinstance(item, CalibrationFrameDiagnostic) for item in diagnostics
        ):
            raise ReconstructionValidationError(
                "diagnostics must contain at least one CalibrationFrameDiagnostic."
            )
        if self.quality_policy is not None and not isinstance(
            self.quality_policy, AutomaticCalibrationQualityPolicy
        ):
            raise TypeError("quality_policy must be AutomaticCalibrationQualityPolicy or None.")
        if not isinstance(self.capabilities, OpenCvCalibrationCapabilities):
            raise TypeError("capabilities must be OpenCvCalibrationCapabilities.")
        reasons = tuple(
            require_nonempty(value, field_name="validation_reason")
            for value in self.validation_reasons
        )
        quality = self.profile.quality
        if quality is None:
            raise ReconstructionValidationError(
                "Automatic calibration results require reprojection quality evidence."
            )
        accepted = sum(
            item.disposition is CalibrationFrameDisposition.ACCEPTED
            for item in diagnostics
        )
        if quality.accepted_view_count != accepted:
            raise ReconstructionValidationError(
                "Profile quality accepted_view_count disagrees with frame diagnostics."
            )
        if quality.rejected_view_count != len(diagnostics) - accepted:
            raise ReconstructionValidationError(
                "Profile quality rejected_view_count disagrees with frame diagnostics."
            )
        if self.profile.validation_state is CameraCalibrationValidationState.VALIDATED and reasons:
            raise ReconstructionValidationError(
                "Validated automatic calibration cannot retain validation rejection reasons."
            )
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "validation_reasons", reasons)

    @property
    def accepted_view_count(self) -> int:
        return sum(item.disposition is CalibrationFrameDisposition.ACCEPTED for item in self.diagnostics)

    @property
    def rejected_view_count(self) -> int:
        return len(self.diagnostics) - self.accepted_view_count

    def as_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile.as_dict(),
            "diagnostics": [item.as_dict() for item in self.diagnostics],
            "quality_policy": self.quality_policy.as_dict() if self.quality_policy else None,
            "validation_reasons": list(self.validation_reasons),
            "capabilities": self.capabilities.as_dict(),
        }


def load_calibration_image_set(paths: Iterable[str | Path]) -> tuple[tuple[str, np.ndarray], ...]:
    """Load a deterministic image set without changing image geometry."""

    loaded: list[tuple[str, np.ndarray]] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise ReconstructionValidationError(f"Calibration image does not exist: {path}")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ReconstructionValidationError(f"Could not decode calibration image: {path}")
        loaded.append((path.name, image))
    if not loaded:
        raise ReconstructionValidationError("Calibration image set must not be empty.")
    return tuple(loaded)


def _normalize_images(
    images: Sequence[np.ndarray | tuple[str, np.ndarray]],
) -> tuple[tuple[str, np.ndarray], ImageSize]:
    if not images:
        raise ReconstructionValidationError("Calibration image set must not be empty.")
    normalized: list[tuple[str, np.ndarray]] = []
    expected_size: ImageSize | None = None
    for index, item in enumerate(images):
        if isinstance(item, tuple):
            if len(item) != 2:
                raise ReconstructionValidationError("Named calibration image entries must be (name, image).")
            name, image = item
        else:
            name, image = f"view-{index:04d}", item
        source_name = require_nonempty(str(name), field_name="source_name")
        array = np.asarray(image)
        if array.dtype != np.uint8 or array.ndim not in {2, 3}:
            raise ReconstructionValidationError(
                "Calibration images must be uint8 grayscale or BGR arrays."
            )
        if array.ndim == 3 and (array.shape[2] != 3):
            raise ReconstructionValidationError("Color calibration images must use three BGR channels.")
        size = ImageSize(width=int(array.shape[1]), height=int(array.shape[0]))
        if expected_size is None:
            expected_size = size
        elif size != expected_size:
            raise ReconstructionValidationError(
                "All automatic-calibration images must have exactly the same resolution."
            )
        normalized.append((source_name, np.ascontiguousarray(array)))
    assert expected_size is not None
    return tuple(normalized), expected_size


def _gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _detect_checkerboard(
    image: np.ndarray,
    target: CheckerboardCalibrationTarget,
) -> np.ndarray | None:
    pattern = (target.inner_corners_x, target.inner_corners_y)
    gray = _gray(image)
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(gray, pattern)
        if found:
            return np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
    found, corners = cv2.findChessboardCorners(
        gray,
        pattern,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        return None
    refined = cv2.cornerSubPix(
        gray,
        np.asarray(corners, dtype=np.float32),
        (11, 11),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3),
    )
    return np.asarray(refined, dtype=np.float32).reshape(-1, 1, 2)


def _create_charuco_board(target: CharucoCalibrationTarget):
    capabilities = inspect_opencv_calibration_capabilities()
    if not capabilities.charuco:
        missing = ", ".join(capabilities.charuco_missing_features) or "unknown ChArUco feature"
        raise ReconstructionValidationError(
            f"ChArUco calibration is unavailable in this OpenCV build: {missing}."
        )
    aruco = cv2.aruco
    if not hasattr(aruco, target.dictionary_name):
        raise ReconstructionValidationError(
            f"OpenCV does not expose ArUco dictionary {target.dictionary_name!r}."
        )
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, target.dictionary_name))
    if hasattr(aruco, "CharucoBoard"):
        board = aruco.CharucoBoard(
            (target.squares_x, target.squares_y),
            target.square_length,
            target.marker_length,
            dictionary,
        )
    else:
        board = aruco.CharucoBoard_create(
            target.squares_x,
            target.squares_y,
            target.square_length,
            target.marker_length,
            dictionary,
        )
    if target.legacy_pattern:
        setter = getattr(board, "setLegacyPattern", None)
        if not callable(setter):
            raise ReconstructionValidationError(
                "This OpenCV build cannot explicitly select the legacy ChArUco pattern."
            )
        setter(True)
    return board, dictionary


def _detect_charuco(image: np.ndarray, board, dictionary) -> tuple[np.ndarray, np.ndarray] | None:
    aruco = cv2.aruco
    if hasattr(aruco, "CharucoDetector"):
        detector = aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, _marker_corners, _marker_ids = detector.detectBoard(image)
    else:
        marker_corners, marker_ids, _rejected = aruco.detectMarkers(image, dictionary)
        if marker_ids is None or len(marker_ids) == 0:
            return None
        count, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            image,
            board,
        )
        if int(count) <= 0:
            return None
    if charuco_ids is None or charuco_corners is None or len(charuco_ids) < 4:
        return None
    return (
        np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 1, 2),
        np.asarray(charuco_ids, dtype=np.int32).reshape(-1, 1),
    )


def _charuco_object_and_image_points(
    board,
    corners: np.ndarray,
    ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matcher = getattr(board, "matchImagePoints", None)
    if callable(matcher):
        object_points, image_points = matcher(corners, ids)
        return (
            np.asarray(object_points, dtype=np.float32).reshape(-1, 3),
            np.asarray(image_points, dtype=np.float32).reshape(-1, 1, 2),
        )
    getter = getattr(board, "getChessboardCorners", None)
    if not callable(getter):
        raise ReconstructionValidationError(
            "This OpenCV ChArUco board cannot expose calibration object points."
        )
    all_points = np.asarray(getter(), dtype=np.float32).reshape(-1, 3)
    flat_ids = ids.reshape(-1)
    if np.any(flat_ids < 0) or np.any(flat_ids >= len(all_points)):
        raise ReconstructionValidationError("ChArUco corner IDs are outside the board definition.")
    return all_points[flat_ids], corners


def _per_view_reprojection_errors(
    object_points: Sequence[np.ndarray],
    image_points: Sequence[np.ndarray],
    rvecs: Sequence[np.ndarray],
    tvecs: Sequence[np.ndarray],
    matrix: np.ndarray,
    distortion: np.ndarray,
) -> tuple[float, ...]:
    errors: list[float] = []
    for obj, img, rvec, tvec in zip(object_points, image_points, rvecs, tvecs, strict=True):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, matrix, distortion)
        delta = np.asarray(img, dtype=np.float64).reshape(-1, 2) - np.asarray(projected, dtype=np.float64).reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))))
    return tuple(errors)


def _evaluate_quality(
    *,
    quality: CalibrationQualityMetrics,
    policy: AutomaticCalibrationQualityPolicy | None,
) -> tuple[CameraCalibrationValidationState, tuple[str, ...]]:
    if policy is None:
        return CameraCalibrationValidationState.UNVALIDATED, ("quality_policy_not_applied",)
    reasons: list[str] = []
    if (quality.accepted_view_count or 0) < policy.minimum_accepted_views:
        reasons.append("insufficient_accepted_views")
    if (
        quality.rms_reprojection_error_px is None
        or quality.rms_reprojection_error_px > policy.maximum_rms_reprojection_error_px
    ):
        reasons.append("rms_reprojection_error_exceeds_policy")
    if (
        quality.mean_reprojection_error_px is None
        or quality.mean_reprojection_error_px > policy.maximum_mean_reprojection_error_px
    ):
        reasons.append("mean_reprojection_error_exceeds_policy")
    if (
        quality.max_reprojection_error_px is None
        or quality.max_reprojection_error_px > policy.maximum_view_reprojection_error_px
    ):
        reasons.append("view_reprojection_error_exceeds_policy")
    return (
        (CameraCalibrationValidationState.VALIDATED, ())
        if not reasons
        else (CameraCalibrationValidationState.REJECTED, tuple(reasons))
    )


def _build_profile(
    *,
    method: CameraCalibrationMethod,
    image_size: ImageSize,
    matrix: np.ndarray,
    distortion: np.ndarray,
    quality: CalibrationQualityMetrics,
    policy: AutomaticCalibrationQualityPolicy | None,
    provenance: str,
    source: InputSourceMetadata | None,
    profile_id: str | None,
    calibration_record_id: str | None,
    camera_id: str | None,
    notes: str,
) -> tuple[CameraCalibrationProfile, tuple[str, ...]]:
    state, reasons = _evaluate_quality(quality=quality, policy=policy)
    resolved_profile_id = require_nonempty(
        profile_id or f"camera-calibration-{uuid4()}", field_name="profile_id"
    )
    record_id = require_nonempty(
        calibration_record_id or f"calibration-record-{uuid4()}",
        field_name="calibration_record_id",
    )
    resolved_camera_id = require_nonempty(
        camera_id or (source.source_id if source is not None else f"camera-{uuid4()}"),
        field_name="camera_id",
    )
    intrinsics = CameraIntrinsics(
        matrix=np.asarray(matrix, dtype=np.float64),
        pixel_space_id=f"calibration-{image_size.width}x{image_size.height}",
        source=IntrinsicsSource.CALIBRATED,
        calibration_status=CalibrationStatus.CALIBRATED,
        calibration_record_id=record_id,
    )
    frame = CoordinateFrame(
        frame_id=f"{resolved_camera_id}-frame",
        kind=CoordinateFrameKind.CAMERA_GENERIC,
        axis_convention=AxisConvention.X_RIGHT_Y_DOWN_Z_FORWARD,
        parent_frame_id="calibration-reconstruction",
    )
    coefficients = tuple(float(value) for value in np.asarray(distortion, dtype=np.float64).reshape(-1))
    camera = ReconstructionCamera(
        camera_id=resolved_camera_id,
        image_size=image_size,
        intrinsics=intrinsics,
        camera_frame=frame,
        distortion_model=(
            CameraDistortionModel.NONE if len(coefficients) == 0 else CameraDistortionModel.BROWN_CONRADY
        ),
        distortion_coefficients=coefficients,
    )
    profile = CameraCalibrationProfile(
        profile_id=resolved_profile_id,
        camera=camera,
        method=method,
        validation_state=state,
        provenance=provenance,
        source_binding=(
            None if source is None else calibration_source_binding_from_metadata(source)
        ),
        quality=quality,
        notes=notes,
    )
    return profile, reasons


def _finish_calibration(
    *,
    method: CameraCalibrationMethod,
    image_size: ImageSize,
    object_points: Sequence[np.ndarray],
    image_points: Sequence[np.ndarray],
    diagnostics: Sequence[CalibrationFrameDiagnostic],
    quality_policy: AutomaticCalibrationQualityPolicy | None,
    provenance: str,
    source: InputSourceMetadata | None,
    profile_id: str | None,
    calibration_record_id: str | None,
    camera_id: str | None,
    notes: str,
) -> AutomaticCalibrationResult:
    if len(object_points) < 3:
        raise ReconstructionValidationError(
            "Automatic camera calibration requires at least three detected target views."
        )
    try:
        rms, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
            list(object_points),
            list(image_points),
            (image_size.width, image_size.height),
            None,
            None,
        )
    except cv2.error as exc:
        raise ReconstructionValidationError(f"OpenCV camera calibration failed: {exc}") from exc
    matrix = np.asarray(matrix, dtype=np.float64)
    distortion = np.asarray(distortion, dtype=np.float64)
    if not np.isfinite(matrix).all() or not np.isfinite(distortion).all() or not isfinite(float(rms)):
        raise ReconstructionValidationError("OpenCV camera calibration returned non-finite values.")
    per_view = _per_view_reprojection_errors(
        object_points, image_points, rvecs, tvecs, matrix, distortion
    )
    quality = CalibrationQualityMetrics(
        rms_reprojection_error_px=float(rms),
        mean_reprojection_error_px=float(np.mean(per_view)),
        max_reprojection_error_px=float(np.max(per_view)),
        accepted_view_count=len(object_points),
        rejected_view_count=len(diagnostics) - len(object_points),
    )
    profile, reasons = _build_profile(
        method=method,
        image_size=image_size,
        matrix=matrix,
        distortion=distortion,
        quality=quality,
        policy=quality_policy,
        provenance=provenance,
        source=source,
        profile_id=profile_id,
        calibration_record_id=calibration_record_id,
        camera_id=camera_id,
        notes=notes,
    )
    accepted_index = 0
    completed: list[CalibrationFrameDiagnostic] = []
    for diagnostic in diagnostics:
        if diagnostic.disposition is CalibrationFrameDisposition.ACCEPTED:
            completed.append(replace(diagnostic, reprojection_error_px=per_view[accepted_index]))
            accepted_index += 1
        else:
            completed.append(diagnostic)
    return AutomaticCalibrationResult(
        profile=profile,
        diagnostics=tuple(completed),
        quality_policy=quality_policy,
        validation_reasons=reasons,
        capabilities=inspect_opencv_calibration_capabilities(),
    )


def calibrate_checkerboard_image_set(
    images: Sequence[np.ndarray | tuple[str, np.ndarray]],
    target: CheckerboardCalibrationTarget,
    *,
    quality_policy: AutomaticCalibrationQualityPolicy | None = None,
    source: InputSourceMetadata | None = None,
    profile_id: str | None = None,
    calibration_record_id: str | None = None,
    camera_id: str | None = None,
) -> AutomaticCalibrationResult:
    """Calibrate from checkerboard images while preserving rejected-view evidence."""

    capabilities = inspect_opencv_calibration_capabilities()
    if not capabilities.checkerboard:
        raise ReconstructionValidationError(
            "This OpenCV build does not expose the required checkerboard calibration APIs."
        )
    if not isinstance(target, CheckerboardCalibrationTarget):
        raise TypeError("target must be a CheckerboardCalibrationTarget.")
    normalized, image_size = _normalize_images(images)
    object_template = target.object_points()
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    diagnostics: list[CalibrationFrameDiagnostic] = []
    for name, image in normalized:
        corners = _detect_checkerboard(image, target)
        if corners is None:
            diagnostics.append(
                CalibrationFrameDiagnostic(
                    source_name=name,
                    image_size=image_size,
                    disposition=CalibrationFrameDisposition.TARGET_NOT_DETECTED,
                    detected_point_count=0,
                )
            )
            continue
        object_points.append(object_template.copy())
        image_points.append(corners)
        diagnostics.append(
            CalibrationFrameDiagnostic(
                source_name=name,
                image_size=image_size,
                disposition=CalibrationFrameDisposition.ACCEPTED,
                detected_point_count=len(corners),
            )
        )
    return _finish_calibration(
        method=CameraCalibrationMethod.CHECKERBOARD,
        image_size=image_size,
        object_points=object_points,
        image_points=image_points,
        diagnostics=diagnostics,
        quality_policy=quality_policy,
        provenance=(
            "automatic_checkerboard_opencv;"
            f"inner_corners={target.inner_corners_x}x{target.inner_corners_y};"
            f"square_length={target.square_length};opencv={capabilities.opencv_version}"
        ),
        source=source,
        profile_id=profile_id,
        calibration_record_id=calibration_record_id,
        camera_id=camera_id,
        notes=(
            "Automatic checkerboard calibration. Target square length affects board/extrinsic "
            "scale only; this camera profile does not qualify scene metric scale."
        ),
    )


def calibrate_charuco_image_set(
    images: Sequence[np.ndarray | tuple[str, np.ndarray]],
    target: CharucoCalibrationTarget,
    *,
    quality_policy: AutomaticCalibrationQualityPolicy | None = None,
    source: InputSourceMetadata | None = None,
    profile_id: str | None = None,
    calibration_record_id: str | None = None,
    camera_id: str | None = None,
) -> AutomaticCalibrationResult:
    """Calibrate from ChArUco images using APIs actually provided by OpenCV."""

    if not isinstance(target, CharucoCalibrationTarget):
        raise TypeError("target must be a CharucoCalibrationTarget.")
    capabilities = inspect_opencv_calibration_capabilities()
    if not capabilities.charuco:
        missing = ", ".join(capabilities.charuco_missing_features) or "unknown ChArUco feature"
        raise ReconstructionValidationError(
            f"ChArUco calibration is unavailable in this OpenCV build: {missing}."
        )
    normalized, image_size = _normalize_images(images)
    board, dictionary = _create_charuco_board(target)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    diagnostics: list[CalibrationFrameDiagnostic] = []
    for name, image in normalized:
        detection = _detect_charuco(image, board, dictionary)
        if detection is None:
            diagnostics.append(
                CalibrationFrameDiagnostic(
                    source_name=name,
                    image_size=image_size,
                    disposition=CalibrationFrameDisposition.TARGET_NOT_DETECTED,
                    detected_point_count=0,
                )
            )
            continue
        corners, ids = detection
        obj, img = _charuco_object_and_image_points(board, corners, ids)
        if len(obj) < 4:
            diagnostics.append(
                CalibrationFrameDiagnostic(
                    source_name=name,
                    image_size=image_size,
                    disposition=CalibrationFrameDisposition.INSUFFICIENT_TARGET_POINTS,
                    detected_point_count=len(obj),
                )
            )
            continue
        object_points.append(obj)
        image_points.append(img)
        diagnostics.append(
            CalibrationFrameDiagnostic(
                source_name=name,
                image_size=image_size,
                disposition=CalibrationFrameDisposition.ACCEPTED,
                detected_point_count=len(obj),
            )
        )
    return _finish_calibration(
        method=CameraCalibrationMethod.CHARUCO,
        image_size=image_size,
        object_points=object_points,
        image_points=image_points,
        diagnostics=diagnostics,
        quality_policy=quality_policy,
        provenance=(
            "automatic_charuco_opencv;"
            f"board={target.squares_x}x{target.squares_y};"
            f"square_length={target.square_length};marker_length={target.marker_length};"
            f"dictionary={target.dictionary_name};legacy_pattern={target.legacy_pattern};"
            f"opencv={capabilities.opencv_version}"
        ),
        source=source,
        profile_id=profile_id,
        calibration_record_id=calibration_record_id,
        camera_id=camera_id,
        notes=(
            "Automatic ChArUco calibration. Target dimensions affect board/extrinsic scale only; "
            "this camera profile does not qualify scene metric scale."
        ),
    )
