"""Phase 9.1 camera-calibration profile contracts and Phase 8 compatibility.

The profile layer is deliberately additive.  It wraps the existing calibrated
``ReconstructionCamera`` authority without changing camera geometry, metric
scale, measurement, or backend contracts.  Metric qualification remains a
separate ``ScaleReferenceProfile`` concern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from core.contracts import CalibrationStatus, ImageSize
from core.contracts.validation import require_nonempty
from core.models.metadata import IntrinsicsSource
from core.reconstruction.camera import CameraDistortionModel, ReconstructionCamera
from core.reconstruction.contracts import InputSourceKind, InputSourceMetadata
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.live_calibration import (
    live_camera_calibration_dict,
    live_camera_from_calibration_dict,
)
from core.reconstruction.validation import require_iso8601_utc


CAMERA_CALIBRATION_PROFILE_SCHEMA_VERSION = 1


class CameraCalibrationMethod(str, Enum):
    """How calibrated intrinsics were obtained.

    Checkerboard and ChArUco are declared now so later Phase 9.1 automatic
    calibration can populate the same stable profile contract without another
    schema redesign.
    """

    CHECKERBOARD = "checkerboard"
    CHARUCO = "charuco"
    MANUAL = "manual"
    IMPORTED_PROFILE = "imported_profile"
    LEGACY_LIVE_JSON = "legacy_live_json"


class CameraCalibrationValidationState(str, Enum):
    """Qualification state of a calibration profile itself.

    ``LEGACY_COMPATIBLE`` preserves Phase 8 behavior for explicit live-camera
    calibration JSON while making the absence of Phase 9 quality evidence
    visible.  It must not be presented as newly validated Phase 9 calibration.
    """

    UNVALIDATED = "unvalidated"
    VALIDATED = "validated"
    REJECTED = "rejected"
    LEGACY_COMPATIBLE = "legacy_compatible"


@dataclass(frozen=True, slots=True)
class CalibrationSourceBinding:
    """Stable, credential-safe identity of the source used for calibration."""

    source_id: str
    kind: InputSourceKind
    display_name: str
    expected_image_size: ImageSize | None = None
    fingerprint: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_id",
            require_nonempty(self.source_id, field_name="source_id"),
        )
        if not isinstance(self.kind, InputSourceKind):
            raise TypeError("kind must be an InputSourceKind.")
        object.__setattr__(
            self,
            "display_name",
            require_nonempty(self.display_name, field_name="display_name"),
        )
        if self.expected_image_size is not None and not isinstance(
            self.expected_image_size, ImageSize
        ):
            raise TypeError("expected_image_size must be an ImageSize or None.")
        object.__setattr__(self, "fingerprint", str(self.fingerprint).strip().lower())

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "kind": self.kind.value,
            "display_name": self.display_name,
            "expected_image_size": (
                self.expected_image_size.as_dict() if self.expected_image_size else None
            ),
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class CalibrationQualityMetrics:
    """Reprojection-quality evidence attached to a calibration profile."""

    rms_reprojection_error_px: float | None = None
    mean_reprojection_error_px: float | None = None
    max_reprojection_error_px: float | None = None
    accepted_view_count: int | None = None
    rejected_view_count: int | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "rms_reprojection_error_px",
            "mean_reprojection_error_px",
            "max_reprojection_error_px",
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            normalized = float(value)
            if not isfinite(normalized) or normalized < 0.0:
                raise ReconstructionValidationError(
                    f"{field_name} must be finite and non-negative when provided."
                )
            object.__setattr__(self, field_name, normalized)

        for field_name in ("accepted_view_count", "rejected_view_count"):
            value = getattr(self, field_name)
            if value is None:
                continue
            if isinstance(value, bool) or int(value) != value or int(value) < 0:
                raise ReconstructionValidationError(
                    f"{field_name} must be a non-negative integer when provided."
                )
            object.__setattr__(self, field_name, int(value))

        mean = self.mean_reprojection_error_px
        maximum = self.max_reprojection_error_px
        if mean is not None and maximum is not None and mean > maximum:
            raise ReconstructionValidationError(
                "mean_reprojection_error_px cannot exceed max_reprojection_error_px."
            )

    @property
    def has_reprojection_evidence(self) -> bool:
        return any(
            value is not None
            for value in (
                self.rms_reprojection_error_px,
                self.mean_reprojection_error_px,
                self.max_reprojection_error_px,
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "rms_reprojection_error_px": self.rms_reprojection_error_px,
            "mean_reprojection_error_px": self.mean_reprojection_error_px,
            "max_reprojection_error_px": self.max_reprojection_error_px,
            "accepted_view_count": self.accepted_view_count,
            "rejected_view_count": self.rejected_view_count,
        }


@dataclass(frozen=True, slots=True)
class CameraCalibrationProfile:
    """Reusable camera-calibration record, intentionally separate from metric scale."""

    profile_id: str
    camera: ReconstructionCamera
    method: CameraCalibrationMethod
    validation_state: CameraCalibrationValidationState
    provenance: str
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source_binding: CalibrationSourceBinding | None = None
    quality: CalibrationQualityMetrics | None = None
    notes: str = ""
    schema_version: int = CAMERA_CALIBRATION_PROFILE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "profile_id",
            require_nonempty(self.profile_id, field_name="profile_id"),
        )
        if not isinstance(self.camera, ReconstructionCamera):
            raise TypeError("camera must be a ReconstructionCamera.")
        if self.camera.intrinsics.source is not IntrinsicsSource.CALIBRATED:
            raise ReconstructionValidationError(
                "Camera calibration profiles require IntrinsicsSource.CALIBRATED."
            )
        if self.camera.intrinsics.calibration_status is not CalibrationStatus.CALIBRATED:
            raise ReconstructionValidationError(
                "Camera calibration profiles require calibrated intrinsics."
            )
        if not self.camera.intrinsics.calibration_record_id:
            raise ReconstructionValidationError(
                "Camera calibration profiles require a calibration record ID."
            )
        if self.camera.distortion_model is CameraDistortionModel.BACKEND_SPECIFIC:
            raise ReconstructionValidationError(
                "Camera calibration profiles require a portable distortion model; "
                "backend-specific distortion cannot be stored in the Phase 8 compatibility schema."
            )
        if not isinstance(self.method, CameraCalibrationMethod):
            raise TypeError("method must be a CameraCalibrationMethod.")
        if not isinstance(self.validation_state, CameraCalibrationValidationState):
            raise TypeError("validation_state must be a CameraCalibrationValidationState.")
        if int(self.schema_version) != CAMERA_CALIBRATION_PROFILE_SCHEMA_VERSION:
            raise ReconstructionValidationError(
                "Unsupported camera-calibration profile schema version."
            )

        provenance = require_nonempty(self.provenance, field_name="provenance")
        created_at = require_iso8601_utc(self.created_at_utc, field_name="created_at_utc")
        if self.source_binding is not None:
            if not isinstance(self.source_binding, CalibrationSourceBinding):
                raise TypeError("source_binding must be CalibrationSourceBinding or None.")
            expected = self.source_binding.expected_image_size
            if expected is not None and expected != self.camera.image_size:
                raise ReconstructionValidationError(
                    "Calibration source resolution must match the calibration camera resolution."
                )
        if self.quality is not None and not isinstance(self.quality, CalibrationQualityMetrics):
            raise TypeError("quality must be CalibrationQualityMetrics or None.")

        if self.method is CameraCalibrationMethod.LEGACY_LIVE_JSON:
            if self.validation_state is not CameraCalibrationValidationState.LEGACY_COMPATIBLE:
                raise ReconstructionValidationError(
                    "Legacy Phase 8 calibration imports must remain LEGACY_COMPATIBLE."
                )
        elif self.validation_state is CameraCalibrationValidationState.LEGACY_COMPATIBLE:
            raise ReconstructionValidationError(
                "LEGACY_COMPATIBLE is reserved for Phase 8 live-calibration imports."
            )

        if (
            self.validation_state is CameraCalibrationValidationState.VALIDATED
            and self.method
            in {CameraCalibrationMethod.CHECKERBOARD, CameraCalibrationMethod.CHARUCO}
            and (self.quality is None or not self.quality.has_reprojection_evidence)
        ):
            raise ReconstructionValidationError(
                "Validated automatic calibration requires reprojection-quality evidence."
            )

        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "created_at_utc", created_at)
        object.__setattr__(self, "notes", str(self.notes).strip())
        object.__setattr__(
            self, "schema_version", CAMERA_CALIBRATION_PROFILE_SCHEMA_VERSION
        )

    @property
    def calibration_record_id(self) -> str:
        return self.camera.intrinsics.calibration_record_id

    @property
    def calibration_image_size(self) -> ImageSize:
        return self.camera.image_size

    @property
    def is_accepted(self) -> bool:
        return self.validation_state in {
            CameraCalibrationValidationState.VALIDATED,
            CameraCalibrationValidationState.LEGACY_COMPATIBLE,
        }

    @property
    def has_phase9_validation_evidence(self) -> bool:
        return self.validation_state is CameraCalibrationValidationState.VALIDATED

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "method": self.method.value,
            "validation_state": self.validation_state.value,
            "created_at_utc": self.created_at_utc,
            "provenance": self.provenance,
            "source_binding": self.source_binding.as_dict() if self.source_binding else None,
            "quality": self.quality.as_dict() if self.quality else None,
            "notes": self.notes,
            # Reuse the qualified Phase 8 camera-calibration representation rather
            # than creating a second camera/distortion authority.
            "camera_calibration": live_camera_calibration_dict(self.camera),
        }


def calibration_source_binding_from_metadata(
    source: InputSourceMetadata,
) -> CalibrationSourceBinding:
    if not isinstance(source, InputSourceMetadata):
        raise TypeError("source must be an InputSourceMetadata.")
    return CalibrationSourceBinding(
        source_id=source.source_id,
        kind=source.kind,
        display_name=source.display_name,
        expected_image_size=source.expected_image_size,
        fingerprint=source.fingerprint,
    )


def camera_calibration_profile_to_json(
    profile: CameraCalibrationProfile,
    *,
    indent: int | None = 2,
) -> str:
    if not isinstance(profile, CameraCalibrationProfile):
        raise TypeError("profile must be a CameraCalibrationProfile.")
    return json.dumps(
        profile.as_dict(),
        indent=indent,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        allow_nan=False,
    )


def camera_calibration_profile_from_json(
    payload: str | bytes | bytearray,
) -> CameraCalibrationProfile:
    try:
        raw = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        raise ReconstructionValidationError(
            "Camera-calibration profile must be valid JSON."
        ) from exc
    if not isinstance(raw, Mapping):
        raise ReconstructionValidationError(
            "Camera-calibration profile root must be an object."
        )
    return camera_calibration_profile_from_dict(raw)


def camera_calibration_profile_from_dict(
    raw: Mapping[str, Any],
) -> CameraCalibrationProfile:
    try:
        if int(raw["schema_version"]) != CAMERA_CALIBRATION_PROFILE_SCHEMA_VERSION:
            raise ReconstructionValidationError(
                "Unsupported camera-calibration profile schema version."
            )
        camera_raw = _mapping(raw["camera_calibration"], "camera_calibration")
        source_raw = raw.get("source_binding")
        quality_raw = raw.get("quality")
        return CameraCalibrationProfile(
            profile_id=str(raw["profile_id"]),
            camera=live_camera_from_calibration_dict(camera_raw),
            method=CameraCalibrationMethod(str(raw["method"])),
            validation_state=CameraCalibrationValidationState(
                str(raw["validation_state"])
            ),
            created_at_utc=str(raw["created_at_utc"]),
            provenance=str(raw["provenance"]),
            source_binding=(
                None
                if source_raw is None
                else _source_binding_from_dict(_mapping(source_raw, "source_binding"))
            ),
            quality=(
                None
                if quality_raw is None
                else _quality_from_dict(_mapping(quality_raw, "quality"))
            ),
            notes=str(raw.get("notes", "")),
        )
    except KeyError as exc:
        raise ReconstructionValidationError(
            f"Camera-calibration profile is missing required field: {exc.args[0]}."
        ) from exc
    except ReconstructionValidationError:
        raise
    except (TypeError, ValueError) as exc:
        raise ReconstructionValidationError(
            f"Camera-calibration profile contains an invalid value: {exc}."
        ) from exc


def camera_calibration_profile_from_live_calibration_dict(
    payload: Mapping[str, Any],
    *,
    profile_id: str | None = None,
    source: InputSourceMetadata | None = None,
    created_at_utc: str | None = None,
    provenance: str = "phase8_live_calibration_json",
    notes: str = "Imported from the Phase 8 live-camera calibration schema.",
) -> CameraCalibrationProfile:
    """Wrap an existing Phase 8 live-calibration record without upgrading evidence."""

    camera = live_camera_from_calibration_dict(payload)
    binding = None if source is None else calibration_source_binding_from_metadata(source)
    return CameraCalibrationProfile(
        profile_id=profile_id or camera.intrinsics.calibration_record_id,
        camera=camera,
        method=CameraCalibrationMethod.LEGACY_LIVE_JSON,
        validation_state=CameraCalibrationValidationState.LEGACY_COMPATIBLE,
        created_at_utc=created_at_utc or datetime.now(timezone.utc).isoformat(),
        provenance=provenance,
        source_binding=binding,
        quality=None,
        notes=notes,
    )


def camera_calibration_profile_to_live_calibration_dict(
    profile: CameraCalibrationProfile,
) -> dict[str, Any]:
    """Export the underlying camera in the unchanged Phase 8 compatibility schema."""

    if not isinstance(profile, CameraCalibrationProfile):
        raise TypeError("profile must be a CameraCalibrationProfile.")
    return live_camera_calibration_dict(profile.camera)


def load_camera_calibration_profile(path: str | Path) -> CameraCalibrationProfile:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ReconstructionValidationError(
            f"Camera-calibration profile does not exist: {source}"
        )
    try:
        return camera_calibration_profile_from_json(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ReconstructionValidationError(
            f"Could not read camera-calibration profile: {source}"
        ) from exc


def save_camera_calibration_profile(
    profile: CameraCalibrationProfile,
    path: str | Path,
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.write_text(
            camera_calibration_profile_to_json(profile) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise ReconstructionValidationError(
            f"Could not write camera-calibration profile: {destination}"
        ) from exc
    return destination


def _source_binding_from_dict(raw: Mapping[str, Any]) -> CalibrationSourceBinding:
    size_raw = raw.get("expected_image_size")
    return CalibrationSourceBinding(
        source_id=str(raw["source_id"]),
        kind=InputSourceKind(str(raw["kind"])),
        display_name=str(raw["display_name"]),
        expected_image_size=(
            None
            if size_raw is None
            else ImageSize(
                int(_mapping(size_raw, "expected_image_size")["width"]),
                int(_mapping(size_raw, "expected_image_size")["height"]),
            )
        ),
        fingerprint=str(raw.get("fingerprint", "")),
    )


def _quality_from_dict(raw: Mapping[str, Any]) -> CalibrationQualityMetrics:
    return CalibrationQualityMetrics(
        rms_reprojection_error_px=_optional_float(raw.get("rms_reprojection_error_px")),
        mean_reprojection_error_px=_optional_float(raw.get("mean_reprojection_error_px")),
        max_reprojection_error_px=_optional_float(raw.get("max_reprojection_error_px")),
        accepted_view_count=_optional_int(raw.get("accepted_view_count")),
        rejected_view_count=_optional_int(raw.get("rejected_view_count")),
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ReconstructionValidationError("Calibration view counts cannot be bool values.")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ReconstructionValidationError(
            "Calibration view counts must be non-negative integers."
        ) from exc
    if normalized != value or normalized < 0:
        raise ReconstructionValidationError(
            "Calibration view counts must be non-negative integers."
        )
    return normalized


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"{field_name} must be an object.")
    return value
