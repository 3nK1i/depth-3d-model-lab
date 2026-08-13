"""Manual calibration creation, profile import, and reusable profile persistence.

Phase 9.1C is intentionally additive.  It builds on the immutable
``CameraCalibrationProfile`` contract from Phase 9.1A and the safe application
layer from Phase 9.1B.  It does not change metric-scale authority, UI settings,
or the Phase 8 live-calibration schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

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
    CameraCalibrationMethod,
    CameraCalibrationProfile,
    CameraCalibrationValidationState,
    calibration_source_binding_from_metadata,
    camera_calibration_profile_from_dict,
    camera_calibration_profile_from_live_calibration_dict,
    camera_calibration_profile_to_json,
    load_camera_calibration_profile,
)
from core.reconstruction.camera import CameraDistortionModel, ReconstructionCamera
from core.reconstruction.contracts import InputSourceMetadata
from core.reconstruction.errors import ReconstructionValidationError


class CalibrationProfileFileFormat(str, Enum):
    """Supported calibration files accepted by the Phase 9.1C importer."""

    PHASE9_PROFILE = "phase9_profile"
    PHASE8_LIVE_JSON = "phase8_live_json"


@dataclass(frozen=True, slots=True)
class ImportedCameraCalibration:
    """One imported profile together with the detected compatibility format."""

    profile: CameraCalibrationProfile
    file_format: CalibrationProfileFileFormat
    source_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.profile, CameraCalibrationProfile):
            raise TypeError("profile must be a CameraCalibrationProfile.")
        if not isinstance(self.file_format, CalibrationProfileFileFormat):
            raise TypeError("file_format must be a CalibrationProfileFileFormat.")
        object.__setattr__(
            self,
            "source_name",
            require_nonempty(self.source_name, field_name="source_name"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile.profile_id,
            "calibration_record_id": self.profile.calibration_record_id,
            "file_format": self.file_format.value,
            "source_name": self.source_name,
            "method": self.profile.method.value,
            "validation_state": self.profile.validation_state.value,
        }


def create_manual_camera_calibration_profile(
    *,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    distortion_model: CameraDistortionModel = CameraDistortionModel.NONE,
    distortion_coefficients: tuple[float, ...] = (),
    profile_id: str | None = None,
    calibration_record_id: str | None = None,
    camera_id: str | None = None,
    source: InputSourceMetadata | None = None,
    provenance: str = "manual_calibration_entry",
    notes: str = "",
    created_at_utc: str | None = None,
    pixel_space_id: str | None = None,
    camera_frame_id: str | None = None,
    reconstruction_frame_id: str = "calibration-reconstruction",
) -> CameraCalibrationProfile:
    """Create an immutable manual calibration profile from explicit parameters.

    Manual entry is structurally validated by the existing camera contracts, but
    remains ``UNVALIDATED``.  Entering plausible intrinsics is not equivalent to
    qualifying their calibration quality or establishing metric scene scale.
    """

    if source is not None and not isinstance(source, InputSourceMetadata):
        raise TypeError("source must be an InputSourceMetadata or None.")
    if not isinstance(distortion_model, CameraDistortionModel):
        raise TypeError("distortion_model must be a CameraDistortionModel.")
    if distortion_model is CameraDistortionModel.BACKEND_SPECIFIC:
        raise ReconstructionValidationError(
            "Manual Phase 9 calibration entry requires a portable distortion model."
        )

    image_size = ImageSize(width, height)
    resolved_profile_id = str(profile_id or f"camera-calibration-{uuid4()}").strip()
    resolved_record_id = str(calibration_record_id or f"calibration-record-{uuid4()}").strip()
    resolved_camera_id = str(
        camera_id or (source.source_id if source is not None else f"camera-{uuid4()}")
    ).strip()
    resolved_pixel_space_id = str(
        pixel_space_id or f"calibration-{image_size.width}x{image_size.height}"
    ).strip()
    resolved_frame_id = str(camera_frame_id or f"{resolved_camera_id}-frame").strip()

    matrix = np.array(
        [[float(fx), 0.0, float(cx)], [0.0, float(fy), float(cy)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    intrinsics = CameraIntrinsics(
        matrix=matrix,
        pixel_space_id=resolved_pixel_space_id,
        source=IntrinsicsSource.CALIBRATED,
        calibration_status=CalibrationStatus.CALIBRATED,
        calibration_record_id=resolved_record_id,
    )
    frame = CoordinateFrame(
        frame_id=resolved_frame_id,
        kind=CoordinateFrameKind.CAMERA_GENERIC,
        axis_convention=AxisConvention.X_RIGHT_Y_DOWN_Z_FORWARD,
        parent_frame_id=require_nonempty(
            reconstruction_frame_id,
            field_name="reconstruction_frame_id",
        ),
    )
    camera = ReconstructionCamera(
        camera_id=resolved_camera_id,
        image_size=image_size,
        intrinsics=intrinsics,
        camera_frame=frame,
        distortion_model=distortion_model,
        distortion_coefficients=tuple(float(value) for value in distortion_coefficients),
    )
    return CameraCalibrationProfile(
        profile_id=resolved_profile_id,
        camera=camera,
        method=CameraCalibrationMethod.MANUAL,
        validation_state=CameraCalibrationValidationState.UNVALIDATED,
        provenance=provenance,
        created_at_utc=created_at_utc or datetime.now(timezone.utc).isoformat(),
        source_binding=(
            None if source is None else calibration_source_binding_from_metadata(source)
        ),
        quality=None,
        notes=notes,
    )


def import_camera_calibration_file(
    path: str | Path,
    *,
    legacy_profile_id: str | None = None,
    source: InputSourceMetadata | None = None,
    created_at_utc: str | None = None,
) -> ImportedCameraCalibration:
    """Import a native Phase 9 profile or unchanged Phase 8 live JSON.

    Native Phase 9 records are returned without rewriting their method,
    validation state, timestamps, or provenance.  Phase 8 live JSON is wrapped
    as ``LEGACY_COMPATIBLE`` and therefore cannot masquerade as new Phase 9
    validation evidence.
    """

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise ReconstructionValidationError(
            f"Camera calibration import file does not exist: {source_path}"
        )
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ReconstructionValidationError(
            "Could not parse camera calibration import JSON."
        ) from exc
    if not isinstance(payload, Mapping):
        raise ReconstructionValidationError(
            "Camera calibration import root must be a JSON object."
        )

    if "camera_calibration" in payload and "profile_id" in payload:
        profile = camera_calibration_profile_from_dict(payload)
        return ImportedCameraCalibration(
            profile=profile,
            file_format=CalibrationProfileFileFormat.PHASE9_PROFILE,
            source_name=source_path.name,
        )

    legacy_required = {"camera_id", "calibration_record_id", "width", "height", "fx", "fy", "cx", "cy"}
    if legacy_required.issubset(payload):
        profile = camera_calibration_profile_from_live_calibration_dict(
            payload,
            profile_id=legacy_profile_id,
            source=source,
            created_at_utc=created_at_utc,
            provenance="phase8_live_calibration_json_import",
            notes=f"Imported from legacy Phase 8 calibration file {source_path.name}.",
        )
        return ImportedCameraCalibration(
            profile=profile,
            file_format=CalibrationProfileFileFormat.PHASE8_LIVE_JSON,
            source_name=source_path.name,
        )

    raise ReconstructionValidationError(
        "Unsupported camera calibration file. Expected a Phase 9 profile or Phase 8 live calibration JSON."
    )


class CameraCalibrationProfileStore:
    """Filesystem-backed reusable profile store with deterministic safe paths."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).expanduser().resolve()

    @property
    def root(self) -> Path:
        return self._root

    def save(self, profile: CameraCalibrationProfile, *, overwrite: bool = False) -> Path:
        if not isinstance(profile, CameraCalibrationProfile):
            raise TypeError("profile must be a CameraCalibrationProfile.")
        if not isinstance(overwrite, bool):
            raise TypeError("overwrite must be a bool.")
        self._root.mkdir(parents=True, exist_ok=True)
        destination = self._profile_path(profile.profile_id)
        if destination.exists() and not overwrite:
            raise ReconstructionValidationError(
                f"Camera calibration profile already exists: {profile.profile_id}"
            )

        temporary = self._root / f".{destination.name}.{uuid4().hex}.tmp"
        try:
            temporary.write_text(
                camera_calibration_profile_to_json(profile) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, destination)
        except OSError as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise ReconstructionValidationError(
                f"Could not persist camera calibration profile: {profile.profile_id}"
            ) from exc
        return destination

    def load(self, profile_id: str) -> CameraCalibrationProfile:
        normalized = require_nonempty(profile_id, field_name="profile_id")
        path = self._profile_path(normalized)
        profile = load_camera_calibration_profile(path)
        if profile.profile_id != normalized:
            raise ReconstructionValidationError(
                "Stored camera calibration profile ID does not match its deterministic store key."
            )
        return profile

    def list_profiles(self) -> tuple[CameraCalibrationProfile, ...]:
        if not self._root.exists():
            return ()
        if not self._root.is_dir():
            raise ReconstructionValidationError(
                f"Camera calibration profile store is not a directory: {self._root}"
            )
        profiles = tuple(
            load_camera_calibration_profile(path)
            for path in sorted(self._root.glob("*.json"))
            if path.is_file()
        )
        return tuple(sorted(profiles, key=lambda item: (item.created_at_utc, item.profile_id)))

    def import_file(
        self,
        path: str | Path,
        *,
        legacy_profile_id: str | None = None,
        source: InputSourceMetadata | None = None,
        created_at_utc: str | None = None,
        overwrite: bool = False,
    ) -> ImportedCameraCalibration:
        imported = import_camera_calibration_file(
            path,
            legacy_profile_id=legacy_profile_id,
            source=source,
            created_at_utc=created_at_utc,
        )
        self.save(imported.profile, overwrite=overwrite)
        return imported

    def contains(self, profile_id: str) -> bool:
        normalized = require_nonempty(profile_id, field_name="profile_id")
        return self._profile_path(normalized).is_file()

    def _profile_path(self, profile_id: str) -> Path:
        normalized = require_nonempty(profile_id, field_name="profile_id")
        digest = sha256(normalized.encode("utf-8")).hexdigest()
        return self._root / f"{digest}.json"
