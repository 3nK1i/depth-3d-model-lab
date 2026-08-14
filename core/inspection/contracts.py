"""Dependency-light contracts for the Phase 7.5 measurement inspection UI."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from core.contracts.validation import ContractValidationError, require_nonempty
from core.measurement.contracts import BoundingBoxKind, ReliabilityLevel


class InspectionRunStatus(str, Enum):
    EMPTY = "empty"
    READY = "ready"
    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL = "partial"
    REJECTED = "rejected"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MaskEditMode(str, Enum):
    SELECT = "select"
    INCLUDE = "include"
    EXCLUDE = "exclude"


class InspectionApprovalState(str, Enum):
    AUTOMATIC = "automatic"
    EDITED = "edited"
    APPROVED = "approved"
    LOCKED = "locked"


class InspectionPointMode(str, Enum):
    RAW = "raw"
    FILTERED = "filtered"


class InspectionBoxMethod(str, Enum):
    AUTO = "auto"
    AXIS_ALIGNED = BoundingBoxKind.AXIS_ALIGNED.value
    CAMERA_ALIGNED = BoundingBoxKind.CAMERA_ALIGNED.value
    ORIENTED = BoundingBoxKind.ORIENTED.value

    @property
    def bounding_box_kind(self) -> BoundingBoxKind | None:
        if self is InspectionBoxMethod.AUTO:
            return None
        return BoundingBoxKind(self.value)


@dataclass(frozen=True, slots=True)
class InspectionObjectSummary:
    observation_id: str
    object_index: int
    class_id: int
    class_name: str
    detection_confidence: float | None
    detection_box_xyxy: tuple[float, float, float, float] | None
    mask_key: str
    raw_points_key: str
    filtered_points_key: str
    raw_colors_key: str = ""
    filtered_colors_key: str = ""
    mask_source: str = ""
    mask_revision: int = 0
    approval_status: str = ""
    track_id: str = ""
    status: str = ""
    extraction_status: str = ""
    fitting_status: str = ""
    measurement_status: str = ""
    preferred_box_id: str = ""
    preferred_box_reason: str = ""
    boxes: tuple[tuple[str, dict[str, Any]], ...] = ()
    measurements: tuple[dict[str, Any], ...] = ()
    diagnostics: tuple[tuple[str, Any], ...] = ()
    instance_name: str = ""
    semantic_source: str = "perception_model"
    semantic_review_status: str = ""
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "observation_id", require_nonempty(self.observation_id, field_name="observation_id"))
        if isinstance(self.object_index, bool) or int(self.object_index) < 0:
            raise ContractValidationError("object_index must be a non-negative integer.")
        if isinstance(self.class_id, bool) or int(self.class_id) < 0:
            raise ContractValidationError("class_id must be a non-negative integer.")
        object.__setattr__(self, "object_index", int(self.object_index))
        object.__setattr__(self, "class_id", int(self.class_id))
        object.__setattr__(self, "class_name", require_nonempty(self.class_name, field_name="class_name"))
        if self.detection_box_xyxy is not None:
            values = tuple(float(value) for value in self.detection_box_xyxy)
            if len(values) != 4 or values[2] <= values[0] or values[3] <= values[1]:
                raise ContractValidationError("detection_box_xyxy must contain one positive-area box.")
            object.__setattr__(self, "detection_box_xyxy", values)
        if self.detection_confidence is not None:
            confidence = float(self.detection_confidence)
            if not 0.0 <= confidence <= 1.0:
                raise ContractValidationError("detection_confidence must remain within [0, 1].")
            object.__setattr__(self, "detection_confidence", confidence)
        for name in ("mask_key", "raw_points_key", "filtered_points_key"):
            object.__setattr__(self, name, require_nonempty(getattr(self, name), field_name=name))
        if isinstance(self.mask_revision, bool) or int(self.mask_revision) < 0:
            raise ContractValidationError("mask_revision must be a non-negative integer.")
        object.__setattr__(self, "mask_revision", int(self.mask_revision))
        object.__setattr__(self, "boxes", tuple((str(name), dict(payload)) for name, payload in self.boxes))
        object.__setattr__(self, "measurements", tuple(dict(value) for value in self.measurements))
        object.__setattr__(self, "diagnostics", tuple((str(name), value) for name, value in self.diagnostics))
        object.__setattr__(self, "instance_name", str(self.instance_name).strip())
        object.__setattr__(self, "semantic_source", str(self.semantic_source).strip() or "perception_model")
        object.__setattr__(self, "semantic_review_status", str(self.semantic_review_status).strip())
        object.__setattr__(self, "warnings", tuple(str(value).strip() for value in self.warnings if str(value).strip()))

    @property
    def reliability_level(self) -> ReliabilityLevel:
        levels = []
        for value in self.measurements:
            if value.get("status") == "available":
                try:
                    levels.append(ReliabilityLevel(value.get("reliability", "unavailable")))
                except ValueError:
                    pass
        order = {
            ReliabilityLevel.HIGH: 4,
            ReliabilityLevel.MEDIUM: 3,
            ReliabilityLevel.LOW: 2,
            ReliabilityLevel.REJECTED: 1,
            ReliabilityLevel.UNAVAILABLE: 0,
        }
        return min(levels, key=lambda item: order[item]) if levels else ReliabilityLevel.UNAVAILABLE

    def box_payload(self, method: InspectionBoxMethod) -> dict[str, Any] | None:
        boxes = dict(self.boxes)
        if method is InspectionBoxMethod.AUTO:
            for name, payload in self.boxes:
                if payload.get("box_id") == self.preferred_box_id:
                    return payload
            return boxes.get("oriented") or boxes.get("camera_aligned") or boxes.get("axis_aligned")
        return boxes.get(method.value)


@dataclass(frozen=True, slots=True)
class InspectionBundleSummary:
    bundle_dir: Path
    schema_version: int
    status: InspectionRunStatus
    arrays_file: str
    state_file: str
    image_width: int
    image_height: int
    depth_model_id: str
    depth_model_name: str
    perception_model_id: str
    perception_model_name: str
    depth_scale: str
    intrinsics_source: str
    calibration_status: str
    confidence_kind: str
    box_method: InspectionBoxMethod
    objects: tuple[InspectionObjectSummary, ...]
    warnings: tuple[str, ...] = ()
    elapsed_seconds: float = 0.0
    calibration_profile_id: str = ""
    calibration_record_id: str = ""
    calibration_method: str = ""
    calibration_validation_state: str = ""
    calibration_application_mode: str = ""
    calibration_rectified: bool = False
    semantic_source: str = "perception_model"
    semantic_workspace_id: str = ""
    semantic_workspace_revision: int = 0

    def __post_init__(self) -> None:
        directory = Path(self.bundle_dir)
        if not directory.exists() or not directory.is_dir():
            raise ContractValidationError("bundle_dir must exist and be a directory.")
        if int(self.schema_version) != 1:
            raise ContractValidationError("Unsupported inspection bundle schema version.")
        if int(self.image_width) <= 0 or int(self.image_height) <= 0:
            raise ContractValidationError("Inspection image dimensions must be positive.")
        object.__setattr__(self, "bundle_dir", directory)
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "image_width", int(self.image_width))
        object.__setattr__(self, "image_height", int(self.image_height))
        object.__setattr__(self, "depth_model_id", require_nonempty(self.depth_model_id, field_name="depth_model_id"))
        object.__setattr__(self, "perception_model_id", require_nonempty(self.perception_model_id, field_name="perception_model_id"))
        object.__setattr__(self, "objects", tuple(self.objects))
        object.__setattr__(self, "warnings", tuple(str(value).strip() for value in self.warnings if str(value).strip()))
        for name in (
            "calibration_profile_id",
            "calibration_record_id",
            "calibration_method",
            "calibration_validation_state",
            "calibration_application_mode",
        ):
            object.__setattr__(self, name, str(getattr(self, name)).strip())
        object.__setattr__(self, "calibration_rectified", bool(self.calibration_rectified))
        object.__setattr__(self, "semantic_source", str(self.semantic_source).strip() or "perception_model")
        object.__setattr__(self, "semantic_workspace_id", str(self.semantic_workspace_id).strip())
        if isinstance(self.semantic_workspace_revision, bool) or int(self.semantic_workspace_revision) < 0:
            raise ContractValidationError("semantic_workspace_revision must be a non-negative integer.")
        object.__setattr__(self, "semantic_workspace_revision", int(self.semantic_workspace_revision))

    @property
    def arrays_path(self) -> Path:
        path = (self.bundle_dir / self.arrays_file).resolve()
        if path.parent != self.bundle_dir.resolve():
            raise ContractValidationError("arrays_file must be inside bundle_dir.")
        return path

    @property
    def state_path(self) -> Path:
        path = (self.bundle_dir / self.state_file).resolve()
        if path.parent != self.bundle_dir.resolve():
            raise ContractValidationError("state_file must be inside bundle_dir.")
        return path
