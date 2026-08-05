"""Versioned JSON/CSV schemas only; export writers are intentionally deferred."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from core.contracts.common import FrameReference, ModelProvenance
from core.contracts.validation import (
    ContractValidationError,
    JsonValue,
    metadata_as_dict,
    normalize_metadata,
    require_nonempty,
)
from core.measurement.contracts import (
    MeasurementEstimate,
    ObjectMeasurementResult,
    UncertaintyComponent,
)


MEASUREMENT_SCHEMA_VERSION = 1


class ArtifactRole(str, Enum):
    RGB_IMAGE = "rgb_image"
    ANNOTATED_IMAGE = "annotated_image"
    INSTANCE_MASK = "instance_mask"
    NUMERICAL_DEPTH = "numerical_depth"
    DEPTH_PREVIEW = "depth_preview"
    RAW_OBJECT_POINT_CLOUD = "raw_object_point_cloud"
    FILTERED_OBJECT_POINT_CLOUD = "filtered_object_point_cloud"
    BOX_3D = "box_3d"
    MEASUREMENT_JSON = "measurement_json"
    MEASUREMENT_CSV = "measurement_csv"
    PROFESSIONAL_REPORT = "professional_report"


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    role: ArtifactRole
    relative_path: str
    media_type: str
    shape: tuple[int, ...] = ()
    dtype: str = ""
    sha256: str = ""

    def __post_init__(self) -> None:
        path = PurePosixPath(require_nonempty(self.relative_path, field_name="relative_path"))
        if path.is_absolute() or ".." in path.parts:
            raise ContractValidationError(
                "Artifact paths must be safe relative POSIX paths without traversal."
            )
        object.__setattr__(self, "relative_path", path.as_posix())
        object.__setattr__(self, "media_type", require_nonempty(self.media_type, field_name="media_type"))
        shape = tuple(int(value) for value in self.shape)
        if any(value <= 0 for value in shape):
            raise ContractValidationError("Artifact shape values must be positive.")
        object.__setattr__(self, "shape", shape)
        digest = str(self.sha256).strip().lower()
        if digest and (len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest)):
            raise ContractValidationError("sha256 must be a 64-character hexadecimal digest.")
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "dtype", str(self.dtype).strip())


@dataclass(frozen=True, slots=True)
class MeasurementManifest:
    application_version: str
    frame_reference: FrameReference
    object_results: tuple[ObjectMeasurementResult, ...]
    depth_model_provenance: tuple[ModelProvenance, ...] = ()
    perception_model_provenance: tuple[ModelProvenance, ...] = ()
    artifacts: tuple[ArtifactReference, ...] = ()
    metadata: Mapping[str, JsonValue] | Sequence[tuple[str, JsonValue]] = field(
        default_factory=tuple
    )
    schema_version: int = MEASUREMENT_SCHEMA_VERSION
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        if int(self.schema_version) != MEASUREMENT_SCHEMA_VERSION:
            raise ContractValidationError(
                f"Unsupported measurement schema version: {self.schema_version}."
            )
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(
            self,
            "application_version",
            require_nonempty(self.application_version, field_name="application_version"),
        )
        object.__setattr__(self, "created_at_utc", require_nonempty(self.created_at_utc, field_name="created_at_utc"))
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))
        artifact_paths = [artifact.relative_path for artifact in self.artifacts]
        if len(set(artifact_paths)) != len(artifact_paths):
            raise ContractValidationError("Artifact relative paths must be unique.")
        for result in self.object_results:
            if result.frame_reference != self.frame_reference:
                raise ContractValidationError(
                    "Every object result must match the manifest frame reference."
                )


MEASUREMENT_CSV_COLUMNS = (
    "session_id",
    "frame_id",
    "observation_id",
    "track_id",
    "class_name",
    "measurement_kind",
    "status",
    "value_si",
    "unit_si",
    "expanded_uncertainty_si",
    "lower_bound_si",
    "upper_bound_si",
    "uncertainty_basis",
    "reliability",
    "valid_observation_count",
    "depth_scale",
    "scale_recovery_status",
    "intrinsics_source",
    "calibration_status",
    "measurement_method",
    "rejection_reasons",
)


def uncertainty_component_to_schema(component: UncertaintyComponent) -> dict[str, Any]:
    return {
        "kind": component.kind.value,
        "contribution_si": component.contribution_si,
        "method": component.method,
        "sample_count": component.sample_count,
        "lower_contribution_si": component.lower_contribution_si,
        "upper_contribution_si": component.upper_contribution_si,
        "note": component.note,
    }


def measurement_to_schema(estimate: MeasurementEstimate) -> dict[str, Any]:
    uncertainty = estimate.uncertainty
    return {
        "measurement_id": estimate.measurement_id,
        "observation_id": estimate.observation_id,
        "kind": estimate.kind.value,
        "status": estimate.status.value,
        "method": estimate.method,
        "unit_si": estimate.unit_si,
        "value_si": estimate.value_si,
        "expanded_uncertainty_si": (
            uncertainty.expanded_uncertainty_si if uncertainty is not None else None
        ),
        "lower_bound_si": estimate.lower_bound_si,
        "upper_bound_si": estimate.upper_bound_si,
        "lower_bound_policy": estimate.lower_bound_policy.value,
        "uncertainty_basis": uncertainty.basis.value if uncertainty is not None else None,
        "uncertainty_display_qualification": (
            uncertainty.display_qualification if uncertainty is not None else None
        ),
        "coverage_factor": uncertainty.coverage_factor if uncertainty is not None else None,
        "coverage_probability": (
            uncertainty.coverage_probability if uncertainty is not None else None
        ),
        "uncertainty_components": (
            [uncertainty_component_to_schema(item) for item in uncertainty.components]
            if uncertainty is not None
            else []
        ),
        "reliability": estimate.reliability.level.value,
        "reliability_score": estimate.reliability.score,
        "valid_observation_count": estimate.valid_observation_count,
        "depth_scale": estimate.depth_scale.value,
        "intrinsics_source": estimate.intrinsics_source.value,
        "calibration_status": estimate.calibration_status.value,
        "scale_recovery_status": estimate.scale_recovery.status.value,
        "rejection_reasons": [value.value for value in estimate.rejection_reasons],
        "warnings": list(estimate.warnings),
        "model_provenance": [value.as_dict() for value in estimate.model_provenance],
        "processing_steps": [value.as_dict() for value in estimate.processing_steps],
    }


def validate_serialized_measurement_bounds(payload: Mapping[str, Any]) -> None:
    """Reject serialized convenience bounds inconsistent with authoritative values."""

    value = payload.get("value_si")
    uncertainty = payload.get("expanded_uncertainty_si")
    if value is None and uncertainty is None:
        if payload.get("lower_bound_si") is not None or payload.get("upper_bound_si") is not None:
            raise ContractValidationError("Bounds cannot be present without a value and uncertainty.")
        return
    if value is None or uncertainty is None:
        raise ContractValidationError("value_si and expanded_uncertainty_si must appear together.")
    value_f = float(value)
    uncertainty_f = float(uncertainty)
    expected_lower = value_f - uncertainty_f
    if payload.get("lower_bound_policy") == "clip_to_zero":
        expected_lower = max(0.0, expected_lower)
    expected_upper = value_f + uncertainty_f
    lower = payload.get("lower_bound_si")
    upper = payload.get("upper_bound_si")
    if lower is not None and not np.isclose(float(lower), expected_lower, rtol=1e-9, atol=1e-12):
        raise ContractValidationError("Serialized lower bound is inconsistent.")
    if upper is not None and not np.isclose(float(upper), expected_upper, rtol=1e-9, atol=1e-12):
        raise ContractValidationError("Serialized upper bound is inconsistent.")


def manifest_to_schema(manifest: MeasurementManifest) -> dict[str, Any]:
    return {
        "schema_version": manifest.schema_version,
        "application_version": manifest.application_version,
        "created_at_utc": manifest.created_at_utc,
        "frame_reference": manifest.frame_reference.as_dict(),
        "depth_model_provenance": [value.as_dict() for value in manifest.depth_model_provenance],
        "perception_model_provenance": [
            value.as_dict() for value in manifest.perception_model_provenance
        ],
        "objects": [
            {
                "observation_id": result.observation_id,
                "tracking_state": result.tracking_state.value,
                "measurements": [measurement_to_schema(value) for value in result.estimates],
                "warnings": list(result.warnings),
            }
            for result in manifest.object_results
        ],
        "artifacts": [
            {
                "role": artifact.role.value,
                "relative_path": artifact.relative_path,
                "media_type": artifact.media_type,
                "shape": list(artifact.shape),
                "dtype": artifact.dtype,
                "sha256": artifact.sha256,
            }
            for artifact in manifest.artifacts
        ],
        "metadata": metadata_as_dict(manifest.metadata),  # type: ignore[arg-type]
    }
