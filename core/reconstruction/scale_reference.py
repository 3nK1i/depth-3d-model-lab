"""Scale/reference contracts for relative and metric reconstruction geometry.

Phase 8.6B keeps raw monocular/SfM/SLAM geometry arbitrary by default and only
introduces metric units when an explicit, traceable scale reference exists.
Camera calibration and metric scale remain independent concerns: a known-size
reference can establish metres per reconstruction unit while estimated camera
intrinsics still make the resulting geometry provisional.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from math import hypot, isfinite
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from core.contracts.validation import require_nonempty
from core.reconstruction.camera_geometry import CameraGeometryProfile, CameraGeometryPurpose
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.sfm_contracts import (
    ReconstructionGauge,
    SfmScaleStatus,
    SfmWorldAlignmentStatus,
)

SCALE_REFERENCE_SCHEMA_VERSION = 1


class ScaleReferenceMode(str, Enum):
    ARBITRARY = "arbitrary"
    KNOWN_DIMENSION = "known_dimension"
    REFERENCE_TARGET = "reference_target"
    EXTERNAL_METRIC_PRIOR = "external_metric_prior"


class ScaleReferenceMethod(str, Enum):
    NONE = "none"
    KNOWN_DISTANCE = "known_distance"
    REFERENCE_TARGET_DISTANCE = "reference_target_distance"
    EXTERNAL_UNIT_SCALE = "external_unit_scale"


class ScaleReferencePurpose(str, Enum):
    RELATIVE_GEOMETRY = "relative_geometry"
    METRIC_COORDINATES = "metric_coordinates"
    CALIBRATED_METRIC_GEOMETRY = "calibrated_metric_geometry"


class MetricGeometryStatus(str, Enum):
    ARBITRARY = "arbitrary"
    PROVISIONAL_METRIC = "provisional_metric"
    CALIBRATED_METRIC = "calibrated_metric"


@dataclass(frozen=True, slots=True)
class ScaleReferenceAssessment:
    purpose: ScaleReferencePurpose
    allowed: bool
    provisional: bool
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.purpose, ScaleReferencePurpose):
            raise TypeError("purpose must be ScaleReferencePurpose.")
        object.__setattr__(self, "allowed", bool(self.allowed))
        object.__setattr__(self, "provisional", bool(self.provisional))
        object.__setattr__(self, "message", str(self.message).strip())

    def as_dict(self) -> dict[str, Any]:
        return {
            "purpose": self.purpose.value,
            "allowed": self.allowed,
            "provisional": self.provisional,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class ReconstructionGeometryAssessment:
    metric_status: MetricGeometryStatus
    relative_reconstruction_allowed: bool
    metric_coordinates_allowed: bool
    calibrated_metric_geometry_allowed: bool
    provisional: bool
    certified_metrology: bool
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.metric_status, MetricGeometryStatus):
            raise TypeError("metric_status must be MetricGeometryStatus.")
        object.__setattr__(
            self,
            "relative_reconstruction_allowed",
            bool(self.relative_reconstruction_allowed),
        )
        object.__setattr__(self, "metric_coordinates_allowed", bool(self.metric_coordinates_allowed))
        object.__setattr__(
            self,
            "calibrated_metric_geometry_allowed",
            bool(self.calibrated_metric_geometry_allowed),
        )
        object.__setattr__(self, "provisional", bool(self.provisional))
        # Phase 8 reconstruction is explicitly estimated geometry, not certified metrology.
        if bool(self.certified_metrology):
            raise ReconstructionValidationError(
                "Phase 8 geometry assessments must not claim certified metrology."
            )
        object.__setattr__(self, "certified_metrology", False)
        object.__setattr__(self, "message", str(self.message).strip())

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_status": self.metric_status.value,
            "relative_reconstruction_allowed": self.relative_reconstruction_allowed,
            "metric_coordinates_allowed": self.metric_coordinates_allowed,
            "calibrated_metric_geometry_allowed": self.calibrated_metric_geometry_allowed,
            "provisional": self.provisional,
            "certified_metrology": self.certified_metrology,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class ScaleReferenceProfile:
    profile_id: str
    mode: ScaleReferenceMode
    method: ScaleReferenceMethod
    gauge: ReconstructionGauge
    reference_id: str = ""
    reference_length_m: float | None = None
    measured_length_reconstruction_units: float | None = None
    reference_relative_uncertainty: float | None = None
    measurement_relative_uncertainty: float | None = None
    validated_reference: bool = False
    provenance: str = ""
    notes: str = ""
    schema_version: int = SCALE_REFERENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "profile_id",
            require_nonempty(self.profile_id, field_name="profile_id"),
        )
        if not isinstance(self.mode, ScaleReferenceMode):
            raise TypeError("mode must be ScaleReferenceMode.")
        if not isinstance(self.method, ScaleReferenceMethod):
            raise TypeError("method must be ScaleReferenceMethod.")
        if not isinstance(self.gauge, ReconstructionGauge):
            raise TypeError("gauge must be ReconstructionGauge.")
        if int(self.schema_version) != SCALE_REFERENCE_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported scale-reference schema version.")

        reference_id = str(self.reference_id).strip()
        provenance = str(self.provenance).strip()
        notes = str(self.notes).strip()
        validated = bool(self.validated_reference)

        ref_length = _optional_positive(self.reference_length_m, "reference_length_m")
        measured = _optional_positive(
            self.measured_length_reconstruction_units,
            "measured_length_reconstruction_units",
        )
        ref_uncertainty = _optional_relative_uncertainty(
            self.reference_relative_uncertainty,
            "reference_relative_uncertainty",
        )
        measurement_uncertainty = _optional_relative_uncertainty(
            self.measurement_relative_uncertainty,
            "measurement_relative_uncertainty",
        )

        if self.mode is ScaleReferenceMode.ARBITRARY:
            if self.method is not ScaleReferenceMethod.NONE:
                raise ReconstructionValidationError("Arbitrary scale requires method=NONE.")
            if self.gauge.scale_status is not SfmScaleStatus.ARBITRARY:
                raise ReconstructionValidationError("Arbitrary scale requires an arbitrary gauge.")
            if self.gauge.unit_to_metre is not None:
                raise ReconstructionValidationError("Arbitrary scale must not declare unit_to_metre.")
            if any(value is not None for value in (ref_length, measured, ref_uncertainty, measurement_uncertainty)):
                raise ReconstructionValidationError(
                    "Arbitrary scale must not contain metric-reference measurements."
                )
            if reference_id or provenance or validated:
                raise ReconstructionValidationError(
                    "Arbitrary scale must not claim a metric reference or provenance."
                )
        else:
            if self.gauge.scale_status is SfmScaleStatus.ARBITRARY:
                raise ReconstructionValidationError("Metric scale references require a metric gauge.")
            if self.gauge.unit_to_metre is None:
                raise ReconstructionValidationError("Metric scale references require unit_to_metre.")
            if not reference_id:
                raise ReconstructionValidationError("Metric scale references require reference_id.")
            if not provenance:
                raise ReconstructionValidationError("Metric scale references require provenance.")

            expected_status = (
                SfmScaleStatus.CALIBRATED_METRIC if validated else SfmScaleStatus.ESTIMATED_METRIC
            )
            if self.gauge.scale_status is not expected_status:
                raise ReconstructionValidationError(
                    "Scale gauge status must match validated_reference semantics."
                )

            if self.mode in {ScaleReferenceMode.KNOWN_DIMENSION, ScaleReferenceMode.REFERENCE_TARGET}:
                if ref_length is None or measured is None:
                    raise ReconstructionValidationError(
                        "Dimension-based scale references require physical and reconstruction lengths."
                    )
                expected_factor = ref_length / measured
                if not np.isclose(
                    expected_factor,
                    float(self.gauge.unit_to_metre),
                    rtol=1e-10,
                    atol=1e-15,
                ):
                    raise ReconstructionValidationError(
                        "unit_to_metre must equal reference_length_m / measured_length_reconstruction_units."
                    )
                combined = _combine_relative_uncertainty(ref_uncertainty, measurement_uncertainty)
                if self.gauge.relative_scale_uncertainty is None:
                    raise ReconstructionValidationError(
                        "Dimension-based metric scale requires relative_scale_uncertainty."
                    )
                if not np.isclose(
                    float(self.gauge.relative_scale_uncertainty),
                    combined,
                    rtol=1e-10,
                    atol=1e-15,
                ):
                    raise ReconstructionValidationError(
                        "Gauge uncertainty must match the combined scale-reference uncertainty."
                    )

            if self.mode is ScaleReferenceMode.KNOWN_DIMENSION:
                if self.method is not ScaleReferenceMethod.KNOWN_DISTANCE:
                    raise ReconstructionValidationError(
                        "Known-dimension references require KNOWN_DISTANCE method."
                    )
                if validated:
                    raise ReconstructionValidationError(
                        "Known-dimension scale is provisional; use REFERENCE_TARGET for a validated reference."
                    )
            elif self.mode is ScaleReferenceMode.REFERENCE_TARGET:
                if self.method is not ScaleReferenceMethod.REFERENCE_TARGET_DISTANCE:
                    raise ReconstructionValidationError(
                        "Reference-target scale requires REFERENCE_TARGET_DISTANCE method."
                    )
            elif self.mode is ScaleReferenceMode.EXTERNAL_METRIC_PRIOR:
                if self.method is not ScaleReferenceMethod.EXTERNAL_UNIT_SCALE:
                    raise ReconstructionValidationError(
                        "External metric priors require EXTERNAL_UNIT_SCALE method."
                    )
                if ref_length is not None or measured is not None:
                    raise ReconstructionValidationError(
                        "External metric priors declare unit_to_metre directly, not a dimension pair."
                    )
                if self.gauge.relative_scale_uncertainty is None:
                    raise ReconstructionValidationError(
                        "External metric priors require explicit relative_scale_uncertainty."
                    )

        object.__setattr__(self, "reference_id", reference_id)
        object.__setattr__(self, "reference_length_m", ref_length)
        object.__setattr__(self, "measured_length_reconstruction_units", measured)
        object.__setattr__(self, "reference_relative_uncertainty", ref_uncertainty)
        object.__setattr__(self, "measurement_relative_uncertainty", measurement_uncertainty)
        object.__setattr__(self, "validated_reference", validated)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "notes", notes)
        object.__setattr__(self, "schema_version", SCALE_REFERENCE_SCHEMA_VERSION)

    @property
    def is_metric(self) -> bool:
        return self.gauge.is_metric

    @property
    def is_calibrated_metric(self) -> bool:
        return self.gauge.scale_status is SfmScaleStatus.CALIBRATED_METRIC

    def assess(self, purpose: ScaleReferencePurpose) -> ScaleReferenceAssessment:
        if not isinstance(purpose, ScaleReferencePurpose):
            raise TypeError("purpose must be ScaleReferencePurpose.")
        if purpose is ScaleReferencePurpose.RELATIVE_GEOMETRY:
            return ScaleReferenceAssessment(
                purpose=purpose,
                allowed=True,
                provisional=not self.is_calibrated_metric,
                message="Relative reconstruction geometry does not require metric scale.",
            )
        if purpose is ScaleReferencePurpose.METRIC_COORDINATES:
            if not self.is_metric:
                return ScaleReferenceAssessment(
                    purpose=purpose,
                    allowed=False,
                    provisional=True,
                    message="Metric coordinates require an explicit metric scale reference.",
                )
            return ScaleReferenceAssessment(
                purpose=purpose,
                allowed=True,
                provisional=not self.is_calibrated_metric,
                message=(
                    "Metric units are available from a traceable scale reference; accuracy remains "
                    "provisional unless both camera geometry and scale reference are calibrated."
                ),
            )
        return ScaleReferenceAssessment(
            purpose=purpose,
            allowed=self.is_calibrated_metric,
            provisional=not self.is_calibrated_metric,
            message=(
                "Calibrated metric scale is available from a validated reference."
                if self.is_calibrated_metric
                else "Calibrated metric geometry requires a validated metric reference."
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "mode": self.mode.value,
            "method": self.method.value,
            "gauge": self.gauge.as_dict(),
            "reference_id": self.reference_id,
            "reference_length_m": self.reference_length_m,
            "measured_length_reconstruction_units": self.measured_length_reconstruction_units,
            "reference_relative_uncertainty": self.reference_relative_uncertainty,
            "measurement_relative_uncertainty": self.measurement_relative_uncertainty,
            "validated_reference": self.validated_reference,
            "provenance": self.provenance,
            "notes": self.notes,
        }


def arbitrary_scale_reference(*, profile_id: str, notes: str = "") -> ScaleReferenceProfile:
    return ScaleReferenceProfile(
        profile_id=profile_id,
        mode=ScaleReferenceMode.ARBITRARY,
        method=ScaleReferenceMethod.NONE,
        gauge=ReconstructionGauge(),
        notes=notes,
    )


def known_dimension_scale_reference(
    *,
    profile_id: str,
    reference_id: str,
    known_length_m: float,
    measured_length_reconstruction_units: float,
    known_length_relative_uncertainty: float = 0.0,
    measurement_relative_uncertainty: float = 0.02,
    provenance: str,
    notes: str = "",
) -> ScaleReferenceProfile:
    physical = _positive(known_length_m, "known_length_m")
    measured = _positive(
        measured_length_reconstruction_units,
        "measured_length_reconstruction_units",
    )
    ref_uncertainty = _relative_uncertainty(
        known_length_relative_uncertainty,
        "known_length_relative_uncertainty",
    )
    measurement_uncertainty = _relative_uncertainty(
        measurement_relative_uncertainty,
        "measurement_relative_uncertainty",
    )
    combined = hypot(ref_uncertainty, measurement_uncertainty)
    source = f"known_dimension:{require_nonempty(reference_id, field_name='reference_id')}"
    return ScaleReferenceProfile(
        profile_id=profile_id,
        mode=ScaleReferenceMode.KNOWN_DIMENSION,
        method=ScaleReferenceMethod.KNOWN_DISTANCE,
        gauge=ReconstructionGauge(
            scale_status=SfmScaleStatus.ESTIMATED_METRIC,
            alignment_status=SfmWorldAlignmentStatus.ARBITRARY,
            unit_to_metre=physical / measured,
            scale_source=source,
            relative_scale_uncertainty=combined,
        ),
        reference_id=reference_id,
        reference_length_m=physical,
        measured_length_reconstruction_units=measured,
        reference_relative_uncertainty=ref_uncertainty,
        measurement_relative_uncertainty=measurement_uncertainty,
        validated_reference=False,
        provenance=provenance,
        notes=notes,
    )


def reference_target_scale_reference(
    *,
    profile_id: str,
    reference_id: str,
    target_length_m: float,
    measured_length_reconstruction_units: float,
    target_relative_uncertainty: float,
    measurement_relative_uncertainty: float = 0.01,
    validated_target: bool = True,
    provenance: str,
    notes: str = "",
) -> ScaleReferenceProfile:
    physical = _positive(target_length_m, "target_length_m")
    measured = _positive(
        measured_length_reconstruction_units,
        "measured_length_reconstruction_units",
    )
    target_uncertainty = _relative_uncertainty(
        target_relative_uncertainty,
        "target_relative_uncertainty",
    )
    measurement_uncertainty = _relative_uncertainty(
        measurement_relative_uncertainty,
        "measurement_relative_uncertainty",
    )
    combined = hypot(target_uncertainty, measurement_uncertainty)
    reference_id = require_nonempty(reference_id, field_name="reference_id")
    status = SfmScaleStatus.CALIBRATED_METRIC if validated_target else SfmScaleStatus.ESTIMATED_METRIC
    source = f"reference_target:{reference_id}"
    return ScaleReferenceProfile(
        profile_id=profile_id,
        mode=ScaleReferenceMode.REFERENCE_TARGET,
        method=ScaleReferenceMethod.REFERENCE_TARGET_DISTANCE,
        gauge=ReconstructionGauge(
            scale_status=status,
            alignment_status=SfmWorldAlignmentStatus.ARBITRARY,
            unit_to_metre=physical / measured,
            scale_source=source,
            relative_scale_uncertainty=combined,
        ),
        reference_id=reference_id,
        reference_length_m=physical,
        measured_length_reconstruction_units=measured,
        reference_relative_uncertainty=target_uncertainty,
        measurement_relative_uncertainty=measurement_uncertainty,
        validated_reference=bool(validated_target),
        provenance=provenance,
        notes=notes,
    )


def external_metric_prior_scale_reference(
    *,
    profile_id: str,
    reference_id: str,
    unit_to_metre: float,
    relative_scale_uncertainty: float,
    validated_prior: bool,
    provenance: str,
    notes: str = "",
) -> ScaleReferenceProfile:
    factor = _positive(unit_to_metre, "unit_to_metre")
    uncertainty = _relative_uncertainty(
        relative_scale_uncertainty,
        "relative_scale_uncertainty",
    )
    reference_id = require_nonempty(reference_id, field_name="reference_id")
    status = SfmScaleStatus.CALIBRATED_METRIC if validated_prior else SfmScaleStatus.ESTIMATED_METRIC
    return ScaleReferenceProfile(
        profile_id=profile_id,
        mode=ScaleReferenceMode.EXTERNAL_METRIC_PRIOR,
        method=ScaleReferenceMethod.EXTERNAL_UNIT_SCALE,
        gauge=ReconstructionGauge(
            scale_status=status,
            alignment_status=SfmWorldAlignmentStatus.ARBITRARY,
            unit_to_metre=factor,
            scale_source=f"external_metric_prior:{reference_id}",
            relative_scale_uncertainty=uncertainty,
        ),
        reference_id=reference_id,
        validated_reference=bool(validated_prior),
        provenance=provenance,
        notes=notes,
    )


def assess_reconstruction_geometry(
    camera_profile: CameraGeometryProfile,
    scale_profile: ScaleReferenceProfile,
) -> ReconstructionGeometryAssessment:
    if not isinstance(camera_profile, CameraGeometryProfile):
        raise TypeError("camera_profile must be CameraGeometryProfile.")
    if not isinstance(scale_profile, ScaleReferenceProfile):
        raise TypeError("scale_profile must be ScaleReferenceProfile.")

    relative_camera = camera_profile.assess(CameraGeometryPurpose.RELATIVE_RECONSTRUCTION)
    metric_scale = scale_profile.assess(ScaleReferencePurpose.METRIC_COORDINATES)
    calibrated_scale = scale_profile.assess(ScaleReferencePurpose.CALIBRATED_METRIC_GEOMETRY)
    calibrated_camera = camera_profile.assess(CameraGeometryPurpose.CALIBRATED_RAY_GEOMETRY)

    calibrated_metric = calibrated_camera.allowed and calibrated_scale.allowed
    if calibrated_metric:
        status = MetricGeometryStatus.CALIBRATED_METRIC
        message = (
            "Camera intrinsics and metric scale reference are calibrated. Geometry can use "
            "calibrated metric coordinates, but Phase 8 does not claim certified metrology."
        )
    elif metric_scale.allowed:
        status = MetricGeometryStatus.PROVISIONAL_METRIC
        message = (
            "Metric coordinates are available, but camera geometry and/or the scale reference "
            "remain provisional. Do not treat the result as calibrated metrology."
        )
    else:
        status = MetricGeometryStatus.ARBITRARY
        message = "Only relative arbitrary-scale reconstruction geometry is established."

    return ReconstructionGeometryAssessment(
        metric_status=status,
        relative_reconstruction_allowed=relative_camera.allowed,
        metric_coordinates_allowed=relative_camera.allowed and metric_scale.allowed,
        calibrated_metric_geometry_allowed=relative_camera.allowed and calibrated_metric,
        provisional=status is not MetricGeometryStatus.CALIBRATED_METRIC,
        certified_metrology=False,
        message=message,
    )


def reconstruction_distance_to_metres(
    distance_reconstruction_units: float,
    scale_profile: ScaleReferenceProfile,
) -> float:
    if not isinstance(scale_profile, ScaleReferenceProfile):
        raise TypeError("scale_profile must be ScaleReferenceProfile.")
    if not scale_profile.is_metric or scale_profile.gauge.unit_to_metre is None:
        raise ReconstructionValidationError(
            "Cannot convert arbitrary reconstruction units to metres without a metric scale reference."
        )
    distance = float(distance_reconstruction_units)
    if not isfinite(distance) or distance < 0.0:
        raise ReconstructionValidationError(
            "distance_reconstruction_units must be finite and non-negative."
        )
    return distance * float(scale_profile.gauge.unit_to_metre)


def reconstruction_points_to_metres(
    points_reconstruction_units: object,
    scale_profile: ScaleReferenceProfile,
) -> np.ndarray:
    if not isinstance(scale_profile, ScaleReferenceProfile):
        raise TypeError("scale_profile must be ScaleReferenceProfile.")
    if not scale_profile.is_metric or scale_profile.gauge.unit_to_metre is None:
        raise ReconstructionValidationError(
            "Cannot convert arbitrary reconstruction points to metres without a metric scale reference."
        )
    points = np.asarray(points_reconstruction_units, dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] != 3 or not np.isfinite(points).all():
        raise ReconstructionValidationError(
            "points_reconstruction_units must be finite with final dimension 3."
        )
    scaled = np.ascontiguousarray(points * float(scale_profile.gauge.unit_to_metre))
    scaled.setflags(write=False)
    return scaled


def scale_reference_profile_to_json(
    profile: ScaleReferenceProfile,
    *,
    indent: int | None = 2,
) -> str:
    if not isinstance(profile, ScaleReferenceProfile):
        raise TypeError("profile must be ScaleReferenceProfile.")
    return json.dumps(
        profile.as_dict(),
        indent=indent,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        allow_nan=False,
    )


def scale_reference_profile_from_json(payload: str | bytes | bytearray) -> ScaleReferenceProfile:
    try:
        raw = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        raise ReconstructionValidationError("Scale-reference profile must be valid JSON.") from exc
    if not isinstance(raw, Mapping):
        raise ReconstructionValidationError("Scale-reference profile root must be an object.")
    return scale_reference_profile_from_dict(raw)


def scale_reference_profile_from_dict(raw: Mapping[str, Any]) -> ScaleReferenceProfile:
    try:
        if int(raw["schema_version"]) != SCALE_REFERENCE_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported scale-reference schema version.")
        gauge_raw = _mapping(raw["gauge"], "gauge")
        gauge = ReconstructionGauge(
            scale_status=SfmScaleStatus(str(gauge_raw["scale_status"])),
            alignment_status=SfmWorldAlignmentStatus(str(gauge_raw["alignment_status"])),
            unit_to_metre=(
                None if gauge_raw.get("unit_to_metre") is None else float(gauge_raw["unit_to_metre"])
            ),
            scale_source=str(gauge_raw.get("scale_source", "")),
            relative_scale_uncertainty=(
                None
                if gauge_raw.get("relative_scale_uncertainty") is None
                else float(gauge_raw["relative_scale_uncertainty"])
            ),
        )
        return ScaleReferenceProfile(
            profile_id=str(raw["profile_id"]),
            mode=ScaleReferenceMode(str(raw["mode"])),
            method=ScaleReferenceMethod(str(raw["method"])),
            gauge=gauge,
            reference_id=str(raw.get("reference_id", "")),
            reference_length_m=(
                None if raw.get("reference_length_m") is None else float(raw["reference_length_m"])
            ),
            measured_length_reconstruction_units=(
                None
                if raw.get("measured_length_reconstruction_units") is None
                else float(raw["measured_length_reconstruction_units"])
            ),
            reference_relative_uncertainty=(
                None
                if raw.get("reference_relative_uncertainty") is None
                else float(raw["reference_relative_uncertainty"])
            ),
            measurement_relative_uncertainty=(
                None
                if raw.get("measurement_relative_uncertainty") is None
                else float(raw["measurement_relative_uncertainty"])
            ),
            validated_reference=bool(raw.get("validated_reference", False)),
            provenance=str(raw.get("provenance", "")),
            notes=str(raw.get("notes", "")),
        )
    except KeyError as exc:
        raise ReconstructionValidationError(
            f"Scale-reference profile is missing required field: {exc.args[0]}."
        ) from exc
    except ReconstructionValidationError:
        raise
    except (TypeError, ValueError) as exc:
        raise ReconstructionValidationError(
            f"Scale-reference profile contains an invalid value: {exc}."
        ) from exc


def load_scale_reference_profile(path: str | Path) -> ScaleReferenceProfile:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ReconstructionValidationError(f"Scale-reference profile does not exist: {source}")
    try:
        return scale_reference_profile_from_json(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ReconstructionValidationError(f"Could not read scale-reference profile: {source}") from exc


def save_scale_reference_profile(profile: ScaleReferenceProfile, path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.write_text(scale_reference_profile_to_json(profile) + "\n", encoding="utf-8")
    except OSError as exc:
        raise ReconstructionValidationError(
            f"Could not write scale-reference profile: {destination}"
        ) from exc
    return destination


def _positive(value: float, field_name: str) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ReconstructionValidationError(f"{field_name} must be finite and positive.")
    return result


def _optional_positive(value: float | None, field_name: str) -> float | None:
    return None if value is None else _positive(value, field_name)


def _relative_uncertainty(value: float, field_name: str) -> float:
    result = float(value)
    if not isfinite(result) or result < 0.0 or result >= 1.0:
        raise ReconstructionValidationError(f"{field_name} must be finite and in [0, 1).")
    return result


def _optional_relative_uncertainty(value: float | None, field_name: str) -> float | None:
    return None if value is None else _relative_uncertainty(value, field_name)


def _combine_relative_uncertainty(first: float | None, second: float | None) -> float:
    if first is None or second is None:
        raise ReconstructionValidationError(
            "Dimension-based scale references require both reference and measurement uncertainty."
        )
    return hypot(first, second)


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"{field_name} must be an object.")
    return value
