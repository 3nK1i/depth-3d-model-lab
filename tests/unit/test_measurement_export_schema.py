from __future__ import annotations

import json

import pytest

from core.contracts import CalibrationStatus, FrameReference, ImageSize, ModelProvenance
from core.contracts.validation import ContractValidationError
from core.measurement import (
    ArtifactReference,
    ArtifactRole,
    MeasurementEstimate,
    MeasurementKind,
    MeasurementManifest,
    MeasurementStatus,
    ObjectMeasurementResult,
    ReliabilityAssessment,
    ReliabilityLevel,
    ScaleRecovery,
    ScaleRecoveryStatus,
    UncertaintyBasis,
    UncertaintyEstimate,
    manifest_to_schema,
    measurement_to_schema,
    validate_serialized_measurement_bounds,
)
from core.models.metadata import DepthScale, IntrinsicsSource


def _estimate() -> MeasurementEstimate:
    provenance = ModelProvenance("depth", "Depth", "depth")
    return MeasurementEstimate(
        measurement_id="distance",
        observation_id="object-1",
        kind=MeasurementKind.CAMERA_DISTANCE,
        status=MeasurementStatus.AVAILABLE,
        method="median-depth",
        unit_si="m",
        reliability=ReliabilityAssessment(ReliabilityLevel.MEDIUM, score=0.7),
        depth_scale=DepthScale.METRIC,
        intrinsics_source=IntrinsicsSource.MODEL_PREDICTED,
        calibration_status=CalibrationStatus.NOT_CALIBRATED,
        scale_recovery=ScaleRecovery(
            ScaleRecoveryStatus.NOT_REQUIRED,
            reliability=ReliabilityLevel.MEDIUM,
        ),
        value_si=1.82,
        uncertainty=UncertaintyEstimate(
            UncertaintyBasis.HEURISTIC_PERTURBATION,
            expanded_uncertainty_si=0.07,
        ),
        valid_observation_count=10,
        model_provenance=(provenance,),
    )


def test_measurement_schema_serializes_derived_bounds_and_basis() -> None:
    payload = measurement_to_schema(_estimate())
    assert payload["lower_bound_si"] == pytest.approx(1.75)
    assert payload["upper_bound_si"] == pytest.approx(1.89)
    assert payload["coverage_probability"] is None
    assert payload["uncertainty_display_qualification"] == "Estimated Uncertainty — Not Calibrated"
    validate_serialized_measurement_bounds(payload)
    json.dumps(payload)


def test_serialized_bound_validation_rejects_inconsistency() -> None:
    payload = measurement_to_schema(_estimate())
    payload["upper_bound_si"] = 999.0
    with pytest.raises(ContractValidationError, match="upper bound"):
        validate_serialized_measurement_bounds(payload)


def test_artifact_paths_are_safe_and_manifest_is_json_serializable() -> None:
    frame = FrameReference(1, 1, 1.0, ImageSize(10, 10))
    result = ObjectMeasurementResult("object-1", frame, (_estimate(),))
    artifact = ArtifactReference(
        ArtifactRole.MEASUREMENT_JSON,
        "objects/object-1/measurement.json",
        "application/json",
    )
    manifest = MeasurementManifest(
        application_version="0.7.0",
        frame_reference=frame,
        object_results=(result,),
        artifacts=(artifact,),
    )
    json.dumps(manifest_to_schema(manifest))

    with pytest.raises(ContractValidationError, match="relative"):
        ArtifactReference(ArtifactRole.RGB_IMAGE, "../rgb.png", "image/png")
