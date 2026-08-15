"""Public object-geometry, measurement, uncertainty and export-schema contracts."""

from core.measurement.contracts import (
    BoundingBox3D,
    BoundingBoxKind,
    LowerBoundPolicy,
    MeasurementEstimate,
    MeasurementKind,
    MeasurementStatus,
    ObjectMeasurementResult,
    ObjectPointCloud,
    RejectionReason,
    ReliabilityAssessment,
    ReliabilityLevel,
    ScaleRecovery,
    ScaleRecoveryStatus,
    ShapeKind,
    ShapeModel,
    UncertaintyBasis,
    UncertaintyComponent,
    UncertaintyComponentKind,
    UncertaintyEstimate,
)
from core.measurement.export_schema import (
    MEASUREMENT_CSV_COLUMNS,
    MEASUREMENT_SCHEMA_VERSION,
    ArtifactReference,
    ArtifactRole,
    MeasurementManifest,
    manifest_to_schema,
    measurement_to_schema,
    validate_serialized_measurement_bounds,
)
from core.measurement.validation import available_measurements, validate_measurement_estimate
from core.tracking.contracts import TrackingState


from core.measurement.interactive_distance import (
    InteractiveDistanceEstimate,
    InteractiveDistanceState,
    estimate_depth_pixel_distance,
    estimate_reconstruction_distance,
    format_interactive_distance,
)

__all__ = [
    "format_interactive_distance",
    "estimate_reconstruction_distance",
    "estimate_depth_pixel_distance",
    "InteractiveDistanceState",
    "InteractiveDistanceEstimate",
    "ArtifactReference",
    "ArtifactRole",
    "BoundingBox3D",
    "BoundingBoxKind",
    "LowerBoundPolicy",
    "MEASUREMENT_CSV_COLUMNS",
    "MEASUREMENT_SCHEMA_VERSION",
    "MeasurementEstimate",
    "MeasurementKind",
    "MeasurementManifest",
    "MeasurementStatus",
    "ObjectMeasurementResult",
    "ObjectPointCloud",
    "RejectionReason",
    "ReliabilityAssessment",
    "ReliabilityLevel",
    "ScaleRecovery",
    "ScaleRecoveryStatus",
    "ShapeKind",
    "ShapeModel",
    "TrackingState",
    "UncertaintyBasis",
    "UncertaintyComponent",
    "UncertaintyComponentKind",
    "UncertaintyEstimate",
    "available_measurements",
    "manifest_to_schema",
    "measurement_to_schema",
    "validate_measurement_estimate",
    "validate_serialized_measurement_bounds",
]


_STEP6_EXPORTS = {
    "MeasurementBatchResult": ("core.measurement.computation_contracts", "MeasurementBatchResult"),
    "MeasurementComputationStatus": ("core.measurement.computation_contracts", "MeasurementComputationStatus"),
    "MeasurementDiagnostics": ("core.measurement.computation_contracts", "MeasurementDiagnostics"),
    "ObjectMeasurementComputationResult": ("core.measurement.computation_contracts", "ObjectMeasurementComputationResult"),
    "MeasurementComputationPolicy": ("core.measurement.policy", "MeasurementComputationPolicy"),
    "MeasurementSelectionPolicy": ("core.measurement.policy", "MeasurementSelectionPolicy"),
    "PerturbationPolicy": ("core.measurement.policy", "PerturbationPolicy"),
    "ReliabilityPolicy": ("core.measurement.policy", "ReliabilityPolicy"),
    "MeasurementComputationRequest": ("core.measurement.pipeline", "MeasurementComputationRequest"),
    "compute_measurements": ("core.measurement.pipeline", "compute_measurements"),
    "make_measurement_request": ("core.measurement.factories", "make_measurement_request"),
}

__all__ += list(_STEP6_EXPORTS)

def __getattr__(name):
    target = _STEP6_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    from importlib import import_module
    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value
