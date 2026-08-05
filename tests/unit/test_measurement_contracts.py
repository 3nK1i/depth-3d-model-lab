from __future__ import annotations

import numpy as np
import pytest

from core.contracts import CalibrationStatus, FrameReference, ImageSize, ModelProvenance
from core.contracts.validation import ContractValidationError
from core.measurement import (
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
    UncertaintyBasis,
    UncertaintyEstimate,
    available_measurements,
)
from core.models.metadata import DepthScale, IntrinsicsSource


def _frame() -> FrameReference:
    return FrameReference(1, 3, 4.0, ImageSize(100, 80))


def _provenance() -> ModelProvenance:
    return ModelProvenance("depth-test", "Depth Test", "depth")


def _reliability(level: ReliabilityLevel = ReliabilityLevel.MEDIUM) -> ReliabilityAssessment:
    return ReliabilityAssessment(level=level, score=0.7, policy_id="phase75-step1")


def _scale(status: ScaleRecoveryStatus = ScaleRecoveryStatus.NOT_REQUIRED) -> ScaleRecovery:
    return ScaleRecovery(status=status, reliability=ReliabilityLevel.MEDIUM)


def _available(kind: MeasurementKind, value: float, uncertainty: float, **overrides) -> MeasurementEstimate:
    values = dict(
        measurement_id=f"m-{kind.value}",
        observation_id="object-1",
        kind=kind,
        status=MeasurementStatus.AVAILABLE,
        method="robust-perturbation",
        unit_si="rad" if "orientation" in kind.value else ("m3" if kind is MeasurementKind.VOLUME else "m"),
        reliability=_reliability(),
        depth_scale=DepthScale.METRIC,
        intrinsics_source=IntrinsicsSource.ESTIMATED,
        calibration_status=CalibrationStatus.NOT_CALIBRATED,
        scale_recovery=_scale(),
        value_si=value,
        uncertainty=UncertaintyEstimate(
            basis=UncertaintyBasis.HEURISTIC_PERTURBATION,
            expanded_uncertainty_si=uncertainty,
            perturbation_sample_count=12,
        ),
        valid_observation_count=12,
        model_provenance=(_provenance(),),
    )
    values.update(overrides)
    return MeasurementEstimate(**values)


def test_uncertainty_basis_does_not_claim_coverage_before_empirical_calibration() -> None:
    with pytest.raises(ContractValidationError, match="empirically calibrated"):
        UncertaintyEstimate(
            basis=UncertaintyBasis.HEURISTIC_PERTURBATION,
            expanded_uncertainty_si=0.01,
            coverage_probability=0.95,
        )
    uncertainty = UncertaintyEstimate(
        basis=UncertaintyBasis.HEURISTIC_PERTURBATION,
        expanded_uncertainty_si=0.012,
    )
    assert uncertainty.display_qualification == "Estimated Uncertainty — Not Calibrated"


def test_authoritative_value_and_uncertainty_derive_consistent_bounds() -> None:
    estimate = _available(MeasurementKind.WIDTH, 0.421, 0.012)
    assert estimate.lower_bound_si == pytest.approx(0.409)
    assert estimate.upper_bound_si == pytest.approx(0.433)

    with pytest.raises(ContractValidationError, match="lower_bound_si"):
        _available(
            MeasurementKind.WIDTH,
            0.421,
            0.012,
            lower_bound_si=0.4,
        )


def test_nonnegative_bounds_require_rejection_or_explicit_clipping() -> None:
    with pytest.raises(ContractValidationError, match="negative"):
        _available(MeasurementKind.WIDTH, 0.01, 0.02)
    clipped = _available(
        MeasurementKind.WIDTH,
        0.01,
        0.02,
        lower_bound_policy=LowerBoundPolicy.CLIP_TO_ZERO,
    )
    assert clipped.lower_bound_si == 0.0


def test_eligibility_is_authoritative_per_measurement_kind() -> None:
    width = MeasurementEstimate(
        measurement_id="width",
        observation_id="object-1",
        kind=MeasurementKind.WIDTH,
        status=MeasurementStatus.UNAVAILABLE,
        method="border-aware-fit",
        unit_si="m",
        reliability=_reliability(ReliabilityLevel.LOW),
        depth_scale=DepthScale.METRIC,
        intrinsics_source=IntrinsicsSource.ESTIMATED,
        calibration_status=CalibrationStatus.NOT_CALIBRATED,
        scale_recovery=_scale(),
        rejection_reasons=(RejectionReason.BORDER_CONTACT,),
    )
    distance = _available(MeasurementKind.CAMERA_DISTANCE, 1.82, 0.07)
    height = _available(MeasurementKind.HEIGHT, 0.188, 0.009)
    result = ObjectMeasurementResult(
        observation_id="object-1",
        frame_reference=_frame(),
        estimates=(width, distance, height),
        warnings=("object touches left border",),
    )
    assert {item.kind for item in available_measurements(result)} == {
        MeasurementKind.CAMERA_DISTANCE,
        MeasurementKind.HEIGHT,
    }


def test_relative_depth_requires_successful_scale_recovery_per_measurement() -> None:
    with pytest.raises(ContractValidationError, match="scale recovery"):
        _available(
            MeasurementKind.HEIGHT,
            0.2,
            0.01,
            depth_scale=DepthScale.RELATIVE,
            scale_recovery=ScaleRecovery(
                status=ScaleRecoveryStatus.NOT_ATTEMPTED,
                reliability=ReliabilityLevel.UNAVAILABLE,
            ),
        )


def test_object_point_cloud_validates_explicit_source_pixel_space() -> None:
    cloud = ObjectPointCloud(
        observation_id="object-1",
        frame_reference=_frame(),
        coordinate_frame_id="camera-depth",
        points_xyz_m=np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.1]]),
        source_pixel_space_id="depth-space",
        source_pixel_space_size=ImageSize(100, 80),
        source_pixel_indices_xy=np.array([[4, 5], [99, 79]]),
        depth_scale=DepthScale.METRIC,
    )
    assert cloud.source_pixel_space_id == "depth-space"

    with pytest.raises(ContractValidationError, match="outside"):
        ObjectPointCloud(
            observation_id="object-1",
            frame_reference=_frame(),
            coordinate_frame_id="camera-depth",
            points_xyz_m=np.array([[0.0, 0.0, 1.0]]),
            source_pixel_space_id="mask-space",
            source_pixel_space_size=ImageSize(10, 8),
            source_pixel_indices_xy=np.array([[10, 0]]),
        )
