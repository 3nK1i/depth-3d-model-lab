"""Measurement-specific reliability assessment."""
from __future__ import annotations
from core.measurement.contracts import MeasurementKind, ReliabilityAssessment, ReliabilityLevel
from core.measurement.geometry import is_orientation_kind
from core.measurement.policy import ReliabilityPolicy
from core.object_geometry.fitting_contracts import ObjectGeometryFitResult, ObjectGeometryFitStatus, OrientationStability


def assess_reliability(*, kind: MeasurementKind, value: float, uncertainty: float, support_points: int, fit: ObjectGeometryFitResult, warnings: tuple[str,...], policy: ReliabilityPolicy, reference_scale: float | None = None) -> ReliabilityAssessment:
    reference=max(float(reference_scale) if reference_scale is not None else abs(value), 0.05 if is_orientation_kind(kind) else 1e-6)
    relative=uncertainty/reference
    passed=[]; failed=[]; explanations=[]
    if relative <= policy.high_relative_uncertainty: passed.append("low_relative_uncertainty")
    elif relative <= policy.medium_relative_uncertainty: passed.append("moderate_relative_uncertainty")
    else: failed.append("high_relative_uncertainty")
    if support_points >= policy.minimum_high_support_points: passed.append("high_point_support")
    elif support_points >= policy.minimum_medium_support_points: passed.append("adequate_point_support")
    else: failed.append("limited_point_support")
    level=ReliabilityLevel.HIGH if relative<=policy.high_relative_uncertainty and support_points>=policy.minimum_high_support_points else ReliabilityLevel.MEDIUM if relative<=policy.medium_relative_uncertainty and support_points>=policy.minimum_medium_support_points else ReliabilityLevel.LOW
    if fit.status is ObjectGeometryFitStatus.PARTIAL and level is ReliabilityLevel.HIGH: level=ReliabilityLevel.MEDIUM
    if is_orientation_kind(kind) and fit.principal_axes is not None:
        stability=fit.object_frame.orientation_stability
        if stability is OrientationStability.WEAK: level=ReliabilityLevel.LOW; failed.append("weak_orientation_stability")
        elif stability in {OrientationStability.AMBIGUOUS,OrientationStability.UNAVAILABLE}: level=ReliabilityLevel.UNAVAILABLE; failed.append("orientation_not_identifiable")
    if warnings: explanations.extend(warnings)
    score=max(0.0,min(1.0,1.0-relative/policy.maximum_relative_uncertainty))
    return ReliabilityAssessment(level,score,tuple(passed),tuple(dict.fromkeys(failed)),tuple(explanations),policy.policy_id)
