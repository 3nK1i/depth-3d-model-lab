"""Authoritative validation entry points for measurement contracts."""

from __future__ import annotations

from core.measurement.contracts import (
    MeasurementEstimate,
    MeasurementStatus,
    ObjectMeasurementResult,
)


def validate_measurement_estimate(estimate: MeasurementEstimate) -> MeasurementEstimate:
    if not isinstance(estimate, MeasurementEstimate):
        raise TypeError("estimate must be a MeasurementEstimate.")
    return estimate


def available_measurements(
    result: ObjectMeasurementResult,
) -> tuple[MeasurementEstimate, ...]:
    """Return per-kind available estimates without applying object-wide rejection."""

    return tuple(
        estimate
        for estimate in result.estimates
        if estimate.status is MeasurementStatus.AVAILABLE
    )
