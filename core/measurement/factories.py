"""Factories for Step 6 measurement computation."""
from core.measurement.pipeline import MeasurementComputationRequest
from core.measurement.policy import MeasurementComputationPolicy

def make_measurement_request(aligned_bundle, extraction_batch, fit_batch, *, observation_ids=(), policy=None, scale_recovery_by_observation=()):
    return MeasurementComputationRequest(aligned_bundle, extraction_batch, fit_batch, tuple(observation_ids), policy or MeasurementComputationPolicy(), tuple(scale_recovery_by_observation))
