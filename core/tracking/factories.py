"""Convenience factories for Step 7 tracking requests."""
from __future__ import annotations
from core.measurement.computation_contracts import MeasurementBatchResult
from core.perception.result import PerceptionFrameResult
from core.tracking.contracts import TrackingContext
from core.tracking.pipeline import TemporalTrackingRequest


def make_tracking_request(perception_result: PerceptionFrameResult, measurement_batch: MeasurementBatchResult, *, depth_model_id: str, perception_model_id: str, worker_generation: int=0, alignment_config_digest: str="") -> TemporalTrackingRequest:
    return TemporalTrackingRequest(perception_result,measurement_batch,TrackingContext(depth_model_id,perception_model_id,worker_generation,alignment_config_digest))
