"""Public Phase 7.5 temporal tracking API with cycle-safe lazy exports."""
from core.tracking.contracts import (
    ObjectTrackState,TrackAssociation,TrackingContext,TrackingResetEvent,
    TrackingResetReason,TrackingState,TrackingUpdateStatus,
)
from core.tracking.policy import AssociationPolicy,TemporalStabilizationPolicy,TemporalTrackingPolicy
from core.tracking.validation import validate_track_state

_LAZY={
    'StabilizedMeasurementEstimate':('core.tracking.temporal_contracts','StabilizedMeasurementEstimate'),
    'TemporalEstimateStatus':('core.tracking.temporal_contracts','TemporalEstimateStatus'),
    'TemporalTrackingBatchResult':('core.tracking.temporal_contracts','TemporalTrackingBatchResult'),
    'TrackedObjectResult':('core.tracking.temporal_contracts','TrackedObjectResult'),
    'TemporalTrackingEngine':('core.tracking.pipeline','TemporalTrackingEngine'),
    'TemporalTrackingRequest':('core.tracking.pipeline','TemporalTrackingRequest'),
    'make_tracking_request':('core.tracking.factories','make_tracking_request'),
    'validate_tracking_result':('core.tracking.validation','validate_tracking_result'),
}
def __getattr__(name):
    if name not in _LAZY: raise AttributeError(name)
    import importlib
    module,attribute=_LAZY[name]; value=getattr(importlib.import_module(module),attribute); globals()[name]=value; return value

__all__=['AssociationPolicy','ObjectTrackState','StabilizedMeasurementEstimate','TemporalEstimateStatus','TemporalStabilizationPolicy','TemporalTrackingBatchResult','TemporalTrackingEngine','TemporalTrackingPolicy','TemporalTrackingRequest','TrackAssociation','TrackedObjectResult','TrackingContext','TrackingResetEvent','TrackingResetReason','TrackingState','TrackingUpdateStatus','make_tracking_request','validate_track_state','validate_tracking_result']
