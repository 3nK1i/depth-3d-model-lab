from __future__ import annotations
import pickle
import pytest
from core.measurement.contracts import MeasurementKind,MeasurementStatus,ReliabilityAssessment,ReliabilityLevel
from core.tracking import (
    ObjectTrackState,StabilizedMeasurementEstimate,TemporalEstimateStatus,
    TrackingContext,TrackingState,
)
from tests.tracking_helpers import frame

def test_tracking_context_and_state_are_pickle_safe():
    f=frame(2)
    state=ObjectTrackState('track-1',TrackingState.CONFIRMED,'obs',f,f,2,2,0,.8,measurement_history_count=2)
    restored=pickle.loads(pickle.dumps(state))
    assert restored==state
    assert TrackingContext('depth','seg',1,'abc').worker_generation==1

def test_reacquired_state_requires_count():
    f=frame(2)
    with pytest.raises(ValueError): ObjectTrackState('track',TrackingState.REACQUIRED,'obs',f,f,2,2,0,.8)

def test_stable_temporal_estimate_derives_bounds():
    estimate=StabilizedMeasurementEstimate('track','obs',MeasurementKind.WIDTH,TemporalEstimateStatus.STABLE,MeasurementStatus.AVAILABLE,'m',ReliabilityAssessment(ReliabilityLevel.HIGH),4,4,1.0,.1)
    assert estimate.lower_bound_si==pytest.approx(.9)
    assert estimate.upper_bound_si==pytest.approx(1.1)

def test_warming_estimate_cannot_expose_value():
    with pytest.raises(ValueError):
        StabilizedMeasurementEstimate('track','obs',MeasurementKind.WIDTH,TemporalEstimateStatus.WARMING_UP,MeasurementStatus.AVAILABLE,'m',ReliabilityAssessment(ReliabilityLevel.UNAVAILABLE),1,1,1.0)
