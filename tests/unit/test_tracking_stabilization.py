import math
import pytest
from core.measurement.contracts import MeasurementKind,MeasurementStatus,ReliabilityLevel
from core.tracking.temporal_contracts import TemporalEstimateStatus
from core.tracking.policy import TemporalStabilizationPolicy
from core.tracking.stabilization import append_available,stabilize_history
from tests.tracking_helpers import estimate,frame

def _history(kind,values):
    history=[]
    for i,value in enumerate(values,1): append_available(history,estimate(frame(i),'obs',kind,value,.02),i,15)
    return history

def test_temporal_median_rejects_large_outlier():
    history=_history(MeasurementKind.WIDTH,[1.0,1.01,.99,5.0,1.0])
    result=stabilize_history(track_id='t',observation_id='obs',kind=MeasurementKind.WIDTH,source_status=MeasurementStatus.AVAILABLE,history=history,policy=TemporalStabilizationPolicy(),confirmed=True)
    assert result.status is TemporalEstimateStatus.STABLE
    assert result.value_si==pytest.approx(1.0,abs=.02)
    assert result.inlier_count==4
    assert any(c.kind.value=='temporal_variation' for c in result.components)

def test_history_warms_up_before_minimum_samples():
    result=stabilize_history(track_id='t',observation_id='obs',kind=MeasurementKind.WIDTH,source_status=MeasurementStatus.AVAILABLE,history=_history(MeasurementKind.WIDTH,[1.0,1.01]),policy=TemporalStabilizationPolicy(minimum_stable_samples=3),confirmed=True)
    assert result.status is TemporalEstimateStatus.WARMING_UP

def test_angle_wrap_is_stabilized_near_pi_not_zero():
    history=_history(MeasurementKind.ORIENTATION_YAW,[3.12,-3.13,3.11,-3.12])
    result=stabilize_history(track_id='t',observation_id='obs',kind=MeasurementKind.ORIENTATION_YAW,source_status=MeasurementStatus.AVAILABLE,history=history,policy=TemporalStabilizationPolicy(),confirmed=True)
    assert result.status is TemporalEstimateStatus.STABLE
    assert abs(abs(result.value_si)-math.pi)<.08

def test_history_is_bounded():
    history=[]
    for i in range(30): append_available(history,estimate(frame(i),'obs',MeasurementKind.WIDTH,1.0,.01),i,5)
    assert len(history)==5 and history[0].frame_id==25
