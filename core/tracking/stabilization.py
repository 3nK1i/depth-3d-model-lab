"""Robust temporal measurement history and stabilization."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

from core.measurement.contracts import (
    MeasurementEstimate, MeasurementKind, MeasurementStatus, ReliabilityAssessment,
    ReliabilityLevel, UncertaintyComponent, UncertaintyComponentKind,
)
from core.tracking.temporal_contracts import StabilizedMeasurementEstimate, TemporalEstimateStatus
from core.tracking.policy import TemporalStabilizationPolicy

_ANGLE_KINDS={MeasurementKind.ORIENTATION_ROLL,MeasurementKind.ORIENTATION_PITCH,MeasurementKind.ORIENTATION_YAW}
_RANK={ReliabilityLevel.HIGH:3,ReliabilityLevel.MEDIUM:2,ReliabilityLevel.LOW:1,ReliabilityLevel.UNAVAILABLE:0,ReliabilityLevel.REJECTED:0}

@dataclass(frozen=True, slots=True)
class MeasurementHistorySample:
    frame_id: int
    value_si: float
    expanded_uncertainty_si: float
    reliability: ReliabilityLevel
    estimate: MeasurementEstimate


def append_available(history: list[MeasurementHistorySample], estimate: MeasurementEstimate, frame_id: int, limit: int) -> None:
    if estimate.status is not MeasurementStatus.AVAILABLE or estimate.value_si is None or estimate.uncertainty is None:
        return
    history.append(MeasurementHistorySample(frame_id,float(estimate.value_si),float(estimate.uncertainty.expanded_uncertainty_si),estimate.reliability.level,estimate))
    if len(history)>limit:
        del history[:-limit]


def _unwrap_angles(values: np.ndarray) -> np.ndarray:
    if values.size <= 1:
        return values
    return np.unwrap(values)


def _inlier_mask(values: np.ndarray, policy: TemporalStabilizationPolicy) -> np.ndarray:
    median=float(np.median(values)); deviations=np.abs(values-median); mad=float(np.median(deviations))
    tolerance=max(policy.minimum_absolute_outlier_tolerance,abs(median)*policy.minimum_relative_outlier_tolerance,policy.outlier_mad_multiplier*1.4826*mad)
    return deviations <= tolerance


def stabilize_history(*, track_id: str, observation_id: str, kind: MeasurementKind, source_status: MeasurementStatus, history: list[MeasurementHistorySample], policy: TemporalStabilizationPolicy, confirmed: bool) -> StabilizedMeasurementEstimate:
    unit = history[-1].estimate.unit_si if history else ({MeasurementKind.VOLUME:"m3",MeasurementKind.ORIENTATION_ROLL:"rad",MeasurementKind.ORIENTATION_PITCH:"rad",MeasurementKind.ORIENTATION_YAW:"rad"}.get(kind,"m"))
    unavailable_reliability=ReliabilityAssessment(ReliabilityLevel.UNAVAILABLE, policy_id="phase7_5_temporal_v1")
    if len(history)<policy.minimum_stable_samples:
        return StabilizedMeasurementEstimate(track_id,observation_id,kind,TemporalEstimateStatus.WARMING_UP,source_status,unit,unavailable_reliability,len(history),len(history),warnings=("Temporal history is still warming up.",))
    if not confirmed:
        return StabilizedMeasurementEstimate(track_id,observation_id,kind,TemporalEstimateStatus.UNAVAILABLE,source_status,unit,unavailable_reliability,len(history),len(history),warnings=("Track is not currently confirmed.",))
    values=np.asarray([s.value_si for s in history],float)
    work=_unwrap_angles(values) if kind in _ANGLE_KINDS else values
    mask=_inlier_mask(work,policy); inliers=work[mask]
    if inliers.size<policy.minimum_stable_samples:
        return StabilizedMeasurementEstimate(track_id,observation_id,kind,TemporalEstimateStatus.UNAVAILABLE,source_status,unit,unavailable_reliability,len(history),int(inliers.size),warnings=("Too few temporal inliers remain after rejection.",))
    value=float(np.median(inliers))
    if kind in _ANGLE_KINDS:
        value=float((value+math.pi)%(2*math.pi)-math.pi)
    source_uncertainty=float(np.median([history[i].expanded_uncertainty_si for i in range(len(history)) if mask[i]]))
    temporal_spread=float(1.4826*np.median(np.abs(inliers-np.median(inliers))))
    if temporal_spread==0.0 and inliers.size>1:
        temporal_spread=float(np.std(inliers))
    combined=math.sqrt(source_uncertainty**2+temporal_spread**2)
    components=(
        UncertaintyComponent(UncertaintyComponentKind.OTHER,source_uncertainty,"median_per_frame_expanded_uncertainty",sample_count=int(inliers.size)),
        UncertaintyComponent(UncertaintyComponentKind.TEMPORAL,temporal_spread,"robust_temporal_mad",sample_count=int(inliers.size)),
    )
    relative=combined/max(abs(value),0.05 if kind in _ANGLE_KINDS else 1e-9)
    source_level=min((s.reliability for i,s in enumerate(history) if mask[i]),key=lambda level:_RANK[level])
    if relative>policy.maximum_temporal_relative_uncertainty:
        level=ReliabilityLevel.LOW
    elif relative>0.15:
        level=ReliabilityLevel.LOW
    elif relative>0.07:
        level=ReliabilityLevel.MEDIUM
    else:
        level=ReliabilityLevel.HIGH
    if _RANK[source_level]<_RANK[level]: level=source_level
    reliability=ReliabilityAssessment(level,score=max(0.0,min(1.0,1.0-relative/policy.maximum_temporal_relative_uncertainty)),passed_checks=("temporal_inlier_count_sufficient",),failed_checks=(("temporal_variation_high",) if relative>0.15 else ()),explanations=(f"{int(inliers.size)}/{len(history)} temporal samples retained.",),policy_id="phase7_5_temporal_v1")
    return StabilizedMeasurementEstimate(track_id,observation_id,kind,TemporalEstimateStatus.STABLE,source_status,unit,reliability,len(history),int(inliers.size),value,combined,value-combined,value+combined,components)
