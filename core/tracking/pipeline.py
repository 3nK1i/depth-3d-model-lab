"""Stateful persistent object tracking and temporal measurement stabilization."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
import time

from core.contracts.common import ProcessingStep
from core.measurement.computation_contracts import MeasurementBatchResult
from core.measurement.contracts import MeasurementKind, MeasurementStatus
from core.perception.result import ObjectObservation, PerceptionFrameResult
from core.tracking.association import ObservationGeometry, greedy_associate, observation_geometry
from core.tracking.contracts import (ObjectTrackState,TrackAssociation,TrackingContext,TrackingResetEvent,TrackingResetReason,TrackingState,TrackingUpdateStatus)
from core.tracking.temporal_contracts import StabilizedMeasurementEstimate,TemporalEstimateStatus,TemporalTrackingBatchResult,TrackedObjectResult
from core.tracking.policy import TemporalTrackingPolicy
from core.tracking.stabilization import MeasurementHistorySample, append_available, stabilize_history


@dataclass(frozen=True, slots=True)
class TemporalTrackingRequest:
    perception_result: PerceptionFrameResult
    measurement_batch: MeasurementBatchResult
    context: TrackingContext

    def __post_init__(self) -> None:
        if self.perception_result.frame_reference != self.measurement_batch.frame_reference:
            raise ValueError("Perception and measurement results must reference the same frame.")


@dataclass(slots=True)
class _RuntimeTrack:
    track_id: str
    class_id: int
    class_name: str
    first_frame: object
    latest_frame: object
    observation_id: str
    geometry: ObservationGeometry
    state: TrackingState = TrackingState.TENTATIVE
    age: int = 1
    hits: int = 1
    missed: int = 0
    reacquisitions: int = 0
    association_confidence: float | None = None
    histories: dict[MeasurementKind,list[MeasurementHistorySample]] = field(default_factory=lambda:defaultdict(list))


class TemporalTrackingEngine:
    """Process-local tracker. No model, GPU, Qt, or open-resource ownership."""

    def __init__(self, policy: TemporalTrackingPolicy | None = None) -> None:
        self.policy=policy or TemporalTrackingPolicy()
        self._tracks: dict[str,_RuntimeTrack]={}
        self._next_track_number=1
        self._last_context: TrackingContext|None=None
        self._last_frame=None

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)

    def reset(self, reason: TrackingResetReason=TrackingResetReason.MANUAL_RESET):
        ids=tuple(sorted(self._tracks)); self._tracks.clear(); self._last_context=None; self._last_frame=None
        return ids,reason

    def _reset_reasons(self, request: TemporalTrackingRequest):
        reasons=[]; frame=request.perception_result.frame_reference
        if self._last_frame is not None:
            if frame.session_id!=self._last_frame.session_id: reasons.append(TrackingResetReason.SOURCE_SESSION_CHANGED)
            if frame.source_size!=self._last_frame.source_size: reasons.append(TrackingResetReason.RESOLUTION_CHANGED)
            if frame.frame_id<=self._last_frame.frame_id and frame.session_id==self._last_frame.session_id: reasons.append(TrackingResetReason.OTHER)
        if self._last_context is not None:
            if request.context.depth_model_id!=self._last_context.depth_model_id: reasons.append(TrackingResetReason.DEPTH_MODEL_CHANGED)
            if request.context.perception_model_id!=self._last_context.perception_model_id: reasons.append(TrackingResetReason.PERCEPTION_MODEL_CHANGED)
            if request.context.worker_generation!=self._last_context.worker_generation: reasons.append(TrackingResetReason.WORKER_GENERATION_CHANGED)
            if request.context.alignment_config_digest!=self._last_context.alignment_config_digest: reasons.append(TrackingResetReason.ALIGNMENT_CONFIGURATION_CHANGED)
        return tuple(dict.fromkeys(reasons))

    def update(self, request: TemporalTrackingRequest) -> TemporalTrackingBatchResult:
        started=time.perf_counter(); frame=request.perception_result.frame_reference; reset_events=[]
        reasons=self._reset_reasons(request)
        if reasons:
            terminated=tuple(sorted(self._tracks)); self._tracks.clear()
            reset_events.append(TrackingResetEvent(frame,reasons,terminated))
        self._last_context=request.context; self._last_frame=frame
        active_geometry={tid:t.geometry for tid,t in self._tracks.items() if t.state is not TrackingState.TERMINATED}
        associations,current_geometry=greedy_associate(active_geometry,request.perception_result.observations,self.policy.association)
        by_track={a.track_id:a for a in associations}; by_obs={a.observation_id:a for a in associations}
        observation_by_id={o.observation_id:o for o in request.perception_result.observations}
        measurement_by_obs={r.observation_id:r for r in request.measurement_batch.object_results}

        for track_id,track in list(self._tracks.items()):
            track.age+=1
            association=by_track.get(track_id)
            if association is None:
                track.missed+=1; track.association_confidence=None
                track.latest_frame=frame
                track.state=TrackingState.TERMINATED if track.missed>self.policy.maximum_missed_frames else TrackingState.LOST
                continue
            observation=observation_by_id[association.observation_id]
            was_lost=track.state is TrackingState.LOST
            track.observation_id=observation.observation_id; track.geometry=current_geometry[observation.observation_id]
            track.latest_frame=frame; track.missed=0; track.hits+=1; track.association_confidence=association.score
            if was_lost:
                track.reacquisitions+=1; track.state=TrackingState.REACQUIRED
            elif track.hits>=self.policy.confirmation_hits:
                track.state=TrackingState.CONFIRMED
            else: track.state=TrackingState.TENTATIVE

        for observation in request.perception_result.observations:
            if observation.observation_id in by_obs: continue
            track_id=f"track-{self._next_track_number:06d}"; self._next_track_number+=1
            state=TrackingState.CONFIRMED if self.policy.confirmation_hits<=1 else TrackingState.TENTATIVE
            self._tracks[track_id]=_RuntimeTrack(track_id,observation.class_id,observation.class_name,frame,frame,observation.observation_id,current_geometry[observation.observation_id],state=state)

        current_track_for_obs={t.observation_id:t for t in self._tracks.values() if t.missed==0 and t.state is not TrackingState.TERMINATED}
        for observation_id,track in current_track_for_obs.items():
            computation=measurement_by_obs.get(observation_id)
            if computation is None: continue
            for estimate in computation.measurement_result.estimates:
                append_available(track.histories[estimate.kind],estimate,frame.frame_id,self.policy.stabilization.history_size)

        results=[]
        all_kinds=tuple(MeasurementKind)
        for track_id in sorted(self._tracks):
            track=self._tracks[track_id]
            current_computation=measurement_by_obs.get(track.observation_id) if track.missed==0 else None
            current_by_kind={e.kind:e for e in current_computation.measurement_result.estimates} if current_computation else {}
            stabilized=[]
            for kind in all_kinds:
                source=current_by_kind.get(kind)
                source_status=source.status if source is not None else MeasurementStatus.UNAVAILABLE
                history=track.histories.get(kind,[])
                estimate=stabilize_history(track_id=track_id,observation_id=track.observation_id,kind=kind,source_status=source_status,history=history,policy=self.policy.stabilization,confirmed=track.state in {TrackingState.CONFIRMED,TrackingState.REACQUIRED})
                stabilized.append(estimate)
            history_count=max((len(v) for v in track.histories.values()),default=0)
            state=ObjectTrackState(track.track_id,track.state,track.observation_id,track.first_frame,track.latest_frame,track.age,track.hits,track.missed,track.association_confidence,track.reacquisitions,None,history_count)
            results.append(TrackedObjectResult(state,track.class_id,track.class_name,tuple(stabilized)))
        terminated=[tid for tid,t in self._tracks.items() if t.state is TrackingState.TERMINATED]
        for tid in terminated: del self._tracks[tid]
        stable_count=sum(m.status is TemporalEstimateStatus.STABLE for r in results for m in r.stabilized_measurements)
        status=TrackingUpdateStatus.EMPTY if not results else TrackingUpdateStatus.SUCCESS if stable_count and all(r.track_state.state in {TrackingState.CONFIRMED,TrackingState.REACQUIRED} for r in results if r.track_state.state is not TrackingState.TERMINATED) else TrackingUpdateStatus.PARTIAL
        step=ProcessingStep(f"tracking-{frame.session_id}-{frame.frame_id}","mask_box_greedy_association_and_robust_temporal_median",input_ids=tuple(o.observation_id for o in request.perception_result.observations),output_ids=tuple(r.track_state.track_id for r in results),duration_ms=(time.perf_counter()-started)*1000,parameters=(("confirmation_hits",self.policy.confirmation_hits),("maximum_missed_frames",self.policy.maximum_missed_frames),("history_size",self.policy.stabilization.history_size)))
        return TemporalTrackingBatchResult(frame,status,tuple(results),associations,tuple(reset_events),(step,))
