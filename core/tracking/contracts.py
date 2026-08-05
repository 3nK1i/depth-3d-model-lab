"""Immutable base contracts for persistent object tracking."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from core.contracts.common import FrameReference
from core.contracts.validation import ContractValidationError,require_nonempty,require_probability

class TrackingState(str,Enum):
    UNTRACKED='untracked'; TENTATIVE='tentative'; CONFIRMED='confirmed'; LOST='lost'; REACQUIRED='reacquired'; TERMINATED='terminated'
class TrackingResetReason(str,Enum):
    SOURCE_SESSION_CHANGED='source_session_changed'; DEPTH_MODEL_CHANGED='depth_model_changed'; PERCEPTION_MODEL_CHANGED='perception_model_changed'; RESOLUTION_CHANGED='resolution_changed'; WORKER_GENERATION_CHANGED='worker_generation_changed'; ALIGNMENT_CONFIGURATION_CHANGED='alignment_configuration_changed'; MANUAL_RESET='manual_reset'; OTHER='other'
class TrackingUpdateStatus(str,Enum):
    SUCCESS='success'; PARTIAL='partial'; EMPTY='empty'

@dataclass(frozen=True,slots=True)
class TrackingContext:
    depth_model_id:str
    perception_model_id:str
    worker_generation:int=0
    alignment_config_digest:str=''
    def __post_init__(self):
        object.__setattr__(self,'depth_model_id',require_nonempty(self.depth_model_id,field_name='depth_model_id'))
        object.__setattr__(self,'perception_model_id',require_nonempty(self.perception_model_id,field_name='perception_model_id'))
        if isinstance(self.worker_generation,bool) or int(self.worker_generation)<0: raise ContractValidationError('worker_generation must be a non-negative integer.')
        object.__setattr__(self,'worker_generation',int(self.worker_generation)); object.__setattr__(self,'alignment_config_digest',str(self.alignment_config_digest).strip())

@dataclass(frozen=True,slots=True)
class ObjectTrackState:
    track_id:str
    state:TrackingState
    current_observation_id:str
    first_frame_reference:FrameReference
    latest_frame_reference:FrameReference
    age_frames:int
    confirmed_observation_count:int
    missed_frame_count:int
    association_confidence:float|None=None
    reacquisition_count:int=0
    reset_reason:TrackingResetReason|None=None
    measurement_history_count:int=0
    def __post_init__(self):
        object.__setattr__(self,'track_id',require_nonempty(self.track_id,field_name='track_id'))
        object.__setattr__(self,'current_observation_id',require_nonempty(self.current_observation_id,field_name='current_observation_id'))
        for field_name in ('age_frames','confirmed_observation_count','missed_frame_count','reacquisition_count','measurement_history_count'):
            value=getattr(self,field_name)
            if isinstance(value,bool) or int(value)<0: raise ContractValidationError(f'{field_name} must be a non-negative integer.')
            object.__setattr__(self,field_name,int(value))
        if self.latest_frame_reference.session_id!=self.first_frame_reference.session_id: raise ContractValidationError('A track cannot span different source sessions.')
        if self.latest_frame_reference.frame_id<self.first_frame_reference.frame_id: raise ContractValidationError('latest_frame_reference must not precede first_frame_reference.')
        if self.association_confidence is not None: object.__setattr__(self,'association_confidence',require_probability(self.association_confidence,field_name='association_confidence'))
        if self.state is TrackingState.REACQUIRED and self.reacquisition_count<=0: raise ContractValidationError('REACQUIRED tracking state requires reacquisition_count > 0.')

@dataclass(frozen=True,slots=True)
class TrackAssociation:
    track_id:str
    observation_id:str
    score:float
    mask_iou:float|None=None
    box_iou:float|None=None
    centre_similarity:float|None=None
    def __post_init__(self):
        object.__setattr__(self,'track_id',require_nonempty(self.track_id,field_name='track_id')); object.__setattr__(self,'observation_id',require_nonempty(self.observation_id,field_name='observation_id'))
        for field_name in ('score','mask_iou','box_iou','centre_similarity'):
            value=getattr(self,field_name)
            if value is not None: object.__setattr__(self,field_name,require_probability(value,field_name=field_name))

@dataclass(frozen=True,slots=True)
class TrackingResetEvent:
    frame_reference:FrameReference
    reasons:tuple[TrackingResetReason,...]
    terminated_track_ids:tuple[str,...]=()
    def __post_init__(self):
        if not self.reasons: raise ContractValidationError('Tracking reset events require at least one reason.')
        if len(set(self.reasons))!=len(self.reasons): raise ContractValidationError('Tracking reset reasons must be unique.')
        object.__setattr__(self,'terminated_track_ids',tuple(require_nonempty(v,field_name='terminated_track_id') for v in self.terminated_track_ids))
