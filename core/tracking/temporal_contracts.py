"""Immutable temporal measurement and per-frame tracking result contracts."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from core.contracts.common import FrameReference,ProcessingStep
from core.contracts.validation import ContractValidationError,require_finite,require_nonempty
from core.measurement.contracts import MeasurementKind,MeasurementStatus,ReliabilityAssessment,UncertaintyBasis,UncertaintyComponent
from core.tracking.contracts import ObjectTrackState,TrackAssociation,TrackingResetEvent,TrackingState,TrackingUpdateStatus

class TemporalEstimateStatus(str,Enum):
    WARMING_UP="warming_up"
    STABLE="stable"
    UNAVAILABLE="unavailable"

@dataclass(frozen=True,slots=True)
class StabilizedMeasurementEstimate:
    track_id:str
    current_observation_id:str
    kind:MeasurementKind
    status:TemporalEstimateStatus
    source_status:MeasurementStatus
    unit_si:str
    reliability:ReliabilityAssessment
    sample_count:int
    inlier_count:int
    value_si:float|None=None
    expanded_uncertainty_si:float|None=None
    lower_bound_si:float|None=None
    upper_bound_si:float|None=None
    components:tuple[UncertaintyComponent,...]=()
    uncertainty_basis:UncertaintyBasis=UncertaintyBasis.TEMPORAL_OBSERVATION
    warnings:tuple[str,...]=()
    def __post_init__(self):
        object.__setattr__(self,'track_id',require_nonempty(self.track_id,field_name='track_id'))
        object.__setattr__(self,'current_observation_id',require_nonempty(self.current_observation_id,field_name='current_observation_id'))
        if not str(self.unit_si).strip(): raise ContractValidationError('unit_si must not be empty.')
        object.__setattr__(self,'unit_si',str(self.unit_si).strip())
        for name in ('sample_count','inlier_count'):
            value=getattr(self,name)
            if isinstance(value,bool) or int(value)<0: raise ContractValidationError(f'{name} must be non-negative.')
            object.__setattr__(self,name,int(value))
        if self.inlier_count>self.sample_count: raise ContractValidationError('inlier_count cannot exceed sample_count.')
        if self.status is TemporalEstimateStatus.STABLE:
            if self.value_si is None or self.expanded_uncertainty_si is None: raise ContractValidationError('Stable temporal estimates require value and uncertainty.')
            value=require_finite(self.value_si,field_name='value_si'); uncertainty=require_finite(self.expanded_uncertainty_si,field_name='expanded_uncertainty_si')
            if uncertainty<0: raise ContractValidationError('expanded_uncertainty_si must not be negative.')
            lower,upper=value-uncertainty,value+uncertainty
            if self.lower_bound_si is not None and abs(float(self.lower_bound_si)-lower)>1e-9: raise ContractValidationError('lower_bound_si is inconsistent with value and uncertainty.')
            if self.upper_bound_si is not None and abs(float(self.upper_bound_si)-upper)>1e-9: raise ContractValidationError('upper_bound_si is inconsistent with value and uncertainty.')
            object.__setattr__(self,'value_si',value); object.__setattr__(self,'expanded_uncertainty_si',uncertainty); object.__setattr__(self,'lower_bound_si',lower); object.__setattr__(self,'upper_bound_si',upper)
        elif any(v is not None for v in (self.value_si,self.expanded_uncertainty_si,self.lower_bound_si,self.upper_bound_si)):
            raise ContractValidationError('Non-stable temporal estimates must not expose numeric values.')
        object.__setattr__(self,'components',tuple(self.components))
        if self.status is TemporalEstimateStatus.STABLE and self.uncertainty_basis is not UncertaintyBasis.TEMPORAL_OBSERVATION: raise ContractValidationError('Stable Step 7 estimates must declare temporal_observation uncertainty basis.')
        object.__setattr__(self,'warnings',tuple(str(v).strip() for v in self.warnings if str(v).strip()))

@dataclass(frozen=True,slots=True)
class TrackedObjectResult:
    track_state:ObjectTrackState
    class_id:int
    class_name:str
    stabilized_measurements:tuple[StabilizedMeasurementEstimate,...]=()
    warnings:tuple[str,...]=()
    def __post_init__(self):
        if isinstance(self.class_id,bool) or int(self.class_id)<0: raise ContractValidationError('class_id must be a non-negative integer.')
        object.__setattr__(self,'class_id',int(self.class_id)); object.__setattr__(self,'class_name',require_nonempty(self.class_name,field_name='class_name'))
        kinds=[v.kind for v in self.stabilized_measurements]
        if len(kinds)!=len(set(kinds)): raise ContractValidationError('Only one stabilized estimate per measurement kind is allowed.')
        if any(v.track_id!=self.track_state.track_id for v in self.stabilized_measurements): raise ContractValidationError('Stabilized measurement belongs to a different track.')
        object.__setattr__(self,'warnings',tuple(str(v).strip() for v in self.warnings if str(v).strip()))

@dataclass(frozen=True,slots=True)
class TemporalTrackingBatchResult:
    frame_reference:FrameReference
    status:TrackingUpdateStatus
    tracked_objects:tuple[TrackedObjectResult,...]
    associations:tuple[TrackAssociation,...]=()
    reset_events:tuple[TrackingResetEvent,...]=()
    processing_steps:tuple[ProcessingStep,...]=()
    warnings:tuple[str,...]=()
    def __post_init__(self):
        ids=[v.track_state.track_id for v in self.tracked_objects]
        if len(ids)!=len(set(ids)): raise ContractValidationError('Track IDs must be unique in a tracking result.')
        if any(v.track_state.latest_frame_reference!=self.frame_reference for v in self.tracked_objects if v.track_state.state is not TrackingState.TERMINATED): raise ContractValidationError('Active track latest frame must match tracking result frame.')
        object.__setattr__(self,'associations',tuple(self.associations)); object.__setattr__(self,'reset_events',tuple(self.reset_events)); object.__setattr__(self,'processing_steps',tuple(self.processing_steps)); object.__setattr__(self,'warnings',tuple(str(v).strip() for v in self.warnings if str(v).strip()))
