from __future__ import annotations
import numpy as np
from core.contracts.common import FrameReference,ImageSize,ModelProvenance
from core.contracts.spatial import CalibrationStatus,PixelCoordinateSpace,PixelSpaceRole
from core.measurement.computation_contracts import MeasurementBatchResult,MeasurementComputationStatus,MeasurementDiagnostics,ObjectMeasurementComputationResult
from core.measurement.contracts import (
    MeasurementEstimate,MeasurementKind,MeasurementStatus,ObjectMeasurementResult,
    ReliabilityAssessment,ReliabilityLevel,ScaleRecovery,ScaleRecoveryStatus,
    UncertaintyBasis,UncertaintyEstimate,
)
from core.models.metadata import DepthScale,IntrinsicsSource,ConfidenceKind
from core.perception.result import (
    ApprovalStatus,BoundingBox2D,DenseMaskPayload,Detection2D,InstanceMask,
    MaskSource,ObjectObservation,PerceptionFrameResult,
)

PROV=ModelProvenance('test-model','Test Model','test')

def frame(frame_id:int, *, session=1, size=(100,80)):
    return FrameReference(session,frame_id,float(frame_id)/10,ImageSize(*size))

def observation(frame_ref, observation_id='obs-0', *, class_id=1, class_name='box', box=(20,20,50,50), mask_shift=0):
    space=PixelCoordinateSpace('rgb',frame_ref,frame_ref.source_size,PixelSpaceRole.RGB)
    bbox=BoundingBox2D(*map(float,box),'rgb')
    det=Detection2D(f'det-{observation_id}',frame_ref,class_id,class_name,.9,bbox,PROV)
    mask=np.zeros((frame_ref.source_size.height,frame_ref.source_size.width),bool)
    x0,y0,x1,y1=map(int,box); x0+=mask_shift; x1+=mask_shift
    mask[max(y0,0):min(y1,mask.shape[0]),max(x0,0):min(x1,mask.shape[1])]=True
    instance=InstanceMask(mask_id=f'mask-{observation_id}',frame_reference=frame_ref,pixel_space=space,payload=DenseMaskPayload(mask),source=MaskSource.NATIVE_INSTANCE,model_provenance=PROV,revision=0,approval_status=ApprovalStatus.NOT_REQUIRED,confidence=.9,confidence_kind=ConfidenceKind.CONFIDENCE,foreground_valid=True)
    return ObjectObservation(observation_id,frame_ref,class_id,class_name,det,(instance,))

def perception(frame_ref, observations):
    return PerceptionFrameResult(frame_ref,tuple(observations),(PROV,))

def estimate(frame_ref, observation_id, kind, value, uncertainty=.02, reliability=ReliabilityLevel.HIGH):
    unit='rad' if kind in {MeasurementKind.ORIENTATION_ROLL,MeasurementKind.ORIENTATION_PITCH,MeasurementKind.ORIENTATION_YAW} else 'm3' if kind is MeasurementKind.VOLUME else 'm'
    unc=UncertaintyEstimate(UncertaintyBasis.HEURISTIC_PERTURBATION,uncertainty,perturbation_sample_count=3)
    return MeasurementEstimate(
        f'{observation_id}-{kind.value}-{frame_ref.frame_id}',observation_id,kind,MeasurementStatus.AVAILABLE,
        'test',unit,ReliabilityAssessment(reliability,.9,policy_id='test'),DepthScale.METRIC,
        IntrinsicsSource.ESTIMATED,CalibrationStatus.NOT_CALIBRATED,ScaleRecovery(ScaleRecoveryStatus.NOT_REQUIRED),
        value_si=value,uncertainty=unc,valid_observation_count=1,model_provenance=(PROV,),
    )

def measurement_batch(frame_ref, values_by_observation):
    results=[]
    for observation_id, values in values_by_observation.items():
        estimates=tuple(estimate(frame_ref,observation_id,kind,value,unc) for kind,(value,unc) in values.items())
        result=ObjectMeasurementResult(observation_id,frame_ref,estimates)
        diagnostics=MeasurementDiagnostics(observation_id,available_kinds=tuple(values))
        results.append(ObjectMeasurementComputationResult(observation_id,frame_ref,MeasurementComputationStatus.SUCCESS,result,diagnostics))
    return MeasurementBatchResult(frame_ref,tuple(results),MeasurementComputationStatus.SUCCESS if results else MeasurementComputationStatus.REJECTED)
