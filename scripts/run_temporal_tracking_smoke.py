#!/usr/bin/env python3
"""Synthetic Step 7 tracking, loss/reacquisition, and stabilization smoke test."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from core.contracts.common import FrameReference,ImageSize,ModelProvenance
from core.contracts.spatial import CalibrationStatus,PixelCoordinateSpace,PixelSpaceRole
from core.measurement.computation_contracts import MeasurementBatchResult,MeasurementComputationStatus,MeasurementDiagnostics,ObjectMeasurementComputationResult
from core.measurement.contracts import MeasurementEstimate,MeasurementKind,MeasurementStatus,ObjectMeasurementResult,ReliabilityAssessment,ReliabilityLevel,ScaleRecovery,ScaleRecoveryStatus,UncertaintyBasis,UncertaintyEstimate
from core.models.metadata import ConfidenceKind,DepthScale,IntrinsicsSource
from core.perception.result import ApprovalStatus,BoundingBox2D,DenseMaskPayload,Detection2D,InstanceMask,MaskSource,ObjectObservation,PerceptionFrameResult
from core.tracking import TemporalEstimateStatus,TemporalTrackingEngine,TrackingState,make_tracking_request
PROV=ModelProvenance('validation-model','Validation Model','validation')

def _frame(fid): return FrameReference(1,fid,fid/10,ImageSize(320,180))
def _observation(f,oid,x):
    space=PixelCoordinateSpace('rgb',f,f.source_size,PixelSpaceRole.RGB); box=(x,50,x+70,135)
    detection=Detection2D(f'det-{oid}',f,1,'test-object',.92,BoundingBox2D(*map(float,box),'rgb'),PROV)
    mask=np.zeros((180,320),bool); mask[50:135,x:x+70]=True
    instance=InstanceMask(f'mask-{oid}',f,space,DenseMaskPayload(mask),MaskSource.NATIVE_INSTANCE,PROV,confidence=.92,confidence_kind=ConfidenceKind.CONFIDENCE)
    return ObjectObservation(oid,f,1,'test-object',detection,(instance,))
def _measurements(f,oid,value):
    uncertainty=UncertaintyEstimate(UncertaintyBasis.HEURISTIC_PERTURBATION,.035,perturbation_sample_count=5)
    estimate=MeasurementEstimate(f'{oid}-width',oid,MeasurementKind.WIDTH,MeasurementStatus.AVAILABLE,'synthetic_frame_measurement','m',ReliabilityAssessment(ReliabilityLevel.HIGH,.9,policy_id='smoke'),DepthScale.METRIC,IntrinsicsSource.ESTIMATED,CalibrationStatus.NOT_CALIBRATED,ScaleRecovery(ScaleRecoveryStatus.NOT_REQUIRED),value_si=value,uncertainty=uncertainty,valid_observation_count=1,model_provenance=(PROV,))
    result=ObjectMeasurementResult(oid,f,(estimate,)); diag=MeasurementDiagnostics(oid,available_kinds=(MeasurementKind.WIDTH,))
    computation=ObjectMeasurementComputationResult(oid,f,MeasurementComputationStatus.SUCCESS,result,diag)
    return MeasurementBatchResult(f,(computation,),MeasurementComputationStatus.SUCCESS)
def _empty_measurements(f): return MeasurementBatchResult(f,(),MeasurementComputationStatus.REJECTED)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='/tmp/phase7_5_step7_tracking.png'); args=parser.parse_args()
    raw=[1.00,1.03,.98,1.01,2.8,1.00,1.02,None,1.01,.99,1.00,1.01]
    engine=TemporalTrackingEngine(); stabilized=[]; states=[]; ids=[]
    for index,value in enumerate(raw,1):
        f=_frame(index)
        if value is None:
            perception=PerceptionFrameResult(f,(),(PROV,)); measurements=_empty_measurements(f)
        else:
            oid=f'obs-{index}'; perception=PerceptionFrameResult(f,(_observation(f,oid,55+index*4),),(PROV,)); measurements=_measurements(f,oid,value)
        result=engine.update(make_tracking_request(perception,measurements,depth_model_id='metric-depth',perception_model_id='segmentation'))
        tracked=result.tracked_objects[0]; states.append(tracked.track_state.state.value); ids.append(tracked.track_state.track_id)
        temporal=next(v for v in tracked.stabilized_measurements if v.kind is MeasurementKind.WIDTH)
        stabilized.append(temporal.value_si if temporal.status is TemporalEstimateStatus.STABLE else np.nan)
        print(f"frame={index:02d} track={tracked.track_state.track_id} state={tracked.track_state.state.value} raw={value} temporal={temporal.status.value} value={temporal.value_si}")
    canvas=np.full((850,1400,3),18,np.uint8)
    cv2.putText(canvas,'Phase 7.5 Step 7 - Temporal Tracking and Measurement Stabilization',(35,55),cv2.FONT_HERSHEY_SIMPLEX,.95,(235,235,235),2,cv2.LINE_AA)
    left,top,width,height=90,120,1220,430
    cv2.rectangle(canvas,(left,top),(left+width,top+height),(80,80,80),1)
    finite=[v for v in raw if v is not None]+[float(v) for v in stabilized if np.isfinite(v)]; lo=min(finite)-.1; hi=max(finite)+.1
    def point(i,v): return (left+int((i-1)/(len(raw)-1)*width),top+height-int((v-lo)/(hi-lo)*height))
    prev=None
    for i,v in enumerate(raw,1):
        if v is None: prev=None; continue
        p=point(i,v); cv2.circle(canvas,p,6,(80,120,255),-1)
        if prev: cv2.line(canvas,prev,p,(80,120,255),2)
        prev=p
    prev=None
    for i,v in enumerate(stabilized,1):
        if not np.isfinite(v): prev=None; continue
        p=point(i,float(v)); cv2.circle(canvas,p,5,(80,230,120),-1)
        if prev: cv2.line(canvas,prev,p,(80,230,120),3)
        prev=p
    cv2.putText(canvas,'raw per-frame width',(100,95),cv2.FONT_HERSHEY_SIMPLEX,.55,(80,120,255),2,cv2.LINE_AA)
    cv2.putText(canvas,'stabilized temporal median',(370,95),cv2.FONT_HERSHEY_SIMPLEX,.55,(80,230,120),2,cv2.LINE_AA)
    y=610
    cv2.putText(canvas,f'persistent_track_id={ids[0]} unique_ids={len(set(ids))}',(60,y),cv2.FONT_HERSHEY_SIMPLEX,.65,(220,220,220),1,cv2.LINE_AA); y+=42
    cv2.putText(canvas,'states: '+' -> '.join(states),(60,y),cv2.FONT_HERSHEY_SIMPLEX,.48,(210,210,210),1,cv2.LINE_AA); y+=45
    cv2.putText(canvas,'Frame 5 is an injected measurement outlier; frame 8 is an object loss; frame 9 is reacquisition.',(60,y),cv2.FONT_HERSHEY_SIMPLEX,.56,(90,195,255),1,cv2.LINE_AA); y+=42
    cv2.putText(canvas,'Uncertainty basis: temporal observation + per-frame estimated uncertainty; not empirically calibrated.',(60,y),cv2.FONT_HERSHEY_SIMPLEX,.52,(190,190,190),1,cv2.LINE_AA)
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(output),canvas): raise RuntimeError(f'Could not write {output}')
    print(f"status=success track_id={ids[0]} unique_ids={len(set(ids))} output={output}")
    return 0
if __name__=='__main__': raise SystemExit(main())
