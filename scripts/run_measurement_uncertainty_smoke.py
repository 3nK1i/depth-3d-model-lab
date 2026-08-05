#!/usr/bin/env python3
"""Dependency-light synthetic Step 6 measurement and uncertainty smoke test."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from core.alignment.contracts import AlignedFrameBundle, AlignedObjectObservation, AlignmentStatus
from core.contracts.common import FrameReference, ImageSize, ModelProvenance
from core.contracts.spatial import AxisConvention, CalibrationStatus, CameraIntrinsics, CoordinateFrame, CoordinateFrameKind, PixelCoordinateSpace, PixelSpaceRole
from core.measurement import MeasurementStatus, ObjectPointCloud, ScaleRecoveryStatus, compute_measurements, make_measurement_request
from core.models.metadata import ConfidenceKind, DepthScale, IntrinsicsSource
from core.object_geometry.contracts import ExtractionStageCounts, ObjectPointExtractionBatchResult, ObjectPointExtractionResult, ObjectPointExtractionStatus, PointCoordinateSemantics
from core.object_geometry.fitting_pipeline import ObjectGeometryFitRequest, fit_object_geometry
from core.perception.result import ApprovalStatus, MaskSource



def _rotation_z(angle: float) -> np.ndarray:
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,-s,0.0],[s,c,0.0],[0.0,0.0,1.0]])


def _cuboid_points() -> np.ndarray:
    extents=np.array([2.0,1.0,0.4]); centre=np.array([0.25,-0.1,3.0]); R=_rotation_z(0.35)
    xs=np.linspace(-1.0,1.0,22); ys=np.linspace(-0.5,0.5,22); zs=np.linspace(-0.2,0.2,7)
    pts=[]
    for x in xs:
        for y in ys:
            pts.extend(((x,y,zs[0]),(x,y,zs[-1])))
    for x in xs:
        for z in zs:
            pts.extend(((x,ys[0],z),(x,ys[-1],z)))
    for y in ys:
        for z in zs:
            pts.extend(((xs[0],y,z),(xs[-1],y,z)))
    points=np.unique(np.asarray(pts,float),axis=0)@R.T+centre
    return points + np.random.default_rng(1).normal(0.0,0.002,points.shape)


def _make_chain():
    width,height=96,72
    frame=FrameReference(9,12,4.5,ImageSize(width,height))
    space=PixelCoordinateSpace('depth',frame,frame.source_size,PixelSpaceRole.DEPTH)
    mask=np.zeros((height,width),bool); mask[2:70,2:94]=True
    observation=AlignedObjectObservation('object-0',0,'synthetic-cuboid',space,source_mask_id='mask-0',source_mask_space_id='rgb',aligned_mask=mask,source_foreground_pixels=int(mask.sum()),aligned_foreground_pixels=int(mask.sum()),foreground_retention_ratio=1.0,mask_source=MaskSource.NATIVE_INSTANCE,approval_status=ApprovalStatus.NOT_REQUIRED,mask_model_provenance=ModelProvenance('seg','Segmentation','segmentation'),detection_model_provenance=ModelProvenance('det','Detection','detection'))
    K=np.array([[80.,0.,width/2],[0.,80.,height/2],[0.,0.,1.]])
    bundle=AlignedFrameBundle(frame,space,np.full((height,width,3),(40,90,150),np.uint8),np.full((height,width),3.0,np.float32),np.ones((height,width),bool),np.ones((height,width),bool),np.full((height,width),0.92,np.float32),np.ones((height,width),bool),ConfidenceKind.CONFIDENCE,CameraIntrinsics(K,'depth',IntrinsicsSource.ESTIMATED,CalibrationStatus.NOT_CALIBRATED),DepthScale.METRIC,1,ModelProvenance('depth','Depth Model','depth'),(ModelProvenance('seg','Segmentation','segmentation'),),(observation,),(),AlignmentStatus.SUCCESS)
    points=_cuboid_points(); camera=CoordinateFrame('camera-depth',CoordinateFrameKind.CAMERA_DEPTH,AxisConvention.X_RIGHT_Y_DOWN_Z_FORWARD,source_frame=frame)
    u=np.rint(K[0,0]*points[:,0]/points[:,2]+K[0,2]).astype(int); v=np.rint(K[1,1]*points[:,1]/points[:,2]+K[1,2]).astype(int)
    pixels=np.column_stack((np.clip(u,0,width-1),np.clip(v,0,height-1)))
    cloud=ObjectPointCloud('object-0',frame,camera.frame_id,points,'depth',frame.source_size,pixels,raw_point_count=len(points),retained_point_count=len(points),mask_id='mask-0',depth_model_provenance=bundle.depth_model_provenance,depth_scale=DepthScale.METRIC,intrinsics_source=IntrinsicsSource.ESTIMATED,scale_recovery_status=ScaleRecoveryStatus.NOT_REQUIRED)
    counts=ExtractionStageCounts(*([len(points)]*12))
    extracted=ObjectPointExtractionResult('object-0',frame,ObjectPointExtractionStatus.SUCCESS,camera,PointCoordinateSemantics.METRIC_METRES,cloud,cloud,counts)
    batch=ObjectPointExtractionBatchResult(frame,camera,(extracted,),ObjectPointExtractionStatus.SUCCESS)
    fitting=fit_object_geometry(ObjectGeometryFitRequest(batch))
    return bundle,batch,fitting


def _fmt(estimate) -> str:
    if estimate.status is not MeasurementStatus.AVAILABLE:
        reasons=", ".join(r.value for r in estimate.rejection_reasons)
        return f"{estimate.kind.value}: {estimate.status.value} ({reasons})"
    u=estimate.uncertainty.expanded_uncertainty_si
    return f"{estimate.kind.value}: {estimate.value_si:.5f} {estimate.unit_si} +/- {u:.5f} [{estimate.lower_bound_si:.5f}, {estimate.upper_bound_si:.5f}] {estimate.reliability.level.value}"


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='/tmp/phase7_5_step6_measurements.png')
    args=parser.parse_args()
    bundle, extraction, fitting=_make_chain()
    result=compute_measurements(make_measurement_request(bundle,extraction,fitting))
    obj=result.object_results[0]
    cloud=extraction.object_results[0].filtered_cloud
    canvas=np.full((900,1400,3),18,np.uint8)
    cv2.putText(canvas,'Phase 7.5 Step 6 - Measurements and Estimated Uncertainty',(35,55),cv2.FONT_HERSHEY_SIMPLEX,1.05,(235,235,235),2,cv2.LINE_AA)
    pts=np.asarray(cloud.points_xyz_m)
    xy=pts[:,:2]
    lo=xy.min(axis=0); hi=xy.max(axis=0); span=np.maximum(hi-lo,1e-9)
    pix=((xy-lo)/span*np.array([580,360])+np.array([60,120])).astype(int)
    for x,y in pix:
        cv2.circle(canvas,(int(x),int(500-y+120)),1,(65,220,110),-1)
    cv2.rectangle(canvas,(35,90),(690,600),(90,90,90),1)
    cv2.putText(canvas,'Synthetic filtered object points - camera X/Y',(50,115),cv2.FONT_HERSHEY_SIMPLEX,.65,(220,220,220),1,cv2.LINE_AA)
    y=110
    for estimate in obj.measurement_result.estimates:
        text=_fmt(estimate)
        color=(210,235,210) if estimate.status is MeasurementStatus.AVAILABLE else (120,170,240)
        cv2.putText(canvas,text,(735,y),cv2.FONT_HERSHEY_SIMPLEX,.48,color,1,cv2.LINE_AA)
        y+=43
    cv2.putText(canvas,f"status={obj.status.value}",(40,660),cv2.FONT_HERSHEY_SIMPLEX,.75,(230,230,230),2,cv2.LINE_AA)
    cv2.putText(canvas,f"selected_box={obj.diagnostics.selected_box_id} ({obj.diagnostics.selected_box_method})",(40,700),cv2.FONT_HERSHEY_SIMPLEX,.62,(210,210,210),1,cv2.LINE_AA)
    cv2.putText(canvas,f"perturbation_samples={obj.diagnostics.perturbation_sample_count}",(40,735),cv2.FONT_HERSHEY_SIMPLEX,.62,(210,210,210),1,cv2.LINE_AA)
    cv2.putText(canvas,'Estimated Uncertainty - Not Calibrated',(40,785),cv2.FONT_HERSHEY_SIMPLEX,.72,(80,190,255),2,cv2.LINE_AA)
    cv2.putText(canvas,'No coverage probability is claimed; values use deterministic heuristic perturbation.',(40,825),cv2.FONT_HERSHEY_SIMPLEX,.56,(190,190,190),1,cv2.LINE_AA)
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(output),canvas):
        raise RuntimeError(f'failed to write {output}')
    available=sum(e.status is MeasurementStatus.AVAILABLE for e in obj.measurement_result.estimates)
    print(f"status={obj.status.value} available={available} total={len(obj.measurement_result.estimates)} output={output}")
    for estimate in obj.measurement_result.estimates:
        print(_fmt(estimate))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
