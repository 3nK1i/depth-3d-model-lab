from __future__ import annotations
import numpy as np
from core.alignment.contracts import AlignedFrameBundle, AlignedObjectObservation, AlignmentStatus
from core.contracts.common import FrameReference, ImageSize, ModelProvenance
from core.contracts.spatial import AxisConvention, CalibrationStatus, CameraIntrinsics, CoordinateFrame, CoordinateFrameKind, PixelCoordinateSpace, PixelSpaceRole
from core.measurement.contracts import ObjectPointCloud, ScaleRecoveryStatus
from core.models.metadata import ConfidenceKind, DepthScale, IntrinsicsSource
from core.object_geometry.contracts import ExtractionStageCounts, ObjectPointExtractionBatchResult, ObjectPointExtractionResult, ObjectPointExtractionStatus, PointCoordinateSemantics
from core.object_geometry.fitting_pipeline import ObjectGeometryFitRequest, fit_object_geometry
from core.perception.result import ApprovalStatus, MaskSource
from tests.object_geometry_fitting_helpers import cuboid_points

def make_chain(*, extents=(2.0,1.0,0.4), angle=0.35, depth_scale=DepthScale.METRIC, border_contacts=(), outliers=8):
    width,height=96,72
    frame=FrameReference(9,12,4.5,ImageSize(width,height))
    space=PixelCoordinateSpace('depth',frame,frame.source_size,PixelSpaceRole.DEPTH)
    mask=np.zeros((height,width),bool); mask[2:70,2:94]=True
    obs=AlignedObjectObservation(
        'object-0',0,'test-object',space,source_mask_id='mask-0',source_mask_space_id='rgb',
        aligned_mask=mask,source_foreground_pixels=int(mask.sum()),aligned_foreground_pixels=int(mask.sum()),
        foreground_retention_ratio=1.0,border_contacts=tuple(border_contacts),mask_source=MaskSource.NATIVE_INSTANCE,
        approval_status=ApprovalStatus.NOT_REQUIRED,mask_model_provenance=ModelProvenance('seg','Segmentation','segmentation'),
        detection_model_provenance=ModelProvenance('det','Detection','detection'),
    )
    K=np.array([[80.,0.,width/2],[0.,80.,height/2],[0.,0.,1.]])
    bundle=AlignedFrameBundle(
        frame,space,np.full((height,width,3),(40,90,150),np.uint8),np.full((height,width),3.0,np.float32),
        np.ones((height,width),bool),np.ones((height,width),bool),np.full((height,width),0.92,np.float32),
        np.ones((height,width),bool),ConfidenceKind.CONFIDENCE,
        CameraIntrinsics(K,'depth',IntrinsicsSource.ESTIMATED,CalibrationStatus.NOT_CALIBRATED),depth_scale,1,
        ModelProvenance('depth','Depth Model','depth'),(ModelProvenance('seg','Segmentation','segmentation'),),(obs,),(),AlignmentStatus.SUCCESS,
    )
    clean=cuboid_points(extents=extents,centre=(0.25,-0.1,3.0),angle=angle,n=22,noise=0.002,outliers=0)
    raw=cuboid_points(extents=extents,centre=(0.25,-0.1,3.0),angle=angle,n=22,noise=0.002,outliers=outliers)
    camera=CoordinateFrame('camera-depth',CoordinateFrameKind.CAMERA_DEPTH,AxisConvention.X_RIGHT_Y_DOWN_Z_FORWARD,source_frame=frame)
    def project_pixels(points):
        u=np.rint(K[0,0]*points[:,0]/points[:,2]+K[0,2]).astype(int)
        v=np.rint(K[1,1]*points[:,1]/points[:,2]+K[1,2]).astype(int)
        return np.column_stack((np.clip(u,0,width-1),np.clip(v,0,height-1)))
    pix_clean=project_pixels(clean)
    pix_raw=project_pixels(raw)
    recovery=ScaleRecoveryStatus.NOT_REQUIRED if depth_scale is DepthScale.METRIC else ScaleRecoveryStatus.NOT_ATTEMPTED
    semantics={DepthScale.METRIC:PointCoordinateSemantics.METRIC_METRES,DepthScale.RELATIVE:PointCoordinateSemantics.RELATIVE_MODEL_UNITS,DepthScale.SYNTHETIC:PointCoordinateSemantics.SYNTHETIC_UNITS}.get(depth_scale,PointCoordinateSemantics.UNKNOWN)
    def cloud(points,pixels,raw_count):
        return ObjectPointCloud('object-0',frame,camera.frame_id,points,'depth',frame.source_size,pixels,
            raw_point_count=raw_count,retained_point_count=len(points),mask_id='mask-0',depth_model_provenance=bundle.depth_model_provenance,
            depth_scale=depth_scale,intrinsics_source=IntrinsicsSource.ESTIMATED,scale_recovery_status=recovery)
    raw_cloud=cloud(raw,pix_raw,len(raw)); filtered=cloud(clean,pix_clean,len(raw))
    nraw,nclean=len(raw),len(clean)
    counts=ExtractionStageCounts(nraw,nraw,nraw,nraw,nraw,nraw,nraw,nraw,nclean,nclean,nclean,nclean)
    extracted=ObjectPointExtractionResult('object-0',frame,ObjectPointExtractionStatus.SUCCESS,camera,semantics,raw_cloud,filtered,counts)
    batch=ObjectPointExtractionBatchResult(frame,camera,(extracted,),ObjectPointExtractionStatus.SUCCESS)
    fit=fit_object_geometry(ObjectGeometryFitRequest(batch))
    return bundle,batch,fit
