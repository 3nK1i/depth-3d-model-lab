from __future__ import annotations
import numpy as np
from core.contracts.common import FrameReference, ImageSize
from core.contracts.spatial import AxisConvention, CoordinateFrame, CoordinateFrameKind
from core.measurement.contracts import ObjectPointCloud
from core.models.metadata import DepthScale, IntrinsicsSource
from core.object_geometry.contracts import ExtractionStageCounts,ObjectPointExtractionBatchResult,ObjectPointExtractionResult,ObjectPointExtractionStatus,PointCoordinateSemantics

def rotation_z(angle):
    c,s=np.cos(angle),np.sin(angle); return np.array([[c,-s,0],[s,c,0],[0,0,1]],float)

def cuboid_points(extents=(2.,1.,.5), centre=(1.,2.,3.), angle=.4, n=16, noise=0., outliers=0):
    xs=np.linspace(-extents[0]/2,extents[0]/2,n); ys=np.linspace(-extents[1]/2,extents[1]/2,n); zs=np.linspace(-extents[2]/2,extents[2]/2,max(4,n//3))
    pts=[]
    for x in xs:
      for y in ys:
       for z in (zs[0],zs[-1]): pts.append((x,y,z))
    for x in xs:
      for z in zs:
       for y in (ys[0],ys[-1]): pts.append((x,y,z))
    for y in ys:
      for z in zs:
       for x in (xs[0],xs[-1]): pts.append((x,y,z))
    p=np.unique(np.array(pts,float),axis=0); R=rotation_z(angle); p=p@R.T+np.array(centre)
    if noise: p += np.random.default_rng(1).normal(0,noise,p.shape)
    if outliers: p=np.vstack([p,np.random.default_rng(2).uniform(-8,8,(outliers,3))+np.array(centre)])
    return p

def extraction_batch(points=None, *, depth_scale=DepthScale.METRIC, semantics=PointCoordinateSemantics.METRIC_METRES):
    p=cuboid_points() if points is None else np.asarray(points,float); frame=FrameReference(1,2,3.0,ImageSize(640,480)); camera=CoordinateFrame("camera-depth",CoordinateFrameKind.CAMERA_DEPTH,AxisConvention.X_RIGHT_Y_DOWN_Z_FORWARD,source_frame=frame)
    cloud=ObjectPointCloud("object-0",frame,camera.frame_id,p,"depth",ImageSize(640,480),raw_point_count=len(p),retained_point_count=len(p),mask_id="mask",depth_scale=depth_scale,intrinsics_source=IntrinsicsSource.ESTIMATED)
    counts=ExtractionStageCounts(*([len(p)]*12))
    result=ObjectPointExtractionResult("object-0",frame,ObjectPointExtractionStatus.SUCCESS,camera,semantics,cloud,cloud,counts)
    return ObjectPointExtractionBatchResult(frame,camera,(result,),ObjectPointExtractionStatus.SUCCESS)
