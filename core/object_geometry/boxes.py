"""Axis-aligned, camera-aligned and oriented box fitting."""
from __future__ import annotations
import numpy as np
from core.measurement.contracts import BoundingBox3D, BoundingBoxKind
from .fitting_contracts import BoxFitDiagnostics

def _fit(points, observation_id, frame_id, kind, R, lower, upper, method):
    p=np.asarray(points,dtype=np.float64); local=p@R
    lo=np.percentile(local,lower,axis=0); hi=np.percentile(local,upper,axis=0); ext=hi-lo
    if np.any(ext<=1e-12): raise ValueError("box has zero extent")
    local_c=(lo+hi)/2; centre=local_c@R.T
    clipped=np.clip(local,lo,hi); dist=np.linalg.norm(local-clipped,axis=1); residual=float(np.sqrt(np.mean(dist**2)))
    inside=np.all((local>=lo)&(local<=hi),axis=1); box_id=f"{observation_id}:{kind.value}"
    box=BoundingBox3D(box_id,observation_id,frame_id,kind,centre,ext,R,method,len(p),residual,f"p{lower:g}-p{upper:g}")
    diag=BoxFitDiagnostics(box_id,lower,upper,len(p),int(inside.sum()),int((~inside).sum()),residual,lo,hi)
    return box,diag

def fit_axis_aligned(points, observation_id, frame_id):
    return _fit(points,observation_id,frame_id,BoundingBoxKind.AXIS_ALIGNED,np.eye(3),0,100,"strict_min_max")
def fit_camera_aligned(points, observation_id, frame_id, lower=1, upper=99):
    return _fit(points,observation_id,frame_id,BoundingBoxKind.CAMERA_ALIGNED,np.eye(3),lower,upper,"camera_percentile")
def fit_oriented(points, observation_id, frame_id, orientation, lower=1, upper=99):
    return _fit(points,observation_id,frame_id,BoundingBoxKind.ORIENTED,np.asarray(orientation),lower,upper,"robust_pca_percentile")
