"""Object-local coordinate-frame construction."""
from __future__ import annotations
import numpy as np
from core.contracts.spatial import AxisConvention, CoordinateFrame, CoordinateFrameKind, RigidTransform3D
from .fitting_contracts import ObjectFrameEstimate, OrientationStability

def make_object_frame(camera_frame, axes):
    object_frame=CoordinateFrame(frame_id=f"object:{axes.observation_id}",kind=CoordinateFrameKind.OBJECT_LOCAL,axis_convention=AxisConvention.CUSTOM,parent_frame_id=camera_frame.frame_id,source_frame=camera_frame.source_frame)
    M=np.eye(4); M[:3,:3]=axes.orientation_matrix; M[:3,3]=axes.robust_centre_xyz
    overall=OrientationStability.STABLE if all(x is OrientationStability.STABLE for x in axes.axis_stability) else (OrientationStability.AMBIGUOUS if any(x is OrientationStability.AMBIGUOUS for x in axes.axis_stability) else OrientationStability.WEAK)
    return ObjectFrameEstimate(camera_frame,object_frame,RigidTransform3D(object_frame.frame_id,camera_frame.frame_id,M,method="robust_pca_object_frame"),"robust_pca",overall,axes.degeneracy)
