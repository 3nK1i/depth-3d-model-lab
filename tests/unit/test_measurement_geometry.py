from __future__ import annotations
import numpy as np
import pytest
from core.measurement.contracts import BoundingBox3D, BoundingBoxKind, MeasurementKind
from core.measurement.geometry import apply_scale_to_values, rotation_matrix_to_rpy, semantic_extents, values_from_box

def test_camera_aligned_extents_map_directly_to_width_height_depth():
    box=BoundingBox3D('b','o','camera',BoundingBoxKind.CAMERA_ALIGNED,np.array([1.,2.,3.]),np.array([2.,1.,.5]),np.eye(3),'test',100)
    values=values_from_box(box)
    assert values[MeasurementKind.WIDTH]==pytest.approx(2.0)
    assert values[MeasurementKind.HEIGHT]==pytest.approx(1.0)
    assert values[MeasurementKind.DEPTH]==pytest.approx(.5)
    assert values[MeasurementKind.CAMERA_DISTANCE]==pytest.approx(np.sqrt(14))
    assert values[MeasurementKind.VOLUME]==pytest.approx(1.0)

def test_oriented_extents_are_assigned_to_camera_semantic_axes():
    R=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    box=BoundingBox3D('b','o','camera',BoundingBoxKind.ORIENTED,np.zeros(3),np.array([3.,2.,1.]),R,'test',100)
    ext=semantic_extents(box)
    assert ext[MeasurementKind.WIDTH]==pytest.approx(2.0)
    assert ext[MeasurementKind.HEIGHT]==pytest.approx(3.0)
    assert ext[MeasurementKind.DEPTH]==pytest.approx(1.0)

def test_rotation_matrix_to_rpy_recovers_yaw():
    a=.4; c,s=np.cos(a),np.sin(a); R=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    roll,pitch,yaw=rotation_matrix_to_rpy(R)
    assert roll==pytest.approx(0,abs=1e-9)
    assert pitch==pytest.approx(0,abs=1e-9)
    assert yaw==pytest.approx(a)

def test_scale_recovery_scales_volume_cubically_but_not_orientation():
    values={MeasurementKind.WIDTH:2.,MeasurementKind.VOLUME:3.,MeasurementKind.ORIENTATION_YAW:.2}
    scaled=apply_scale_to_values(values,4.)
    assert scaled[MeasurementKind.WIDTH]==8.
    assert scaled[MeasurementKind.VOLUME]==192.
    assert scaled[MeasurementKind.ORIENTATION_YAW]==.2
