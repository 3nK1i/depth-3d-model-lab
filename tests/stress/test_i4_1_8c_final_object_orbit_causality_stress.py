from __future__ import annotations

import numpy as np

from core.reconstruction.offline_dense_i4_final_object_orbit_causality import build_i4_1_8c_final_object_orbit_causality
from core.reconstruction.offline_dense_i4_retained_core_object_support import _geometric_median
from tests.unit.test_i4_1_8c_final_object_orbit_causality import _pose, _summaries, _support


def _transform() -> np.ndarray:
    axis=np.asarray([1.0,-2.0,0.5]); axis/=np.linalg.norm(axis); a=np.deg2rad(47.0)
    k=np.asarray([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]],dtype=np.float64)
    r=np.eye(3)+np.sin(a)*k+(1-np.cos(a))*(k@k)
    x=np.eye(4); x[:3,:3]=r; x[:3,3]=np.asarray([5.0,-3.0,2.0]); return x


def _build(points, target, q, qsrc, b, bsrc):
    s7,s8,s8b=_summaries(target)
    return build_i4_1_8c_final_object_orbit_causality(
        source_i4_1_7b_summary=s7, source_i4_1_8_summary=s8, source_i4_1_8b_summary=s8b,
        qualified_camera_matrices=q, qualified_source_sequence_indices=qsrc,
        production_bound_camera_matrices=b, production_bound_source_sequence_indices=bsrc,
        isolated_core_points=points, retained_core_voxel_size=0.05,
    )


def test_i4_1_8c_causality_and_locked_metrics_are_rigid_transform_invariant() -> None:
    points,target=_support(); angles=np.linspace(-110,110,118)
    q=np.asarray([_pose(a,target) for a in angles]); qsrc=np.arange(118)*10
    pos=np.rint(np.linspace(0,117,40)).astype(np.int64); b=q[pos]; bsrc=qsrc[pos]
    base=_build(points,target,q,qsrc,b,bsrc)
    x=_transform(); r=x[:3,:3]; t=x[:3,3]
    moved_points=points@r.T+t; moved_target=_geometric_median(moved_points)
    moved_q=np.einsum('ij,njk->nik',x,q); moved_b=np.einsum('ij,njk->nik',x,b)
    moved=_build(moved_points,moved_target,moved_q,qsrc,moved_b,bsrc)
    assert base['causality']['classification']==moved['causality']['classification']=='final_object_orbit_qualified'
    for name in ('qualified_path','production_bound'):
        a=base['pose_sets'][name]['metrics']; c=moved['pose_sets'][name]['metrics']
        assert np.isclose(a['azimuth_coverage_degrees'],c['azimuth_coverage_degrees'],atol=1e-7)
        assert a['occupied_azimuth_bins']==c['occupied_azimuth_bins']
        assert np.isclose(a['maximum_opposed_view_angle_degrees'],c['maximum_opposed_view_angle_degrees'],atol=1e-7)
        assert np.isclose(a['target_facing_fraction'],c['target_facing_fraction'],atol=1e-7)
