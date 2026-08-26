from __future__ import annotations

import numpy as np

from core.reconstruction.offline_dense_i4_retained_core_object_support import (
    build_i4_1_8b_retained_core_object_support_isolation,
)
from tests.unit.test_i4_1_8b_retained_core_object_support import _scene, _source_summary


def _transform() -> np.ndarray:
    axis = np.asarray([1.0, -2.0, 0.7], dtype=np.float64); axis /= np.linalg.norm(axis)
    angle = np.deg2rad(53.0)
    k = np.asarray([[0.0,-axis[2],axis[1]],[axis[2],0.0,-axis[0]],[-axis[1],axis[0],0.0]])
    r = np.eye(3) + np.sin(angle)*k + (1.0-np.cos(angle))*(k@k)
    x = np.eye(4); x[:3,:3] = r; x[:3,3] = np.asarray([6.0,-4.0,3.0])
    return x


def _build(poses, sources, points, matrix, support, confidence):
    source, diag = _source_summary()
    return build_i4_1_8b_retained_core_object_support_isolation(
        source_i4_1_8_summary=source,
        source_i4_1_8a_summary=diag,
        production_selected_camera_matrices=poses,
        production_selected_source_sequence_indices=sources,
        retained_core_points=points,
        retained_core_voxel_size=0.08,
        exact_view_observation_matrix=matrix,
        core_support_counts=support,
        core_filter_confidence=confidence,
    )


def test_i4_1_8b_object_support_isolation_is_rigid_transform_equivariant() -> None:
    poses, sources, points, matrix, support, confidence = _scene()
    base = _build(poses, sources, points, matrix, support, confidence)
    x = _transform(); r = x[:3,:3]; t = x[:3,3]
    moved_poses = np.einsum('ij,njk->nik', x, poses)
    moved_points = points @ r.T + t
    moved = _build(moved_poses, sources, moved_points, matrix, support, confidence)
    assert base['diagnosis']['classification'] == moved['diagnosis']['classification'] == 'retained_core_object_support_isolated'
    assert base['counts']['isolated_support_points'] == moved['counts']['isolated_support_points']
    a = np.asarray(base['selected_support']['target_reconstruction'], dtype=np.float64)
    b = np.asarray(moved['selected_support']['target_reconstruction'], dtype=np.float64)
    assert np.allclose(a @ r.T + t, b, atol=1e-7, rtol=1e-7)
    assert np.isclose(
        base['selected_support']['selection_score'], moved['selected_support']['selection_score'], atol=1e-8, rtol=1e-8
    )
    assert np.isclose(
        base['selected_support']['target_bootstrap_stability']['maximum_drift_voxels'],
        moved['selected_support']['target_bootstrap_stability']['maximum_drift_voxels'],
        atol=1e-7, rtol=1e-7,
    )
