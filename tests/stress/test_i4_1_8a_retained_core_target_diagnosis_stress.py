from __future__ import annotations

import numpy as np

from core.reconstruction.offline_dense_i4_orbit_geometry_sampling_diagnosis import trajectory_circle_target
from core.reconstruction.offline_dense_i4_orbit_qualification import orbit_coverage_metrics
from core.reconstruction.offline_dense_i4_retained_core_target_diagnosis import (
    build_i4_1_8a_retained_core_target_diagnosis,
)


def _pose(center: np.ndarray, target: np.ndarray) -> np.ndarray:
    z = (target - center) / np.linalg.norm(target - center)
    up = np.asarray([0.0, 0.0, 1.0])
    if abs(float(np.dot(up, z))) > 0.95:
        up = np.asarray([0.0, 1.0, 0.0])
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4); pose[:3, :3] = np.column_stack([x, y, z]); pose[:3, 3] = center
    return pose


def _scene() -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray([0.0, 0.0, 1.0])
    angles = np.deg2rad(np.linspace(-110.0, 110.0, 40))
    poses = np.asarray([
        _pose(target + np.asarray([4.0*np.cos(a), 4.0*np.sin(a), 0.1*np.sin(2*a)]), target)
        for a in angles
    ])
    xs = np.linspace(3.0, 9.0, 80); ys = np.linspace(-4.0, 4.0, 60)
    floor = np.asarray([[x, y, 0.0] for x in xs for y in ys])
    rng = np.random.default_rng(81)
    obj = target + rng.uniform([-0.5,-0.4,-0.5], [0.5,0.4,0.5], size=(1400,3))
    return poses, np.vstack([floor, obj])


def _summary(poses: np.ndarray, core: np.ndarray) -> tuple[dict, dict, np.ndarray]:
    current = np.median(core, axis=0)
    metric = orbit_coverage_metrics(poses, current)
    source = {
        "phase":"I4.1.8", "status":"degraded",
        "stage_gates":{"production_fusion":True,"formal_filtering":True,"canonical_persistence":True,"ordinary_i2_bound_scene_quality":True},
        "authority":{"metric_scale_claimed":False,"measurement_authoritative":False,"surface_mesh_generated":False,"dense_consistency_threshold_relaxation_authorized":False},
        "final_assessment":{"failed_gates":["selected_azimuth_coverage","selected_opposed_view_diversity"],"selected_orbit":metric},
    }
    circle,_ = trajectory_circle_target(poses)
    b = {"phase":"I4.1.7B","status":"qualified_for_i4_1_8","target_correction":{"target_reconstruction":circle.tolist()}}
    return source,b,current


def _axis_permutation_transform() -> np.ndarray:
    # Proper rotation: x->y, y->z, z->x plus translation.  Coordinate-wise
    # medians transform exactly under axis permutation/sign changes, while the
    # dominant plane and camera-attention diagnosis must remain equivalent.
    x = np.eye(4)
    x[:3,:3] = np.asarray([[0.0,0.0,1.0],[1.0,0.0,0.0],[0.0,1.0,0.0]])
    x[:3,3] = np.asarray([7.0,-3.0,5.0])
    return x


def test_i4_1_8a_bias_diagnosis_survives_rigid_axis_permutation_and_translation() -> None:
    poses,core = _scene(); source,b,current = _summary(poses,core)
    base = build_i4_1_8a_retained_core_target_diagnosis(
        source_i4_1_8_summary=source, source_i4_1_7b_summary=b,
        production_selected_camera_matrices=poses, retained_core_points=core,
        retained_core_voxel_size=0.08, current_retained_core_target=current,
    )
    x = _axis_permutation_transform()
    moved_poses = np.einsum("ij,njk->nik", x, poses)
    moved_core = core @ x[:3,:3].T + x[:3,3]
    moved_current = current @ x[:3,:3].T + x[:3,3]
    moved_source,moved_b,_ = _summary(moved_poses,moved_core)
    moved = build_i4_1_8a_retained_core_target_diagnosis(
        source_i4_1_8_summary=moved_source, source_i4_1_7b_summary=moved_b,
        production_selected_camera_matrices=moved_poses, retained_core_points=moved_core,
        retained_core_voxel_size=0.08, current_retained_core_target=moved_current,
    )
    assert base["diagnosis"]["classification"] == moved["diagnosis"]["classification"] == "whole_core_median_dominant_plane_bias_supported"
    assert np.isclose(
        base["retained_core"]["dominant_plane"]["core_inlier_fraction"],
        moved["retained_core"]["dominant_plane"]["core_inlier_fraction"],
        atol=1e-10,
    )
    for name in ("production_bound_trajectory_circle", "retained_core_camera_attention_candidate"):
        a = base["targets"][name]["orbit_evaluation"]["current_metric"]
        z = moved["targets"][name]["orbit_evaluation"]["current_metric"]
        for key in ("azimuth_coverage_degrees","maximum_opposed_view_angle_degrees","target_facing_fraction","standoff_p90_to_p10_ratio"):
            assert np.isclose(float(a[key]), float(z[key]), rtol=1e-8, atol=1e-8)
