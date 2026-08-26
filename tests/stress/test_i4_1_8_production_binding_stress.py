from __future__ import annotations

import numpy as np

from core.reconstruction.offline_dense_i4_production_binding import (
    I418ProductionBinding,
    build_final_i4_1_assessment,
    deterministic_capped_positions,
)


def _rigid_transform() -> np.ndarray:
    a = np.deg2rad(37.0)
    c, s = np.cos(a), np.sin(a)
    t = np.eye(4)
    t[:3, :3] = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    t[:3, 3] = np.asarray([8.0, -3.0, 5.0])
    return t


def _scene() -> tuple[np.ndarray, np.ndarray]:
    target = np.zeros(3)
    angles = np.deg2rad(np.linspace(-110.0, 110.0, 40))
    poses = []
    for a in angles:
        center = np.asarray([4.0 * np.cos(a), 4.0 * np.sin(a), 0.2 * np.sin(a)])
        z = (target - center) / np.linalg.norm(target - center)
        up = np.asarray([0.0, 0.0, 1.0])
        x = np.cross(up, z); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        pose = np.eye(4)
        pose[:3, :3] = np.column_stack([x, y, z])
        pose[:3, 3] = center
        poses.append(pose)
    rng = np.random.default_rng(9)
    half = rng.normal(0.0, 0.1, size=(1000, 3))
    core = np.concatenate([half, -half], axis=0)
    return np.asarray(poses), core


def test_final_core_orbit_qualification_is_rigid_transform_invariant() -> None:
    poses, core = _scene()
    binding = I418ProductionBinding(
        plan_name="global_anchors_plus_local_neighbors",
        qualified_source_count=40,
        final_fusion_cap=40,
        selected_positions=tuple(range(40)),
        selected_source_sequence_indices=tuple(range(40)),
    )
    base = build_final_i4_1_assessment(
        input_global_views=40,
        registered_global_views=40,
        production_alignment_accepted_views=40,
        registered_camera_matrices=poses,
        production_selected_camera_matrices=poses,
        retained_core_points=core,
        i2_summary={"phase": "I2", "automated_pass": True},
        binding=binding,
    )
    x = _rigid_transform()
    moved_poses = np.einsum("ij,njk->nik", x, poses)
    moved_core = (core @ x[:3, :3].T) + x[:3, 3]
    moved = build_final_i4_1_assessment(
        input_global_views=40,
        registered_global_views=40,
        production_alignment_accepted_views=40,
        registered_camera_matrices=moved_poses,
        production_selected_camera_matrices=moved_poses,
        retained_core_points=moved_core,
        i2_summary={"phase": "I2", "automated_pass": True},
        binding=binding,
    )
    assert base["pass"] == moved["pass"] == True
    for key in (
        "azimuth_coverage_degrees",
        "maximum_opposed_view_angle_degrees",
        "target_facing_fraction",
        "standoff_p90_to_p10_ratio",
    ):
        assert np.isclose(base["selected_orbit"][key], moved["selected_orbit"][key], rtol=1e-9, atol=1e-9)


def test_capped_selector_is_stable_for_large_qualified_paths() -> None:
    for count in (40, 41, 79, 118, 120, 500, 5000):
        positions = deterministic_capped_positions(count, 40)
        assert len(positions) == min(count, 40)
        assert positions[0] == 0
        assert positions[-1] == count - 1
        assert len(set(positions)) == len(positions)
