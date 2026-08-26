from __future__ import annotations

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_i4_orbit_qualification import (
    I4OrbitQualificationPolicy,
    build_i4_1_orbit_assessment,
    camera_poses_from_sfm_report,
    orbit_coverage_metrics,
    robust_object_target,
    select_camera_poses,
)


def _look_at_pose(angle_deg: float, *, radius: float = 4.0, height: float = 0.2, face_target: bool = True) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    center = np.asarray([radius * np.cos(theta), radius * np.sin(theta), height], dtype=np.float64)
    target = np.zeros(3, dtype=np.float64)
    z = target - center
    z /= np.linalg.norm(z)
    if not face_target:
        z = -z
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(up, z))) > 0.95:
        up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x); y /= np.linalg.norm(y)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.column_stack([x, y, z])
    matrix[:3, 3] = center
    return matrix


def _circle(count: int = 12, *, radius: float = 4.0, face_target: bool = True) -> np.ndarray:
    return np.asarray([_look_at_pose(i * 360.0 / count, radius=radius, face_target=face_target) for i in range(count)])


def _core() -> np.ndarray:
    return np.asarray([(x, y, z) for x in np.linspace(-0.5, 0.5, 5) for y in np.linspace(-0.4, 0.4, 5) for z in np.linspace(-0.3, 0.3, 5)], dtype=np.float64)


def _summaries(count: int = 12, *, i2_pass: bool = True):
    sfm = {"input_keyframes": count}
    i1 = {
        "phase": "I1", "status": "pass",
        "sfm": {"registered_view_count": count},
        "dense_qualification": {"alignment_accepted_views": count},
    }
    i2 = {"phase": "I2", "automated_pass": i2_pass}
    return sfm, i1, i2


def test_i4_1_policy_defaults_are_object_orbit_conservative():
    p = I4OrbitQualificationPolicy()
    assert p.minimum_registered_fraction == pytest.approx(0.80)
    assert p.minimum_selected_azimuth_coverage_degrees == pytest.approx(150.0)
    assert p.minimum_opposed_view_angle_degrees == pytest.approx(120.0)
    assert p.minimum_selected_views == 10


def test_i4_1_policy_rejects_invalid_standoff_ratio():
    with pytest.raises(ReconstructionValidationError):
        I4OrbitQualificationPolicy(maximum_standoff_p90_to_p10_ratio=0.9)


def test_i4_1_robust_target_ignores_sparse_far_outlier():
    core = _core()
    points = np.vstack([core, np.asarray([[1000.0, -900.0, 700.0]])])
    target = robust_object_target(points)
    assert np.linalg.norm(target) < 0.1


def test_i4_1_camera_report_extracts_dependency_neutral_poses():
    matrices = _circle(4)
    report = {"sfm_result": {"cameras": [
        {"keyframe_id": f"kf-{i}", "T_reconstruction_from_camera": matrix.tolist()}
        for i, matrix in enumerate(matrices)
    ]}}
    ids, poses = camera_poses_from_sfm_report(report)
    assert ids == ("kf-0", "kf-1", "kf-2", "kf-3")
    assert np.allclose(poses, matrices)


def test_i4_1_camera_report_rejects_duplicate_keyframe_ids():
    matrix = _look_at_pose(0.0)
    report = {"sfm_result": {"cameras": [
        {"keyframe_id": "same", "T_reconstruction_from_camera": matrix.tolist()},
        {"keyframe_id": "same", "T_reconstruction_from_camera": _look_at_pose(90).tolist()},
        {"keyframe_id": "other", "T_reconstruction_from_camera": _look_at_pose(180).tolist()},
    ]}}
    with pytest.raises(ReconstructionValidationError):
        camera_poses_from_sfm_report(report)


def test_i4_1_select_camera_poses_preserves_qualified_order():
    matrices = _circle(4)
    ids, poses = select_camera_poses(("a", "b", "c", "d"), matrices, ("d", "b", "a"))
    assert ids == ("d", "b", "a")
    assert np.allclose(poses[0], matrices[3])
    assert np.allclose(poses[1], matrices[1])


def test_i4_1_select_camera_poses_fails_closed_on_missing_selected_view():
    with pytest.raises(ReconstructionValidationError):
        select_camera_poses(("a", "b", "c"), _circle(3), ("a", "missing", "c"))


def test_i4_1_full_circle_has_broad_azimuth_coverage_and_bins():
    metrics = orbit_coverage_metrics(_circle(12), np.zeros(3), azimuth_bin_count=12)
    assert metrics["azimuth_coverage_degrees"] > 320.0
    assert metrics["occupied_azimuth_bins"] >= 11
    assert metrics["maximum_opposed_view_angle_degrees"] > 170.0


def test_i4_1_partial_arc_has_narrow_coverage():
    matrices = np.asarray([_look_at_pose(value) for value in np.linspace(0.0, 90.0, 10)])
    metrics = orbit_coverage_metrics(matrices, np.zeros(3))
    assert metrics["azimuth_coverage_degrees"] < 100.0


def test_i4_1_target_facing_fraction_is_one_for_look_at_orbit():
    metrics = orbit_coverage_metrics(_circle(12), np.zeros(3), facing_angle_degrees=30.0)
    assert metrics["target_facing_fraction"] == pytest.approx(1.0)
    assert metrics["facing_angle_degrees"]["p90"] < 1e-5


def test_i4_1_target_facing_detects_backwards_camera_axes():
    metrics = orbit_coverage_metrics(_circle(12, face_target=False), np.zeros(3), facing_angle_degrees=60.0)
    assert metrics["target_facing_fraction"] == pytest.approx(0.0)


def test_i4_1_standoff_ratio_is_scale_invariant_for_constant_radius():
    a = orbit_coverage_metrics(_circle(12, radius=2.0), np.zeros(3))
    b = orbit_coverage_metrics(_circle(12, radius=20.0), np.zeros(3))
    assert a["standoff_p90_to_p10_ratio"] == pytest.approx(b["standoff_p90_to_p10_ratio"], abs=1e-10)
    assert a["standoff_p90_to_p10_ratio"] < 1.01


def test_i4_1_assessment_passes_clean_object_orbit():
    matrices = _circle(12)
    ids = tuple(f"kf-{i}" for i in range(12))
    sfm, i1, i2 = _summaries(12)
    result = build_i4_1_orbit_assessment(
        sfm_summary=sfm,
        i1_summary=i1,
        i2_summary=i2,
        registered_camera_matrices=matrices,
        registered_camera_ids=ids,
        selected_keyframe_ids=ids,
        core_points=_core(),
    )
    assert result["pass"] is True
    assert all(result["gates"].values())


def test_i4_1_assessment_rejects_narrow_arc_even_when_i2_passes():
    matrices = np.asarray([_look_at_pose(value) for value in np.linspace(0.0, 90.0, 12)])
    ids = tuple(f"kf-{i}" for i in range(12))
    sfm, i1, i2 = _summaries(12, i2_pass=True)
    result = build_i4_1_orbit_assessment(
        sfm_summary=sfm,
        i1_summary=i1,
        i2_summary=i2,
        registered_camera_matrices=matrices,
        registered_camera_ids=ids,
        selected_keyframe_ids=ids,
        core_points=_core(),
    )
    assert result["pass"] is False
    assert result["gates"]["ordinary_i2_baseline_quality"] is True
    assert result["gates"]["selected_azimuth_coverage"] is False
