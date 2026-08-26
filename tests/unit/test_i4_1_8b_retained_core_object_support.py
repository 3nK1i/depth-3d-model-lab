from __future__ import annotations

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_i4_orbit_qualification import I4OrbitQualificationPolicy
from core.reconstruction.offline_dense_i4_retained_core_object_support import (
    I418BObjectSupportIsolationPolicy,
    build_i4_1_8b_retained_core_object_support_isolation,
)


def _pose(center: np.ndarray, target: np.ndarray) -> np.ndarray:
    z = (target - center) / np.linalg.norm(target - center)
    up = np.asarray([0.0, 0.0, 1.0])
    if abs(float(np.dot(up, z))) > 0.95:
        up = np.asarray([0.0, 1.0, 0.0])
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    p = np.eye(4); p[:3, :3] = np.column_stack([x, y, z]); p[:3, 3] = center
    return p


def _source_summary() -> tuple[dict, dict]:
    source = {
        "phase": "I4.1.8",
        "status": "degraded",
        "stage_gates": {
            "production_fusion": True,
            "formal_filtering": True,
            "canonical_persistence": True,
            "ordinary_i2_bound_scene_quality": True,
        },
        "authority": {
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "surface_mesh_generated": False,
            "dense_consistency_threshold_relaxation_authorized": False,
        },
        "final_assessment": {
            "failed_gates": ["selected_azimuth_coverage", "selected_azimuth_bins", "selected_opposed_view_diversity"]
        },
    }
    diag = {
        "phase": "I4.1.8A",
        "status": "diagnosis_ready",
        "diagnosis": {"classification": "trajectory_circle_not_yet_retained_core_supported"},
        "authority": {
            "production_target_changed": False,
            "orbit_thresholds_changed": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "surface_mesh_generated": False,
        },
    }
    return source, diag


def _scene(*, twin: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(4182)
    target = np.asarray([0.0, 0.0, 1.0])
    angles = np.deg2rad(np.linspace(-110.0, 110.0, 40))
    poses = np.asarray([
        _pose(target + np.asarray([4.2*np.cos(a), 4.2*np.sin(a), 0.18*np.sin(2*a)]), target)
        for a in angles
    ])
    # Scene/background Core is broad and mostly off the cameras' optical center.
    xs = np.linspace(2.5, 8.5, 70); ys = np.linspace(-4.0, 4.0, 50)
    floor = np.asarray([[x, y, 0.0] for x in xs for y in ys], dtype=np.float64)
    object_a = target + rng.uniform([-0.48, -0.42, -0.48], [0.48, 0.42, 0.48], size=(1600, 3))
    if twin:
        object_b = target + np.asarray([1.25, 0.0, 0.0]) + rng.uniform(
            [-0.48, -0.42, -0.48], [0.48, 0.42, 0.48], size=(1600, 3)
        )
        points = np.vstack([floor, object_a, object_b])
    else:
        points = np.vstack([floor, object_a])

    n_floor = len(floor); n_obj = len(object_a); views = len(poses)
    matrix = np.zeros((views, len(points)), dtype=bool)
    # Floor voxels are seen only by local windows of views.
    for j in range(n_floor):
        center_rank = int((j * 17) % views)
        width = 3 + (j % 3)
        ranks = [(center_rank + k) % views for k in range(width)]
        matrix[ranks, j] = True
    # Object support persists over broad but not necessarily all view windows.
    for local in range(n_obj):
        start = local % 10
        ranks = np.arange(start, views - (9 - start), 2, dtype=np.int64)
        matrix[ranks, n_floor + local] = True
    if twin:
        offset = n_floor + n_obj
        for local in range(n_obj):
            start = local % 10
            ranks = np.arange(start, views - (9 - start), 2, dtype=np.int64)
            matrix[ranks, offset + local] = True

    support = matrix.sum(axis=0, dtype=np.uint32)
    confidence = np.where(np.arange(len(points)) >= n_floor, 0.92, 0.70).astype(np.float64)
    sources = np.rint(np.linspace(0, 1333, views)).astype(np.int64)
    return poses, sources, points, matrix, support, confidence


def _build(*, twin: bool = False, qualification_policy: I4OrbitQualificationPolicy | None = None):
    poses, sources, points, matrix, support, confidence = _scene(twin=twin)
    source, diag = _source_summary()
    result = build_i4_1_8b_retained_core_object_support_isolation(
        source_i4_1_8_summary=source,
        source_i4_1_8a_summary=diag,
        production_selected_camera_matrices=poses,
        production_selected_source_sequence_indices=sources,
        retained_core_points=points,
        retained_core_voxel_size=0.08,
        exact_view_observation_matrix=matrix,
        core_support_counts=support,
        core_filter_confidence=confidence,
        qualification_policy=qualification_policy,
    )
    return result, points


def test_i4_1_8b_isolates_retained_core_object_support_without_promoting_authority() -> None:
    result, _ = _build()
    assert result["status"] == "object_support_isolated"
    assert result["diagnosis"]["classification"] == "retained_core_object_support_isolated"
    assert result["diagnosis"]["object_support_isolation_ready"] is True
    selected = result["selected_support"]
    assert selected is not None
    target = np.asarray(selected["target_reconstruction"], dtype=np.float64)
    assert np.linalg.norm(target - np.asarray([0.0, 0.0, 1.0])) < 0.30
    assert selected["active_view_fraction"] >= 0.8
    assert selected["active_source_span_fraction"] >= 0.8
    assert selected["posthoc_orbit_used_for_selection_or_readiness"] is False
    assert result["evidence"]["component_selection_uses_locked_i4_1_gates"] is False
    assert result["authority"]["production_target_changed"] is False
    assert result["decision"]["authorizes_target_change"] is False
    assert result["diagnosis"]["recommends_i4_1_8c_final_target_binding_design"] is True


def test_i4_1_8b_isolation_readiness_does_not_depend_on_orbit_gate_pass() -> None:
    impossible = I4OrbitQualificationPolicy(
        minimum_selected_azimuth_coverage_degrees=180.0,
        minimum_selected_occupied_azimuth_bins=12,
        minimum_opposed_view_angle_degrees=180.0,
    )
    result, _ = _build(qualification_policy=impossible)
    assert result["diagnosis"]["object_support_isolation_ready"] is True
    assert result["selected_support"]["posthoc_orbit_diagnosis"]["robust_locked_orbit_pass"] is False
    assert result["selected_support"]["posthoc_orbit_used_for_selection_or_readiness"] is False


def test_i4_1_8b_fails_closed_when_exact_attribution_disagrees_with_saved_support() -> None:
    poses, sources, points, matrix, support, confidence = _scene()
    source, diag = _source_summary()
    bad = support.copy(); bad[0] += 1
    with pytest.raises(ReconstructionValidationError, match="support counts"):
        build_i4_1_8b_retained_core_object_support_isolation(
            source_i4_1_8_summary=source,
            source_i4_1_8a_summary=diag,
            production_selected_camera_matrices=poses,
            production_selected_source_sequence_indices=sources,
            retained_core_points=points,
            retained_core_voxel_size=0.08,
            exact_view_observation_matrix=matrix,
            core_support_counts=bad,
            core_filter_confidence=confidence,
        )


def test_i4_1_8b_rejects_unrelated_source_failure() -> None:
    poses, sources, points, matrix, support, confidence = _scene()
    source, diag = _source_summary(); source["final_assessment"]["failed_gates"].append("selected_view_count")
    with pytest.raises(ReconstructionValidationError, match="unrelated"):
        build_i4_1_8b_retained_core_object_support_isolation(
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
