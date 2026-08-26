from __future__ import annotations

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_i4_orbit_geometry_sampling_diagnosis import trajectory_circle_target
from core.reconstruction.offline_dense_i4_orbit_qualification import orbit_coverage_metrics
from core.reconstruction.offline_dense_i4_retained_core_target_diagnosis import (
    build_i4_1_8a_retained_core_target_diagnosis,
)


def _pose(center: np.ndarray, target: np.ndarray) -> np.ndarray:
    z = target - center
    z = z / np.linalg.norm(z)
    up = np.asarray([0.0, 0.0, 1.0])
    if abs(float(np.dot(up, z))) > 0.95:
        up = np.asarray([0.0, 1.0, 0.0])
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([x, y, z])
    pose[:3, 3] = center
    return pose


def _orbit(*, count: int = 40, coverage: float = 220.0, target=(0.0, 0.0, 1.0)) -> np.ndarray:
    target_v = np.asarray(target, dtype=np.float64)
    angles = np.deg2rad(np.linspace(-coverage / 2.0, coverage / 2.0, count))
    return np.asarray([
        _pose(
            target_v + np.asarray([4.0 * np.cos(a), 4.0 * np.sin(a), 0.1 * np.sin(2.0 * a)]),
            target_v,
        )
        for a in angles
    ])


def _biased_core() -> np.ndarray:
    # Dominant floor is deliberately asymmetric relative to the object.  Its
    # coordinate-wise median is therefore a scene/floor target far from the
    # camera-attended object at (0,0,1).
    xs = np.linspace(3.0, 9.0, 90)
    ys = np.linspace(-4.0, 4.0, 70)
    floor = np.asarray([[x, y, 0.0] for x in xs for y in ys], dtype=np.float64)
    rng = np.random.default_rng(418)
    object_points = np.asarray([0.0, 0.0, 1.0]) + rng.uniform(
        [-0.5, -0.4, -0.5], [0.5, 0.4, 0.5], size=(1600, 3)
    )
    return np.vstack([floor, object_points])


def _source(poses: np.ndarray, current: np.ndarray, *, stage_pass: bool = True) -> dict:
    metric = orbit_coverage_metrics(poses, current)
    failed = []
    if metric["azimuth_coverage_degrees"] < 150.0:
        failed.append("selected_azimuth_coverage")
    if metric["occupied_azimuth_bins"] < 5:
        failed.append("selected_azimuth_bins")
    if metric["maximum_opposed_view_angle_degrees"] < 120.0:
        failed.append("selected_opposed_view_diversity")
    return {
        "phase": "I4.1.8",
        "status": "degraded",
        "stage_gates": {
            "production_fusion": stage_pass,
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
            "failed_gates": failed,
            "selected_orbit": metric,
        },
    }


def _b(poses: np.ndarray) -> dict:
    target, _ = trajectory_circle_target(poses)
    return {
        "phase": "I4.1.7B",
        "status": "qualified_for_i4_1_8",
        "target_correction": {"target_reconstruction": target.tolist()},
    }


def _run(core: np.ndarray, poses: np.ndarray, source: dict | None = None) -> dict:
    current = np.median(core, axis=0)
    return build_i4_1_8a_retained_core_target_diagnosis(
        source_i4_1_8_summary=source or _source(poses, current),
        source_i4_1_7b_summary=_b(poses),
        production_selected_camera_matrices=poses,
        retained_core_points=core,
        retained_core_voxel_size=0.08,
        current_retained_core_target=current,
    )


def test_detects_dominant_plane_whole_core_median_bias_without_promoting_target() -> None:
    poses = _orbit()
    result = _run(_biased_core(), poses)
    assert result["status"] == "diagnosis_ready"
    assert result["diagnosis"]["classification"] == "whole_core_median_dominant_plane_bias_supported"
    assert result["diagnosis"]["recommends_i4_1_8b_bounded_target_correction_design"] is True
    assert result["retained_core"]["dominant_plane"]["strong_dominant_plane"] is True
    assert result["diagnosis"]["evidence_flags"]["retained_core_attention_candidate_is_robust"] is True
    assert result["decision"]["authorizes_target_change"] is False
    assert result["authority"]["production_target_changed"] is False
    assert result["authority"]["metric_scale_claimed"] is False


def test_attention_candidate_is_derived_from_retained_core_and_near_true_object() -> None:
    poses = _orbit()
    result = _run(_biased_core(), poses)
    candidate = np.asarray(
        result["targets"]["retained_core_camera_attention_candidate"]["target_reconstruction"],
        dtype=np.float64,
    )
    assert np.linalg.norm(candidate - np.asarray([0.0, 0.0, 1.0])) < 0.25
    diag = result["targets"]["retained_core_camera_attention_candidate"]["diagnostics"]
    assert diag["dominant_plane_exclusion_applied"] is True
    assert diag["production_target_authority"] is False


def test_current_metric_is_reproduced_and_coordinate_median_axis_sensitivity_is_reported() -> None:
    poses = _orbit()
    result = _run(_biased_core(), poses)
    reproduction = result["current_target_reproduction"]
    assert reproduction["exact_within_numeric_tolerance"] is True
    assert reproduction["source_orbit_metric_reproduced"] is True
    probe = result["retained_core"]["coordinate_median_invariance_probe"]
    assert probe["maximum_drift_voxels"] > 1.0
    assert result["diagnosis"]["evidence_flags"]["coordinatewise_core_median_is_axis_sensitive"] is True


def test_circle_alone_does_not_authorize_correction_when_core_has_no_object_support() -> None:
    poses = _orbit()
    xs = np.linspace(3.0, 9.0, 90)
    ys = np.linspace(-4.0, 4.0, 70)
    floor = np.asarray([[x, y, 0.0] for x in xs for y in ys], dtype=np.float64)
    # Add a tiny off-plane patch far from the actual object so attention cannot
    # establish the minimum retained-Core object-support population.
    rng = np.random.default_rng(3)
    patch = np.asarray([7.0, 3.0, 1.0]) + rng.normal(0.0, 0.03, size=(40, 3))
    core = np.vstack([floor, patch])
    result = _run(core, poses)
    assert result["diagnosis"]["classification"] == "trajectory_circle_not_yet_retained_core_supported"
    assert result["diagnosis"]["recommends_i4_1_8b_bounded_target_correction_design"] is False
    assert result["decision"]["authorizes_target_change"] is False


def test_unrelated_failed_stage_blocks_target_bias_classification() -> None:
    poses = _orbit()
    core = _biased_core()
    current = np.median(core, axis=0)
    source = _source(poses, current, stage_pass=False)
    result = _run(core, poses, source=source)
    assert result["diagnosis"]["classification"] == "unrelated_i4_1_8_failure_present"
    assert result["diagnosis"]["recommends_i4_1_8b_bounded_target_correction_design"] is False


def test_refuses_promoted_metric_authority() -> None:
    poses = _orbit()
    core = _biased_core()
    current = np.median(core, axis=0)
    source = _source(poses, current)
    source["authority"]["metric_scale_claimed"] = True
    with pytest.raises(ReconstructionValidationError):
        _run(core, poses, source=source)
