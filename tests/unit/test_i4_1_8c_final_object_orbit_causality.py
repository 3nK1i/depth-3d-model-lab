from __future__ import annotations

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_i4_final_object_orbit_causality import (
    build_i4_1_8c_final_object_orbit_causality,
    apply_i4_1_8c_manual_closure,
)
from core.reconstruction.offline_dense_i4_retained_core_object_support import _geometric_median


def _pose(angle_deg: float, target: np.ndarray, *, radius: float = 4.0) -> np.ndarray:
    a = np.deg2rad(float(angle_deg))
    center = target + np.asarray([radius*np.cos(a), radius*np.sin(a), 0.15*np.sin(2*a)])
    z = target - center; z /= np.linalg.norm(z)
    up = np.asarray([0.0, 0.0, 1.0])
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    p = np.eye(4); p[:3,:3] = np.column_stack([x,y,z]); p[:3,3] = center
    return p


def _summaries(target: np.ndarray) -> tuple[dict, dict, dict]:
    auth = {
        "metric_scale_claimed": False,
        "measurement_authoritative": False,
        "surface_mesh_generated": False,
        "dense_consistency_threshold_relaxation_authorized": False,
    }
    s7 = {
        "phase": "I4.1.7B", "status": "qualified_for_i4_1_8", "authority": dict(auth),
    }
    s8 = {
        "phase": "I4.1.8", "status": "degraded", "authority": dict(auth),
        "stage_gates": {
            "production_fusion": True,
            "formal_filtering": True,
            "canonical_persistence": True,
            "ordinary_i2_bound_scene_quality": True,
        },
        "final_assessment": {
            "failed_gates": ["selected_azimuth_coverage", "selected_azimuth_bins", "selected_opposed_view_diversity"]
        },
    }
    s8b = {
        "phase": "I4.1.8B", "status": "object_support_isolated", "authority": dict(auth) | {
            "production_target_changed": False,
        },
        "diagnosis": {
            "classification": "retained_core_object_support_isolated",
            "object_support_isolation_ready": True,
            "recommends_i4_1_8c_final_target_binding_design": True,
        },
        "selected_support": {
            "target_reconstruction": [float(v) for v in target],
            "target_bootstrap_stability": {"stable": True, "maximum_drift_voxels": 0.2},
            "readiness_gates": {
                "minimum_point_count": True,
                "broad_view_support": True,
                "broad_source_span": True,
                "view_quartile_coverage": True,
                "unique_best_component": True,
                "object_compact_relative_to_core": True,
                "target_bootstrap_stable": True,
            },
            "posthoc_orbit_used_for_selection_or_readiness": False,
        },
    }
    return s7, s8, s8b


def _support() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(41803)
    # Symmetric-ish compact support; saved target is always computed from the
    # same exact points before entering the policy.
    points = rng.normal(size=(1200,3)) * np.asarray([0.25,0.22,0.18]) + np.asarray([0.1,-0.2,1.0])
    return points, _geometric_median(points)


def _build(qualified_angles: np.ndarray, bound_positions: np.ndarray):
    points, target = _support(); s7,s8,s8b = _summaries(target)
    qualified = np.asarray([_pose(a,target) for a in qualified_angles])
    qsrc = np.arange(len(qualified_angles), dtype=np.int64) * 10
    bound = qualified[bound_positions]
    bsrc = qsrc[bound_positions]
    return build_i4_1_8c_final_object_orbit_causality(
        source_i4_1_7b_summary=s7,
        source_i4_1_8_summary=s8,
        source_i4_1_8b_summary=s8b,
        qualified_camera_matrices=qualified,
        qualified_source_sequence_indices=qsrc,
        production_bound_camera_matrices=bound,
        production_bound_source_sequence_indices=bsrc,
        isolated_core_points=points,
        retained_core_voxel_size=0.05,
    )


def test_i4_1_8c_classifies_reconstructed_capture_insufficient_when_118_and_40_fail() -> None:
    qa = np.linspace(-47.0,47.0,118)
    bp = np.rint(np.linspace(0,117,40)).astype(np.int64)
    result = _build(qa,bp)
    assert result["causality"]["classification"] == "qualified_path_object_orbit_insufficient"
    assert result["causality"]["qualified_path_pass"] is False
    assert result["causality"]["production_bound_pass"] is False
    assert result["decision"]["recapture_required"] is True
    assert result["decision"]["production_binding_correction_required"] is False
    assert result["authority"]["production_sampling_integration_authorized"] is False


def test_i4_1_8c_detects_binding_loss_without_searching_camera_subset() -> None:
    points,target = _support(); s7,s8,s8b = _summaries(target)
    qa = np.linspace(-110.0,110.0,118)
    bp = np.rint(np.linspace(0,117,40)).astype(np.int64)
    # The complete qualified path has only six distant standoff outliers (<10%),
    # while the bounded 40-view subset contains all six (>10%).  The same
    # source poses are used in both sets; no synthetic pose mismatch is allowed.
    outliers = set(int(v) for v in bp[10:16])
    qualified = np.asarray([
        _pose(a,target,radius=(14.0 if idx in outliers else 4.0))
        for idx,a in enumerate(qa)
    ])
    qsrc=np.arange(118,dtype=np.int64)*10
    bound=qualified[bp]; bsrc=qsrc[bp]
    result = build_i4_1_8c_final_object_orbit_causality(
        source_i4_1_7b_summary=s7, source_i4_1_8_summary=s8, source_i4_1_8b_summary=s8b,
        qualified_camera_matrices=qualified, qualified_source_sequence_indices=qsrc,
        production_bound_camera_matrices=bound, production_bound_source_sequence_indices=bsrc,
        isolated_core_points=points, retained_core_voxel_size=0.05,
    )
    assert result["causality"]["qualified_path_pass"] is True
    assert result["causality"]["production_bound_pass"] is False
    assert result["causality"]["classification"] == "production_binding_lost_object_angular_coverage"
    assert result["decision"]["production_binding_correction_required"] is True
    assert result["decision"]["recapture_required"] is False
    assert result["causality"]["camera_subset_search_performed"] is False


def test_i4_1_8c_authorizes_final_target_only_when_actual_bound_path_passes() -> None:
    qa = np.linspace(-110.0,110.0,118)
    bp = np.rint(np.linspace(0,117,40)).astype(np.int64)
    result = _build(qa,bp)
    assert result["causality"]["qualified_path_pass"] is True
    assert result["causality"]["production_bound_pass"] is True
    assert result["causality"]["classification"] == "final_object_orbit_qualified"
    assert result["status"] == "ready_for_final_manual_review"
    assert result["target"]["production_target_authority_ready"] is True
    assert result["decision"]["final_i4_1_automated_qualification_pass"] is True
    assert result["decision"]["manual_orbit_review_required"] is True
    assert result["decision"]["i4_1_closed"] is False
    assert result["authority"]["i4_2_detail_preservation_authorized"] is False


def test_i4_1_8c_fails_closed_when_saved_target_does_not_match_isolated_support() -> None:
    points,target = _support(); s7,s8,s8b = _summaries(target + np.asarray([0.2,0.0,0.0]))
    qa = np.linspace(-100,100,20); poses=np.asarray([_pose(a,target) for a in qa]); src=np.arange(20)
    with pytest.raises(ReconstructionValidationError, match="does not reproduce exactly"):
        build_i4_1_8c_final_object_orbit_causality(
            source_i4_1_7b_summary=s7, source_i4_1_8_summary=s8, source_i4_1_8b_summary=s8b,
            qualified_camera_matrices=poses, qualified_source_sequence_indices=src,
            production_bound_camera_matrices=poses, production_bound_source_sequence_indices=src,
            isolated_core_points=points, retained_core_voxel_size=0.05,
        )


def test_i4_1_8c_rejects_orbit_feedback_selected_object_support() -> None:
    points,target = _support(); s7,s8,s8b = _summaries(target)
    s8b["selected_support"]["posthoc_orbit_used_for_selection_or_readiness"] = True
    qa=np.linspace(-100,100,20); poses=np.asarray([_pose(a,target) for a in qa]); src=np.arange(20)
    with pytest.raises(ReconstructionValidationError, match="orbit-gate feedback"):
        build_i4_1_8c_final_object_orbit_causality(
            source_i4_1_7b_summary=s7, source_i4_1_8_summary=s8, source_i4_1_8b_summary=s8b,
            qualified_camera_matrices=poses, qualified_source_sequence_indices=src,
            production_bound_camera_matrices=poses, production_bound_source_sequence_indices=src,
            isolated_core_points=points, retained_core_voxel_size=0.05,
        )


def test_i4_1_8c_rejects_bound_sources_outside_qualified_path() -> None:
    points,target = _support(); s7,s8,s8b = _summaries(target)
    qa=np.linspace(-100,100,20); q=np.asarray([_pose(a,target) for a in qa]); qsrc=np.arange(20)*10
    b=q[:10]; bsrc=qsrc[:10].copy(); bsrc[-1]=999
    with pytest.raises(ReconstructionValidationError, match="subset"):
        build_i4_1_8c_final_object_orbit_causality(
            source_i4_1_7b_summary=s7, source_i4_1_8_summary=s8, source_i4_1_8b_summary=s8b,
            qualified_camera_matrices=q, qualified_source_sequence_indices=qsrc,
            production_bound_camera_matrices=b, production_bound_source_sequence_indices=bsrc,
            isolated_core_points=points, retained_core_voxel_size=0.05,
        )


def test_i4_1_8c_explicit_manual_review_can_close_i4_1_only_after_automated_pass() -> None:
    qa=np.linspace(-110.0,110.0,118); bp=np.rint(np.linspace(0,117,40)).astype(np.int64)
    result=_build(qa,bp)
    closed=apply_i4_1_8c_manual_closure(result, manual_review_confirmed=True)
    assert closed["status"] == "i4_1_closed"
    assert closed["decision"]["i4_1_closed"] is True
    assert closed["decision"]["i4_2_detail_preservation_authorized"] is True
    assert closed["authority"]["i4_2_detail_preservation_authorized"] is True


def test_i4_1_8c_manual_review_cannot_override_automated_failure() -> None:
    qa=np.linspace(-47.0,47.0,118); bp=np.rint(np.linspace(0,117,40)).astype(np.int64)
    result=_build(qa,bp)
    with pytest.raises(ReconstructionValidationError, match="automated qualification did not pass"):
        apply_i4_1_8c_manual_closure(result, manual_review_confirmed=True)
