from __future__ import annotations

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_i4_clean_static_orbit_qualification import (
    I417QualificationPolicy,
    assemble_selected_pose_sequence,
    build_clean_static_orbit_assessment,
    selected_plan_payload,
    validate_prior_decoupled_evidence,
)
from scripts.run_i4_1_6b_decoupled_dense_ab import (
    _baseline_selected_path,
    _localization_rejection_summary,
    _partition_extra_localization_candidates,
)


def _look_at_pose(angle_deg: float, *, radius: float = 4.0, face_target: bool = True) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    center = np.asarray([radius * np.cos(theta), radius * np.sin(theta), 0.2], dtype=np.float64)
    z = -center
    z /= np.linalg.norm(z)
    if not face_target:
        z = -z
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x); y /= np.linalg.norm(y)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.column_stack([x, y, z])
    matrix[:3, 3] = center
    return matrix


def _arc(count: int = 12, degrees: float = 220.0) -> np.ndarray:
    return np.asarray([_look_at_pose(v) for v in np.linspace(-degrees / 2.0, degrees / 2.0, count)])


def _prior() -> dict:
    return {
        "phase": "I4.1.6B",
        "decision": {
            "single_authoritative_sfm_frame": True,
            "production_default_changed": False,
            "production_sampling_integration_authorized": False,
            "dense_consistency_threshold_relaxation_authorized": False,
            "metric_scale_claimed": False,
            "mesh_authorization": False,
            "real_dense_ab_validation_pass": False,
            "recommended_plan": "uniform_localized_dense_pool",
        },
    }


def _plan(selected: int = 12, *, aligned: int = 92, pair_fraction: float = 0.40) -> dict:
    return {
        "plan_name": "uniform_localized_dense_pool",
        "planned_candidates": 102,
        "alignment_accepted": aligned,
        "alignment_fraction": aligned / 102.0,
        "pair_pass_fraction": pair_fraction,
        "selected_qualified_views": selected,
        "full_span": True,
        "dense_gate_pass": aligned / 102.0 >= 0.80 and selected >= 10,
        "selected_source_sequence_indices": list(range(selected)),
    }


def _assessment(*, matrices: np.ndarray, capture: bool = True, plan: dict | None = None):
    plan = plan or _plan(selected=len(matrices))
    return build_clean_static_orbit_assessment(
        input_global_views=40,
        registered_global_views=40,
        registered_camera_matrices=_arc(40, 230.0),
        selected_camera_matrices=matrices,
        selected_source_indices=list(range(len(matrices))),
        selected_pose_provenance=["global_sfm_anchor"] * len(matrices),
        provisional_target_reconstruction=np.zeros(3),
        target_diagnostics={"inlier_fraction": 1.0},
        plan_payload=plan,
        baseline_payload={"selected_qualified_views": 2, "pair_pass_fraction": 0.02},
        prior_guardrails=validate_prior_decoupled_evidence(_prior()),
        capture_confirmed_static=capture,
    )


def test_i4_1_7_policy_keeps_locked_orbit_thresholds():
    policy = I417QualificationPolicy()
    assert policy.minimum_registered_fraction == pytest.approx(0.80)
    assert policy.minimum_dense_alignment_fraction == pytest.approx(0.80)
    assert policy.minimum_selected_views == 10
    assert policy.minimum_selected_azimuth_coverage_degrees == pytest.approx(150.0)
    assert policy.minimum_opposed_view_angle_degrees == pytest.approx(120.0)


def test_i4_1_7_policy_rejects_invalid_fraction():
    with pytest.raises(ReconstructionValidationError):
        I417QualificationPolicy(minimum_dense_alignment_fraction=1.1)


def test_i4_1_7_accepts_prior_horse_guardrails_without_requiring_horse_pass():
    result = validate_prior_decoupled_evidence(_prior())
    assert result["guardrails_pass"] is True
    assert result["prior_real_dense_ab_validation_pass"] is False


def test_i4_1_7_rejects_prior_production_default_change():
    payload = _prior()
    payload["decision"]["production_default_changed"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_prior_decoupled_evidence(payload)


def test_i4_1_7_selects_named_plan_payload():
    summary = {"plans": [_plan(), {**_plan(), "plan_name": "global_anchors_plus_local_neighbors"}]}
    result = selected_plan_payload(summary, "global_anchors_plus_local_neighbors")
    assert result["plan_name"] == "global_anchors_plus_local_neighbors"


def test_i4_1_7_assembles_global_and_localized_poses_in_selected_order():
    a = _look_at_pose(-100.0); b = _look_at_pose(0.0); c = _look_at_pose(100.0)
    matrices, provenance = assemble_selected_pose_sequence(
        [3, 8, 12],
        global_pose_by_source={3: a, 12: c},
        localized_pose_by_source={8: b},
    )
    assert np.allclose(matrices[0], a)
    assert np.allclose(matrices[1], b)
    assert provenance == ("global_sfm_anchor", "localized_extra_observation", "global_sfm_anchor")


def test_i4_1_7_pose_assembly_fails_closed_when_selected_pose_missing():
    with pytest.raises(ReconstructionValidationError):
        assemble_selected_pose_sequence([1, 2], global_pose_by_source={1: np.eye(4)}, localized_pose_by_source={})


def test_i4_1_7_clean_broad_orbit_passes_all_gates():
    result = _assessment(matrices=_arc(12, 220.0))
    assert result["pass"] is True
    assert all(result["gates"].values())
    assert result["decision"]["i4_1_8_production_binding_authorized"] is True
    assert result["authority"]["production_sampling_integration_authorized"] is False
    assert result["target"]["production_target_authority"] is False


def test_i4_1_7_requires_explicit_static_capture_confirmation():
    result = _assessment(matrices=_arc(12, 220.0), capture=False)
    assert result["pass"] is False
    assert result["gates"]["capture_confirmed_static"] is False
    assert result["decision"]["i4_1_8_production_binding_authorized"] is False


def test_i4_1_7_rejects_narrow_selected_orbit_even_when_dense_gate_passes():
    result = _assessment(matrices=_arc(12, 90.0))
    assert result["pass"] is False
    assert result["gates"]["selected_view_count"] is True
    assert result["gates"]["selected_azimuth_coverage"] is False

def test_i4_1_7_baseline_reader_accepts_current_successful_i1_schema():
    selected = {"orders": [1, 2, 3], "view_count": 3}
    result = _baseline_selected_path(
        {"dense_qualification": {"selected_pass_path": selected}}
    )
    assert result == selected


def test_i4_1_7_baseline_reader_prefers_current_schema_over_legacy_fallback():
    current = {"orders": [4, 5, 6], "view_count": 3}
    legacy = {"orders": [1], "view_count": 1}
    result = _baseline_selected_path(
        {
            "dense_qualification": {"selected_pass_path": current},
            "selected_path": legacy,
        }
    )
    assert result == current


def test_i4_1_7_baseline_reader_keeps_legacy_i1_schema_compatible():
    selected = {"orders": [7, 8], "view_count": 2}
    result = _baseline_selected_path({"selected_path": selected})
    assert result == selected


def test_i4_1_7_baseline_reader_fails_closed_without_supported_path_schema():
    with pytest.raises(ReconstructionValidationError, match="selected_pass_path"):
        _baseline_selected_path({"dense_qualification": {}})



def test_i4_1_7_accepts_saturated_baseline_with_more_selected_views_and_no_pair_regression():
    matrices = _arc(20, 220.0)
    plan = _plan(selected=20, pair_fraction=1.0)
    result = build_clean_static_orbit_assessment(
        input_global_views=40,
        registered_global_views=40,
        registered_camera_matrices=_arc(40, 230.0),
        selected_camera_matrices=matrices,
        selected_source_indices=list(range(20)),
        selected_pose_provenance=["global_sfm_anchor"] * 20,
        provisional_target_reconstruction=np.zeros(3),
        target_diagnostics={"inlier_fraction": 1.0},
        plan_payload=plan,
        baseline_payload={"selected_qualified_views": 12, "pair_pass_fraction": 1.0},
        prior_guardrails=validate_prior_decoupled_evidence(_prior()),
        capture_confirmed_static=True,
    )
    assert result["gates"]["real_dense_plan_improves_over_global_only"] is True
    assert result["counts"]["baseline_pair_pass_fraction_saturated"] is True
    assert result["counts"]["dense_baseline_comparison_rule"] == "saturated_baseline_no_pair_regression_plus_selected_gain"
    assert result["pass"] is True


def test_i4_1_7_saturated_baseline_still_rejects_pair_quality_regression():
    matrices = _arc(20, 220.0)
    plan = _plan(selected=20, pair_fraction=0.99)
    result = build_clean_static_orbit_assessment(
        input_global_views=40,
        registered_global_views=40,
        registered_camera_matrices=_arc(40, 230.0),
        selected_camera_matrices=matrices,
        selected_source_indices=list(range(20)),
        selected_pose_provenance=["global_sfm_anchor"] * 20,
        provisional_target_reconstruction=np.zeros(3),
        target_diagnostics={},
        plan_payload=plan,
        baseline_payload={"selected_qualified_views": 12, "pair_pass_fraction": 1.0},
        prior_guardrails=validate_prior_decoupled_evidence(_prior()),
        capture_confirmed_static=True,
    )
    assert result["gates"]["real_dense_plan_improves_over_global_only"] is False
    assert result["counts"]["pair_quality_regressed"] is True
    assert result["pass"] is False


def test_i4_1_6b_partitions_only_accepted_localizations_for_da3(tmp_path):
    evidence = tmp_path / "localization_evidence"
    evidence.mkdir()
    (evidence / "candidate_00000010.json").write_text(
        '{"source_sequence_index":10,"accepted":true,"status":"localized"}',
        encoding="utf-8",
    )
    (evidence / "candidate_00000020.json").write_text(
        '{"source_sequence_index":20,"accepted":false,"status":"rejected_inlier_gate","inlier_count":12}',
        encoding="utf-8",
    )
    (evidence / "candidate_00000030.json").write_text(
        '{"source_sequence_index":30,"accepted":true,"status":"localized"}',
        encoding="utf-8",
    )

    accepted, rejected = _partition_extra_localization_candidates(tmp_path, [10, 20, 30])

    assert accepted == [10, 30]
    assert set(rejected) == {20}
    assert rejected[20]["status"] == "rejected_inlier_gate"


def test_i4_1_6b_localization_rejection_is_preserved_as_non_aligned_candidate():
    summary = _localization_rejection_summary(
        source_index=848,
        localization={
            "source_sequence_index": 848,
            "accepted": False,
            "status": "rejected_inlier_gate",
            "inlier_count": 12,
        },
    )

    assert summary["source_sequence_index"] == 848
    assert summary["alignment_accepted"] is False
    assert summary["localized_observation"] is False
    assert summary["localization_status"] == "rejected_inlier_gate"
    assert summary["da3_skipped_due_to_localization_rejection"] is True
    assert summary["da3_npz"] is None


def test_i4_1_6b_partition_fails_closed_when_localization_evidence_missing(tmp_path):
    (tmp_path / "localization_evidence").mkdir()
    with pytest.raises(ReconstructionValidationError):
        _partition_extra_localization_candidates(tmp_path, [848])
