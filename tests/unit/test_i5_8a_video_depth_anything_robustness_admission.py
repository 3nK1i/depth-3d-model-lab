from __future__ import annotations

import pytest

from core.models.runtime_config import _VIDEO_DEPTH_ANYTHING_PROFILE_VARIANTS
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_video_depth_anything_robustness import (
    I58A_FROZEN_G3_VDA_STATUSES,
    I58A_RUNTIME_VARIANTS,
    I58A_SCENARIO_IDS,
    I58AAdmissionPolicy,
    build_i5_8a_video_depth_anything_robustness_admission,
    validate_i4_3g3c_for_i5_8,
    validate_i5_7c_for_i5_8,
    validate_i5_8a_receipt,
)
from tests.i5_8a_helpers import build_args, g3c_closure, i57c_closure, receipt


def test_policy_freezes_existing_vda_streaming_contract() -> None:
    policy = I58AAdmissionPolicy().as_dict()
    assert policy["provider_id"] == "video-depth-anything-temporal"
    assert policy["provider_implementation"] == "video_depth_anything"
    assert policy["temporal_mode"] == "official_streaming_cache"
    assert policy["stateful_temporal_sequence"] is True
    assert policy["source_order_required"] is True
    assert policy["stable_pixel_grid_required"] is True
    assert policy["worker_model_loads_once_per_sequence"] is True
    assert policy["warmup_reset_before_user_sequence"] is True
    assert policy["monotonic_worker_frame_ids"] is True
    assert policy["canonical_alignment"] == "per_registered_frame_affine_to_authoritative_sfm"
    assert policy["per_frame_alignment_performed"] is True
    assert policy["sequence_global_alignment_performed"] is False
    assert policy["permanent_global_view_cap"] is None


def test_policy_freezes_existing_vda_production_thresholds() -> None:
    config = I58AAdmissionPolicy().as_dict()["production_config"]
    assert config == {
        "minimum_correspondences": 8,
        "maximum_relative_residual": 0.35,
        "voxel_scene_fraction": 0.004,
        "pixel_stride": 2,
        "max_voxels": 1_500_000,
        "minimum_accepted_views": 2,
    }


def test_runtime_variant_matrix_matches_existing_production_profiles() -> None:
    assert I58A_RUNTIME_VARIANTS == _VIDEO_DEPTH_ANYTHING_PROFILE_VARIANTS
    assert I58AAdmissionPolicy().as_dict()["runtime_profile_resolution"] == "hardware_resolved_compute_profile_auto"
    assert I58AAdmissionPolicy().as_dict()["checkpoint_must_match_resolved_profile"] is True


@pytest.mark.parametrize("kwargs", [
    {"temporal_mode": "whole_sequence"},
    {"per_frame_alignment_performed": False},
    {"sequence_global_alignment_performed": True},
    {"runtime_profile_resolution": "fixed_vits"},
    {"checkpoint_must_match_resolved_profile": False},
    {"permanent_global_view_cap": 12},
    {"result_driven_retuning": True},
    {"provider_comparison": True},
])
def test_policy_drift_fails_closed(kwargs: dict) -> None:
    with pytest.raises(ReconstructionValidationError):
        I58AAdmissionPolicy(**kwargs)


def test_i57c_must_authorize_i58() -> None:
    payload = i57c_closure()
    payload["decision"]["i5_8_authorized"] = False
    with pytest.raises(ReconstructionValidationError):
        validate_i5_7c_for_i5_8(payload)


def test_i57c_authority_drift_fails_closed() -> None:
    payload = i57c_closure()
    payload["authority"]["metric_scale_claimed"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_7c_for_i5_8(payload)


def test_g3c_frozen_vda_statuses_are_immutable() -> None:
    payload = validate_i4_3g3c_for_i5_8(g3c_closure())
    observed = {
        sid: payload["scenario_synthesis"][sid]["production_statuses"]["video_depth_anything"]
        for sid in I58A_FROZEN_G3_VDA_STATUSES
    }
    assert observed == I58A_FROZEN_G3_VDA_STATUSES


def test_g3c_motion_degraded_cannot_be_rewritten_to_pass() -> None:
    payload = g3c_closure()
    payload["scenario_synthesis"]["g3_temporal_motion"]["production_statuses"]["video_depth_anything"] = "pass"
    with pytest.raises(ReconstructionValidationError):
        validate_i4_3g3c_for_i5_8(payload)


def test_receipt_rejects_hidden_vda_inference() -> None:
    payload = receipt()
    payload["scope"]["video_depth_anything_inference"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_8a_receipt(payload)


def test_build_admits_exact_six_frozen_castle_sfm_scopes() -> None:
    result = build_i5_8a_video_depth_anything_robustness_admission(**build_args())
    assert set(result["scenario_matrix"]) == set(I58A_SCENARIO_IDS)
    assert all(node["admitted_for_i5_8b"] for node in result["scenario_matrix"].values())
    assert all(node["video_depth_anything_execution_required"] for node in result["scenario_matrix"].values())


def test_low_overlap_has_vda_specific_registered_keyframe_spacing_semantics() -> None:
    result = build_i5_8a_video_depth_anything_robustness_admission(**build_args())
    row = result["scenario_matrix"]["castle_low_overlap"]
    assert row["scientific_interpretation"] == "registered_keyframe_spacing_and_cache_continuity_stress_not_mvs_overlap_equivalence"
    assert row["original_full_video_continuity_claimed"] is False
    assert row["mvs_specific_metrics_are_video_depth_anything_criteria"] is False


def test_symmetry_preserves_only_authoritative_registered_subset() -> None:
    result = build_i5_8a_video_depth_anything_robustness_admission(**build_args())
    row = result["scenario_matrix"]["castle_symmetry"]
    assert row["input_images"] == 19
    assert row["registered_images"] == 12
    assert row["source_registered_scope_only"] is True


def test_provider_contract_preserves_native_metric_and_intrinsics_only_as_evidence() -> None:
    result = build_i5_8a_video_depth_anything_robustness_admission(**build_args())
    contract = result["provider_contract"]
    assert contract["native_metric_depth_preserved_as_evidence"] is True
    assert contract["native_estimated_intrinsics_preserved_as_evidence"] is True
    assert contract["native_metric_depth_promoted_to_canonical_metric_scale"] is False
    assert contract["native_estimated_intrinsics_consumed_for_canonical_projection"] is False
    assert contract["canonical_projection_pose_source"] == "pycolmap_sfm"


def test_upstream_exclusions_are_not_vda_failures() -> None:
    result = build_i5_8a_video_depth_anything_robustness_admission(**build_args())
    exclusions = result["excluded_or_deferred_scenarios"]
    assert exclusions["castle_blur"]["status"] == "BLOCKED_UPSTREAM"
    assert exclusions["castle_underexposure"]["status"] == "BLOCKED_UPSTREAM"
    assert exclusions["castle_clutter"]["status"] == "NOT_FORMALLY_ADMITTED"
    assert exclusions["kitti_forward_motion"]["status"] == "BLOCKED_UPSTREAM"
    assert exclusions["difficult_handheld_horse"]["status"] == "DEFERRED_NOT_ADMITTED"
    assert all(node["video_depth_anything_failure"] is False for node in exclusions.values())


def test_admission_is_execution_free_and_authorizes_only_i58b() -> None:
    result = build_i5_8a_video_depth_anything_robustness_admission(**build_args())
    scope = result["scope"]
    assert scope["video_depth_anything_inference"] is False
    assert scope["sfm_rerun"] is False
    assert scope["mvs_rerun"] is False
    assert scope["provider_comparison_started"] is False
    decision = result["decision"]
    assert decision["i5_7_complete"] is True
    assert decision["i5_8a_complete"] is True
    assert decision["i5_8b_authorized"] is True
    assert decision["i5_8b_started"] is False
    assert decision["i5_9_started"] is False


def test_authority_and_roadmap_guards_remain_locked() -> None:
    result = build_i5_8a_video_depth_anything_robustness_admission(**build_args())
    authority = result["authority"]
    assert authority["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert authority["video_depth_anything_pose_authority"] is False
    assert authority["native_estimated_intrinsics_are_canonical_authority"] is False
    assert authority["native_metric_depth_establishes_physical_metric_scale"] is False
    assert authority["metric_scale_claimed"] is False
    assert authority["measurement_authoritative"] is False
    assert authority["certified_metrology"] is False
    assert authority["permanent_global_view_cap"] is None
    assert result["roadmap_guard"]["i5_10_mp_sfm_deferred"] is True
    assert result["roadmap_guard"]["i5_13_auto_deferred"] is True
