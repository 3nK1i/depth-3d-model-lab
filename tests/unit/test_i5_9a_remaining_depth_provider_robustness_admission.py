from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_remaining_depth_provider_robustness import (
    I59A_B1_PROVIDERS,
    I59A_B2_PROVIDERS,
    I59A_PROVIDER_IMPLEMENTATIONS,
    I59A_SCENARIO_IDS,
    I59AAdmissionPolicy,
    build_i5_9a_remaining_depth_provider_robustness_admission,
    provider_robustness_specs,
    validate_i4_3g4c_for_i5_9,
    validate_i5_8c_for_i5_9,
)
from tests.i5_9a_helpers import build_args, clone, g4c_closure, i58c_closure


def test_provider_inventory_is_exactly_the_six_g4_single_image_providers():
    assert I59A_PROVIDER_IMPLEMENTATIONS == (
        "metric3d_v2", "unidepth_v2", "moge_2", "depth_pro", "depth_anything_v2", "marigold"
    )


def test_b1_b2_groups_partition_inventory_without_overlap():
    assert set(I59A_B1_PROVIDERS).isdisjoint(I59A_B2_PROVIDERS)
    assert set(I59A_B1_PROVIDERS) | set(I59A_B2_PROVIDERS) == set(I59A_PROVIDER_IMPLEMENTATIONS)
    assert I59A_B1_PROVIDERS == ("metric3d_v2", "unidepth_v2", "moge_2", "depth_pro")
    assert I59A_B2_PROVIDERS == ("depth_anything_v2", "marigold")


def test_provider_specs_preserve_production_configs_and_alignment_semantics():
    specs = provider_robustness_specs()
    assert tuple(specs) == I59A_PROVIDER_IMPLEMENTATIONS
    for spec in specs.values():
        assert spec["production_config"]["max_voxels"] == 1_500_000
        assert spec["production_config"]["minimum_accepted_views"] == 2
        assert spec["production_config"]["minimum_correspondences"] == 8
        assert spec["production_config"]["maximum_relative_residual"] == 0.35
        assert spec["permanent_global_view_cap"] is None
        assert spec["fusion_capacity_drops_change_status"] is False
    assert specs["depth_anything_v2"]["alignment_semantics"] == "per_view_affine_inverse_depth_to_authoritative_sfm"
    assert specs["marigold"]["alignment_semantics"] == "per_view_affine_native_domain_to_authoritative_sfm"
    for provider in I59A_B1_PROVIDERS:
        assert specs[provider]["alignment_semantics"] == "per_view_sparse_correspondence_scale_to_authoritative_sfm"


def test_historical_g4_anchor_is_preserved_exactly():
    summary = validate_i4_3g4c_for_i5_9(g4c_closure())
    assert summary["complete_room"] == {p: "pass" for p in I59A_PROVIDER_IMPLEMENTATIONS}
    assert summary["low_texture"]["depth_anything_v2"] == "rejected"
    assert summary["low_texture"]["marigold"] == "rejected"
    for p in I59A_B1_PROVIDERS:
        assert summary["low_texture"][p] == "pass"
    assert summary["weak_overlap"] == {p: "pass" for p in I59A_PROVIDER_IMPLEMENTATIONS}


def test_g4_low_texture_rejections_cannot_be_silently_upgraded():
    payload = g4c_closure()
    payload["scenario_synthesis"]["g4_single_image_low_texture"]["production_statuses"]["marigold"] = "pass"
    with pytest.raises(ReconstructionValidationError):
        validate_i4_3g4c_for_i5_9(payload)


def test_i58c_authorizes_i59_without_claiming_clean_vda_quality():
    result = validate_i5_8c_for_i5_9(i58c_closure())
    assert result["i5_9_authorized"] is True
    assert result["i5_9_started"] is False


def test_i58c_cannot_claim_clean_quality_pass():
    payload = i58c_closure()
    payload["quality_status"] = "pass"
    with pytest.raises(ReconstructionValidationError):
        validate_i5_8c_for_i5_9(payload)


def test_policy_is_execution_free_and_keeps_b1_before_b2():
    policy = I59AAdmissionPolicy()
    data = policy.as_dict()
    assert data["scenario_ids"] == list(I59A_SCENARIO_IDS)
    assert data["batches"]["I5.9B1"]["providers"] == list(I59A_B1_PROVIDERS)
    assert data["batches"]["I5.9B2"]["providers"] == list(I59A_B2_PROVIDERS)
    assert data["batches"]["execution_order"] == ["I5.9B1", "I5.9B2"]
    assert data["provider_comparison"] is False
    assert data["result_driven_retuning"] is False


def test_build_admission_preserves_six_current_castle_foundations_and_exclusions():
    result = build_i5_9a_remaining_depth_provider_robustness_admission(**build_args())
    assert result["pass"] is True
    assert tuple(result["scenario_matrix"]) == I59A_SCENARIO_IDS
    assert result["decision"]["i5_9a_complete"] is True
    assert result["decision"]["i5_9b1_authorized"] is True
    assert result["decision"]["i5_9b1_started"] is False
    assert result["decision"]["i5_9b2_authorized"] is True
    assert result["decision"]["i5_9c_started"] is False
    assert result["exclusions"]["castle_blur"]["classification"] == "blocked_upstream"
    assert result["exclusions"]["castle_underexposure"]["classification"] == "blocked_upstream"
    assert result["exclusions"]["kitti_forward_motion"]["classification"] == "blocked_upstream"
    assert result["exclusions"]["castle_clutter"]["classification"] == "not_formally_admitted"


def test_scenario_interpretations_are_single_image_specific_not_temporal_or_mvs_equivalence():
    result = build_i5_9a_remaining_depth_provider_robustness_admission(**build_args())
    assert "stateless" in result["scenario_matrix"]["castle_control"]["scientific_interpretation"]
    assert "not_mvs_overlap_equivalence" in result["scenario_matrix"]["castle_low_overlap"]["scientific_interpretation"]
    assert "independent_frame" in result["scenario_matrix"]["castle_dynamic_proxy"]["scientific_interpretation"]


def test_admission_does_not_start_later_roadmap_phases_or_change_authority():
    result = build_i5_9a_remaining_depth_provider_robustness_admission(**build_args())
    scope = result["scope"]
    assert scope["single_image_provider_inference"] is False
    assert scope["sfm_rerun"] is False
    assert scope["mvs_rerun"] is False
    assert scope["mp_sfm_started"] is False
    assert scope["auto_policy_implemented"] is False
    assert scope["hybrid_authorized"] is False
    authority = result["authority"]
    assert authority["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert authority["metric_scale_claimed"] is False
    assert authority["permanent_global_view_cap"] is None


def test_mutating_production_max_voxels_is_rejected_by_policy():
    with pytest.raises(ReconstructionValidationError):
        I59AAdmissionPolicy(production_max_voxels=2_000_000)
