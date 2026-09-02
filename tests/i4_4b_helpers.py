from __future__ import annotations

from copy import deepcopy

from core.reconstruction.cross_provider_qualification import qualification_scenario_matrix
from core.reconstruction.offline_dense_cross_provider_visual_verification import (
    build_i4_4b_cross_provider_visual_verification,
    VisualVerificationEvidenceBinding,
)
from core.reconstruction.offline_dense_foundation_regression import (
    RegressionEvidenceBinding,
    build_i4_4a_focused_regression,
)
from core.reconstruction.single_image_provider_matched_view_review import (
    I43G4C_PROVIDER_IMPLEMENTATIONS,
    I43G4C_REVIEW_DIMENSIONS,
)
from core.reconstruction.temporal_provider_matched_view_review import (
    I43G3C_PROVIDER_IMPLEMENTATIONS,
    I43G3C_REVIEW_DIMENSIONS,
)
from tests.i4_4a_helpers import passing_h2, passing_receipt as passing_i44a_receipt


def _scenario_ids(substep: str) -> tuple[str, ...]:
    return tuple(item.scenario_id for item in qualification_scenario_matrix() if item.substep == substep)


def _dimensions(scenario_id: str) -> tuple[str, ...]:
    matches = [item for item in qualification_scenario_matrix() if item.scenario_id == scenario_id]
    return tuple(matches[0].manual_review_dimensions)


def passing_i4_4a() -> dict:
    return build_i4_4a_focused_regression(
        i4_3h2_closure=passing_h2(),
        i4_3h2_binding=RegressionEvidenceBinding("/evidence/h2.json", "1" * 64),
        regression_receipt=passing_i44a_receipt(),
        regression_binding=RegressionEvidenceBinding("/evidence/i44a_receipt.json", "2" * 64),
    )


def passing_g2() -> dict:
    dims = _dimensions("g2_complete_room_mvs_da3")
    payload = {
        "schema_version": 1,
        "phase": "I4.3G2",
        "status": "formally_closed",
        "scenario_count": 5,
        "all_required_g2_scenarios_accounted": True,
        "all_required_manual_reviews_complete": True,
        "fresh_scenario_count": 4,
        "historical_scenario_count": 1,
        "g2c_manual_qualification_complete": True,
        "i4_3g2_closed": True,
        "i4_3g3_authorized": True,
        "synthesis": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "auto_provider_selection_implemented": False,
            "hybrid_provider_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "provider_promoted": False,
            "provider_selected": False,
            "surface_mesh_generated": False,
            "universal_provider_winner_declared": False,
        },
        "scenario_synthesis": {
            "g2_complete_room_mvs_da3": {
                "controlled_reveal_complete": True,
                "evidence_mode": "fresh_production_plus_matched_view",
                "manual_review_complete": True,
                "global_provider_winner_declared": False,
                "provider_preferences": {dimension: "colmap_mvs" for dimension in dims},
                "review_scope": "localized_authoritative_registered_views",
            },
            "g2_house_mvs_da3_historical": {
                "evidence_mode": "historical_read_only",
                "manual_review_complete": True,
                "global_provider_winner_declared": False,
                "provider_reexecution_performed": False,
                "historical_verdict": "mvs_better_but_imperfect",
            },
            "g2_low_texture_mvs_da3": {
                "controlled_reveal_complete": True,
                "evidence_mode": "fresh_production_plus_matched_view",
                "manual_review_complete": True,
                "global_provider_winner_declared": False,
                "manual_comparison_assessable": False,
                "provider_preferences": {dimension: None for dimension in dims},
                "review_scope": "localized_authoritative_registered_views",
            },
            "g2_temple_ring_mvs_da3": {
                "controlled_reveal_complete": True,
                "evidence_mode": "fresh_production_plus_matched_view",
                "manual_review_complete": True,
                "global_provider_winner_declared": False,
                "provider_preferences": {dimension: "colmap_mvs" for dimension in dims},
            },
            "g2_weak_overlap_mvs_da3": {
                "controlled_reveal_complete": True,
                "evidence_mode": "fresh_production_plus_matched_view",
                "manual_review_complete": True,
                "global_provider_winner_declared": False,
                "provider_preferences": {dimension: ("da3" if dimension == "global_completeness" else "colmap_mvs") for dimension in dims},
                "review_scope": "authoritative_registered_views",
            },
        },
    }
    return payload


def passing_g3() -> dict:
    dims = tuple(I43G3C_REVIEW_DIMENSIONS)
    statuses = {
        "g3_temporal_static_sweep": {"rolling_depth": "degraded", "video_depth_anything": "pass"},
        "g3_temporal_motion": {"rolling_depth": "rejected", "video_depth_anything": "degraded"},
        "g3_temporal_low_texture": {"rolling_depth": "pass", "video_depth_anything": "pass"},
    }
    return {
        "schema_version": 1,
        "phase": "I4.3G3C",
        "status": "formally_closed",
        "pass": True,
        "scenario_count": 3,
        "all_required_temporal_scenarios_accounted": True,
        "all_required_manual_reviews_complete": True,
        "i4_3g3_closed": True,
        "i4_3g4_authorized": True,
        "scenario_synthesis": {
            scenario_id: {
                "production_statuses": state,
                "provider_preferences": {dimension: "video_depth_anything" for dimension in dims},
                "manual_review_complete": True,
                "matched_view_diagnostic_secondary": True,
                "production_rejection_overridden": False,
                "global_provider_winner_declared": False,
            }
            for scenario_id, state in statuses.items()
        },
        "decision": {
            "production_native_evidence_primary": True,
            "matched_view_diagnostic_secondary": True,
            "scenario_specific_verdicts_preserved": True,
            "quality_failures_preserved": True,
            "provider_selected": False,
            "provider_promoted": False,
            "universal_provider_winner_declared": False,
            "global_scalar_score_created": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "authoritative_sfm_modified": False,
            "provider_selected": False,
            "provider_promoted": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "permanent_global_view_cap": None,
        },
    }


def passing_g4() -> dict:
    dims = tuple(I43G4C_REVIEW_DIMENSIONS)
    statuses = {
        "g4_single_image_complete_room": {provider: "pass" for provider in I43G4C_PROVIDER_IMPLEMENTATIONS},
        "g4_single_image_low_texture": {
            **{provider: "pass" for provider in I43G4C_PROVIDER_IMPLEMENTATIONS},
            "depth_anything_v2": "rejected",
            "marigold": "rejected",
        },
        "g4_single_image_weak_overlap": {provider: "pass" for provider in I43G4C_PROVIDER_IMPLEMENTATIONS},
    }
    not_assessable = {dimension: [] for dimension in dims}
    not_assessable["cross_view_stability_after_sfm_alignment"] = ["marigold", "depth_anything_v2"]
    not_assessable["false_geometry_and_floaters"] = ["marigold", "depth_anything_v2"]
    not_assessable["global_completeness"] = ["marigold", "depth_anything_v2"]
    not_assessable["local_shape_plausibility"] = ["marigold", "depth_anything_v2"]
    return {
        "schema_version": 1,
        "phase": "I4.3G4C",
        "status": "formally_closed",
        "pass": True,
        "scenario_count": 3,
        "all_required_single_image_scenarios_accounted": True,
        "all_required_manual_reviews_complete": True,
        "all_six_provider_outcomes_accounted_per_scenario": True,
        "i4_3g4_closed": True,
        "i4_3g5_authorized": True,
        "scenario_synthesis": {
            scenario_id: {
                "production_statuses": state,
                "provider_preferences": {dimension: ["moge_2"] for dimension in dims},
                "provider_not_assessable": deepcopy(not_assessable) if scenario_id == "g4_single_image_low_texture" else {dimension: [] for dimension in dims},
                "manual_review_complete": True,
                "matched_view_diagnostic_secondary": True,
                "production_rejection_overridden": False,
                "provider_ranking_created": False,
                "global_provider_winner_declared": False,
            }
            for scenario_id, state in statuses.items()
        },
        "decision": {
            "production_native_evidence_primary": True,
            "matched_view_diagnostic_secondary": True,
            "scenario_specific_verdicts_preserved": True,
            "quality_rejections_preserved": True,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "universal_provider_winner_declared": False,
            "global_scalar_score_created": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "authoritative_sfm_modified": False,
            "provider_selected": False,
            "provider_promoted": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "permanent_global_view_cap": None,
        },
    }


def passing_receipt() -> dict:
    return {
        "schema_version": 1,
        "phase": "I4.4B_VISUAL_VERIFICATION_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i4_4b_tests": True,
            "affected_visual_unit_regression": True,
            "affected_visual_integration_regression": True,
            "affected_visual_stress_regression": True,
            "import_probe": True,
            "runner_help_probe": True,
            "provider_execution_started": False,
            "sfm_rerun": False,
            "visual_review_rerun": False,
            "manual_review_rerun": False,
        },
        "counts": {
            "focused_tests": 27,
            "affected_test_files": 27,
            "affected_unit_tests": 132,
            "affected_integration_tests": 38,
            "affected_stress_tests": 54,
            "affected_total_tests": 224,
        },
        "return_codes": {
            "focused": 0,
            "affected_unit": 0,
            "affected_integration": 0,
            "affected_stress": 0,
            "import_probe": 0,
            "runner_help_probe": 0,
        },
    }


def build_passing_i4_4b() -> dict:
    return build_i4_4b_cross_provider_visual_verification(
        i4_3h2_closure=passing_h2(),
        i4_3h2_binding=VisualVerificationEvidenceBinding("/evidence/h2.json", "1" * 64),
        i4_4a_result=passing_i4_4a(),
        i4_4a_binding=VisualVerificationEvidenceBinding("/evidence/i44a.json", "2" * 64),
        i4_3g2_closure=passing_g2(),
        i4_3g2_binding=VisualVerificationEvidenceBinding("/evidence/g2.json", "3" * 64),
        i4_3g3c_closure=passing_g3(),
        i4_3g3c_binding=VisualVerificationEvidenceBinding("/evidence/g3c.json", "4" * 64),
        i4_3g4c_closure=passing_g4(),
        i4_3g4c_binding=VisualVerificationEvidenceBinding("/evidence/g4c.json", "5" * 64),
        visual_receipt=passing_receipt(),
        visual_receipt_binding=VisualVerificationEvidenceBinding("/evidence/i44b_receipt.json", "6" * 64),
    )


def clone(value: dict) -> dict:
    return deepcopy(value)
