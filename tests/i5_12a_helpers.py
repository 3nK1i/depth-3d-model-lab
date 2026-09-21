from __future__ import annotations

from copy import deepcopy

from core.reconstruction.i5_hybrid_admission import I512A_CLEANLINESS_DIMENSIONS


def binding(path: str, char: str) -> dict[str, str]:
    return {"path": path, "sha256": char * 64}


def i511d() -> dict:
    return {
        "schema_version": 1,
        "phase": "I5.11D",
        "status": "bounded_failure_recovery_evidence_qualified_and_closed",
        "pass": True,
        "scope": {"hybrid_authorized": False, "auto_policy_implemented": False},
        "decision": {
            "i5_11_complete": True,
            "i5_12_admission_review_authorized": True,
            "i5_12_started": False,
            "hybrid_authorized": False,
            "auto_policy_implemented": False,
        },
        "next": "I5.12A hybrid experiment evidence-justification and admission review (hybrid execution not yet authorized).",
    }


def i56c() -> dict:
    return {
        "phase": "I5.6C",
        "status": "mvs_robustness_qualified_and_closed",
        "pass": True,
        "quality_status": "pass",
        "qualification": {
            "current_production_colmap_mvs_path_qualified_within_admitted_envelope": True,
            "universal_mvs_robustness_claimed": False,
            "provider_ranking_claimed": False,
        },
        "limitations": {
            "runtime_performance_envelope_complete": False,
            "resource_envelope_complete": False,
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "permanent_global_view_cap": None,
        },
        "decision": {
            "i5_6_complete": True,
            "hybrid_authorized": False,
            "production_default_modified": False,
        },
    }


def i52a(g2_binding: dict[str, str] | None = None) -> dict:
    g2_binding = g2_binding or binding("/evidence/g2.json", "4")
    return {
        "phase": "I5.2A",
        "status": "low_overlap_weak_parallax_scenario_admitted",
        "pass": True,
        "evidence_bindings": {"i4_3g2_final_closure": deepcopy(g2_binding)},
        "historical_baseline": {
            "scenario_id": "g2_weak_overlap_mvs_da3",
            "historical_review_scope": "authoritative_registered_views",
            "historical_partial_registration_preserved": True,
        },
        "decision": {"hybrid_authorized": False},
    }


def g2(*, justified: bool = False, completeness_da3: bool = True) -> dict:
    dims = (
        "global_completeness",
        "legitimate_hole_recovery",
        "false_geometry_and_floaters",
        "surface_noise_and_roughness",
        "thin_structure_preservation",
        "edge_definition",
        "small_detail_preservation",
        "false_bridging_or_thickening",
    )
    prefs = {dimension: "colmap_mvs" for dimension in dims}
    if completeness_da3:
        prefs["global_completeness"] = "da3"
    if justified:
        prefs["legitimate_hole_recovery"] = "da3"
    return {
        "phase": "I4.3G2",
        "status": "formally_closed",
        "all_required_g2_scenarios_accounted": True,
        "all_required_manual_reviews_complete": True,
        "i4_3g2_closed": True,
        "synthesis": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "provider_selected": False,
            "provider_promoted": False,
            "universal_provider_winner_declared": False,
            "hybrid_provider_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "scenario_synthesis": {
            "g2_weak_overlap_mvs_da3": {
                "controlled_reveal_complete": True,
                "manual_review_complete": True,
                "global_provider_winner_declared": False,
                "evidence_mode": "fresh_production_plus_matched_view",
                "partial_registration_preserved": True,
                "full_scenario_claim_permitted": False,
                "review_scope": "authoritative_registered_views",
                "provider_preferences": prefs,
            }
        },
    }


def i43a_text() -> str:
    return """A `hybrid` family is reserved in the contract only as a future hypothesis.
No concrete hybrid provider is registered or authorized by I4.3A.
A later hybrid experiment is allowed only if cross-provider evidence shows that MVS is cleaner while AI depth meaningfully recovers legitimate MVS holes.
"""


def receipt() -> dict:
    return {
        "schema_version": 1,
        "phase": "I5.12A_ADMISSION_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i5_12a_tests": True,
            "predecessor_i5_11d_regression": True,
            "mvs_robustness_regression": True,
            "g2_visual_evidence_regression": True,
            "i5_2a_binding_regression": True,
            "import_probe": True,
            "runner_help_probe": True,
            "provider_execution": False,
            "sfm_execution": False,
            "mvs_execution": False,
            "da3_execution": False,
            "hybrid_execution": False,
            "auto_policy_implemented": False,
            "production_default_modified": False,
            "thresholds_modified": False,
        },
        "counts": {
            "focused_tests": 24,
            "predecessor_total_tests": 40,
            "combined_exercised_tests": 64,
        },
        "return_codes": {
            "focused": 0,
            "predecessor_unit": 0,
            "predecessor_integration": 0,
            "predecessor_stress": 0,
            "import_probe": 0,
            "runner_help_probe": 0,
        },
    }


def build_args(*, justified: bool = False) -> dict:
    g2b = binding("/evidence/g2.json", "4")
    return {
        "i5_11d_closure": i511d(),
        "i5_11d_binding": binding("/evidence/i511d.json", "1"),
        "i5_6c_closure": i56c(),
        "i5_6c_binding": binding("/evidence/i56c.json", "2"),
        "i5_2a_admission": i52a(g2b),
        "i5_2a_binding": binding("/evidence/i52a.json", "3"),
        "i4_3g2_closure": g2(justified=justified),
        "i4_3g2_binding": g2b,
        "i4_3a_phase_history_text": i43a_text(),
        "i4_3a_binding": binding("/repo/i43a.md", "5"),
        "admission_receipt": receipt(),
        "admission_receipt_binding": binding("/evidence/i512a_receipt.json", "6"),
    }


def clone(value):
    return deepcopy(value)
