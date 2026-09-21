from __future__ import annotations

from typing import Any


def make_binding(name: str, fill: str) -> dict[str, str]:
    return {"path": f"/evidence/{name}", "sha256": fill * 64}


def make_receipt() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": "I5.14A_FINAL_QUALIFICATION_READINESS_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i5_14a_tests": True,
            "i5_closure_regression": True,
            "evidence_only_readiness_review": True,
            "import_probe": True,
            "runner_help_probe": True,
            "provider_execution": False,
            "sfm_execution": False,
            "mvs_execution": False,
            "scene_classifier_execution": False,
            "production_auto_integration": False,
            "production_default_modified": False,
            "provider_promoted": False,
        },
        "counts": {
            "focused_tests": 1,
            "predecessor_unit_tests": 1,
            "predecessor_integration_tests": 1,
            "predecessor_stress_tests": 1,
            "predecessor_total_tests": 3,
            "combined_exercised_tests": 4,
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


def make_closures() -> dict[str, dict[str, Any]]:
    return {
        "i5_1c": {"phase": "I5.1C", "status": "difficult_handheld_robustness_qualified", "pass": True, "qualification": {"quality_status": "degraded"}},
        "i5_2c": {"phase": "I5.2C", "status": "low_overlap_weak_parallax_robustness_qualified", "pass": True, "qualification": {"quality_status": "pass"}},
        "i5_3c": {"phase": "I5.3C", "status": "blur_exposure_clutter_symmetry_robustness_qualified", "pass": True, "qualification": {"quality_status": "degraded"}},
        "i5_4c": {"phase": "I5.4C", "status": "forward_motion_kitti_robustness_qualified_and_closed", "pass": True, "quality_status": "rejected", "qualification": {"quality_status": "rejected"}},
        "i5_5c": {"phase": "I5.5C", "status": "dynamic_scene_handling_robustness_qualified_and_closed", "pass": True, "quality_status": "pass", "qualification": {"quality_status": "pass", "natural_dynamic_scene_ground_truth": False}},
        "i5_6c": {"phase": "I5.6C", "status": "mvs_robustness_qualified_and_closed", "pass": True, "quality_status": "pass", "qualification": {"quality_status": "pass", "runtime_performance_qualification_claimed": False, "resource_robustness_qualification_claimed": False, "universal_mvs_robustness_claimed": False}},
        "i5_7c": {"phase": "I5.7C", "status": "rolling_depth_robustness_qualified_and_closed", "pass": True, "quality_status": "pass"},
        "i5_8c": {"phase": "I5.8C", "status": "video_depth_anything_robustness_qualified_and_closed", "pass": True, "quality_status": "degraded"},
        "i5_9c": {"phase": "I5.9C", "status": "remaining_depth_provider_robustness_qualified_and_closed", "pass": True, "quality_status": "mixed_terminal_quality"},
        "i5_10d": {"phase": "I5.10D", "status": "mp_sfm_evidence_qualified_and_closed", "pass": True, "quality_status": "bounded_matching_policy_sensitivity_candidate_rejected", "qualification": {"matching_policy_sensitivity_observed": True, "production_quality_recovery_observed": False}},
        "i5_11d": {"phase": "I5.11D", "status": "bounded_failure_recovery_evidence_qualified_and_closed", "pass": True, "quality_status": "bounded_fault_recovery_pass", "qualification": {"universal_failure_recovery_claimed": False}},
        "i5_12a": {"phase": "I5.12A", "status": "hybrid_evidence_justification_review_complete", "pass": True, "quality_status": "hybrid_not_justified", "decision": {"hybrid_experiment_justified": False, "i5_12_complete": True, "i5_13_authorized": True}},
        "i5_13d": {"phase": "I5.13D", "status": "auto_policy_production_integration_decision_complete", "pass": True, "quality_status": "offline_auto_qualified_production_integration_deferred", "qualification": {"offline_policy_qualified": True, "offline_replay_qualified": True, "production_auto_authorized": False, "production_auto_qualified": False, "integration_decision": "defer_production_auto"}, "decision": {"i5_13_complete": True, "i5_14_authorized": True, "production_auto_authorized": False, "production_default_modified": False}},
    }


def make_bindings() -> dict[str, dict[str, str]]:
    return {key: make_binding(f"{key}.json", chr(97 + (idx % 6))) for idx, key in enumerate(make_closures())}
