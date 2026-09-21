from __future__ import annotations

from copy import deepcopy

I510D_SHA = "d43bcccb0f800fc5e6ff4108d25e2757e8ef5efca1fe14b53012027b8763f309"
POST_I510D_DELTA_SHA = "773f63c31d27d5746fe788f9a0463792a6d5061676fea6f44cb004f99e4a6247"


def synthetic_i510d_closure() -> dict:
    return {
        "schema_version": 1,
        "phase": "I5.10D",
        "status": "mp_sfm_evidence_qualified_and_closed",
        "pass": True,
        "quality_status": "bounded_matching_policy_sensitivity_candidate_rejected",
        "scope": {
            "evidence_only_qualification_and_closure": True,
            "sfm_execution": False,
            "mvs_execution": False,
            "provider_inference": False,
            "baseline_rerun": False,
            "candidate_rerun": False,
            "retry_until_pass": False,
            "result_driven_retuning": False,
            "threshold_relaxation": False,
            "adaptive_image_reduction": False,
            "production_default_modified": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "candidate_reconstruction_is_evaluation_only": True,
            "candidate_can_replace_production_default": False,
            "independent_sfm_gauges_preserved": True,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "permanent_global_view_cap": None,
        },
        "decision": {
            "i5_10a_complete": True,
            "i5_10b_complete": True,
            "i5_10c_complete": True,
            "i5_10d_complete": True,
            "i5_10_complete": True,
            "i5_11_authorized": True,
            "i5_11_started": False,
            "i5_12_started": False,
            "i5_13_started": False,
            "i5_14_started": False,
            "production_default_modified": False,
            "threshold_relaxation_authorized": False,
            "universal_matching_strategy_winner_declared": False,
            "hybrid_authorized": False,
            "auto_policy_implemented": False,
        },
        "next": "I5.11 OOM / disk / crash / recovery robustness.",
    }


def synthetic_i511a_receipt() -> dict:
    return {
        "schema_version": 1,
        "phase": "I5.11A_ADMISSION_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i5_11a_tests": True,
            "engine_hardening_regression": True,
            "failure_classification_regression": True,
            "recovery_ux_regression": True,
            "processing_supervisor_regression": True,
            "import_probe": True,
            "runner_help_probe": True,
            "fault_injection": False,
            "real_gpu_oom": False,
            "real_disk_fill": False,
            "sfm_execution": False,
            "mvs_execution": False,
            "provider_inference": False,
            "production_default_modified": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "counts": {
            "focused_tests": 16,
            "predecessor_unit_tests": 40,
            "predecessor_integration_tests": 10,
            "predecessor_stress_tests": 10,
            "predecessor_total_tests": 60,
            "combined_exercised_tests": 76,
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


def binding(path: str, sha: str) -> dict[str, str]:
    return {"path": path, "sha256": sha}


def mutate(payload: dict, *path: str, value) -> dict:
    out = deepcopy(payload)
    node = out
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return out
