from __future__ import annotations

from copy import deepcopy


def passing_i4_4d() -> dict:
    return {
        "phase": "I4.4D",
        "status": "formally_closed",
        "pass": True,
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "permanent_global_view_cap": None,
            "authoritative_sfm_modified": False,
            "provider_selected": False,
            "provider_promoted": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "authority_audit": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "default_reconstruction_scale_status": "arbitrary",
            "permanent_global_view_cap": None,
        },
        "decision": {
            "i4_3_closed": True,
            "i4_4a_complete": True,
            "i4_4b_complete": True,
            "i4_4c_complete": True,
            "i4_4d_final_closure_complete": True,
            "i4_4_offline_dense_foundation_frozen": True,
            "i4_closed": True,
            "i5_authorized": True,
            "i5_started": False,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "universal_provider_winner_declared": False,
            "global_scalar_score_created": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "handoff_to_i5": {
            "i5_authorized": True,
            "i5_work_started": False,
            "first_boundary": "I5.1 difficult handheld captures",
            "evidence_driven_auto_policy_deferred_to_i5_13": True,
            "hybrid_experiment_deferred_to_i5_12_if_justified": True,
            "offline_dense_foundation_authority_locked": True,
            "dense_provider_metric_scale_not_established_by_i4": True,
        },
        "next": "I5.1 difficult handheld captures.",
    }


def i4_1_4_doc() -> str:
    return """# I4.1.4 — Decoupled Global-Coverage / Local-Density Sampling Diagnosis
On the Horse orbit this improved DA3/SfM alignment to 36/40, but the dense-consistency path still collapsed to two views.
Threshold relaxation is never authorized by this milestone.
"""


def i4_1_5_doc() -> str:
    return """# I4.1.5 — Decoupled Global/Local Sampling Counterfactual
I4.1.4 established a mixed failure: the Horse capture regime is difficult **and** local temporal density materially affects dense-consistency success.
The Horse sequence remains an I5 robustness case; a cleaner static object/connector orbit remains required for final I4 qualification when capture-regime evidence is strong.
"""


def i4_1_6b_doc() -> str:
    return """# I4.1.6B — Real Decoupled Dense A/B Validation
1. `uniform_localized_dense_pool`
2. `global_anchors_plus_local_neighbors`
A production-binding follow-up is authorized only when real evidence gives a full-span plan with >=80% DA3/SfM alignment acceptance, >=10 connected pass-qualified views, and a positive gain over the global-only baseline.
Dense threshold relaxation is never authorized by this step.
"""


def i4_1_7_doc() -> str:
    return """# I4.1.7 — Clean Static Object / Connector Orbit Qualification
It follows the difficult Horse I4.1.6B result, which proved that decoupled sampling materially improves real dense consistency but did not reach the 10-view production gate.
No threshold relaxation is allowed.
"""


def passing_receipt() -> dict:
    return {
        "schema_version": 1,
        "phase": "I5.1A_BASELINE_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i5_1a_tests": True,
            "affected_i4_horse_regression": True,
            "i4_4d_handoff_regression": True,
            "import_probe": True,
            "runner_help_probe": True,
            "provider_execution_started": False,
            "sfm_rerun": False,
            "dense_pair_rerun": False,
            "current_difficult_handheld_replay_started": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "counts": {
            "focused_tests": 30,
            "affected_unit_tests": 25,
            "affected_integration_tests": 8,
            "affected_stress_tests": 9,
            "affected_total_tests": 42,
            "combined_exercised_tests": 72,
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


def clone(value):
    return deepcopy(value)
