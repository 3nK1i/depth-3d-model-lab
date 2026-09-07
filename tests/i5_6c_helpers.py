from __future__ import annotations

from copy import deepcopy

from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_mvs_robustness import (
    I56B_RECOVERY_PHASE,
    I56B_RECOVERY_SCENARIO_ID,
    I56B_SCENARIO_IDS,
    build_i5_6b_mvs_robustness_replay,
)
from tests.i5_6b_helpers import admission, receipt, scenario_record

I56A_SHA = "a" * 64
I56B_SHA = "b" * 64
I56C_RECEIPT_SHA = "c" * 64


def clone(payload: dict) -> dict:
    return deepcopy(payload)


def binding(path: str, sha: str) -> I51EvidenceBinding:
    return I51EvidenceBinding(path, sha)


def _artifact_bindings() -> dict:
    return {
        key: {"path": f"/evidence/{key}.json", "sha256": (str(index % 10) * 64)}
        for index, key in enumerate(
            (
                "runtime_preflight",
                "original_replay_receipt",
                "recovery_validation_receipt",
                "recovered_scenario_record",
                "recovery_provider_summary",
                "recovered_native_manifest",
            ),
            start=1,
        )
    }


def recovery_provenance() -> dict:
    return {
        "schema_version": 1,
        "phase": I56B_RECOVERY_PHASE,
        "recovery_required": True,
        "interrupted_parent_exit_code": 130,
        "recovered_scenario_id": I56B_RECOVERY_SCENARIO_ID,
        "scenarios_1_through_5_reused_without_rerun": True,
        "scenario_6_native_artifacts_reused_without_rerun": True,
        "worker_result_missing": True,
        "normal_provider_summary_missing": True,
        "worker_termination_cause": "unresolved_abrupt_termination_after_native_fusion_artifact_write",
        "native_fusion_structurally_validated": True,
        "authoritative_inputs_revalidated": True,
        "sfm_rerun": False,
        "patch_match_rerun": False,
        "stereo_fusion_rerun": False,
        "mvs_rerun": False,
        "thresholds_modified": False,
        "production_default_modified": False,
        "retry_until_pass": False,
        "artifact_bindings": _artifact_bindings(),
    }


def passing_i5_6b(*, directional_reduction: bool = True, recovered: bool = True) -> dict:
    a = admission()
    point_support = {
        "castle_control": (1000, 4.0),
        "castle_low_overlap": (850 if directional_reduction else 1000, 3.7 if directional_reduction else 4.0),
        "castle_weak_parallax": (1020, 4.1),
        "castle_overexposure": (900 if directional_reduction else 1000, 3.9 if directional_reduction else 4.0),
        "castle_symmetry": (600 if directional_reduction else 1000, 2.8 if directional_reduction else 4.0),
        "castle_dynamic_proxy": (800 if directional_reduction else 1000, 4.6),
    }
    records = {}
    for sid in I56B_SCENARIO_IDS:
        points, support = point_support[sid]
        selected = a["scenario_matrix"][sid]["registered_images"]
        records[sid] = scenario_record(sid, selected=selected, points=points, support=support)
        records[sid]["authority"].update(
            {
                "single_authoritative_sfm_frame_preserved": True,
                "sfm_rerun": False,
                "bundle_adjustment_rerun": False,
                "pose_refinement": False,
                "surface_mesh_generated": False,
                "production_filter_policy_changed": False,
                "provider_promotion_performed": False,
            }
        )
    recovery = recovery_provenance() if recovered else None
    if recovered:
        dynamic = records[I56B_RECOVERY_SCENARIO_ID]
        dynamic["provider_adapter_returned"] = False
        dynamic["recovered_from_existing_native_artifacts"] = True
        dynamic["execution_result_provenance"] = "reconstructed_from_existing_native_artifacts_not_original_adapter_return"
        dynamic["recovery_provenance"] = {
            "schema_version": 1,
            "phase": I56B_RECOVERY_PHASE,
            "scenario_id": I56B_RECOVERY_SCENARIO_ID,
            "recovery_required": True,
            "native_worker_finalization_incomplete": True,
            "worker_result_present": False,
            "worker_state_stale_at_stereo_fusion": True,
            "native_patchmatch_outputs_recovered": True,
            "native_stereo_fusion_artifacts_recovered": True,
            "native_fusion_structurally_validated": True,
            "authoritative_inputs_revalidated": True,
            "provider_adapter_returned": False,
            "execution_result_reconstructed_from_native_evidence": True,
            "worker_termination_cause": "unresolved_abrupt_termination_after_native_fusion_artifact_write",
            "sfm_rerun": False,
            "patch_match_rerun": False,
            "stereo_fusion_rerun": False,
            "mvs_rerun": False,
            "thresholds_modified": False,
            "production_default_modified": False,
            "result_driven_retuning": False,
            "retry_until_pass": False,
        }
        dynamic["native_metrics"]["wall_time_s"] = None
        dynamic["native_metrics"]["peak_vram_mib"] = None
        dynamic["native_metrics"]["wall_time_unavailable_due_interrupted_parent_finalization"] = True
        dynamic["native_metrics"]["peak_resource_metrics_incomplete_due_missing_worker_result"] = True
    result = build_i5_6b_mvs_robustness_replay(
        i5_6a_admission=a,
        i5_6a_binding=binding("/evidence/i5_6a.json", I56A_SHA),
        replay_receipt=receipt(),
        replay_receipt_binding=binding("/evidence/i5_6b_receipt.json", "d" * 64),
        runtime_preflight={"mvs_runtime_ready": True, "provider_available": True, "cuda_devices": 1},
        scenario_records=records,
        recovery_provenance=recovery,
    )
    return result


def degraded_i5_6b() -> dict:
    a = admission()
    records = {}
    for sid in I56B_SCENARIO_IDS:
        selected = a["scenario_matrix"][sid]["registered_images"]
        records[sid] = scenario_record(sid, selected=selected)
        records[sid]["authority"].update(
            {
                "single_authoritative_sfm_frame_preserved": True,
                "sfm_rerun": False,
                "bundle_adjustment_rerun": False,
                "pose_refinement": False,
                "surface_mesh_generated": False,
                "production_filter_policy_changed": False,
                "provider_promotion_performed": False,
            }
        )
    records["castle_low_overlap"] = scenario_record("castle_low_overlap", status="rejected", selected=6)
    records["castle_low_overlap"]["authority"].update(
        {
            "single_authoritative_sfm_frame_preserved": True,
            "sfm_rerun": False,
            "bundle_adjustment_rerun": False,
            "pose_refinement": False,
            "surface_mesh_generated": False,
            "production_filter_policy_changed": False,
            "provider_promotion_performed": False,
        }
    )
    return build_i5_6b_mvs_robustness_replay(
        i5_6a_admission=a,
        i5_6a_binding=binding("/evidence/i5_6a.json", I56A_SHA),
        replay_receipt=receipt(),
        replay_receipt_binding=binding("/evidence/i5_6b_receipt.json", "d" * 64),
        runtime_preflight={"mvs_runtime_ready": True, "provider_available": True, "cuda_devices": 1},
        scenario_records=records,
    )


def passing_i5_6c_receipt() -> dict:
    return {
        "schema_version": 1,
        "phase": "I5.6C_QUALIFICATION_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i5_6c_tests": True,
            "predecessor_i5_6b_regression": True,
            "predecessor_i5_6a_regression": True,
            "evidence_only_closure_contract_regression": True,
            "interrupted_recovery_provenance_regression": True,
            "directional_native_support_preservation_regression": True,
            "authority_guard_regression": True,
            "roadmap_guard_regression": True,
            "import_probe": True,
            "runner_help_probe": True,
            "sfm_rerun": False,
            "mvs_rerun": False,
            "patch_match_rerun": False,
            "stereo_fusion_rerun": False,
            "provider_comparison_started": False,
            "da3_rerun": False,
            "thresholds_modified": False,
            "production_default_modified": False,
            "retry_until_pass": False,
            "mp_sfm_started": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "counts": {
            "focused_tests": 20,
            "predecessor_unit_tests": 80,
            "predecessor_integration_tests": 20,
            "predecessor_stress_tests": 25,
            "predecessor_total_tests": 125,
            "combined_exercised_tests": 145,
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
