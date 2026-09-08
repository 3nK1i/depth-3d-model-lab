from __future__ import annotations

from copy import deepcopy

from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_rolling_depth_robustness import I57A_FROZEN_G3_ROLLING_DEPTH_STATUSES
from tests.i5_7b_helpers import admission, receipt, recovery_artifact_bindings, recovery_provenance, provider_summary, scenario_record
from core.reconstruction.i5_rolling_depth_robustness_replay import (
    I57B_RECOVERY_SCENARIO_ID,
    build_i5_7b_recovered_control_record,
    build_i5_7b_rolling_depth_robustness_replay,
)

I57A_SHA = "040ee6ce45f32e9f3f51087ce5f5b8764ced9a33e978d8024d04867efa5fae61"
I57B_SHA = "516264b62866bf0b02ce80006d7ba72b15cef23576d065435f792b391d5f29df"
I57C_RECEIPT_SHA = "c" * 64


def clone(payload: dict) -> dict:
    return deepcopy(payload)


def binding(path: str, sha: str) -> I51EvidenceBinding:
    return I51EvidenceBinding(path, sha)


def _recovery_bindings() -> dict:
    return {
        "original_attempt_manifest": {"path": "/evidence/original.sha256_manifest.txt", "sha256": "1" * 64},
        "original_console_log": {"path": "/evidence/original.console.log", "sha256": "2" * 64},
        "original_attempt_failure": {"path": "/evidence/i5_7b_attempt_failure.json", "sha256": "3" * 64},
        "original_scenario_failure": {"path": "/evidence/01_castle_control/i5_7b_attempt_failure.json", "sha256": "4" * 64},
        "provider_summary": {"path": "/evidence/provider_summary.json", "sha256": "5" * 64},
        "canonical_npz": {"path": "/evidence/canonical.npz", "sha256": "6" * 64},
        "canonical_ply": {"path": "/evidence/canonical.ply", "sha256": "7" * 64},
        "native_worker_evidence": {"path": "/evidence/rolling_depth_evidence.npz", "sha256": "8" * 64},
        "native_result_manifest": {"path": "/evidence/rolling_depth_result_manifest.json", "sha256": "9" * 64},
    }


def passing_i5_7b_final() -> dict:
    a = admission()
    records = {}
    provenance = recovery_provenance()
    provenance["current_frozen_sfm_revalidated"] = True
    control = build_i5_7b_recovered_control_record(
        provider_summary=provider_summary("pass", views=19),
        artifact_bindings=recovery_artifact_bindings(),
        recovery_provenance=provenance,
    )
    records[I57B_RECOVERY_SCENARIO_ID] = control
    for sid in (
        "castle_low_overlap",
        "castle_weak_parallax",
        "castle_overexposure",
        "castle_symmetry",
        "castle_dynamic_proxy",
    ):
        records[sid] = scenario_record(sid)
    result = build_i5_7b_rolling_depth_robustness_replay(
        i5_7a_admission=a,
        i5_7a_binding=binding("/evidence/i5_7a.json", I57A_SHA),
        replay_receipt=receipt(),
        replay_receipt_binding=binding("/evidence/i5_7b_receipt.json", "d" * 64),
        runtime_preflight={
            "rolling_depth_runtime_ready": True,
            "provider_id": "rolling-depth-temporal",
            "runtime_lock_valid": True,
            "checkpoint_ready": True,
            "cuda_available": True,
            "device": "cuda:0",
            "precision": "fp16",
            "production_config_unchanged": True,
            "source_revision": "frozen-test-revision",
        },
        scenario_records=records,
    )
    # Match the actual recovery-production-native final artifact emitted by the CLI.
    result["scope"]["recovery_used"] = True
    result["scope"]["castle_control_provider_rerun"] = False
    result["scope"]["five_remaining_scenarios_executed_in_recovery_attempt"] = True
    result["recovery_provenance"] = deepcopy(control["recovery"])
    assert result["historical_temporal_anchors_preserved"]["g3_temporal_motion"]["rolling_depth_production_status"] == I57A_FROZEN_G3_ROLLING_DEPTH_STATUSES["g3_temporal_motion"]
    return result


def passing_i5_7c_receipt() -> dict:
    return {
        "schema_version": 1,
        "phase": "I5.7C_QUALIFICATION_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i5_7c_tests": True,
            "predecessor_i5_7b_regression": True,
            "predecessor_i5_7a_regression": True,
            "historical_g3_preservation_regression": True,
            "recovery_provenance_regression": True,
            "authority_guard_regression": True,
            "roadmap_guard_regression": True,
            "import_probe": True,
            "runner_help_probe": True,
            "rolling_depth_inference": False,
            "sfm_rerun": False,
            "mvs_rerun": False,
            "provider_comparison_started": False,
            "thresholds_modified": False,
            "production_default_modified": False,
            "retry_until_pass": False,
            "mp_sfm_started": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "counts": {
            "focused_tests": 20,
            "predecessor_unit_tests": 100,
            "predecessor_integration_tests": 20,
            "predecessor_stress_tests": 30,
            "predecessor_total_tests": 150,
            "combined_exercised_tests": 170,
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
