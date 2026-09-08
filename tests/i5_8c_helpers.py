from __future__ import annotations

from copy import deepcopy

from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_video_depth_anything_robustness_replay import (
    I58B_SCENARIO_IDS,
    build_i5_8b_video_depth_anything_robustness_replay,
)
from tests import i5_8b_helpers

I58A_SHA = "1d635cdcefe02e1ee3b66dfaa8af4df49fb952e58cbf4446b50c2d562369d90a"
I58B_SHA = "24eab2c95869ea96792c6599eeb94ce2801d5920d4a37b9a015a5428e568bb63"
I58C_RECEIPT_SHA = "c" * 64
ACTUAL_CHECKPOINT_SHA = "3c28432b4e1f0d7bb31cad5151b6313b49457db5aa58d82e85bfb0f8b1311b33"
ACTUAL_SOURCE_REVISION = "4f5ae23172ba60fd7bc11ef671cca678842c7072"
ACTUAL_CHECKPOINT = "/home/eiatech/Desktop/depth_3d_model_lab/checkpoints/video-depth-anything/metric_video_depth_anything_vits.pth"
CAPACITY_DROPS = {
    "castle_control": 3678651,
    "castle_low_overlap": 215631,
    "castle_weak_parallax": 925581,
    "castle_overexposure": 3518689,
    "castle_symmetry": 3213160,
    "castle_dynamic_proxy": 3744324,
}


def clone(payload):
    return deepcopy(payload)


def binding(path: str, sha: str) -> I51EvidenceBinding:
    return I51EvidenceBinding(path, sha)


def passing_i5_8a() -> dict:
    return i5_8b_helpers.admission()


def passing_i5_8b_final() -> dict:
    args = i5_8b_helpers.build_args()
    args["scenario_records"] = {
        sid: i5_8b_helpers.scenario_record(sid, "degraded") for sid in I58B_SCENARIO_IDS
    }
    result = build_i5_8b_video_depth_anything_robustness_replay(**args)
    result["quality_status"] = "degraded"
    preflight = result["runtime_preflight"]
    preflight["resolved_profile"] = "efficient"
    preflight["resolved_encoder"] = "vits"
    preflight["device"] = "cuda:0"
    preflight["precision"] = "fp16"
    preflight["resolved_checkpoint"] = ACTUAL_CHECKPOINT
    preflight["checkpoint_sha256"] = ACTUAL_CHECKPOINT_SHA
    preflight["source_revision"] = ACTUAL_SOURCE_REVISION
    preflight["runtime_config"]["profile"] = "efficient"
    preflight["runtime_config"]["hardware_tier"] = "efficient"
    preflight["runtime_config"]["encoder"] = "vits"
    preflight["runtime_config"]["checkpoint"] = ACTUAL_CHECKPOINT
    preflight["runtime_config"]["device"] = "cuda:0"
    preflight["runtime_config"]["precision"] = "fp16"
    for sid in I58B_SCENARIO_IDS:
        record = result["scenario_records"][sid]
        views = record["source_registered_views"]
        drops = CAPACITY_DROPS[sid]
        record["runtime_status"] = "degraded"
        record["native_metrics"]["alignment_accepted_views"] = views
        record["native_metrics"]["alignment_rejected_views"] = 0
        record["native_metrics"]["alignment_accepted_fraction"] = 1.0
        record["native_metrics"]["fusion_capacity_drops"] = drops
        record["native_metrics"]["canonical_point_count"] = 1500000
        record["runtime_profile"]["resolved_profile"] = "efficient"
        record["runtime_profile"]["resolved_encoder"] = "vits"
        record["runtime_profile"]["resolved_checkpoint"] = ACTUAL_CHECKPOINT
        record["execution_result"]["status"] = "degraded"
        record["execution_result"]["degradation_reasons"] = [f"fusion_capacity_drops:{drops}"]
        record["execution_result"]["failure_reason"] = ""
    return result


def passing_i5_8c_receipt() -> dict:
    return {
        "schema_version": 1,
        "phase": "I5.8C_QUALIFICATION_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i5_8c_tests": True,
            "predecessor_i5_8b_regression": True,
            "predecessor_i5_8a_regression": True,
            "historical_g3_preservation_regression": True,
            "fusion_capacity_provenance_regression": True,
            "authority_guard_regression": True,
            "roadmap_guard_regression": True,
            "import_probe": True,
            "runner_help_probe": True,
            "video_depth_anything_inference": False,
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
            "focused_tests": 18,
            "predecessor_unit_tests": 60,
            "predecessor_integration_tests": 10,
            "predecessor_stress_tests": 10,
            "predecessor_total_tests": 80,
            "combined_exercised_tests": 98,
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
