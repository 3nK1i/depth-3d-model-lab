from __future__ import annotations

from copy import deepcopy

from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_remaining_depth_provider_robustness import (
    build_i5_9a_remaining_depth_provider_robustness_admission,
)
from core.reconstruction.i5_remaining_depth_provider_robustness_replay import (
    I59B1_I56A_FROZEN_SHA256,
    I59B1_I59A_FROZEN_SHA256,
    I59B1_PROVIDERS,
    I59B1_SCENARIO_IDS,
)
from tests.i5_9a_helpers import build_args as i59a_build_args


def clone(value):
    return deepcopy(value)


def binding(path: str, sha256: str) -> I51EvidenceBinding:
    return I51EvidenceBinding(path, sha256)


def admission() -> dict:
    return build_i5_9a_remaining_depth_provider_robustness_admission(**i59a_build_args())


def receipt() -> dict:
    return {
        "schema_version": 1,
        "phase": "I5.9B1_REPLAY_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i5_9b1_tests": True,
            "b1_provider_regression": True,
            "i5_9a_admission_regression": True,
            "i5_6a_source_binding_regression": True,
            "authority_guard_regression": True,
            "import_probe": True,
            "runner_help_probe": True,
            "real_provider_inference_executed_by_receipt": False,
            "sfm_rerun": False,
            "mvs_rerun": False,
            "thresholds_modified": False,
            "production_default_modified": False,
            "result_driven_retuning": False,
            "retry_until_pass": False,
            "provider_comparison_started": False,
            "mp_sfm_started": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "counts": {"focused_tests": 1, "predecessor_total_tests": 1, "combined_exercised_tests": 2},
    }


def preflight() -> dict:
    runtimes = {
        provider: {
            "available": True,
            "cuda_ready": True,
            "provider_id": f"{provider}-provider",
            "worker_script": f"/repo/scripts/{provider}_worker.py",
            "runtime_config": {"device": "cuda:0", "precision": "fp16"},
            "environment": {"available": True},
        }
        for provider in I59B1_PROVIDERS
    }
    plans = {}
    source_counts = {
        "castle_control": 19,
        "castle_low_overlap": 6,
        "castle_weak_parallax": 7,
        "castle_overexposure": 19,
        "castle_symmetry": 12,
        "castle_dynamic_proxy": 19,
    }
    for sid in I59B1_SCENARIO_IDS:
        plans[sid] = {
            "source_integrity_verified": True,
            "authoritative_sfm_reused_without_rerun": True,
            "registered_views": source_counts[sid],
            "providers": {
                provider: {
                    "source_view_count": source_counts[sid],
                    "selected_view_count": source_counts[sid],
                    "provider_inference_started": False,
                }
                for provider in I59B1_PROVIDERS
            },
        }
    return {
        "schema_version": 1,
        "phase": "I5.9B1",
        "status": "runtime_source_preflight_pass",
        "pass": True,
        "providers": list(I59B1_PROVIDERS),
        "scenario_ids": list(I59B1_SCENARIO_IDS),
        "expected_provider_scenario_runs": 24,
        "runtime": runtimes,
        "scenario_plans": plans,
        "scope": {
            "provider_inference_started": False,
            "sfm_rerun": False,
            "mvs_rerun": False,
            "provider_comparison_started": False,
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "authoritative_sfm_modified": False,
            "provider_selected": False,
            "provider_promoted": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "permanent_global_view_cap": None,
        },
    }


def record(sid: str, provider: str, *, status: str = "pass") -> dict:
    source = {
        "castle_control": 19,
        "castle_low_overlap": 6,
        "castle_weak_parallax": 7,
        "castle_overexposure": 19,
        "castle_symmetry": 12,
        "castle_dynamic_proxy": 19,
    }[sid]
    accepted = source if status != "rejected" else 1
    payload = {
        "schema_version": 1,
        "phase": "I5.9B1",
        "scenario_id": sid,
        "provider_implementation": provider,
        "runtime_status": status,
        "source_registered_views": source,
        "selected_views": source,
        "processed_views": source,
        "accepted_views": accepted,
        "adaptive_selection": True,
        "plan_reason_codes": ["all_registered_views_fit_provider_budget"],
        "input_integrity_revalidated_after": True,
        "fusion_capacity_drops": 123 if status != "rejected" else 0,
        "fusion_capacity_drops_change_status": False,
        "native_metrics": {
            "wall_time_s": 10.0,
            "peak_vram_mib": 1024.0,
            "canonical_point_count": 1000 if status != "rejected" else None,
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "authoritative_sfm_modified": False,
            "sfm_rerun": False,
            "bundle_adjustment_rerun": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "production_filter_policy_changed": False,
            "provider_promotion_performed": False,
            "permanent_global_view_cap": None,
        },
    }
    if status == "rejected":
        payload["rejection"] = {
            "recognized_existing_quality_gate": True,
            "terminal": True,
            "threshold_changed": False,
            "reason": "synthetic recognized quality rejection",
        }
    return payload


def records(*, overrides: dict[tuple[str, str], str] | None = None) -> dict:
    overrides = overrides or {}
    return {
        sid: {
            provider: record(sid, provider, status=overrides.get((sid, provider), "pass"))
            for provider in I59B1_PROVIDERS
        }
        for sid in I59B1_SCENARIO_IDS
    }


def build_replay_args(*, overrides: dict[tuple[str, str], str] | None = None) -> dict:
    return {
        "i5_9a_admission": admission(),
        "i5_9a_binding": binding("/evidence/i5_9a.json", I59B1_I59A_FROZEN_SHA256),
        "i5_6a_binding": binding("/evidence/i5_6a.json", I59B1_I56A_FROZEN_SHA256),
        "replay_receipt": receipt(),
        "replay_receipt_binding": binding("/evidence/i5_9b1_receipt.json", "c" * 64),
        "runtime_preflight": preflight(),
        "runtime_preflight_binding": binding("/evidence/i5_9b1_preflight.json", "d" * 64),
        "records": records(overrides=overrides),
    }
