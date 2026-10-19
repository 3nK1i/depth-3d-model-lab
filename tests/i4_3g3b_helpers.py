from __future__ import annotations

from copy import deepcopy
from typing import Any

from core.reconstruction.cross_provider_qualification import qualification_metric_registry
from core.reconstruction.temporal_provider_qualification import build_i4_3g3a_preflight_snapshot
from tests.i4_3g3a_helpers import all_bindings, passing_g1_snapshot, passing_g2_closure, runtime_probe


def passing_g3a_snapshot() -> dict[str, Any]:
    vda = runtime_probe("video_depth_anything")
    vda.update({
        "project_root": "/opt/depth3d/depth_3d_model_lab",
        "python_executable": "/opt/depth3d/envs/depth/bin/python",
        "checkpoint_path": "/opt/depth3d/depth_3d_model_lab/checkpoints/video_depth_anything_vits.pth",
    })
    rolling = runtime_probe("rolling_depth")
    rolling.update({
        "python_executable": "/opt/depth3d/envs/depth3d-rolling-depth/bin/python",
        "repository_root": "/opt/depth3d/third_party/RollingDepth",
        "runtime_lock_path": "/opt/depth3d/third_party/RollingDepth/.depth3d_i4_3d2_runtime.json",
        "checkpoint_path": "/opt/depth3d/third_party/RollingDepth/checkpoints/rollingdepth",
    })
    return build_i4_3g3a_preflight_snapshot(
        i4_3g1_snapshot=passing_g1_snapshot(),
        i4_3g2_closure=passing_g2_closure(),
        scenario_bindings=all_bindings(),
        video_depth_anything_runtime=vda,
        rolling_depth_runtime=rolling,
    )


def sfm_runtime(*, ready: bool = True) -> dict[str, Any]:
    return {
        "available": ready,
        "backend_ready": ready,
        "required_apis_available": ready,
        "recommended_version": ready,
        "cuda_ready": ready,
        "cuda_devices": 1 if ready else 0,
        "python_executable": "/opt/depth3d/envs/depth/bin/python",
        "version": "4.1.1",
    }


def current_vda_runtime(snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    source = (snapshot or passing_g3a_snapshot())["runtime"]["video_depth_anything"]
    return deepcopy(source)


def current_rolling_runtime(snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    source = (snapshot or passing_g3a_snapshot())["runtime"]["rolling_depth"]
    return deepcopy(source)


def production_plan(implementation: str, *, scrambled: bool = False) -> dict[str, Any]:
    source = ["kf000", "kf001", "kf002", "kf003", "kf004"]
    selected = ["kf000", "kf002", "kf004"]
    if scrambled:
        selected = ["kf004", "kf000", "kf002"]
    return {
        "implementation": implementation,
        "adaptive_selection_evaluated": True,
        "permanent_global_view_cap": None,
        "requirements": {
            "stateful_sequence": True,
            "source_order_required": True,
        },
        "source_order_keyframe_ids": source,
        "selected_keyframe_ids": selected,
        "selected_view_count": len(selected),
    }


def provider_evidence(
    implementation: str = "rolling_depth",
    *,
    scenario_id: str = "g3_temporal_static_sweep",
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    optional_reasons: dict[str, str] = {}
    required_values = {
        "execution_completed": True,
        "source_input_identity_preserved": True,
        "authoritative_sfm_preserved": True,
        "adaptive_plan_bound": True,
        "provider_provenance_complete": True,
        "forbidden_authority_claims_absent": True,
        "wall_time_s": 1.25,
        "peak_vram_mib": 1234.0,
        "selected_view_count": 3,
        "processed_view_count": 3,
        "aligned_view_fraction": 1.0,
    }
    for spec in qualification_metric_registry():
        if spec.metric_id in required_values:
            metrics[spec.metric_id] = required_values[spec.metric_id]
        elif spec.required:
            raise AssertionError(f"Unhandled required metric: {spec.metric_id}")
        else:
            optional_reasons[spec.metric_id] = "not_available_from_frozen_temporal_adapter_in_g3b"
    return {
        "schema_version": 1,
        "scenario_id": scenario_id,
        "provider_implementation": implementation,
        "comparison_mode": "production_native",
        "attempt": "attempt-001",
        "status": "pass",
        "source_identity": {"source_file_sha256": "1" * 64},
        "authoritative_sfm_identity": {"snapshot_sha256": "2" * 64},
        "adaptive_plan_identity": {"sha256": "3" * 64},
        "provider_provenance": {"implementation": implementation},
        "runtime_resources": {"measurement_scope": "device_wide_nvidia_smi_peak_used_memory"},
        "metrics": metrics,
        "optional_metric_reasons": optional_reasons,
        "artifacts": [],
        "temporal_sequence": {
            "source_order_preserved": True,
            "provider_specific_temporal_semantics_preserved": True,
            "matched_view_contiguous_subsequence_used": False,
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
