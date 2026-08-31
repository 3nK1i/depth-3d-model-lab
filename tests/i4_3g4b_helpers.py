from __future__ import annotations

from copy import deepcopy
from typing import Any

from core.reconstruction.single_image_provider_qualification import (
    I43G4A_PROVIDER_IMPLEMENTATIONS,
    build_i4_3g4a_preflight_snapshot,
)
from core.reconstruction.single_image_provider_production_qualification import (
    G4BScenarioBinding,
)
from tests.i4_3g4a_helpers import (
    all_bindings,
    all_runtime_probes,
    passing_g1_snapshot,
    passing_g3c_closure,
)


def _identity_fields(implementation: str) -> dict[str, Any]:
    common = {
        "python_executable": "/env/bin/python",
        "project_root": "/project",
    }
    values: dict[str, dict[str, Any]] = {
        "metric3d_v2": {
            "source_directory": "/project/third_party/Metric3D",
            "ready_marker": "/project/third_party/Metric3D/.ready",
            "checkpoint_path": "/project/checkpoints/metric3d.pth",
            "source_revision": "a" * 40,
        },
        "unidepth_v2": {
            "source_directory": "/project/third_party/UniDepth",
            "compatibility_marker": "/project/third_party/UniDepth/.compat",
            "checkpoint_path": "/project/checkpoints/unidepth-v2/vits14",
            "source_revision": "b" * 40,
            "checkpoint_expected_revision": "1" * 40,
            "variant": "small",
            "repository": "lpiccinelli/unidepth-v2-vits14",
        },
        "moge_2": {
            "distribution_name": "moge",
            "distribution_version": "1.0",
            "distribution_root": "/env/site-packages/moge",
            "source_repository": "https://github.com/microsoft/MoGe",
            "source_revision": "c" * 40,
            "variant": "small",
            "checkpoint_repository": "Ruicheng/moge-2-vits-normal",
            "checkpoint_revision": "2" * 40,
        },
        "depth_pro": {
            "distribution_name": "depth-pro",
            "distribution_version": "0.1",
            "distribution_root": "/env/site-packages/depth_pro",
            "source_repository": "https://github.com/apple/ml-depth-pro",
            "source_revision": "d" * 40,
            "checkpoint_path": "/project/checkpoints/depth_pro.pt",
            "checkpoint_sha256": "d" * 64,
        },
        "depth_anything_v2": {
            "source_directory": "/project/third_party/Depth-Anything-V2",
            "source_marker": "/project/third_party/Depth-Anything-V2/.ready",
            "source_repository": "https://github.com/DepthAnything/Depth-Anything-V2",
            "source_revision": "e" * 40,
            "variant": "small",
            "encoder": "vits",
            "checkpoint_repository": "depth-anything/Depth-Anything-V2-Small",
            "checkpoint_declared_revision": "main",
            "checkpoint_path": "/project/checkpoints/dav2-vits.pth",
            "checkpoint_sha256": "e" * 64,
        },
        "marigold": {
            "distribution_name": "diffusers",
            "distribution_version": "0.30",
            "distribution_root": "/env/site-packages/diffusers",
            "profile": "efficient",
            "checkpoint_repository": "prs-eth/marigold-depth-v1-1",
            "checkpoint_resolved_revision": "f" * 40,
            "checkpoint_path": "/project/checkpoints/marigold/depth-v1-1",
            "checkpoint_marker_sha256": "f" * 64,
            "model_index_sha256": "0" * 64,
        },
    }
    common.update(values[implementation])
    return common


def passing_g4a_snapshot() -> dict[str, Any]:
    probes = all_runtime_probes()
    for implementation in I43G4A_PROVIDER_IMPLEMENTATIONS:
        probes[implementation].update(_identity_fields(implementation))
    snapshot = build_i4_3g4a_preflight_snapshot(
        i4_3g1_snapshot=passing_g1_snapshot(),
        i4_3g3c_closure=passing_g3c_closure(),
        scenario_bindings=all_bindings(),
        runtime_probes=probes,
        free_disk_bytes=80 * 1024**3,
        minimum_free_disk_bytes=32 * 1024**3,
    )
    snapshot["predecessor_evidence"] = {
        "i4_3g1_snapshot": {"path": "/evidence/g1.json", "sha256": "a" * 64},
        "i4_3g3c_final_closure": {"path": "/evidence/g3c.json", "sha256": "b" * 64},
    }
    return snapshot


def current_runtime(implementation: str) -> dict[str, Any]:
    payload = deepcopy(passing_g4a_snapshot()["runtime"][implementation])
    payload.pop("g4a_ready", None)
    return payload


def g4b_binding(scenario_id: str = "g4_single_image_complete_room") -> G4BScenarioBinding:
    raw = next(item for item in passing_g4a_snapshot()["scenario_bindings"] if item["scenario_id"] == scenario_id)
    return G4BScenarioBinding(
        scenario_id=raw["scenario_id"],
        predecessor_g2_scenario_id=raw["predecessor_g2_scenario_id"],
        source_directory=raw["source_directory"],
        source_image_count=raw["source_image_count"],
        source_asset_set_sha256=raw["source_asset_set_sha256"],
        source_manifest_path=raw["source_manifest_path"],
        source_manifest_sha256=raw["source_manifest_sha256"],
        authoritative_sfm_snapshot_path=raw["authoritative_sfm_snapshot_path"],
        authoritative_sfm_snapshot_sha256=raw["authoritative_sfm_snapshot_sha256"],
        authoritative_sfm_frame_id=raw["authoritative_sfm_frame_id"],
        registered_view_count=raw["registered_view_count"],
        landmark_count=raw["landmark_count"],
        g2b_production_snapshot_path=raw["g2b_production_snapshot_path"],
        g2b_production_snapshot_sha256=raw["g2b_production_snapshot_sha256"],
    )


def plan_dict(implementation: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "implementation": implementation,
        "source_view_count": 8,
        "selected_view_count": 6,
        "source_order_keyframe_ids": [f"kf-{i}" for i in range(8)],
        "selected_keyframe_ids": [f"kf-{i}" for i in range(6)],
        "adaptive_selection_evaluated": True,
        "reduction_applied": True,
        "requirements": {
            "stateful_sequence": False,
            "source_order_required": False,
        },
        "permanent_global_view_cap": None,
    }


def provider_evidence(implementation: str, *, status: str = "pass") -> dict[str, Any]:
    rejected = status == "rejected"
    reason = {
        "metric3d_v2": "I4.3E1 insufficient Metric3D/SfM alignment-accepted views: accepted=1 required=2.",
        "unidepth_v2": "I4.3E2 insufficient UniDepthV2/SfM alignment-accepted views: accepted=1 required=2.",
        "moge_2": "I4.3E3 insufficient MoGe-2/SfM alignment-accepted views: accepted=1 required=2.",
        "depth_pro": "I4.3E4 insufficient Depth Pro/SfM alignment-accepted views: accepted=1 required=2.",
        "depth_anything_v2": "I4.3E5 insufficient Depth Anything V2/SfM alignment-accepted views: accepted=1 required=2.",
        "marigold": "I4.3E6 insufficient Marigold/SfM alignment-accepted views: accepted=1 required=2.",
    }[implementation]
    metrics = {
        "execution_completed": not rejected,
        "source_input_identity_preserved": True,
        "authoritative_sfm_preserved": True,
        "adaptive_plan_bound": True,
        "provider_provenance_complete": True,
        "forbidden_authority_claims_absent": True,
        "wall_time_s": 1.25,
        "peak_vram_mib": 1024.0,
        "selected_view_count": 6,
        "processed_view_count": 6,
        "aligned_view_fraction": 1.0 / 6.0 if rejected else 1.0,
    }
    optional_reasons = {
        "peak_host_memory_mib": "not_measured",
        "sparse_correspondence_coverage": "not_available",
        "alignment_median_relative_residual": "not_available",
        "alignment_p95_relative_residual": "not_available",
        "inter_view_consistency_ratio": "deferred_to_g4c",
        "inter_view_median_relative_residual": "deferred_to_g4c",
        "registered_view_surface_coverage": "not_available",
        "canonical_point_count": "not_available" if rejected else "covered_below",
        "detached_component_fraction": "not_available",
        "unsupported_geometry_fraction": "not_available",
    }
    if not rejected:
        metrics["canonical_point_count"] = 1000
        optional_reasons.pop("canonical_point_count")
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "scenario_id": "g4_single_image_complete_room",
        "provider_implementation": implementation,
        "comparison_mode": "production_native",
        "attempt": 1,
        "status": status,
        "source_identity": {},
        "authoritative_sfm_identity": {},
        "adaptive_plan_identity": {},
        "provider_provenance": {},
        "runtime_resources": {},
        "metrics": metrics,
        "optional_metric_reasons": optional_reasons,
        "artifacts": [{"path": "/evidence/native.json", "sha256": "9" * 64}],
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
    if rejected:
        evidence["rejection"] = {
            "terminal": True,
            "recognized_existing_quality_gate": True,
            "reason": reason,
            "canonical_geometry_generated": False,
            "threshold_changed": False,
            "provider_execution_reran": False,
            "scenario_execution_continues_to_other_provider": True,
        }
    return evidence
