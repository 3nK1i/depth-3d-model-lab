from __future__ import annotations

from copy import deepcopy
from typing import Any

from core.reconstruction.cross_provider_qualification import (
    qualification_run_evidence_schema,
    qualification_scenario_matrix,
)
from core.reconstruction.single_image_provider_qualification import (
    I43G4A_G2_SCENARIO_MAP,
    I43G4A_PROVIDER_IMPLEMENTATIONS,
    I43G4A_SCENARIOS,
    SingleImageScenarioBinding,
)


def passing_g1_snapshot() -> dict[str, Any]:
    return {
        "phase": "I4.3G1",
        "status": "pass",
        "pass": True,
        "coverage": {
            "g4_scenarios": 3,
            "provider_implementation_count": 10,
            "metric_count": 21,
        },
        "scenario_matrix": [item.as_dict() for item in qualification_scenario_matrix()],
        "run_evidence_schema": qualification_run_evidence_schema(),
        "fairness": {
            "production_native_comparison_primary": True,
            "matched_view_diagnostic_secondary": True,
            "same_authoritative_sfm_within_static_scenario": True,
            "same_source_identity_within_scenario": True,
            "provider_specific_adaptive_plans_persisted": True,
            "manual_review_same_camera_and_framing": True,
            "independent_visual_auto_fit_forbidden": True,
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
        "predecessor_evidence": {
            "i4_3a_sha256": "a" * 64,
            "i4_3f3_sha256": "f" * 64,
        },
    }


def passing_g3c_closure() -> dict[str, Any]:
    authority = {
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
    }
    return {
        "phase": "I4.3G3C",
        "status": "formally_closed",
        "pass": True,
        "i4_3g3_closed": True,
        "i4_3g4_authorized": True,
        "all_required_temporal_scenarios_accounted": True,
        "all_required_manual_reviews_complete": True,
        "authority": authority,
        "decision": {
            "quality_failures_preserved": True,
            "scenario_specific_verdicts_preserved": True,
        },
        "scenario_synthesis": {
            "g3_temporal_motion": {
                "production_statuses": {
                    "rolling_depth": "rejected",
                    "video_depth_anything": "degraded",
                }
            }
        },
    }


def g2b_snapshot(g4_scenario_id: str) -> dict[str, Any]:
    g2 = I43G4A_G2_SCENARIO_MAP[g4_scenario_id]
    return {
        "phase": "I4.3G2B",
        "status": "production_native_complete",
        "pass": True,
        "scenario": {
            "scenario_id": g2,
            "source_directory": f"/data/{g2}",
            "source_image_count": 12,
            "source_asset_set_sha256": "1" * 64,
        },
        "source_identity": {
            "path": f"/evidence/{g2}/source_identity.json",
            "sha256": "2" * 64,
        },
        "authoritative_sfm": {
            "snapshot": {
                "path": f"/evidence/{g2}/authoritative_sfm/sfm_result.json",
                "sha256": "3" * 64,
            },
            "frame_id": f"g2b:{g2}:pycolmap_sfm:session",
            "registered_view_count": 8,
            "landmark_count": 1024,
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "same_authoritative_sfm_shared_by_both_providers": True,
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


def scenario_binding(g4_scenario_id: str) -> SingleImageScenarioBinding:
    g2 = I43G4A_G2_SCENARIO_MAP[g4_scenario_id]
    return SingleImageScenarioBinding(
        scenario_id=g4_scenario_id,
        predecessor_g2_scenario_id=g2,
        source_directory=f"/data/{g2}",
        source_image_count=12,
        source_asset_set_sha256="1" * 64,
        source_manifest_path=f"/evidence/{g2}/source_identity.json",
        source_manifest_sha256="2" * 64,
        authoritative_sfm_snapshot_path=f"/evidence/{g2}/authoritative_sfm/sfm_result.json",
        authoritative_sfm_snapshot_sha256="3" * 64,
        authoritative_sfm_frame_id=f"g2b:{g2}:pycolmap_sfm:session",
        registered_view_count=8,
        landmark_count=1024,
        g2b_production_snapshot_path=f"/evidence/{g2}/i4_3g2b_production_native_snapshot.json",
        g2b_production_snapshot_sha256="4" * 64,
    )


def all_bindings() -> tuple[SingleImageScenarioBinding, ...]:
    return tuple(scenario_binding(sid) for sid in I43G4A_SCENARIOS)


def runtime_probe(implementation: str, *, ready: bool = True) -> dict[str, Any]:
    common: dict[str, Any] = {
        "available": ready,
        "adapter_importable": ready,
        "torch_available": ready,
        "cuda_available": ready,
        "checkpoint_ready": ready,
        "device_names": ["NVIDIA RTX Test"] if ready else [],
        "error": "" if ready else "not ready",
    }
    if implementation == "metric3d_v2":
        common.update({"source_ready": ready, "source_revision": "a" * 40})
    elif implementation == "unidepth_v2":
        common.update({
            "source_ready": ready,
            "source_checkout_matches_marker": ready,
            "official_runtime_importable": ready,
            "official_runtime_source_matches_checkout": ready,
            "source_revision": "b" * 40,
        })
    elif implementation == "moge_2":
        common.update({
            "source_revision_pinned": ready,
            "official_runtime_importable": ready,
            "official_runtime_source_matches_distribution": ready,
            "checkpoint_revision": "c" * 40,
        })
    elif implementation == "depth_pro":
        common.update({
            "source_revision_resolved": ready,
            "requirement_commit_pinned": False,
            "official_runtime_importable": ready,
            "official_runtime_source_matches_distribution": ready,
            "checkpoint_sha256": "d" * 64,
        })
    elif implementation == "depth_anything_v2":
        common.update({
            "source_checkout_matches_marker": ready,
            "installer_ref_commit_pinned": False,
            "official_runtime_importable": ready,
            "official_runtime_source_matches_checkout": ready,
            "checkpoint_revision_commit_pinned": False,
            "checkpoint_sha256": "e" * 64,
        })
    elif implementation == "marigold":
        common.update({
            "official_runtime_importable": ready,
            "official_runtime_source_matches_distribution": ready,
            "runtime_requirement_exact_pinned": False,
            "checkpoint_revision_pinned": ready,
            "checkpoint_marker_sha256": "f" * 64,
            "model_index_sha256": "0" * 64,
        })
    else:
        raise AssertionError(implementation)
    return common


def all_runtime_probes() -> dict[str, dict[str, Any]]:
    return {implementation: runtime_probe(implementation) for implementation in I43G4A_PROVIDER_IMPLEMENTATIONS}


def copied_runtime_probes() -> dict[str, dict[str, Any]]:
    return deepcopy(all_runtime_probes())
