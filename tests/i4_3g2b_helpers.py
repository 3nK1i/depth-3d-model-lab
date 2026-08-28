from __future__ import annotations

from pathlib import Path
from typing import Any

from core.reconstruction.mvs_da3_qualification import build_source_identity


def make_source(root: Path, *, count: int = 6) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (root / f"frame_{index:03d}.png").write_bytes(b"G2B" + bytes([index]) * 17)
    return root


def authority() -> dict[str, Any]:
    return {
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


def g1_snapshot() -> dict[str, Any]:
    ids = (
        ("g2_complete_room_mvs_da3", ["production_native", "matched_view_diagnostic"]),
        ("g2_temple_ring_mvs_da3", ["production_native", "matched_view_diagnostic"]),
        ("g2_house_mvs_da3_historical", ["historical_read_only"]),
        ("g2_low_texture_mvs_da3", ["production_native", "matched_view_diagnostic"]),
        ("g2_weak_overlap_mvs_da3", ["production_native", "matched_view_diagnostic"]),
    )
    return {
        "phase": "I4.3G1",
        "status": "pass",
        "pass": True,
        "decision": {
            "i4_3g2_mvs_da3_qualification_authorized": True,
            "global_scalar_score_forbidden": True,
            "universal_provider_winner_assumed": False,
        },
        "authority": authority(),
        "scenario_matrix": [
            {
                "scenario_id": sid,
                "substep": "I4.3G2",
                "provider_implementations": ["colmap_mvs", "da3"],
                "comparison_modes": modes,
            }
            for sid, modes in ids
        ],
    }


def g2a_snapshot(source_dir: Path) -> dict[str, Any]:
    identity = build_source_identity(source_dir, minimum_images=3).as_dict()
    fresh_ids = (
        "g2_complete_room_mvs_da3",
        "g2_temple_ring_mvs_da3",
        "g2_low_texture_mvs_da3",
        "g2_weak_overlap_mvs_da3",
    )
    bindings = [
        {
            "scenario_id": sid,
            "status": "bound",
            "ready_for_real_execution": True,
            "binding_method": "test_frozen_binding",
            "source_identity": identity,
        }
        for sid in fresh_ids
    ]
    bindings.insert(2, {
        "scenario_id": "g2_house_mvs_da3_historical",
        "status": "verified_historical_read_only",
        "ready_for_real_execution": False,
        "historical_read_only": True,
        "new_real_execution_allowed": False,
        "historical_evidence_verified": True,
        "source_identity": identity,
    })
    return {
        "phase": "I4.3G2A",
        "status": "ready",
        "pass": True,
        "binding_summary": {
            "fresh_scenario_bindings_ready": 4,
            "historical_binding_verified": True,
            "all_g2_sources_bound": True,
            "missing_explicit_fresh_bindings": [],
        },
        "decision": {"i4_3g2b_real_execution_authorized": True},
        "authority": authority(),
        "scenario_bindings": bindings,
    }


def mvs_runtime(*, ready: bool = True) -> dict[str, Any]:
    return {
        "available": ready,
        "mvs_api_ready": ready,
        "patchmatch_cuda_ready": ready,
        "recommended_version": ready,
        "cuda_distribution": ready,
        "cuda_devices": 1 if ready else 0,
        "pycolmap_version": "4.1.1",
    }


def da3_runtime(*, ready: bool = True) -> dict[str, Any]:
    return {
        "available": ready,
        "cuda_available": ready,
        "depth_anything_3_importable": ready,
        "turing_compatibility_patch_applied": ready,
        "runtime_lock_valid": ready,
        "device_names": ["GPU"] if ready else [],
        "source_revision": "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4" if ready else "",
    }
