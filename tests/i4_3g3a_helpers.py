from __future__ import annotations

from core.reconstruction.temporal_provider_qualification import TemporalScenarioBinding, TemporalVideoIdentity


def passing_g1_snapshot() -> dict:
    scenarios = []
    specs = (
        ("g3_temporal_static_sweep", "ordered_video_static_scene", "mixed"),
        ("g3_temporal_motion", "ordered_video_dynamic_scene", "mixed"),
        ("g3_temporal_low_texture", "ordered_video_low_texture", "low"),
    )
    for sid, source_kind, texture in specs:
        scenarios.append({
            "scenario_id": sid,
            "substep": "I4.3G3",
            "provider_implementations": ["rolling_depth", "video_depth_anything"],
            "source_kind": source_kind,
            "comparison_modes": ["production_native", "matched_view_diagnostic"],
            "manual_review_required": True,
            "manual_review_dimensions": [
                "temporal_flicker", "temporal_drift", "static_geometry_consistency",
                "motion_boundary_stability", "transient_artifacts", "textureless_region_stability",
            ],
            "new_real_execution_required": True,
            "temporal_order_required": True,
            "texture_regime": texture,
            "overlap_regime": "temporal_contiguous",
        })
    return {
        "phase": "I4.3G1", "status": "pass", "pass": True,
        "coverage": {"g3_scenarios": 3},
        "scenario_matrix": scenarios,
        "decision_rules": {
            "production_native_plan_is_primary_operating_envelope": True,
            "matched_view_diagnostic_is_secondary_causal_diagnostic": True,
            "matched_view_mode_must_preserve_provider_semantics": True,
            "temporal_matched_view_mode_requires_contiguous_source_order": True,
            "same_source_asset_identity_required_within_scenario": True,
            "same_manual_review_camera_and_framing_required": True,
            "independent_visual_auto_fit_forbidden": True,
            "scenario_specific_verdicts_required": True,
            "global_scalar_score_forbidden": True,
            "auto_policy_implemented": False,
            "hybrid_experiment_authorized": False,
            "provider_promotion_performed": False,
            "universal_provider_winner_assumed": False,
        },
        "fairness": {
            "production_native_comparison_primary": True,
            "matched_view_diagnostic_secondary": True,
            "same_source_identity_within_scenario": True,
            "provider_specific_adaptive_plans_persisted": True,
            "temporal_contiguous_source_order_required": True,
            "manual_review_same_camera_and_framing": True,
            "independent_visual_auto_fit_forbidden": True,
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm", "permanent_global_view_cap": None,
            "authoritative_sfm_modified": False, "provider_selected": False, "provider_promoted": False,
            "auto_policy_implemented": False, "hybrid_authorized": False, "metric_scale_claimed": False,
            "measurement_authoritative": False, "certified_metrology": False, "surface_mesh_generated": False,
        },
    }


def passing_g2_closure() -> dict:
    return {
        "phase": "I4.3G2", "status": "formally_closed", "i4_3g2_closed": True, "i4_3g3_authorized": True,
        "synthesis": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "auto_provider_selection_implemented": False, "hybrid_provider_authorized": False,
            "metric_scale_claimed": False, "measurement_authoritative": False, "certified_metrology": False,
            "provider_promoted": False, "provider_selected": False, "surface_mesh_generated": False,
            "universal_provider_winner_declared": False,
        },
    }


def runtime_probe(implementation: str, *, available: bool = True) -> dict:
    common = {
        "available": available, "source_revision": "a" * 40, "source_ready": True,
        "checkpoint_ready": True, "torch_available": True,
    }
    if implementation == "video_depth_anything":
        return {**common, "adapter_importable": True, "upstream_importable": True, "cuda_available": True}
    if implementation == "rolling_depth":
        return {
            **common, "runtime_lock_valid": True, "upstream_importable": True,
            "modified_diffusers_ready": True, "ffmpeg_available": True, "cuda_available": True,
        }
    raise ValueError(implementation)


def scenario_binding(scenario_id: str, index: int = 0) -> TemporalScenarioBinding:
    source_sha = f"{index + 1:064x}"[-64:]
    fingerprint = f"{index + 101:064x}"[-64:]
    manifest_sha = f"{index + 201:064x}"[-64:]
    identity = TemporalVideoIdentity(
        source_path=f"/tmp/{scenario_id}.mp4",
        source_file_sha256=source_sha,
        asset_set_sha256=source_sha,
        file_size_bytes=1024 + index,
        reader_fingerprint=fingerprint,
        expected_frame_rate_hz=30.0,
        reported_frame_count=120,
    )
    return TemporalScenarioBinding(
        scenario_id=scenario_id,
        source_identity=identity,
        manifest_path=f"/tmp/{scenario_id}.source.json",
        manifest_sha256=manifest_sha,
        semantic_binding_method="explicit_engineer_scenario_binding",
    )


def all_bindings() -> tuple[TemporalScenarioBinding, ...]:
    return tuple(scenario_binding(sid, index) for index, sid in enumerate((
        "g3_temporal_static_sweep", "g3_temporal_motion", "g3_temporal_low_texture"
    )))
