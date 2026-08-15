from __future__ import annotations


def make_valid_phase9_4h_evidence() -> dict:
    fingerprint = "a" * 64
    revision = "937e29912570c372bed6747a5c9bf85fed877bae"
    scene_id = "prs-phase94-source"
    return {
        "runtime": {
            "ready": True,
            "checkout_revision": revision,
            "checkout_clean": True,
            "production_heavy_modules": [],
        },
        "source_scene": {
            "status": "PASS",
            "canonical_geometry_authoritative": True,
            "metric_geometry_status": "arbitrary",
            "unit_to_metre": None,
        },
        "bridge": {
            "status": "PASS",
            "scene_id": scene_id,
            "dataset_fingerprint_sha256": fingerprint,
            "source_frame_fingerprints_exact": True,
            "canonical_camera_binding_exact": True,
            "gaussian_scene_role": "appearance_only",
            "da3_reexecuted": False,
            "gaussian_backend_started": False,
        },
        "optimization": {
            "status": "PASS",
            "scene_id": scene_id,
            "dataset_fingerprint_sha256": fingerprint,
            "runtime_checkout_revision": revision,
            "real_cuda_execution": True,
            "optimization_completed": True,
            "preview_render_written": True,
            "peak_vram_mib": 420.0,
            "finite_loss": True,
            "initial_l1": 0.31,
            "final_l1": 0.23,
            "da3_reexecuted": False,
            "graphdeco_started": False,
            "nerf_started": False,
        },
        "persistence": {
            "status": "PASS",
            "parent_scene_id": scene_id,
            "dataset_fingerprint_sha256": fingerprint,
            "source_scene_unchanged": True,
            "restart_reload_verified": True,
            "gaussian_file_reference_count": 7,
            "gaussian_backend_reexecuted": False,
            "da3_reexecuted": False,
            "nerf_started": False,
        },
        "reload": {
            "status": "PASS",
            "restart_reload_verified": True,
            "canonical_geometry_available": True,
            "gaussian_artifact_available": True,
        },
        "viewer": {
            "interactive_appearance_ready": True,
            "picking_source": "canonical_geometry_only",
        },
        "controls": {
            "scene_layer_visibility_only": True,
            "persisted_gaussian_modified": False,
        },
        "resources": {
            "cycles": 12,
            "release_checks_passed": 12,
            "all_numpy_staging_released": True,
            "gpu_vram_measured": False,
            "timings_are_cpu_ui_prep_only": True,
        },
        "measurement": {
            "state": "unavailable",
            "metric_value_exposed": False,
        },
        "authority": {
            "canonical_geometry_authoritative": True,
            "gaussian_canonical_geometry_authoritative": False,
            "gaussian_metric_scale_authoritative": False,
            "gaussian_measurement_authoritative": False,
            "gaussian_qc_authoritative": False,
            "certified_metrology": False,
            "production_heavy_modules_after": [],
        },
    }
