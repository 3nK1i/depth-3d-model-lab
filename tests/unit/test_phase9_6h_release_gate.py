from __future__ import annotations

from copy import deepcopy

from app.reconstruction_release_gate import qualified_release_limitations, validate_phase96h_evidence


def _evidence() -> dict:
    refs = [
        {"path": f"/tmp/ref-{i}", "role": f"role-{i}", "sha256": "a" * 64, "size_bytes": i + 1}
        for i in range(3)
    ]
    manual = {
        "canonical_scene_visible": True,
        "camera_trajectory_visible": True,
        "estimated_metric_ruler_uncertainty_visible": True,
        "restart_restore_pass": True,
        "nerf_interactive_pass": True,
        "static_preview_nonowning_pass": True,
        "resource_handoff_pass": True,
        "no_orphan_cuda_process_after_exit": True,
        "authority_wording_pass": True,
        "appearance_path": "nerf",
    }
    return {
        "video": {"readable": True, "duration_s": 173.0, "fps": 30.0, "frame_count": 5190, "width": 1280, "height": 720},
        "e2_full_scene": {
            "phase": "9.6E2", "pass": True, "selected_keyframes": 87, "window_count": 85,
            "geometry_scale": "relative_arbitrary_scale", "unit_to_metre": None,
            "loop_closure": False, "quality_certified": False, "camera_calibrated": False,
            "canonical_geometry_authoritative": False, "metric_scale_authoritative": False,
            "measurement_authoritative": False, "qc_authoritative": False, "certified_metrology": False,
            "scene_id": "prs-source",
        },
        "persistent_scene": {
            "canonical_geometry_authoritative": True, "certified_metrology": False,
            "scene_fingerprint_sha256": "b" * 64, "camera_count": 87, "references_verified": True,
        },
        "e3_nerf": {
            "phase": "9.6E3", "pass": True, "source_scene_id": "prs-source", "persistent_scene_id": "prs-nerf",
            "input_frame_count": 87, "training_steps": 64, "nerf_scene_role": "appearance_only", "quality_certified": False,
            "canonical_geometry_authoritative": False, "metric_scale_authoritative": False,
            "measurement_authoritative": False, "qc_authoritative": False, "certified_metrology": False,
        },
        "state_export": {
            "schema": "depth3d.digital-twin.state-export.v1", "scene_id": "prs-nerf",
            "artifact_payloads_embedded": False, "portable_payload_bundle": False,
            "restore_requires_referenced_files": True, "canonical_geometry_authoritative": False,
            "metric_scale_authoritative": False, "measurement_authoritative": False,
            "qc_authoritative": False, "certified_metrology": False,
            "scene_fingerprint_sha256": "c" * 64, "workspace_fingerprint_sha256": "d" * 64,
            "referenced_payload_count": 3, "referenced_payloads": refs,
        },
        "integrity": {"persistent_scene_manifest_exists": True, "state_export_references_verified": True},
        "manual": manual,
        "limitations": qualified_release_limitations(),
    }


def test_phase96h_complete_functional_release_evidence_passes() -> None:
    result = validate_phase96h_evidence(_evidence())
    assert result.passed is True
    assert result.reasons == ()


def test_phase96h_rejects_short_sequence_and_missing_manual_gate() -> None:
    evidence = _evidence()
    evidence["video"]["duration_s"] = 87.0
    evidence["manual"]["restart_restore_pass"] = False
    result = validate_phase96h_evidence(evidence)
    assert result.passed is False
    assert "source_video_duration_outside_2_to_3_minute_gate" in result.reasons
    assert "manual_gate_missing:restart_restore_pass" in result.reasons


def test_phase96h_rejects_authority_or_quality_promotion() -> None:
    evidence = _evidence()
    evidence["e3_nerf"]["metric_scale_authoritative"] = True
    evidence["limitations"]["photorealistic_quality_certified"] = True
    result = validate_phase96h_evidence(evidence)
    assert result.passed is False
    assert "e3_authority_violation:metric_scale_authoritative" in result.reasons
    assert "release_limitation_overclaimed:photorealistic_quality_certified" in result.reasons


def test_phase96h_rejects_lineage_or_reference_integrity_drift() -> None:
    evidence = deepcopy(_evidence())
    evidence["e3_nerf"]["source_scene_id"] = "different-scene"
    evidence["state_export"]["scene_id"] = "different-revision"
    evidence["integrity"]["state_export_references_verified"] = False
    result = validate_phase96h_evidence(evidence)
    assert result.passed is False
    assert "e3_source_scene_lineage_mismatch" in result.reasons
    assert "state_export_scene_lineage_mismatch" in result.reasons
    assert "state_export_reference_integrity_failed" in result.reasons
