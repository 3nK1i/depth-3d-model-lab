"""Cross-layer deterministic integration gates for final Phase 8.7 readiness."""
from __future__ import annotations

from pathlib import Path

from core.reconstruction import (
    PersistentReconstructionSceneState,
    build_reconstruction_endurance_campaign,
    persistent_scene_from_pipeline_result,
    read_persistent_reconstruction_scene,
    verify_persistent_scene_file_references,
    write_persistent_reconstruction_scene,
)
from tests.reconstruction_f5_helpers import run_f5_fixture
from tests.reconstruction_g3_helpers import (
    checkpoint_rejection_evidence,
    clean_evidence,
    optional_degradation_evidence,
    required_failure_evidence,
)


def test_g4_integrated_f1_to_f5_scene_is_ready_and_authority_safe(tmp_path: Path) -> None:
    _fixture, _providers, result, snapshots = run_f5_fixture(tmp_path / "run", request_id="phase8-7g4-final")
    scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)
    manifest = write_persistent_reconstruction_scene(scene, (tmp_path / "scene").resolve())
    restored = read_persistent_reconstruction_scene(manifest)
    verify_persistent_scene_file_references(restored)
    assert restored.state is PersistentReconstructionSceneState.READY
    assert restored.canonical_geometry_authoritative is True
    assert restored.certified_metrology is False
    payload = restored.as_dict()
    assert payload["appearance"]["authority"] == "appearance_only"
    assert payload["appearance"]["gaussian_artifact_id"]
    assert payload["appearance"]["nerf_artifact_id"]
    assert payload["reconstruction_ui_enabled"] is False


def test_g4_final_campaign_keeps_g3_failure_and_degradation_boundaries(tmp_path: Path) -> None:
    report = build_reconstruction_endurance_campaign(
        (
            clean_evidence(0),
            clean_evidence(1),
            optional_degradation_evidence(2, tmp_path),
            required_failure_evidence(3),
            checkpoint_rejection_evidence(4),
        ),
        backward_compatibility_verified=True,
        resource_cleanup_verified=True,
        authority_stability_verified=True,
        checkpoint_safety_verified=True,
    )
    payload = report.as_dict()
    assert report.passed
    assert payload["stale_state_detected"] is False
    assert payload["unique_clean_scene_ids"] is True
    assert payload["full_regression_deferred_to"] == "8.7G4"
    assert payload["reconstruction_ui_enabled"] is False
