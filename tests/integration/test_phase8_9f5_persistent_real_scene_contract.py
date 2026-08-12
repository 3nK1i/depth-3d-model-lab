from __future__ import annotations

from pathlib import Path

from core.reconstruction import ReconstructionArtifactAuthority, ReconstructionArtifactKind

ROOT = Path(__file__).resolve().parents[2]


def test_existing_persistent_scene_contract_is_reused_not_reimplemented() -> None:
    runner = (ROOT / "scripts/run_phase8_9f5_persistent_multi_artifact_scene.py").read_text(encoding="utf-8")
    assert "persistent_scene_from_pipeline_payload" in runner
    assert "write_persistent_reconstruction_scene" in runner
    assert "read_persistent_reconstruction_scene" in runner
    assert "verify_persistent_scene_file_references" in runner
    assert "class PersistentReconstructionScene" not in runner


def test_existing_authority_contract_keeps_neural_artifacts_appearance_only() -> None:
    assert ReconstructionArtifactKind.GAUSSIAN_SCENE.value == "gaussian_scene"
    assert ReconstructionArtifactKind.NERF_SCENE.value == "nerf_scene"
    assert ReconstructionArtifactAuthority.APPEARANCE_ONLY.value == "appearance_only"
    text = (ROOT / "core/reconstruction/persistent_scene.py").read_text(encoding="utf-8")
    assert "Gaussian/NeRF persistent artifacts must remain appearance-only" in text
    assert "Persistent scene requires canonical geometry authority" in text


def test_f5_prepares_existing_app_preview_contract_without_backend_execution() -> None:
    runner = (ROOT / "scripts/run_phase8_9f5_persistent_multi_artifact_scene.py").read_text(encoding="utf-8")
    assert "build_gaussian_visualization_presentation" in runner
    assert "build_nerf_visualization_presentation" in runner
    assert "preview.png" in runner
    assert "gaussian_training_started\": False" in runner
    assert "nerfacto_training_started\": False" in runner
