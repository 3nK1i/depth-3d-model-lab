from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from core.reconstruction import (
    GAUSSIAN_SPLATTING_SCHEMA_VERSION,
    INSPECTION_QC_SCHEMA_VERSION,
    NERF_E4_SCHEMA_VERSION,
    NERF_SCHEMA_VERSION,
    NEURAL_SCENE_DATASET_SCHEMA_VERSION,
    RECONSTRUCTION_ORCHESTRATION_SCHEMA_VERSION,
    SEMANTIC_GEOMETRY_SCHEMA_VERSION,
    SURFACE_MESH_SCHEMA_VERSION,
    GaussianSceneArtifact,
    NerfSceneArtifact,
    ReconstructionPipelineRunResult,
)

ROOT = Path(__file__).resolve().parents[2]


def test_existing_phase87_contract_schema_versions_are_unchanged() -> None:
    assert SURFACE_MESH_SCHEMA_VERSION == 1
    assert SEMANTIC_GEOMETRY_SCHEMA_VERSION == 1
    assert INSPECTION_QC_SCHEMA_VERSION == 1
    assert NEURAL_SCENE_DATASET_SCHEMA_VERSION == 1
    assert GAUSSIAN_SPLATTING_SCHEMA_VERSION == 1
    assert NERF_SCHEMA_VERSION == 1
    assert NERF_E4_SCHEMA_VERSION == 1
    assert RECONSTRUCTION_ORCHESTRATION_SCHEMA_VERSION == 1


def test_gaussian_and_nerf_artifacts_remain_separate_appearance_contracts() -> None:
    gaussian_fields = {field.name for field in fields(GaussianSceneArtifact)}
    nerf_fields = {field.name for field in fields(NerfSceneArtifact)}
    assert "gaussian_count" in gaussian_fields
    assert "checkpoint_path" in nerf_fields
    assert "canonical_geometry_authoritative" in gaussian_fields
    assert "canonical_geometry_authoritative" in nerf_fields


def test_f1_result_does_not_introduce_persistent_scene_contract_early() -> None:
    names = {field.name for field in fields(ReconstructionPipelineRunResult)}
    assert "persistent_scene" not in names
    assert "scene_database" not in names
    assert "canonical_geometry_artifact_id" in names


def test_f1_core_has_no_real_backend_or_future_da3_runtime_imports() -> None:
    text = (ROOT / "core/reconstruction/orchestration.py").read_text(encoding="utf-8").lower()
    forbidden = (
        "import nerfstudio",
        "from nerfstudio",
        "import tinycudann",
        "from tinycudann",
        "import gsplat",
        "from gsplat",
        "import torch",
        "from torch",
        "import depth_anything_3",
        "from depth_anything_3",
        "import depth_anything_3_api",
    )
    assert not any(token in text for token in forbidden)


def test_f1_does_not_add_heavy_integration_dependencies_to_pyproject() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    for token in ("nerfstudio", "tinycudann", "tiny-cuda-nn", "depth-anything-3", "depth_anything_3"):
        assert token not in text


def test_existing_model_agnostic_inference_controller_is_not_coupled_to_reconstruction_orchestrator() -> None:
    candidates = (
        ROOT / "app/inference_controller.py",
        ROOT / "core/inference_controller.py",
        ROOT / "core/inference/controller.py",
    )
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    assert path is not None, "InferenceController source must remain discoverable."
    text = path.read_text(encoding="utf-8").lower()
    assert "reconstructionpipelineorchestrator" not in text
    assert "core.reconstruction.orchestration" not in text


def test_reconstruction_ui_remains_disabled_until_phase88() -> None:
    text = (ROOT / "config/app.yaml").read_text(encoding="utf-8").lower()
    assert "reconstruction_mode_enabled: true" not in text


def test_f1_has_no_persistent_reconstruction_scene_implementation() -> None:
    for path in (ROOT / "core/reconstruction").glob("orchestration*.py"):
        text = path.read_text(encoding="utf-8")
        assert "class PersistentReconstructionScene" not in text
