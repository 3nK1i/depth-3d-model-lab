from __future__ import annotations
from dataclasses import fields
from pathlib import Path

from core.reconstruction import (
    GAUSSIAN_SPLATTING_SCHEMA_VERSION,
    INSPECTION_QC_SCHEMA_VERSION,
    NEURAL_SCENE_DATASET_SCHEMA_VERSION,
    SEMANTIC_GEOMETRY_SCHEMA_VERSION,
    SURFACE_MESH_SCHEMA_VERSION,
    GaussianSceneArtifact,
    NerfSceneArtifact,
    NeuralSceneDataset,
)

ROOT = Path(__file__).resolve().parents[2]


def test_existing_phase87_contract_versions_remain_unchanged() -> None:
    assert SURFACE_MESH_SCHEMA_VERSION == 1
    assert SEMANTIC_GEOMETRY_SCHEMA_VERSION == 1
    assert INSPECTION_QC_SCHEMA_VERSION == 1
    assert NEURAL_SCENE_DATASET_SCHEMA_VERSION == 1
    assert GAUSSIAN_SPLATTING_SCHEMA_VERSION == 1


def test_neural_scene_dataset_is_reused_not_rewritten_for_nerf() -> None:
    names = [field.name for field in fields(NeuralSceneDataset)]
    assert 'frames' in names
    assert 'pose_convention' in names
    assert 'nerf_scene' not in names


def test_gaussian_artifact_contract_is_not_modified_into_nerf() -> None:
    gaussian = [field.name for field in fields(GaussianSceneArtifact)]
    nerf = [field.name for field in fields(NerfSceneArtifact)]
    assert 'gaussian_count' in gaussian
    assert 'checkpoint_path' not in gaussian
    assert 'checkpoint_path' in nerf


def test_nerf_dependencies_are_not_added_to_production_pyproject() -> None:
    text = (ROOT / 'pyproject.toml').read_text(encoding='utf-8').lower()
    assert 'nerfstudio' not in text
    assert 'tinycudann' not in text
    assert 'tiny-cuda-nn' not in text


def test_production_reconstruction_core_does_not_directly_import_nerf_runtime() -> None:
    for path in (ROOT / 'core/reconstruction').glob('*.py'):
        text = path.read_text(encoding='utf-8')
        forbidden = ('import nerfstudio', 'from nerfstudio', 'import tinycudann', 'from tinycudann')
        assert not any(token in text for token in forbidden), path.name


def test_nerf_contract_is_appearance_only_by_design() -> None:
    names = [field.name for field in fields(NerfSceneArtifact)]
    assert 'canonical_geometry_authoritative' in names
    assert 'certified_metrology' in names


def test_reconstruction_ui_remains_disabled() -> None:
    text = (ROOT / 'config/app.yaml').read_text(encoding='utf-8').lower()
    assert 'reconstruction_mode_enabled: true' not in text
