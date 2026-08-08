from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from core.reconstruction import (
    GAUSSIAN_SPLATTING_SCHEMA_VERSION,
    INSPECTION_QC_SCHEMA_VERSION,
    NEURAL_SCENE_DATASET_SCHEMA_VERSION,
    SEMANTIC_GEOMETRY_SCHEMA_VERSION,
    SURFACE_MESH_SCHEMA_VERSION,
    GaussianBackendKind,
    GaussianSceneArtifact,
    GaussianUsageScope,
    NeuralSceneDataset,
    gaussian_backend_provenance,
)

ROOT = Path(__file__).resolve().parents[2]


def test_locked_contract_schema_versions_are_unchanged() -> None:
    assert SURFACE_MESH_SCHEMA_VERSION == 1
    assert SEMANTIC_GEOMETRY_SCHEMA_VERSION == 1
    assert INSPECTION_QC_SCHEMA_VERSION == 1
    assert NEURAL_SCENE_DATASET_SCHEMA_VERSION == 1
    assert GAUSSIAN_SPLATTING_SCHEMA_VERSION == 1


def test_d1_gaussian_artifact_fields_remain_backward_compatible() -> None:
    names = [field.name for field in fields(GaussianSceneArtifact)]
    assert "backend_provenance" in names
    assert "optimization_state" in names
    assert "canonical_geometry_authoritative" in names
    assert "certified_metrology" in names
    assert "qualification_status" not in names


def test_d1_neural_scene_contract_remains_independent_of_backend() -> None:
    names = [field.name for field in fields(NeuralSceneDataset)]
    assert "frames" in names
    assert "pose_convention" in names
    assert "gaussian_backend" not in names
    assert "graphdeco_root" not in names


def test_graphdeco_research_only_policy_is_preserved() -> None:
    provenance = gaussian_backend_provenance(GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL)
    assert provenance.usage_scope is GaussianUsageScope.RESEARCH_EVALUATION_ONLY
    assert provenance.commercial_use_permitted_by_backend_license is False
    assert provenance.explicit_commercial_consent_required is True


def test_gsplat_permissive_policy_is_preserved() -> None:
    provenance = gaussian_backend_provenance(GaussianBackendKind.GSPLAT_EXTERNAL)
    assert provenance.usage_scope is GaussianUsageScope.PERMISSIVE_GENERAL_USE
    assert provenance.license_name == "Apache-2.0"


def test_production_dependency_manifest_remains_gaussian_backend_free() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert '"gsplat' not in text
    assert '"nerfstudio' not in text
    assert "diff-gaussian-rasterization" not in text
    assert "simple-knn" not in text


def test_reconstruction_ui_stays_disabled() -> None:
    text = (ROOT / "config/app.yaml").read_text(encoding="utf-8").lower()
    assert "reconstruction_mode_enabled: true" not in text


def test_phase8_7d2_does_not_rewrite_locked_d1_validation_script() -> None:
    text = (ROOT / "scripts/run_phase8_7d1_tests.sh").read_text(encoding="utf-8")
    assert "test_reconstruction_gaussian_splatting.py" in text
    assert "test_phase8_7d1_backward_compatibility.py" in text
