from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from core.reconstruction import (
    INSPECTION_QC_SCHEMA_VERSION,
    SEMANTIC_GEOMETRY_SCHEMA_VERSION,
    SURFACE_MESH_SCHEMA_VERSION,
    GaussianBackendKind,
    GaussianSceneArtifact,
    GaussianUsageScope,
    NeuralSceneDataset,
    gaussian_backend_provenance,
    SurfaceMeshResult,
)

ROOT = Path(__file__).resolve().parents[2]


def test_existing_87a_surface_contract_version_is_unchanged() -> None:
    assert SURFACE_MESH_SCHEMA_VERSION == 1


def test_existing_87b_semantic_contract_version_is_unchanged() -> None:
    assert SEMANTIC_GEOMETRY_SCHEMA_VERSION == 1


def test_existing_87c_inspection_contract_version_is_unchanged() -> None:
    assert INSPECTION_QC_SCHEMA_VERSION == 1


def test_surface_mesh_result_fields_are_not_rewritten() -> None:
    names = [field.name for field in fields(SurfaceMeshResult)]
    assert "mesh" in names
    assert "statistics" in names
    assert "gaussian_scene" not in names


def test_gaussian_dependencies_are_not_added_to_production_pyproject() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert '"gsplat' not in text
    assert '"nerfstudio' not in text
    assert "gaussian-splatting" not in text


def test_production_core_does_not_directly_import_gaussian_backends() -> None:
    for relative in (
        "core/reconstruction/gaussian_splatting.py",
        "core/reconstruction/neural_scene.py",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        forbidden = (
            "import gsplat",
            "from gsplat",
            "import nerfstudio",
            "from nerfstudio",
            "import gaussian_renderer",
            "from gaussian_renderer",
            "import diff_gaussian_rasterization",
            "from diff_gaussian_rasterization",
            "import simple_knn",
            "from simple_knn",
        )
        assert not any(token in text for token in forbidden)


def test_gaussian_artifact_contract_is_appearance_only_by_design() -> None:
    names = [field.name for field in fields(GaussianSceneArtifact)]
    assert "canonical_geometry_authoritative" in names
    assert "certified_metrology" in names


def test_neural_scene_dataset_retains_explicit_pose_convention() -> None:
    names = [field.name for field in fields(NeuralSceneDataset)]
    assert "pose_convention" in names
    assert "T_world_from_camera" not in names



def test_graphdeco_reference_backend_cannot_be_marked_permissive() -> None:
    provenance = gaussian_backend_provenance(
        GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL
    )
    assert provenance.usage_scope is GaussianUsageScope.RESEARCH_EVALUATION_ONLY
    assert provenance.commercial_use_permitted_by_backend_license is False
    assert provenance.explicit_commercial_consent_required is True


def test_gsplat_backend_remains_separate_permissive_path() -> None:
    provenance = gaussian_backend_provenance(GaussianBackendKind.GSPLAT_EXTERNAL)
    assert provenance.usage_scope is GaussianUsageScope.PERMISSIVE_GENERAL_USE
    assert provenance.license_name == "Apache-2.0"

def test_reconstruction_ui_remains_disabled() -> None:
    text = (ROOT / "config/app.yaml").read_text(encoding="utf-8").lower()
    assert "reconstruction_mode_enabled: true" not in text
