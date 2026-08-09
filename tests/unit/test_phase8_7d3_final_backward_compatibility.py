from pathlib import Path

from core.reconstruction import (
    GAUSSIAN_SPLATTING_SCHEMA_VERSION,
    INSPECTION_QC_SCHEMA_VERSION,
    NEURAL_SCENE_DATASET_SCHEMA_VERSION,
    SEMANTIC_GEOMETRY_SCHEMA_VERSION,
    SURFACE_MESH_SCHEMA_VERSION,
    GaussianBackendKind,
    GaussianUsageScope,
    gaussian_backend_provenance,
)

ROOT = Path(__file__).resolve().parents[2]


def test_locked_schema_versions_remain_unchanged_at_d3_closure() -> None:
    assert SURFACE_MESH_SCHEMA_VERSION == 1
    assert SEMANTIC_GEOMETRY_SCHEMA_VERSION == 1
    assert INSPECTION_QC_SCHEMA_VERSION == 1
    assert NEURAL_SCENE_DATASET_SCHEMA_VERSION == 1
    assert GAUSSIAN_SPLATTING_SCHEMA_VERSION == 1


def test_backend_license_provenance_remains_separate() -> None:
    graphdeco = gaussian_backend_provenance(GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL)
    gsplat = gaussian_backend_provenance(GaussianBackendKind.GSPLAT_EXTERNAL)
    assert graphdeco.usage_scope is GaussianUsageScope.RESEARCH_EVALUATION_ONLY
    assert graphdeco.explicit_commercial_consent_required is True
    assert gsplat.usage_scope is GaussianUsageScope.PERMISSIVE_GENERAL_USE
    assert gsplat.license_name == "Apache-2.0"


def test_no_gaussian_heavy_dependency_added_to_production_manifest() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    for token in ("gsplat", "nerfstudio", "gaussian-splatting", "diff-gaussian-rasterization", "simple-knn"):
        assert token not in text


def test_reconstruction_ui_remains_disabled_at_d3_closure() -> None:
    text = (ROOT / "config" / "app.yaml").read_text(encoding="utf-8").lower()
    assert "reconstruction_mode_enabled: true" not in text
