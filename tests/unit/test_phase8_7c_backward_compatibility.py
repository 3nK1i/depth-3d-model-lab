from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from core.reconstruction import (
    CAMERA_GEOMETRY_SCHEMA_VERSION,
    DENSE_FUSION_SCHEMA_VERSION,
    DENSE_GEOMETRY_QUALITY_SCHEMA_VERSION,
    INSPECTION_QC_SCHEMA_VERSION,
    SCALE_REFERENCE_SCHEMA_VERSION,
    SEMANTIC_GEOMETRY_SCHEMA_VERSION,
    SURFACE_MESH_SCHEMA_VERSION,
    Semantic3DObject,
    SemanticGeometryResult,
    SurfaceMesh,
    SurfaceMeshResult,
)


def test_phase8_7c_adds_new_schema_without_rewriting_prior_schemas() -> None:
    assert CAMERA_GEOMETRY_SCHEMA_VERSION == 1
    assert SCALE_REFERENCE_SCHEMA_VERSION == 1
    assert DENSE_FUSION_SCHEMA_VERSION == 1
    assert DENSE_GEOMETRY_QUALITY_SCHEMA_VERSION == 1
    assert SURFACE_MESH_SCHEMA_VERSION == 1
    assert SEMANTIC_GEOMETRY_SCHEMA_VERSION == 1
    assert INSPECTION_QC_SCHEMA_VERSION == 1


def test_semantic_3d_object_contract_is_unchanged() -> None:
    assert tuple(field.name for field in fields(Semantic3DObject)) == (
        "object_id",
        "class_id",
        "class_name",
        "semantic_role",
        "confidence",
        "vertex_indices",
        "face_indices",
        "centroid_reconstruction",
        "bounds_min_reconstruction",
        "bounds_max_reconstruction",
        "principal_axes_reconstruction",
        "orientation_reliable",
        "observation_ids",
        "frame_ids",
        "mask_ids",
        "track_ids",
        "camera_profile_ids",
        "pose_layers",
        "mean_vertex_quality",
        "median_vertex_uncertainty_reconstruction_units",
        "maximum_vertex_uncertainty_reconstruction_units",
        "scale_profile_id",
        "metric_geometry_status",
        "unit_to_metre",
        "relative_scale_uncertainty",
        "certified_metrology",
    )


def test_semantic_result_contract_is_unchanged() -> None:
    assert tuple(field.name for field in fields(SemanticGeometryResult)) == (
        "session_id",
        "scene_profile",
        "pose_convention",
        "reconstruction_frame_id",
        "scale_profile_id",
        "metric_geometry_status",
        "unit_to_metre",
        "relative_scale_uncertainty",
        "certified_metrology",
        "source_surface_schema_version",
        "policy",
        "projections",
        "objects",
        "statistics",
        "schema_version",
    )


def test_surface_mesh_contracts_are_unchanged() -> None:
    assert tuple(field.name for field in fields(SurfaceMesh)) == (
        "vertices_reconstruction",
        "faces",
        "colors_rgb",
        "normals_reconstruction",
        "vertex_quality_scores",
        "vertex_quality_codes",
        "vertex_uncertainty_reconstruction_units",
        "source_point_indices",
        "face_component_ids",
    )
    assert "mesh" in {field.name for field in fields(SurfaceMeshResult)}


def test_reconstruction_ui_remains_disabled() -> None:
    text = Path("app/widgets/mode_tabs.py").read_text(encoding="utf-8")
    enabled_block = text.split("_ENABLED_MODES", 1)[1].split("}", 1)[0]
    assert "RECONSTRUCTION" not in enabled_block


def test_phase8_7c_keeps_native_backends_isolated() -> None:
    for path in Path("core/reconstruction").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "import pycolmap" not in source
        assert "from pycolmap" not in source
        assert "import stella_vslam" not in source
        assert "from stella_vslam" not in source


def test_phase8_7c_does_not_add_neural_rendering_or_ui() -> None:
    text = Path("core/reconstruction/inspection_qc.py").read_text(encoding="utf-8")
    assert "GaussianSplattingBackend" not in text
    assert "NeRFBackend" not in text
    assert "PyQt6" not in text
    assert "vispy" not in text.lower()


def test_phase8_7c_preserves_non_certified_metrology_semantics() -> None:
    text = Path("core/reconstruction/inspection_qc.py").read_text(encoding="utf-8")
    assert "certified_metrology" in text
    assert "Phase 8.7C cannot claim certified metrology" in text
