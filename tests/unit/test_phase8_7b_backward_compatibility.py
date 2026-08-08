from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from core.perception import ObjectObservation
from core.reconstruction import (
    CAMERA_GEOMETRY_SCHEMA_VERSION,
    DENSE_FUSION_SCHEMA_VERSION,
    DENSE_GEOMETRY_QUALITY_SCHEMA_VERSION,
    SCALE_REFERENCE_SCHEMA_VERSION,
    SEMANTIC_GEOMETRY_SCHEMA_VERSION,
    SURFACE_MESH_SCHEMA_VERSION,
    DenseFusionResult,
    DenseGeometryQualityReport,
    SurfaceMesh,
    SurfaceMeshResult,
)


def test_prior_phase_schema_versions_remain_unchanged() -> None:
    assert CAMERA_GEOMETRY_SCHEMA_VERSION == 1
    assert SCALE_REFERENCE_SCHEMA_VERSION == 1
    assert DENSE_FUSION_SCHEMA_VERSION == 1
    assert DENSE_GEOMETRY_QUALITY_SCHEMA_VERSION == 1
    assert SURFACE_MESH_SCHEMA_VERSION == 1
    assert SEMANTIC_GEOMETRY_SCHEMA_VERSION == 1


def test_surface_mesh_contract_is_not_rewritten() -> None:
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


def test_surface_mesh_result_contract_is_not_rewritten() -> None:
    assert tuple(field.name for field in fields(SurfaceMeshResult)) == (
        "session_id",
        "scene_profile",
        "camera_profile_id",
        "scale_profile_id",
        "metric_geometry_status",
        "pose_convention",
        "reconstruction_frame_id",
        "scale_status",
        "world_alignment_status",
        "unit_to_metre",
        "relative_scale_uncertainty",
        "certified_metrology",
        "source_frame_ids",
        "source_pose_layers",
        "source_quality_schema_version",
        "quality_limiters",
        "policy",
        "mesh",
        "statistics",
        "schema_version",
    )


def test_phase75_object_observation_contract_is_not_rewritten() -> None:
    assert tuple(field.name for field in fields(ObjectObservation)) == (
        "observation_id",
        "frame_reference",
        "class_id",
        "class_name",
        "detection",
        "masks",
        "selected_mask_id",
        "track_id",
        "border_contacts",
        "estimated_occlusion_fraction",
        "warnings",
        "refinements",
    )


def test_dense_and_quality_contracts_are_not_rewritten() -> None:
    assert "point_cloud" in {field.name for field in fields(DenseFusionResult)}
    assert "point_quality" in {field.name for field in fields(DenseGeometryQualityReport)}


def test_reconstruction_ui_remains_disabled() -> None:
    text = Path("app/widgets/mode_tabs.py").read_text(encoding="utf-8")
    enabled_block = text.split("_ENABLED_MODES", 1)[1].split("}", 1)[0]
    assert "RECONSTRUCTION" not in enabled_block


def test_phase8_7b_keeps_native_backends_isolated() -> None:
    for path in Path("core/reconstruction").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "import pycolmap" not in source
        assert "from pycolmap" not in source
        assert "import stella_vslam" not in source
        assert "from stella_vslam" not in source


def test_phase8_7b_does_not_add_qc_or_neural_rendering() -> None:
    text = Path("core/reconstruction/semantic_geometry.py").read_text(encoding="utf-8")
    assert "GaussianSplattingBackend" not in text
    assert "NeRFBackend" not in text
    assert "InspectionResult" not in text
    assert "Pass / Warn / Reject" not in text
