from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from core.reconstruction import (
    CAMERA_GEOMETRY_SCHEMA_VERSION,
    DENSE_FUSION_SCHEMA_VERSION,
    DENSE_GEOMETRY_QUALITY_SCHEMA_VERSION,
    SCALE_REFERENCE_SCHEMA_VERSION,
    SURFACE_MESH_SCHEMA_VERSION,
    DenseFusionResult,
    DenseGeometryQualityReport,
    DensePointCloud,
    ReconstructionGauge,
    SfmScaleStatus,
    SfmWorldAlignmentStatus,
)


def test_prior_phase_schema_versions_remain_unchanged() -> None:
    assert CAMERA_GEOMETRY_SCHEMA_VERSION == 1
    assert SCALE_REFERENCE_SCHEMA_VERSION == 1
    assert DENSE_FUSION_SCHEMA_VERSION == 1
    assert DENSE_GEOMETRY_QUALITY_SCHEMA_VERSION == 1
    assert SURFACE_MESH_SCHEMA_VERSION == 1


def test_dense_point_cloud_contract_is_not_rewritten() -> None:
    assert tuple(field.name for field in fields(DensePointCloud)) == (
        "points_reconstruction",
        "colors_rgb",
        "view_observation_counts",
        "mean_confidence",
        "voxel_size_reconstruction_units",
    )


def test_dense_fusion_result_contract_is_not_rewritten() -> None:
    assert tuple(field.name for field in fields(DenseFusionResult)) == (
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
        "certified_metrology",
        "source_frame_ids",
        "source_pose_layers",
        "point_cloud",
        "statistics",
        "schema_version",
    )


def test_geometry_quality_report_contract_is_not_rewritten() -> None:
    assert tuple(field.name for field in fields(DenseGeometryQualityReport)) == (
        "session_id",
        "scene_profile",
        "camera_profile_id",
        "scale_profile_id",
        "metric_geometry_status",
        "pose_convention",
        "reconstruction_frame_id",
        "certified_metrology",
        "policy",
        "evidence",
        "point_quality",
        "summary",
        "quality_limiters",
        "unobserved_semantics",
        "schema_version",
    )


def test_raw_reconstruction_gauge_default_remains_arbitrary() -> None:
    gauge = ReconstructionGauge()
    assert gauge.scale_status is SfmScaleStatus.ARBITRARY
    assert gauge.alignment_status is SfmWorldAlignmentStatus.ARBITRARY
    assert gauge.unit_to_metre is None


def test_reconstruction_ui_remains_disabled() -> None:
    text = Path("app/widgets/mode_tabs.py").read_text(encoding="utf-8")
    enabled_block = text.split("_ENABLED_MODES", 1)[1].split("}", 1)[0]
    assert "RECONSTRUCTION" not in enabled_block


def test_phase8_7a_keeps_native_backends_isolated() -> None:
    for path in Path("core/reconstruction").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "import pycolmap" not in source
        assert "from pycolmap" not in source
        assert "import stella_vslam" not in source
        assert "from stella_vslam" not in source


def test_phase8_7a_does_not_add_semantics_qc_or_neural_rendering() -> None:
    text = Path("core/reconstruction/surface_mesh.py").read_text(encoding="utf-8")
    assert "GaussianSplattingBackend" not in text
    assert "NeRFBackend" not in text
    assert "Semantic3DObject" not in text
    assert "InspectionResult" not in text
