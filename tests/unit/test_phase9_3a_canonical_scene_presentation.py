from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.reconstruction_canonical_scene_presentation import (
    CanonicalSceneResourceState,
    CanonicalSceneScaleState,
    build_canonical_scene_presentation,
)
from core.reconstruction import (
    MetricEvidenceValidationState,
    build_persistent_scene_authority_snapshot,
    known_size_reference_evidence,
    persistent_scene_from_pipeline_result,
    reference_target_metric_evidence,
)
from core.reconstruction.calibration_management import create_manual_camera_calibration_profile
from tests.reconstruction_f5_helpers import run_f5_fixture

ROOT = Path(__file__).resolve().parents[2]


def _profile():
    return create_manual_camera_calibration_profile(
        width=640,
        height=480,
        fx=520.0,
        fy=521.0,
        cx=320.0,
        cy=240.0,
        profile_id="phase9-3a-profile",
        calibration_record_id="phase9-3a-record",
        camera_id="phase9-3a-camera",
        provenance="phase9_3a_test",
    )


def _qualified_evidence():
    return reference_target_metric_evidence(
        evidence_id="phase9-3a-qualified-target",
        reconstruction_frame_id="reconstruction-world",
        reference_id="phase9-3a-target",
        target_length_m=0.25,
        measured_length_reconstruction_units=5.0,
        target_relative_uncertainty=0.005,
        measurement_relative_uncertainty=0.02,
        provenance="phase9_3a_test",
        validation_state=MetricEvidenceValidationState.VALIDATED,
        validation_record_id="phase9-3a-qualified-validation",
    )


def _provisional_evidence():
    return known_size_reference_evidence(
        evidence_id="phase9-3a-provisional-size",
        reconstruction_frame_id="reconstruction-world",
        reference_id="phase9-3a-size",
        known_length_m=0.50,
        measured_length_reconstruction_units=10.0,
        known_length_relative_uncertainty=0.01,
        measurement_relative_uncertainty=0.03,
        provenance="phase9_3a_test",
    )


def _scene(tmp_path: Path, *, request_id: str = "phase9-3a-scene", materialized: bool = True):
    _fixture, _providers, result, snapshots = run_f5_fixture(tmp_path / request_id, request_id=request_id)
    if materialized:
        surface = (tmp_path / request_id / "canonical_surface.obj").resolve()
        surface.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
        snapshots = dict(snapshots)
        snapshots["surface_mesh"] = {
            "surface_obj_path": str(surface),
            "metric_geometry_status": "arbitrary",
            "canonical_geometry_authoritative": True,
            "certified_metrology": False,
        }
    return persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)


def test_3a_empty_state_is_neutral_and_non_authoritative() -> None:
    presentation = build_canonical_scene_presentation(None)
    assert presentation.available is False
    assert presentation.canonical_geometry_authoritative is False
    assert presentation.resource_state is CanonicalSceneResourceState.NO_SCENE
    assert presentation.scale_state is CanonicalSceneScaleState.NO_SCENE
    assert presentation.certified_metrology is False
    assert presentation.scene_text == "No persistent scene"
    assert presentation.viewer_scale_banner_text == "VIEWER Z-UP • ARBITRARY GAUGE"


def test_3a_persistent_scene_exposes_canonical_authority_without_inventing_scale(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    presentation = build_canonical_scene_presentation(scene)
    assert presentation.scene_id == scene.scene_id
    assert presentation.artifact_id == scene.canonical_geometry_artifact_id
    assert presentation.canonical_geometry_authoritative is True
    assert presentation.resource_state is CanonicalSceneResourceState.MATERIALIZED
    assert presentation.viewer_resource_ready is True
    assert presentation.scale_state is CanonicalSceneScaleState.NOT_QUALIFIED
    assert presentation.calibration_available is False
    assert presentation.certified_metrology is False
    assert presentation.viewer_scale_banner_text == "VIEWER Z-UP • ARBITRARY GAUGE"
    assert "rendering deferred to 9.3B" in presentation.state_text


def test_3a_missing_surface_is_distinct_from_canonical_authority(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    reference = next(
        item for item in scene.file_references
        if item.artifact_id == scene.canonical_geometry_artifact_id and item.role == "surface_obj_path"
    )
    Path(reference.path).unlink()
    presentation = build_canonical_scene_presentation(scene)
    assert presentation.canonical_geometry_authoritative is True
    assert presentation.resource_state is CanonicalSceneResourceState.MISSING
    assert presentation.viewer_resource_ready is False
    assert presentation.resource_text == "Referenced surface missing"
    assert "viewer resource unavailable" in presentation.state_text


def test_3a_legacy_scene_without_surface_reference_remains_authoritative_but_not_viewer_ready(tmp_path: Path) -> None:
    scene = _scene(tmp_path, materialized=False)
    presentation = build_canonical_scene_presentation(scene)
    assert presentation.canonical_geometry_authoritative is True
    assert presentation.resource_state is CanonicalSceneResourceState.NOT_REFERENCED
    assert presentation.viewer_resource_ready is False
    assert presentation.scale_state is CanonicalSceneScaleState.NOT_QUALIFIED


def test_3a_degraded_scene_remains_explicit_without_losing_canonical_authority(tmp_path: Path) -> None:
    _fixture, _providers, result, snapshots = run_f5_fixture(
        tmp_path / "degraded",
        request_id="phase9-3a-degraded",
        degrade_gaussian=True,
    )
    surface = (tmp_path / "degraded_surface.obj").resolve()
    surface.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    snapshots = dict(snapshots)
    snapshots["surface_mesh"] = {
        "surface_obj_path": str(surface),
        "metric_geometry_status": "arbitrary",
        "canonical_geometry_authoritative": True,
        "certified_metrology": False,
    }
    scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)
    presentation = build_canonical_scene_presentation(scene)
    assert presentation.scene_state == "degraded"
    assert presentation.scene_text == "Degraded persistent scene"
    assert presentation.canonical_geometry_authoritative is True
    assert presentation.viewer_resource_ready is True
    assert presentation.scale_state is CanonicalSceneScaleState.NOT_QUALIFIED


def test_3a_validated_reference_target_exposes_qualified_metric_state_and_uncertainty(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    authority = build_persistent_scene_authority_snapshot(
        scene,
        camera_calibration_profile=_profile(),
        metric_qualification_evidence=_qualified_evidence(),
    )
    presentation = build_canonical_scene_presentation(scene, authority)
    assert presentation.calibration_available is True
    assert presentation.scale_state is CanonicalSceneScaleState.QUALIFIED
    assert presentation.scale_relative_uncertainty == pytest.approx((0.005**2 + 0.02**2) ** 0.5)
    assert presentation.scale_unit_to_metre == pytest.approx(0.05)
    assert presentation.metric_evidence_id == "phase9-3a-qualified-target"
    assert presentation.metric_assessment_state == "qualified"
    assert presentation.certified_metrology is False
    assert presentation.viewer_scale_banner_text == "VIEWER Z-UP • METRIC SCALE QUALIFIED"


def test_3a_known_size_evidence_is_provisional_not_qualified(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    authority = build_persistent_scene_authority_snapshot(
        scene,
        camera_calibration_profile=_profile(),
        metric_qualification_evidence=_provisional_evidence(),
    )
    presentation = build_canonical_scene_presentation(scene, authority)
    assert presentation.scale_state is CanonicalSceneScaleState.PROVISIONAL
    assert presentation.metric_assessment_state == "provisional"
    assert presentation.viewer_scale_banner_text == "VIEWER Z-UP • PROVISIONAL METRIC SCALE"
    assert presentation.certified_metrology is False


def test_3a_calibration_without_metric_evidence_never_establishes_scale(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    authority = build_persistent_scene_authority_snapshot(
        scene,
        camera_calibration_profile=_profile(),
    )
    presentation = build_canonical_scene_presentation(scene, authority)
    assert presentation.calibration_available is True
    assert presentation.scale_state is CanonicalSceneScaleState.NOT_QUALIFIED
    assert presentation.scale_relative_uncertainty is None
    assert presentation.scale_unit_to_metre is None
    assert presentation.viewer_scale_banner_text == "VIEWER Z-UP • ARBITRARY GAUGE"


def test_3a_rejects_authority_sidecar_bound_to_another_scene(tmp_path: Path) -> None:
    scene_a = _scene(tmp_path, request_id="phase9-3a-a")
    scene_b = _scene(tmp_path, request_id="phase9-3a-b")
    authority_b = build_persistent_scene_authority_snapshot(
        scene_b,
        camera_calibration_profile=_profile(),
        metric_qualification_evidence=_qualified_evidence(),
    )
    with pytest.raises(ValueError, match="not bound"):
        build_canonical_scene_presentation(scene_a, authority_b)


def test_3a_presentation_module_has_no_heavy_runtime_or_mesh_parser_imports() -> None:
    path = ROOT / "app/reconstruction_canonical_scene_presentation.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    for forbidden in (
        "pycolmap",
        "vispy",
        "trimesh",
        "open3d",
        "gsplat",
        "nerfstudio",
        "depth_anything_3",
        "tinycudann",
    ):
        assert forbidden not in imported
    source = path.read_text(encoding="utf-8")
    assert "surface_obj_path" in source
    assert "open(" not in source
    assert "read_text(" not in source
    assert "read_bytes(" not in source
