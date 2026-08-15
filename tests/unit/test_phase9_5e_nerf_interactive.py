from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from app.reconstruction_nerf_interactive import (
    PHASE9_5E_DEFAULT_RENDER_HEIGHT,
    PHASE9_5E_DEFAULT_RENDER_WIDTH,
    build_interactive_nerf_presentation,
    verify_interactive_nerf_integrity,
    turntable_state_to_nerf_render_request,
)
from core.reconstruction import persist_nerf_representation
from core.reconstruction.errors import ReconstructionValidationError
from tests.phase9_5d_helpers import make_phase9_5d_source


def _persist(tmp_path: Path):
    scene, artifact, preview, runtime, manifest, summary, console = make_phase9_5d_source(tmp_path)
    result = persist_nerf_representation(
        base_scene=scene,
        nerf_artifact=artifact,
        preview_path=preview,
        runtime_evidence_path=runtime,
        source_optimization_manifest_path=manifest,
        source_optimization_summary_path=summary,
        source_worker_console_path=console,
        expected_dataset_fingerprint_sha256=artifact.dataset_fingerprint_sha256,
        output_dir=(tmp_path / "persistent").resolve(),
    )
    return result


def test_5e_builds_ready_interactive_presentation_from_9_5d_scene(tmp_path: Path) -> None:
    result = _persist(tmp_path)
    presentation = build_interactive_nerf_presentation(result.scene)
    assert presentation is not None
    assert presentation.ready is True
    assert presentation.scene_id == result.scene.scene_id
    assert presentation.artifact_id == result.scene.nerf_artifact.artifact_id
    assert Path(presentation.checkpoint_path).is_file()
    assert Path(presentation.immutable_config_path).is_file()
    assert Path(presentation.dataset_root).is_dir()
    assert Path(presentation.transforms_path).is_file()
    assert presentation.scene_role == "appearance_only"
    assert presentation.canonical_geometry_authoritative is False
    assert presentation.metric_scale_authoritative is False
    assert presentation.measurement_authoritative is False
    assert presentation.qc_authoritative is False
    assert presentation.certified_metrology is False


def test_5e_turntable_request_is_finite_right_handed_opengl_c2w() -> None:
    request = turntable_state_to_nerf_render_request(
        request_id=7,
        center_viewer=(0.1, -0.2, 0.3),
        distance=3.0,
        azimuth_degrees=32.0,
        elevation_degrees=24.0,
        fov_degrees=45.0,
        scene_id="prs-5e",
        artifact_id="nerf-5e",
    )
    matrix = np.asarray(request.camera_to_world_gl, dtype=np.float64)
    rotation = matrix[:3, :3]
    assert matrix.shape == (4, 4)
    assert np.isfinite(matrix).all()
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-7)
    assert request.width == PHASE9_5E_DEFAULT_RENDER_WIDTH
    assert request.height == PHASE9_5E_DEFAULT_RENDER_HEIGHT
    assert request.fx > 0.0
    payload = request.as_protocol_payload()
    assert payload["scene_role"] == "appearance_only"
    assert payload["canonical_geometry_authoritative"] is False
    assert payload["metric_scale_authoritative"] is False
    assert payload["measurement_authoritative"] is False
    assert payload["qc_authoritative"] is False
    assert payload["certified_metrology"] is False


def test_5e_camera_mapping_is_presentation_only_and_preserves_center_target() -> None:
    request = turntable_state_to_nerf_render_request(
        request_id=1,
        center_viewer=(0.0, 0.0, 0.0),
        distance=2.0,
        azimuth_degrees=0.0,
        elevation_degrees=0.0,
        fov_degrees=60.0,
        scene_id="scene",
        artifact_id="artifact",
    )
    matrix = np.asarray(request.camera_to_world_gl)
    # VisPy TurntableCamera azimuth=elevation=0 places the viewer camera on
    # -Y. The viewer->reconstruction axis map sends viewer -Y to reconstruction -Z.
    assert matrix[0, 3] == pytest.approx(0.0)
    assert matrix[1, 3] == pytest.approx(0.0)
    assert matrix[2, 3] == pytest.approx(-2.0)
    # OpenGL camera -Z is the forward direction; it points back to the target.
    forward = -matrix[:3, 2]
    assert np.allclose(forward, [0.0, 0.0, 1.0], atol=1e-7)


def test_5e_rejects_authority_promotion_in_runtime_load_spec(tmp_path: Path) -> None:
    result = _persist(tmp_path)
    spec_path = Path(result.runtime_load_spec_path)
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    payload["measurement_authoritative"] = True
    spec_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ReconstructionValidationError, match="authority"):
        build_interactive_nerf_presentation(result.scene)


def test_5e_rejects_wrong_persisted_load_strategy(tmp_path: Path) -> None:
    result = _persist(tmp_path)
    spec_path = Path(result.runtime_load_spec_path)
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    payload["load_strategy"] = "rewrite-qualified-training-config"
    spec_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ReconstructionValidationError, match="load strategy"):
        build_interactive_nerf_presentation(result.scene)


def test_5e_rejects_missing_durable_checkpoint(tmp_path: Path) -> None:
    result = _persist(tmp_path)
    Path(result.artifact.checkpoint_path).unlink()
    with pytest.raises(ReconstructionValidationError, match="checkpoint"):
        build_interactive_nerf_presentation(result.scene)


def test_5e_integrity_gate_rejects_same_size_checkpoint_tamper(tmp_path: Path) -> None:
    result = _persist(tmp_path)
    presentation = build_interactive_nerf_presentation(result.scene)
    assert presentation is not None
    checkpoint = Path(presentation.checkpoint_path)
    original = checkpoint.read_bytes()
    checkpoint.write_bytes(bytes((value ^ 0x01) for value in original))
    assert checkpoint.stat().st_size == len(original)
    with pytest.raises(ReconstructionValidationError, match="checksum"):
        verify_interactive_nerf_integrity(result.scene, presentation)


def test_5e_camera_state_api_belongs_to_reconstruction_canvas() -> None:
    canvas_path = Path(__file__).resolve().parents[2] / "app/widgets/viewport/vispy_canvas.py"
    tree = ast.parse(canvas_path.read_text(encoding="utf-8"))
    classes = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    reconstruction = classes["ReconstructionCanvas"]
    point_cloud = classes["PointCloudCanvas"]
    reconstruction_methods = {
        node.name for node in reconstruction.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    point_cloud_methods = {
        node.name for node in point_cloud.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "interactive_render_camera_state" in reconstruction_methods
    assert "interactive_render_camera_state" not in point_cloud_methods
