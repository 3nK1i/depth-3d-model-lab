from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from app.reconstruction_canonical_geometry import load_canonical_viewer_mesh
from app.reconstruction_canonical_semantics import load_persistent_canonical_semantics
from core.contracts.validation import metadata_as_dict
from tests.phase9_3d_helpers import make_phase9_3d_scene

ROOT = Path(__file__).resolve().parents[2]


def test_3d_semantic_serialization_preserves_exact_existing_surface_membership(tmp_path: Path) -> None:
    scene, semantic, _surface = make_phase9_3d_scene(tmp_path)
    semantic_artifact = next(item for item in scene.artifacts if item.kind.value == "semantic_geometry")
    payload = metadata_as_dict(semantic_artifact.payload_snapshot)

    assert len(payload["objects"]) == len(semantic.objects) == 3
    for persisted, runtime in zip(payload["objects"], semantic.objects, strict=True):
        assert persisted["vertex_indices"] == runtime.vertex_indices.tolist()
        assert persisted["face_indices"] == runtime.face_indices.tolist()
        assert persisted["vertex_count"] == len(runtime.vertex_indices)
        assert persisted["face_count"] == len(runtime.face_indices)
        assert persisted["certified_metrology"] is False


def test_3d_binds_persisted_semantic_membership_to_loaded_canonical_mesh(tmp_path: Path) -> None:
    scene, semantic, _surface = make_phase9_3d_scene(tmp_path)
    mesh = load_canonical_viewer_mesh(scene)
    assert mesh is not None

    presentation = load_persistent_canonical_semantics(scene, mesh)

    assert presentation is not None
    assert presentation.object_ids == tuple(item.object_id for item in semantic.objects)
    assert presentation.surface_bound_object_count == 3
    assert presentation.surface_highlight_object_count == 3
    assert presentation.certified_metrology is False
    assert "canonical geometry authority unchanged" in presentation.authority_text
    for bound, runtime in zip(presentation.objects, semantic.objects, strict=True):
        np.testing.assert_array_equal(bound.vertex_indices, runtime.vertex_indices)
        np.testing.assert_array_equal(bound.face_indices, runtime.face_indices)
        assert bound.surface_faces.shape[0] == len(runtime.face_indices)
        assert bound.surface_vertices_viewer.shape[1:] == (3,)
        assert bound.surface_faces.shape[1:] == (3,)


def test_3d_legacy_detailed_semantic_snapshot_remains_valid_without_inventing_membership(tmp_path: Path) -> None:
    scene, semantic, _surface = make_phase9_3d_scene(tmp_path, legacy_membership=True)
    mesh = load_canonical_viewer_mesh(scene)
    assert mesh is not None

    presentation = load_persistent_canonical_semantics(scene, mesh)

    assert presentation is not None
    assert len(presentation.objects) == len(semantic.objects)
    assert presentation.surface_bound_object_count == 0
    assert presentation.surface_highlight_object_count == 0
    assert all(item.vertex_indices.size == 0 for item in presentation.objects)
    assert all(item.face_indices.size == 0 for item in presentation.objects)
    assert "membership unavailable" in presentation.binding_text


def test_3d_metadata_only_semantics_do_not_fabricate_surface_membership(tmp_path: Path) -> None:
    scene, _semantic, _surface = make_phase9_3d_scene(
        tmp_path, include_semantic_snapshot=False
    )
    mesh = load_canonical_viewer_mesh(scene)
    assert mesh is not None

    assert load_persistent_canonical_semantics(scene, mesh) is None


def test_3d_rejects_out_of_range_persisted_face_membership_without_revoking_mesh(tmp_path: Path) -> None:
    def corrupt(payload, surface) -> None:
        payload["objects"][0]["face_indices"] = [surface.mesh.face_count]
        payload["objects"][0]["face_count"] = 1

    scene, _semantic, _surface = make_phase9_3d_scene(tmp_path, mutate_semantic=corrupt)
    mesh = load_canonical_viewer_mesh(scene)
    assert mesh is not None

    with pytest.raises(ValueError, match="face_indices exceed"):
        load_persistent_canonical_semantics(scene, mesh)
    assert scene.canonical_geometry_authoritative is True
    assert scene.certified_metrology is False


def test_3d_rejects_partial_membership_and_metrology_claims(tmp_path: Path) -> None:
    def partial(payload, _surface) -> None:
        payload["objects"][0].pop("face_indices")

    scene, _semantic, _surface = make_phase9_3d_scene(
        tmp_path / "partial", mutate_semantic=partial
    )
    mesh = load_canonical_viewer_mesh(scene)
    assert mesh is not None
    with pytest.raises(ValueError, match="persist vertex_indices and face_indices together"):
        load_persistent_canonical_semantics(scene, mesh)

    def metrology(payload, _surface) -> None:
        payload["certified_metrology"] = True

    scene, _semantic, _surface = make_phase9_3d_scene(
        tmp_path / "metrology", request_id="phase9-3d-metrology", mutate_semantic=metrology
    )
    mesh = load_canonical_viewer_mesh(scene)
    assert mesh is not None
    with pytest.raises(ValueError, match="certified metrology"):
        load_persistent_canonical_semantics(scene, mesh)


def test_3d_rejects_semantic_binding_to_different_scene_identity(tmp_path: Path) -> None:
    scene, _semantic, _surface = make_phase9_3d_scene(tmp_path)
    mesh = load_canonical_viewer_mesh(scene)
    assert mesh is not None
    wrong = replace(mesh, scene_id="different-scene")

    with pytest.raises(ValueError, match="scene identity"):
        load_persistent_canonical_semantics(scene, wrong)


def test_3d_loader_does_not_import_inference_or_heavy_backends() -> None:
    path = ROOT / "app/reconstruction_canonical_semantics.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    for forbidden in (
        "ultralytics",
        "sam2",
        "pycolmap",
        "gsplat",
        "nerfstudio",
        "depth_anything_3",
        "tinycudann",
    ):
        assert forbidden not in imported
    assert "YOLO(" not in source
    assert "SAM2" not in source
    assert "build_semantic_3d_geometry(" not in source
    assert "certified_metrology=False" in source


def test_3d_real_vispy_selected_surface_clear_never_submits_zero_vertex_mesh() -> None:
    path = ROOT / "app/widgets/viewport/vispy_canvas.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    clear = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_clear_selected_canonical_semantic_mesh"
    )
    set_data_calls = [
        node
        for node in ast.walk(clear)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "set_data"
    ]
    assert set_data_calls == []
    segment = ast.get_source_segment(source, clear) or ""
    assert "visible = False" in segment
    assert "parent = None" in segment
    assert "scene.visuals.Mesh(" in segment
    assert "_selected_canonical_semantic_surface_loaded = False" in segment
