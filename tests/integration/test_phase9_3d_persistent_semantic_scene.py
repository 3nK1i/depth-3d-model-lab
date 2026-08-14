from __future__ import annotations

from pathlib import Path

import numpy as np

from app.reconstruction_canonical_geometry import load_canonical_viewer_mesh
from app.reconstruction_canonical_semantics import load_persistent_canonical_semantics
from core.reconstruction import (
    read_persistent_reconstruction_scene,
    verify_persistent_scene_file_references,
    write_persistent_reconstruction_scene,
)
from tests.phase9_3d_helpers import make_phase9_3d_scene


def test_3d_persistent_roundtrip_preserves_semantic_surface_membership(tmp_path: Path) -> None:
    scene, semantic, _surface = make_phase9_3d_scene(tmp_path / "source")
    manifest = write_persistent_reconstruction_scene(scene, (tmp_path / "persistent").resolve())
    loaded = read_persistent_reconstruction_scene(manifest)
    verify_persistent_scene_file_references(loaded)

    mesh = load_canonical_viewer_mesh(loaded)
    assert mesh is not None
    presentation = load_persistent_canonical_semantics(loaded, mesh)
    assert presentation is not None
    assert presentation.object_ids == tuple(item.object_id for item in semantic.objects)
    for bound, runtime in zip(presentation.objects, semantic.objects, strict=True):
        np.testing.assert_array_equal(bound.vertex_indices, runtime.vertex_indices)
        np.testing.assert_array_equal(bound.face_indices, runtime.face_indices)
    assert loaded.canonical_geometry_authoritative is True
    assert loaded.certified_metrology is False


def test_3d_reloaded_scene_produces_deterministic_semantic_bindings(tmp_path: Path) -> None:
    scene, _semantic, _surface = make_phase9_3d_scene(tmp_path / "source")
    manifest = write_persistent_reconstruction_scene(scene, (tmp_path / "persistent").resolve())
    first = read_persistent_reconstruction_scene(manifest)
    second = read_persistent_reconstruction_scene(manifest)
    first_mesh = load_canonical_viewer_mesh(first)
    second_mesh = load_canonical_viewer_mesh(second)
    assert first_mesh is not None and second_mesh is not None
    first_semantics = load_persistent_canonical_semantics(first, first_mesh)
    second_semantics = load_persistent_canonical_semantics(second, second_mesh)
    assert first_semantics is not None and second_semantics is not None

    assert first.scene_fingerprint_sha256 == second.scene_fingerprint_sha256
    assert first_semantics.object_ids == second_semantics.object_ids
    for left, right in zip(first_semantics.objects, second_semantics.objects, strict=True):
        np.testing.assert_array_equal(left.vertex_indices, right.vertex_indices)
        np.testing.assert_array_equal(left.face_indices, right.face_indices)
        np.testing.assert_array_equal(left.surface_vertices_viewer, right.surface_vertices_viewer)
        np.testing.assert_array_equal(left.surface_faces, right.surface_faces)


def test_3d_legacy_roundtrip_keeps_semantic_context_without_surface_highlight(tmp_path: Path) -> None:
    scene, semantic, _surface = make_phase9_3d_scene(
        tmp_path / "source", legacy_membership=True
    )
    manifest = write_persistent_reconstruction_scene(scene, (tmp_path / "persistent").resolve())
    loaded = read_persistent_reconstruction_scene(manifest)
    mesh = load_canonical_viewer_mesh(loaded)
    assert mesh is not None
    presentation = load_persistent_canonical_semantics(loaded, mesh)

    assert presentation is not None
    assert len(presentation.objects) == len(semantic.objects)
    assert presentation.surface_bound_object_count == 0
    assert presentation.surface_highlight_object_count == 0
    assert loaded.canonical_geometry_authoritative is True
