from __future__ import annotations

import pytest

from app.reconstruction_scene_runtime_health import (
    ReconstructionSceneRuntimeHealth,
    build_reconstruction_scene_bind_plan,
    build_reconstruction_scene_runtime_health,
)


def test_3i_bind_plan_preserves_view_and_selection_only_for_exact_same_scene() -> None:
    same = build_reconstruction_scene_bind_plan("scene-a", "scene-a", selected_object_id="obj-7")
    assert same.same_scene_reload is True
    assert same.preserve_view is True
    assert same.fit_after_bind is False
    assert same.preserve_selected_object_id == "obj-7"

    changed = build_reconstruction_scene_bind_plan("scene-a", "scene-b", selected_object_id="obj-7")
    assert changed.same_scene_reload is False
    assert changed.preserve_view is False
    assert changed.fit_after_bind is True
    assert changed.preserve_selected_object_id == ""

    cleared = build_reconstruction_scene_bind_plan("scene-a", "", selected_object_id="obj-7")
    assert cleared.fit_after_bind is False
    assert cleared.preserve_selected_object_id == ""


def test_3i_runtime_health_is_diagnostics_only_and_formats_bounded_resource_state() -> None:
    health = build_reconstruction_scene_runtime_health(
        scene_id="scene-a",
        bind_latency_ms=12.5,
        bind_count=3,
        same_scene_reload_count=2,
        canonical_vertex_count=1189,
        canonical_face_count=8190,
        canonical_camera_count=6,
        semantic_object_count=3,
        mesh_visual_reused=True,
        view_preserved=True,
        selection_preserved=True,
        canonical_geometry_authoritative=True,
    )
    assert "12.5 ms UI bind" in health.bind_text
    assert "1,189 vertices" in health.resources_text
    assert "6 cameras" in health.resources_text
    assert "Same-scene reload" in health.lifecycle_text
    assert "mesh reused" in health.lifecycle_text
    assert "selection preserved" in health.lifecycle_text
    assert health.certified_metrology is False
    assert health.as_dict()["diagnostics_only"] is True
    assert health.as_dict()["canonical_geometry_authoritative"] is True


def test_3i_runtime_health_rejects_impossible_or_metrology_promoting_state() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        ReconstructionSceneRuntimeHealth(
            scene_id="scene-a",
            bind_latency_ms=1.0,
            bind_count=1,
            same_scene_reload_count=2,
            canonical_vertex_count=0,
            canonical_face_count=0,
            canonical_camera_count=0,
            semantic_object_count=0,
            mesh_visual_reused=False,
            view_preserved=False,
            selection_preserved=False,
            canonical_geometry_authoritative=True,
            certified_metrology=False,
        )
    with pytest.raises(ValueError, match="certified metrology"):
        ReconstructionSceneRuntimeHealth(
            scene_id="scene-a",
            bind_latency_ms=1.0,
            bind_count=1,
            same_scene_reload_count=0,
            canonical_vertex_count=3,
            canonical_face_count=1,
            canonical_camera_count=0,
            semantic_object_count=0,
            mesh_visual_reused=False,
            view_preserved=False,
            selection_preserved=False,
            canonical_geometry_authoritative=True,
            certified_metrology=True,
        )


def test_3i_empty_health_remains_safe_without_calibration_or_scene() -> None:
    health = build_reconstruction_scene_runtime_health(
        scene_id="",
        bind_latency_ms=0.0,
        bind_count=0,
        same_scene_reload_count=0,
        canonical_vertex_count=0,
        canonical_face_count=0,
        canonical_camera_count=0,
        semantic_object_count=0,
        mesh_visual_reused=False,
        view_preserved=False,
        selection_preserved=False,
        canonical_geometry_authoritative=False,
    )
    assert health.bind_text == "No persistent scene"
    assert "No canonical viewer resources" in health.resources_text
    assert health.certified_metrology is False
