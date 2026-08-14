from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt

from app.reconstruction_canonical_cameras import build_canonical_camera_trajectory_snapshot
from app.main_window import MainWindow
from core.reconstruction import persistent_scene_from_pipeline_result, write_persistent_reconstruction_scene
from core.reconstruction.orchestration import ReconstructionArtifactKind
from app.widgets.mode_tabs import AppMode
from tests.reconstruction_f5_helpers import run_f5_fixture


def _matrices(count: int = 6) -> np.ndarray:
    poses = []
    for index in range(count):
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = np.asarray(
            [-1.5 + index * 0.6, 0.08 * (index % 2), -1.1 + 0.24 * index],
            dtype=np.float64,
        )
        poses.append(matrix)
    return np.asarray(poses, dtype=np.float64)


def _scene(tmp_path: Path, *, cameras: bool = True, bad_convention: bool = False, request_id: str = "phase9-3c-ui"):
    _fixture, _providers, result, snapshots = run_f5_fixture(tmp_path / request_id, request_id=request_id)
    base = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)
    camera_artifact = next(
        item for item in base.artifacts if item.kind is ReconstructionArtifactKind.CAMERA_TRAJECTORY
    )
    surface = (tmp_path / f"{request_id}-surface.obj").resolve()
    surface.write_text(
        "v -1 -1 -0.5 1 0 0\n"
        "v 1 -1 -0.5 0 1 0\n"
        "v 1 1 -0.5 0 0 1\n"
        "v -1 1 -0.5 1 1 0\n"
        "v -1 -1 0.5 1 0 1\n"
        "v 1 -1 0.5 0 1 1\n"
        "v 1 1 0.5 1 1 1\n"
        "v -1 1 0.5 0.5 0.5 0.5\n"
        "f 1 2 3\nf 1 3 4\nf 5 7 6\nf 5 8 7\n"
        "f 1 5 6\nf 1 6 2\nf 2 6 7\nf 2 7 3\n"
        "f 3 7 8\nf 3 8 4\nf 4 8 5\nf 4 5 1\n",
        encoding="utf-8",
    )
    updated = dict(snapshots)
    updated["surface_mesh"] = {
        "surface_obj_path": str(surface),
        "surface_vertex_count": 8,
        "surface_face_count": 12,
        "metric_geometry_status": "arbitrary",
        "canonical_geometry_authoritative": True,
        "certified_metrology": False,
    }
    if cameras:
        matrices = _matrices()
        snapshot = build_canonical_camera_trajectory_snapshot(
            matrices_reconstruction=matrices,
            camera_ids=tuple(f"cam-{index:03d}" for index in range(matrices.shape[0])),
            pose_layers=("optimized",) * matrices.shape[0],
            provider_id=camera_artifact.provider_id,
            reconstruction_frame_id="reconstruction-world",
        )
        if bad_convention:
            snapshot["pose_convention"] = "T_camera_from_reconstruction"
        updated["camera_trajectory"] = snapshot
    return persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=updated)


def _window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    window = MainWindow()
    qtbot.addWidget(window)
    return window


def test_3c_persistent_scene_renders_world_frame_camera_frustums_and_trajectory(qtbot, monkeypatch, tmp_path: Path) -> None:
    window = _window(qtbot, monkeypatch)
    page = window._reconstruction_page
    scene = _scene(tmp_path)

    page.set_persistent_scene(scene, "persistent_reconstruction_scene.json")

    canvas = page._reconstruction_canvas
    assert canvas.canonical_mesh_loaded is True
    assert canvas.canonical_camera_loaded is True
    assert canvas.canonical_camera_scene_id == scene.scene_id
    assert canvas.canonical_camera_persisted_count == 6
    assert canvas.canonical_camera_count == 6
    assert canvas.canonical_camera_frustum_display_count == 6
    assert canvas.canonical_camera_trajectory_display_count == 6
    assert canvas.canonical_camera_frustums_visible is True
    assert canvas.canonical_camera_trajectory_visible is True
    assert page.canonical_camera_presentation is not None
    assert page.canonical_camera_presentation.certified_metrology is False
    assert "Canonical reconstruction origin" in page._canonical_scene_world_value.text()
    assert "6 persisted poses" in page._canonical_scene_cameras_value.text()
    assert "frustums + trajectory" in page._canonical_scene_cameras_value.text()
    assert page._viewport_state.text() == "Canonical scene • cameras + trajectory"
    assert page._viewport_safety.text() == "VIEWER Z-UP • ARBITRARY GAUGE"
    assert canvas.fit_calls >= 2
    window.close()


def test_3c_legacy_scene_does_not_invent_camera_poses(qtbot, monkeypatch, tmp_path: Path) -> None:
    window = _window(qtbot, monkeypatch)
    page = window._reconstruction_page
    scene = _scene(tmp_path, cameras=False, request_id="phase9-3c-legacy")

    # Switching from a scene that had camera context must not leave stale poses
    # influencing the legacy scene's viewer state or Fit behavior.
    page.set_persistent_scene(_scene(tmp_path, request_id="phase9-3c-before-legacy"), "before.json")
    assert page._reconstruction_canvas.canonical_camera_loaded is True
    page.set_persistent_scene(scene, "scene.json")

    assert page._reconstruction_canvas.canonical_mesh_loaded is True
    assert page._reconstruction_canvas.canonical_camera_loaded is False
    assert page.canonical_camera_presentation is None
    assert page._canonical_scene_cameras_value.text() == "Camera trajectory not persisted"
    assert page._viewport_state.text() == "Canonical surface mesh"
    assert scene.canonical_geometry_authoritative is True
    assert scene.certified_metrology is False
    window.close()


def test_3c_invalid_persisted_camera_context_fails_closed_without_revoking_geometry(qtbot, monkeypatch, tmp_path: Path) -> None:
    window = _window(qtbot, monkeypatch)
    page = window._reconstruction_page
    good = _scene(tmp_path, request_id="phase9-3c-good")
    bad = _scene(tmp_path, bad_convention=True, request_id="phase9-3c-bad")

    page.set_persistent_scene(good, "good.json")
    assert page._reconstruction_canvas.canonical_camera_loaded is True

    page.set_persistent_scene(bad, "bad.json")

    assert page._reconstruction_canvas.canonical_mesh_loaded is True
    assert page._reconstruction_canvas.canonical_camera_loaded is False
    assert page.canonical_camera_presentation is None
    assert page._canonical_scene_cameras_value.text() == "Persistent camera context rejected"
    assert "pose_convention" in page._canonical_scene_cameras_value.toolTip()
    assert bad.canonical_geometry_authoritative is True
    assert bad.certified_metrology is False
    window.close()


def test_3c_existing_trajectory_and_camera_frustum_controls_apply_to_canonical_context(qtbot, monkeypatch, tmp_path: Path) -> None:
    window = _window(qtbot, monkeypatch)
    page = window._reconstruction_page
    page.set_persistent_scene(_scene(tmp_path), "scene.json")
    canvas = page._reconstruction_canvas

    trajectory = page._viewport_tool_buttons["trajectory"]
    frustums = page._viewport_tool_buttons["keyframes"]
    trajectory.setChecked(False)
    frustums.setChecked(False)
    assert canvas.canonical_camera_trajectory_visible is False
    assert canvas.canonical_camera_frustums_visible is False

    trajectory.setChecked(True)
    frustums.setChecked(True)
    assert canvas.canonical_camera_trajectory_visible is True
    assert canvas.canonical_camera_frustums_visible is True
    assert "camera trajectory" in trajectory.toolTip().lower()
    assert "camera frustums" in frustums.toolTip().lower()
    window.close()


def test_3c_controller_load_reload_restores_persisted_canonical_cameras(qtbot, monkeypatch, tmp_path: Path) -> None:
    window = _window(qtbot, monkeypatch)
    controller = window._reconstruction_controller
    page = window._reconstruction_page
    scene = _scene(tmp_path)
    manifest = write_persistent_reconstruction_scene(scene, (tmp_path / "persistent").resolve())

    loaded = controller.load_persistent_scene(manifest, verify_file_references=True)
    assert loaded.scene_id == scene.scene_id
    assert page._reconstruction_canvas.canonical_camera_loaded is True
    first_ids = page._reconstruction_canvas.canonical_camera_ids

    reloaded = controller.reload_persistent_scene()
    assert reloaded.scene_fingerprint_sha256 == scene.scene_fingerprint_sha256
    assert page._reconstruction_canvas.canonical_camera_loaded is True
    assert page._reconstruction_canvas.canonical_camera_ids == first_ids
    assert page._reconstruction_canvas.canonical_mesh_loaded is True
    window.close()


def test_3c_clear_shutdown_and_1366_layout_release_camera_presentation(qtbot, monkeypatch, tmp_path: Path) -> None:
    window = _window(qtbot, monkeypatch)
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    window.resize(1366, 768)
    window.show()
    qtbot.wait(20)
    page = window._reconstruction_page
    scene = _scene(tmp_path)
    page.set_persistent_scene(scene, "scene.json")

    assert page._settings_scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert page._settings_scroll.horizontalScrollBar().maximum() == 0
    assert page._reconstruction_canvas.canonical_camera_loaded is True

    page.set_persistent_scene(None, "")
    assert page.canonical_camera_presentation is None
    assert page._reconstruction_canvas.canonical_camera_loaded is False
    assert page._canonical_scene_world_value.text() == "No canonical scene"

    page.set_persistent_scene(scene, "scene.json")
    assert page._reconstruction_canvas.canonical_camera_loaded is True
    page.shutdown_viewport()
    assert page.canonical_camera_presentation is None
    assert page._reconstruction_canvas.canonical_camera_loaded is False
    assert page._reconstruction_canvas.closed is True
    window.close()
