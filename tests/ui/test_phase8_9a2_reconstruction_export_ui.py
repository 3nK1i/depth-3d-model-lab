from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFileDialog, QSizePolicy

from app.main_window import MainWindow
from app.offline_reconstruction_controller import OfflineReconstructionRunState
from app.widgets.mode_tabs import AppMode
from core.reconstruction import (
    CanonicalSurfaceMeshExportSource,
    ReconstructionArtifactKind,
    SurfaceMeshResult,
    verify_reconstruction_export_bundle,
)
from tests.reconstruction_f3_helpers import make_f3_fixture


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    win._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    yield win
    win.close()


def _sparse_payload(root: Path, *, session_id: str | None = None) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    sid = session_id or str(uuid4())
    sparse = root / "sparse_reconstruction.ply"
    sparse.write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n0 0 1\n",
        encoding="utf-8",
    )
    sfm = root / "sfm_result.json"
    sfm.write_text(
        json.dumps(
            {
                "session_id": sid,
                "backend_id": "phase8-9a2-ui-fixture",
                "backend_version": "1",
                "gauge": {"scale_status": "arbitrary", "unit_to_metre": None},
            }
        ),
        encoding="utf-8",
    )
    return {
        "session_id": sid,
        "status": "completed",
        "pass": True,
        "camera_pose_available": True,
        "canonical_geometry_authoritative": False,
        "certified_metrology": False,
        "artifacts": {
            "sparse_reconstruction_ply": str(sparse.resolve()),
            "sfm_result_json": str(sfm.resolve()),
        },
        "sfm_summary": {
            "registered_keyframes": 1,
            "input_keyframes": 1,
            "landmarks": 1,
            "mean_reprojection_error_px": 0.25,
        },
    }


def _canonical_source() -> CanonicalSurfaceMeshExportSource:
    fixture = make_f3_fixture(request_id="phase8-9a2-ui-canonical")
    result = fixture.orchestrator.run()
    surface_artifact = next(
        stage.artifact
        for stage in result.stages
        if stage.artifact is not None
        and stage.artifact.kind is ReconstructionArtifactKind.SURFACE_MESH
    )
    surface = fixture.store.get(surface_artifact.payload_ref, SurfaceMeshResult)
    return CanonicalSurfaceMeshExportSource(
        artifact=surface_artifact,
        result=surface,
        canonical_geometry_artifact_id=result.canonical_geometry_artifact_id,
    )


def test_a2_idle_ui_starts_without_false_export_availability(window) -> None:
    page = window._reconstruction_page
    assert page._reconstruction_export_state_value.text() == "No materialized export evidence"
    assert not page._reconstruction_export_button.isEnabled()
    assert not page._reconstruction_export_open_button.isEnabled()
    assert "separate from Save Scene" in page._reconstruction_export_button.toolTip()


def test_a2_completed_sparse_result_enables_truthful_arbitrary_scale_export(window, tmp_path: Path) -> None:
    page = window._reconstruction_page
    page._on_offline_result_ready(_sparse_payload(tmp_path / "worker"), str(tmp_path / "worker"))
    assert page._reconstruction_export_button.isEnabled()
    assert page._reconstruction_export_state_value.text() == "Sparse SfM ready • arbitrary scale"
    tooltip = page._reconstruction_export_state_value.toolTip().lower()
    assert "supporting geometry" in tooltip
    assert "not canonical geometry" in tooltip
    assert "certified metrology" in tooltip


def test_a2_sparse_export_creates_verified_non_destructive_bundle(
    window, qtbot, monkeypatch, tmp_path: Path
) -> None:
    page = window._reconstruction_page
    payload = _sparse_payload(tmp_path / "worker")
    page._on_offline_result_ready(payload, str(tmp_path / "worker"))
    parent = (tmp_path / "exports").resolve()
    parent.mkdir()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(parent))

    page._reconstruction_export_button.click()
    qtbot.wait(10)
    first = page.last_reconstruction_export_dir
    assert first is not None and first.is_dir()
    files = verify_reconstruction_export_bundle(first)
    assert {item.role for item in files} == {"sparse_point_cloud", "sparse_sfm_result"}
    assert all(not item.canonical_geometry_authoritative for item in files)
    assert all(not item.certified_metrology for item in files)
    assert page._reconstruction_export_open_button.isEnabled()
    assert page._reconstruction_export_state_value.text() == "Export complete • 2 files"

    # A1 rejects replacement. A2 therefore allocates a fresh sibling rather than deleting/replacing.
    page._reconstruction_export_button.click()
    qtbot.wait(10)
    second = page.last_reconstruction_export_dir
    assert second is not None and second.is_dir() and second != first
    assert first.is_dir()
    assert second.name.endswith("_02")


def test_a2_canonical_export_requires_explicit_materialized_a1_source(
    window, qtbot, monkeypatch, tmp_path: Path
) -> None:
    page = window._reconstruction_page
    source = _canonical_source()
    page.set_reconstruction_canonical_surface_export_source(source)
    assert page._reconstruction_export_button.isEnabled()
    assert page._reconstruction_export_state_value.text() == "Canonical mesh ready"

    parent = (tmp_path / "canonical_exports").resolve()
    parent.mkdir()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(parent))
    page._reconstruction_export_button.click()
    qtbot.wait(10)
    output = page.last_reconstruction_export_dir
    assert output is not None
    files = verify_reconstruction_export_bundle(output)
    assert len(files) == 1
    assert files[0].role == "canonical_surface_mesh"
    assert files[0].canonical_geometry_authoritative
    assert not files[0].certified_metrology


def test_a2_loaded_persistent_scene_does_not_manufacture_mesh_payload(window, qtbot, tmp_path: Path) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    fixture = make_f3_fixture(request_id="phase8-9a2-ui-persistent")
    result = fixture.orchestrator.run()
    scene = controller.adopt_pipeline_result(result, (tmp_path / "scene").resolve())
    qtbot.wait(10)
    assert scene.canonical_geometry_authoritative
    assert not scene.certified_metrology
    assert page._canonical_surface_export_source is None
    assert not page._reconstruction_export_button.isEnabled()
    assert page._reconstruction_export_state_value.text() == "No materialized export evidence"


def test_a2_active_runtime_state_blocks_export(window, tmp_path: Path) -> None:
    page = window._reconstruction_page
    page._on_offline_result_ready(_sparse_payload(tmp_path / "worker"), str(tmp_path / "worker"))
    assert page._reconstruction_export_button.isEnabled()
    page.set_offline_state(OfflineReconstructionRunState.RUNNING)
    assert not page._reconstruction_export_button.isEnabled()
    assert page._reconstruction_export_state_value.text() == "Stop reconstruction before exporting"


def test_a2_export_controls_preserve_1366_vertical_only_sidebar(window, qtbot, tmp_path: Path) -> None:
    page = window._reconstruction_page
    page._on_offline_result_ready(_sparse_payload(tmp_path / "worker"), str(tmp_path / "worker"))
    window.resize(1366, 768)
    window.show()
    qtbot.wait(20)
    assert page._settings_scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert page._settings_scroll.horizontalScrollBar().maximum() == 0
    assert page._settings_scroll.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    for button in (page._reconstruction_export_button, page._reconstruction_export_open_button):
        assert button.minimumWidth() == 0
        assert button.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Ignored
