from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")
PIL = pytest.importorskip("PIL.Image")

from PyQt6.QtWidgets import QApplication

from app.application_controller import CameraEnumerator
from app.main_window import MainWindow
from app.widgets.mode_tabs import AppMode
from core.reconstruction import (
    GaussianSceneArtifact,
    NerfSceneArtifact,
    persistent_scene_from_pipeline_result,
    write_persistent_reconstruction_scene,
)
from tests.reconstruction_f5_helpers import run_f5_fixture


def _preview(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = PIL.new("RGB", (320, 180), (30, 40, 55))
    image.save(path)


def _activate(page, label: str, app: QApplication) -> None:
    index = page._appearance_representation_combo.findText(label)
    assert index >= 0
    page._appearance_representation_combo.setCurrentIndex(index)
    page._appearance_representation_combo.activated.emit(index)
    app.processEvents()


def test_f6_actual_mainwindow_load_reload_and_switch_contract(
    tmp_path: Path, monkeypatch, qapp, qtbot
) -> None:
    fixture, _providers, result, snapshots = run_f5_fixture(tmp_path / "fixture", request_id="phase8-9f6-ui")
    gaussian = fixture.store.find_unique(GaussianSceneArtifact)
    nerf = fixture.store.find_unique(NerfSceneArtifact)
    gaussian_preview = Path(gaussian.artifact_dir) / "preview.png"
    nerf_preview = Path(nerf.artifact_dir) / "preview.png"
    _preview(gaussian_preview)
    _preview(nerf_preview)
    (Path(gaussian.artifact_dir) / "qualification_metrics.json").write_text(
        json.dumps({"render_written": True, "preview_path": str(gaussian_preview.resolve())}),
        encoding="utf-8",
    )
    (Path(nerf.artifact_dir) / "e2_runtime_evidence.json").write_text(
        json.dumps({"preview_path": str(nerf_preview.resolve())}), encoding="utf-8"
    )
    scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)
    manifest = write_persistent_reconstruction_scene(scene, tmp_path / "scene")

    monkeypatch.setattr(CameraEnumerator, "start", lambda self: None)
    app = qapp
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1366, 768)
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    page = window._reconstruction_page
    controller = window._reconstruction_controller

    loaded = controller.load_persistent_scene(manifest, verify_file_references=True)
    app.processEvents()
    assert loaded.scene_id == scene.scene_id
    assert page.active_view_representation == "canonical_geometry"

    _activate(page, "Gaussian", app)
    assert page.gaussian_preview_visible
    assert page.active_view_representation == "gaussian_scene"

    _activate(page, "NeRF", app)
    assert page.nerf_preview_visible
    assert not page.gaussian_preview_visible
    assert page.active_view_representation == "nerf_scene"

    reloaded = controller.reload_persistent_scene()
    app.processEvents()
    assert reloaded.scene_id == scene.scene_id
    assert reloaded.scene_fingerprint_sha256 == scene.scene_fingerprint_sha256
    assert page.active_view_representation == "canonical_geometry"
    assert page._appearance_authority_value.text() == "Appearance only • canonical geometry remains authoritative"
    assert reloaded.certified_metrology is False

    window.close()
    app.processEvents()
