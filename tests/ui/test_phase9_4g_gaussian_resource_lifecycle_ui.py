from __future__ import annotations

from pathlib import Path

import pytest

from app.main_window import MainWindow
from tests.phase9_4e_helpers import make_phase9_4e_persistent_scene


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    yield win
    win.close()


def _page_with_scene(window, tmp_path: Path):
    scene, manifest, _artifact = make_phase9_4e_persistent_scene(tmp_path)
    page = window._reconstruction_page
    page.set_persistent_scene(scene, str(manifest))
    return page


def test_94g_hide_releases_source_staging_and_reload_preserves_display_settings(window, tmp_path: Path, qtbot) -> None:
    page = _page_with_scene(window, tmp_path)
    page._gaussian_interactive_button.click(); qtbot.wait(10)
    first_scene = page._interactive_gaussian_scene
    assert first_scene is not None
    assert page.gaussian_runtime_health is not None
    assert page.gaussian_runtime_health.resident is True
    assert page._gaussian_load_count == 1

    page._gaussian_density_combo.setCurrentIndex(page._gaussian_density_combo.findData(50))
    page._gaussian_size_slider.setValue(125)
    page._gaussian_opacity_slider.setValue(70)
    qtbot.wait(10)
    assert page._reconstruction_canvas.interactive_gaussian_display_count == 32

    page._gaussian_hide_button.click(); qtbot.wait(10)
    assert page._interactive_gaussian_scene is None
    assert page._gaussian_display_buffers is None
    assert page.interactive_gaussian_visible is False
    assert page._reconstruction_canvas.canonical_mesh_loaded is True
    assert page._gaussian_release_count == 1
    assert page.gaussian_runtime_health is not None
    assert page.gaussian_runtime_health.resident is False
    assert page.gaussian_runtime_health.source_staging_bytes == 0
    assert "Released" in page._gaussian_resources_value.text()

    page._gaussian_interactive_button.click(); qtbot.wait(10)
    assert page._interactive_gaussian_scene is not None
    assert page._interactive_gaussian_scene is not first_scene
    assert page._gaussian_load_count == 2
    assert page._gaussian_density_combo.currentData() == 50
    assert page._gaussian_size_slider.value() == 125
    assert page._gaussian_opacity_slider.value() == 70
    assert page._reconstruction_canvas.interactive_gaussian_display_count == 32


def test_94g_preview_and_scene_switch_release_interactive_resources(window, tmp_path: Path, qtbot) -> None:
    page = _page_with_scene(window, tmp_path)
    page._gaussian_interactive_button.click(); qtbot.wait(10)
    page._gaussian_show_button.click(); qtbot.wait(10)
    assert page.gaussian_preview_visible is True
    assert page._interactive_gaussian_scene is None
    assert page._gaussian_display_buffers is None
    assert page._gaussian_release_count == 1

    page._gaussian_interactive_button.click(); qtbot.wait(10)
    scene2, manifest2, _artifact2 = make_phase9_4e_persistent_scene(tmp_path / "second")
    page.set_persistent_scene(scene2, str(manifest2)); qtbot.wait(10)
    assert page._interactive_gaussian_scene is None
    assert page._gaussian_display_buffers is None
    assert page._gaussian_release_count >= 2
    assert page._reconstruction_canvas.canonical_mesh_loaded is True
