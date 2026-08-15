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


def test_94f_gaussian_display_controls_are_compact_and_presentation_only(window, tmp_path: Path, qtbot) -> None:
    page = _page_with_scene(window, tmp_path)
    assert page._gaussian_density_combo.isEnabled()
    assert page._gaussian_size_slider.isEnabled()
    assert page._gaussian_opacity_slider.isEnabled()
    assert page._gaussian_scene_layer_toggle.isEnabled()
    assert page._gaussian_scene_layer_toggle.isChecked() is False

    page._gaussian_interactive_button.click(); qtbot.wait(10)
    assert page.interactive_gaussian_visible is True
    assert page._gaussian_scene_layer_toggle.isChecked() is True
    assert page._reconstruction_canvas.canonical_mesh_loaded is True
    assert page._reconstruction_canvas.interactive_gaussian_display_count == 64

    index = page._gaussian_density_combo.findData(50)
    assert index >= 0
    page._gaussian_density_combo.setCurrentIndex(index); qtbot.wait(10)
    assert page._reconstruction_canvas.interactive_gaussian_display_count == 32
    assert "32/64 splats" in page._gaussian_display_value.text()

    page._gaussian_size_slider.setValue(140)
    page._gaussian_opacity_slider.setValue(50)
    qtbot.wait(10)
    assert page._gaussian_size_value.text() == "140%"
    assert page._gaussian_opacity_value.text() == "50%"
    assert page._reconstruction_canvas.canonical_mesh_loaded is True


def test_94f_scene_layer_toggle_hides_and_restores_gaussian_without_touching_canonical_mesh(window, tmp_path: Path, qtbot) -> None:
    page = _page_with_scene(window, tmp_path)
    page._gaussian_interactive_button.click(); qtbot.wait(10)
    assert page.interactive_gaussian_visible
    assert page._reconstruction_canvas.canonical_mesh_loaded

    page._gaussian_scene_layer_toggle.setChecked(False); qtbot.wait(10)
    assert page.interactive_gaussian_visible is False
    assert page._reconstruction_canvas.canonical_mesh_loaded is True

    page._gaussian_scene_layer_toggle.setChecked(True); qtbot.wait(10)
    assert page.interactive_gaussian_visible is True
    assert page._reconstruction_canvas.canonical_mesh_loaded is True


def test_94f_reset_and_preview_switching_keep_controls_truthful(window, tmp_path: Path, qtbot) -> None:
    page = _page_with_scene(window, tmp_path)
    page._gaussian_interactive_button.click(); qtbot.wait(10)
    page._gaussian_density_combo.setCurrentIndex(page._gaussian_density_combo.findData(25))
    page._gaussian_size_slider.setValue(150)
    page._gaussian_opacity_slider.setValue(40)
    page._gaussian_reset_display_button.click(); qtbot.wait(10)
    assert page._gaussian_density_combo.currentData() == 100
    assert page._gaussian_size_slider.value() == 100
    assert page._gaussian_opacity_slider.value() == 100
    assert page._reconstruction_canvas.interactive_gaussian_display_count == 64

    page._gaussian_show_button.click(); qtbot.wait(10)
    assert page.gaussian_preview_visible is True
    assert page.interactive_gaussian_visible is False
    assert page._gaussian_scene_layer_toggle.isChecked() is False
    assert page._reconstruction_canvas.canonical_mesh_loaded is True
