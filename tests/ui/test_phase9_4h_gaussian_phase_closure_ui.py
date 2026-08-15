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


def test_94h_gaussian_view_controls_and_release_preserve_canonical_scene(window, tmp_path: Path, qtbot) -> None:
    scene, manifest, _artifact = make_phase9_4e_persistent_scene(tmp_path)
    page = window._reconstruction_page
    page.set_persistent_scene(scene, str(manifest))
    assert page._reconstruction_canvas.canonical_mesh_loaded is True

    page._gaussian_interactive_button.click(); qtbot.wait(10)
    assert page.interactive_gaussian_visible is True
    assert page._reconstruction_canvas.canonical_mesh_loaded is True
    assert "canonical picking" in page._viewport_state.text().lower()

    page._gaussian_density_combo.setCurrentIndex(page._gaussian_density_combo.findData(50))
    page._gaussian_size_slider.setValue(125)
    page._gaussian_opacity_slider.setValue(70)
    qtbot.wait(10)
    assert page._reconstruction_canvas.interactive_gaussian_display_count == 32

    page._gaussian_hide_button.click(); qtbot.wait(10)
    assert page.interactive_gaussian_visible is False
    assert page._interactive_gaussian_scene is None
    assert page._gaussian_display_buffers is None
    assert page._reconstruction_canvas.canonical_mesh_loaded is True

    page._gaussian_interactive_button.click(); qtbot.wait(10)
    assert page.interactive_gaussian_visible is True
    assert page._gaussian_density_combo.currentData() == 50
    assert page._gaussian_size_slider.value() == 125
    assert page._gaussian_opacity_slider.value() == 70
    assert page._reconstruction_canvas.canonical_mesh_loaded is True
