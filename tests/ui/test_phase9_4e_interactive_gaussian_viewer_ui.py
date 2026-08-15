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


def test_94e_interactive_gaussian_button_explores_persisted_appearance_and_preserves_canonical_picking(
    window, tmp_path: Path, qtbot
) -> None:
    scene, manifest, _artifact = make_phase9_4e_persistent_scene(tmp_path)
    page = window._reconstruction_page
    page.set_persistent_scene(scene, str(manifest))
    assert page.selected_appearance_representation is not None
    assert page.selected_appearance_representation.representation_kind == "gaussian_scene"
    assert page._gaussian_interactive_button.isEnabled() is True
    assert page._gaussian_interactive_button.text() == "Interactive"
    assert page._gaussian_show_button.text() == "Preview"
    assert page._gaussian_hide_button.text() == "Hide"

    page._gaussian_interactive_button.click()
    qtbot.wait(10)
    assert page.interactive_gaussian_visible is True
    assert page._reconstruction_canvas.interactive_gaussian_visible is True
    assert page._reconstruction_canvas.interactive_gaussian_display_count == 64
    assert page._viewport_state.text() == "Interactive Gaussian appearance • canonical picking"
    assert "appearance" in page._gaussian_state_value.text().lower()
    # Canonical surface remains loaded beneath the appearance representation and
    # therefore remains the ruler/picking authority.
    assert page._reconstruction_canvas.canonical_mesh_loaded is True

    page._gaussian_hide_button.click()
    qtbot.wait(10)
    assert page.interactive_gaussian_visible is False
    assert page._reconstruction_canvas.interactive_gaussian_visible is False
    assert page._reconstruction_canvas.canonical_mesh_loaded is True


def test_94e_static_preview_and_interactive_gaussian_are_mutually_exclusive(window, tmp_path: Path, qtbot) -> None:
    scene, manifest, _artifact = make_phase9_4e_persistent_scene(tmp_path)
    page = window._reconstruction_page
    page.set_persistent_scene(scene, str(manifest))
    page._gaussian_interactive_button.click(); qtbot.wait(10)
    assert page.interactive_gaussian_visible
    page._gaussian_show_button.click(); qtbot.wait(10)
    assert page.interactive_gaussian_visible is False
    assert page.gaussian_preview_visible is True
