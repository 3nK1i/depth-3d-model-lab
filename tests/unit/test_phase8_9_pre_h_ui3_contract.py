from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_reconstruction_toolbar_actions_are_wired_to_real_canvas_behaviors() -> None:
    page = (ROOT / "app/widgets/reconstruction_page.py").read_text(encoding="utf-8")

    for call in (
        "self._reconstruction_canvas.fit_scene",
        "self._reconstruction_canvas.focus_current_camera",
        "self._reconstruction_canvas.reset_view",
        "self._reconstruction_canvas.set_grid_visible",
        "self._reconstruction_canvas.set_trajectory_visible",
        "self._reconstruction_canvas.set_sparse_landmarks_visible",
        "self._reconstruction_canvas.set_keyframes_visible",
        "self._reconstruction_canvas.set_offline_sparse_envelope_visible",
        "self._reconstruction_canvas.set_semantic_objects_visible",
        "self._sync_diagnostics_visibility",
    ):
        assert call in page

    for mode in ("orbit", "pan", "zoom"):
        assert f'("{mode}", "{mode}"' in page

    # Semantic presentation is data-conditional, not a fake enabled control.
    assert "semantic_button.setEnabled(False)" in page
    assert "semantic_button.setEnabled(bool(presentation.objects))" in page


def test_reconstruction_fullscreen_reuses_focus_mode_without_recreating_canvas() -> None:
    page = (ROOT / "app/widgets/reconstruction_page.py").read_text(encoding="utf-8")
    main = (ROOT / "app/main_window.py").read_text(encoding="utf-8")

    assert "viewport_fullscreen_requested = pyqtSignal()" in page
    assert 'self._viewport_tool_buttons["fullscreen"]' in page
    assert "def set_viewport_focus_mode(self, enabled: bool)" in page
    assert "without recreating its GL surface" in page
    assert "self._reconstruction_canvas = ReconstructionCanvas(self)" in page

    assert "self._reconstruction_page.viewport_fullscreen_requested.connect(" in main
    assert 'self._toggle_panel_focus("reconstruction")' in main
    assert 'target not in {None, "source", "depth", "viewport", "reconstruction"}' in main
    assert "self._reconstruction_page.set_viewport_focus_mode(True)" in main
    assert "self._reconstruction_page.set_viewport_focus_mode(False)" in main
    assert "self._reconstruction_page.set_viewport_fullscreen_active(" in main


def test_reconstruction_fullscreen_is_presentation_only() -> None:
    page = (ROOT / "app/widgets/reconstruction_page.py").read_text(encoding="utf-8")
    block = page.split("def set_viewport_focus_mode", 1)[1].split(
        "def set_viewport_fullscreen_active", 1
    )[0]

    for forbidden in (
        "start_live",
        "start_offline",
        "shutdown",
        "export_reconstruction_bundle",
        "set_semantic_geometry",
        "set_reconstruction_measurement",
        "certified_metrology",
    ):
        assert forbidden not in block



def test_reconstruction_canvas_escape_is_forwarded_to_focus_exit() -> None:
    page = Path("app/widgets/reconstruction_page.py").read_text(encoding="utf-8")
    main = Path("app/main_window.py").read_text(encoding="utf-8")

    assert "viewport_escape_requested = pyqtSignal()" in page
    assert "self._reconstruction_canvas.escape_pressed.connect(" in page
    assert "self.viewport_escape_requested.emit" in page
    assert "self._reconstruction_page.viewport_escape_requested.connect(" in main
    assert "self._exit_focus_mode" in main


def test_reconstruction_idle_copy_is_biased_below_world_origin_overlay() -> None:
    page = Path("app/widgets/reconstruction_page.py").read_text(encoding="utf-8")
    block = page.split("center = QVBoxLayout(self._viewport_placeholder)", 1)[1].split(
        "grid.addWidget(self._viewport_placeholder, 0, 0)", 1
    )[0]

    assert "center.addStretch(4)" in block
    assert "center.addStretch(3)" in block
