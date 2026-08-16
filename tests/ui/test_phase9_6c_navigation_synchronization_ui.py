from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "app/widgets/reconstruction_page.py"
CANVAS = ROOT / "app/widgets/viewport/vispy_canvas.py"


def test_96c_ui_projects_one_shared_navigation_source_into_existing_workspace() -> None:
    page = PAGE.read_text(encoding="utf-8")
    assert "build_digital_twin_navigation_sync" in page
    assert "digital_twin_navigation_changed = pyqtSignal(object)" in page
    assert "def _refresh_digital_twin_navigation_sync" in page
    assert 'self._add_status_row(\n            workspace_details_layout, "Navigation", self._digital_twin_navigation_value\n        )' in page
    assert 'getattr(canvas, "interactive_render_camera_state", None)' in page
    assert "callable(camera_state_getter)" in page
    assert "gaussian_interactive=self.interactive_gaussian_visible" in page
    assert "nerf_interactive=self.interactive_nerf_visible" in page
    assert "gaussian_preview_visible=self.gaussian_preview_visible" in page
    assert "nerf_preview_visible=self.nerf_preview_visible" in page


def test_96c_navigation_events_and_scene_switch_keep_existing_canonical_camera_contract() -> None:
    page = PAGE.read_text(encoding="utf-8")
    canvas = CANVAS.read_text(encoding="utf-8")

    assert page.count("self._refresh_digital_twin_navigation_sync()") >= 7
    assert "def _fit_reconstruction_view" in page
    assert "def _focus_reconstruction_camera" in page
    assert "def _reset_reconstruction_view" in page
    assert "def _on_reconstruction_view_interaction_ended" in page
    assert "same-scene reload preserves the user's current orbit/pan/zoom" in page
    assert "self._hide_interactive_gaussian()" in page
    assert "self._hide_nerf_visualization(refresh=False)" in page

    persistent_assignment = page.index("self._persistent_scene = scene")
    assert page.index("self._hide_interactive_gaussian()") < persistent_assignment
    assert page.index("self._hide_nerf_visualization(refresh=False)") < persistent_assignment

    assert "def interactive_render_camera_state" in canvas
    assert "def set_interactive_gaussian_scene" in canvas
    for forbidden in ("nerfstudio", "tinycudann", "gsplat"):
        assert forbidden not in canvas
