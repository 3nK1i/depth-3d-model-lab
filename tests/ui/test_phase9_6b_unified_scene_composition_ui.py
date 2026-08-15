from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "app/widgets/reconstruction_page.py"


def test_96b_ui_binds_scene_composition_into_existing_workspace_without_new_renderer() -> None:
    page = PAGE.read_text(encoding="utf-8")
    assert "build_digital_twin_scene_composition" in page
    assert "digital_twin_scene_composition_changed = pyqtSignal(object)" in page
    assert "def _refresh_digital_twin_scene_composition" in page
    assert "self._digital_twin_composition_value.setText(composition.composition_text)" in page
    assert "composition.visibility_text" in page
    assert "composition.authority_text" in page
    assert 'self._settings_group("DIGITAL TWIN WORKSPACE")' in page
    assert page.count('self._settings_group("SCENE LAYERS")') == 1


def test_96b_ui_composes_existing_canvas_state_without_mutating_locked_canvas_or_controller() -> None:
    page = PAGE.read_text(encoding="utf-8")
    assert "canonical_surface_visible=bool(canvas and canvas.canonical_mesh_visible)" in page
    assert "semantic_presentation=self._canonical_semantic_presentation" in page
    assert "camera_presentation=self._canonical_camera_presentation" in page
    assert "appearance_presentation=self._appearance_presentation" in page
    assert "gaussian_visible=bool(self.interactive_gaussian_visible or self.gaussian_preview_visible)" in page
    assert "nerf_visible=bool(self.interactive_nerf_visible or self.nerf_preview_visible)" in page

    canvas = (ROOT / "app/widgets/viewport/vispy_canvas.py").read_text(encoding="utf-8")
    controller = (ROOT / "app/reconstruction_controller.py").read_text(encoding="utf-8")
    assert "def set_canonical_mesh(" in canvas
    assert "measurement_point_picked = pyqtSignal(object)" in canvas
    assert "class ReconstructionController(QObject):" in controller
