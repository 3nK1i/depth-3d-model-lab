from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "app/widgets/reconstruction_page.py"
PREVIEW = ROOT / "app/widgets/reconstruction_appearance_preview.py"
CANVAS = ROOT / "app/widgets/viewport/vispy_canvas.py"


def test_5e_nerf_group_keeps_clear_three_mode_hierarchy() -> None:
    page = PAGE.read_text(encoding="utf-8")
    start = page.index('self._nerf_visualization_group = self._settings_group("NERF VISUALIZATION")')
    end = page.index('self._appearance_resource_group = self._settings_group("RESOURCE & FALLBACK")', start)
    group = page[start:end]
    assert group.count('QPushButton("Interactive")') == 1
    assert group.count('QPushButton("Preview")') == 1
    assert group.count('QPushButton("Hide")') == 1
    assert "canonical geometry remains" in group.lower()


def test_5e_interactive_overlay_is_mouse_transparent_and_measurement_pauses_it() -> None:
    page = PAGE.read_text(encoding="utf-8")
    preview = PREVIEW.read_text(encoding="utf-8")
    assert "set_mouse_passthrough" in preview
    assert "WA_TransparentForMouseEvents" in preview
    assert "self._appearance_preview.set_mouse_passthrough(True)" in page
    assert "Canonical measurement active • NeRF appearance paused" in page
    assert 'self._reconstruction_canvas.interaction_mode == "measure"' in page


def test_5e_canvas_exposes_presentation_camera_without_neural_backend() -> None:
    canvas = CANVAS.read_text(encoding="utf-8")
    assert "def interactive_render_camera_state" in canvas
    for forbidden in ("nerfstudio", "tinycudann", "Nerfacto"):
        assert forbidden not in canvas
