from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "app/widgets/reconstruction_page.py"
PREVIEW = ROOT / "app/widgets/reconstruction_appearance_preview.py"


def _nerf_group() -> str:
    page = PAGE.read_text(encoding="utf-8")
    start = page.index('self._nerf_visualization_group = self._settings_group("NERF VISUALIZATION")')
    end = page.index('self._appearance_resource_group = self._settings_group("RESOURCE & FALLBACK")', start)
    return page[start:end]


def test_95f_nerf_group_adds_compact_presentation_controls_without_losing_three_modes() -> None:
    group = _nerf_group()
    assert 'QLabel("Render detail")' in group
    assert 'QLabel("Opacity")' in group
    assert 'QLabel("Auto refresh")' in group
    assert 'QPushButton("Refresh view")' in group
    assert 'QPushButton("Reset display")' in group
    assert group.count('QPushButton("Interactive")') == 1
    assert group.count('QPushButton("Preview")') == 1
    assert group.count('QPushButton("Hide")') == 1
    assert "never modify the persisted NeRF artifact" in group


def test_95f_scene_layers_adds_one_explicit_nerf_appearance_toggle() -> None:
    page = PAGE.read_text(encoding="utf-8")
    assert page.count('QLabel("NeRF appearance")') == 1
    assert 'setObjectName("reconstructionNerfSceneLayerToggle")' in page
    assert "_on_nerf_scene_layer_toggled" in page
    assert "canonical geometry remains the only picking/measurement authority" in page.lower()


def test_95f_opacity_blending_changes_only_display_pixmap() -> None:
    preview = PREVIEW.read_text(encoding="utf-8")
    assert "def set_image_opacity" in preview
    assert "QGraphicsOpacityEffect" in preview
    assert "self._opacity_effect.setOpacity" in preview
    assert "self._source" in preview


def test_95f_manual_refresh_never_changes_canonical_navigation_or_ruler_guard() -> None:
    page = PAGE.read_text(encoding="utf-8")
    assert "not force and not self._nerf_display_settings.auto_refresh" in page
    assert "self._reconstruction_canvas.interaction_mode == \"measure\"" in page
    assert "Canonical measurement active • NeRF appearance paused" in page
    assert "turntable_state_to_nerf_render_request" in page
