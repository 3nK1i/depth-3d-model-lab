from __future__ import annotations

from pathlib import Path

import pytest

from app.reconstruction_visualization import adaptive_camera_frustum_depth


def test_current_camera_frustum_depth_is_bounded_and_adaptive() -> None:
    assert adaptive_camera_frustum_depth(0.08) == pytest.approx(0.05)
    assert adaptive_camera_frustum_depth(0.28) == pytest.approx(0.0952)
    assert adaptive_camera_frustum_depth(0.34) == pytest.approx(0.1156)

    with pytest.raises(ValueError):
        adaptive_camera_frustum_depth(0.0)
    with pytest.raises(ValueError):
        adaptive_camera_frustum_depth(float("nan"))


def test_live_canvas_uses_adaptive_camera_frustum_depth() -> None:
    source = Path("app/widgets/viewport/vispy_canvas.py").read_text(encoding="utf-8")
    update_live_pose = source.split("def update_live_pose", 1)[1].split("def clear_live_state", 1)[0]

    assert "adaptive_camera_frustum_depth(self._world_axis_length)" in update_live_pose
    assert "depth=adaptive_camera_frustum_depth" in update_live_pose


def test_reconstruction_gizmo_is_borderless_screen_space_overlay() -> None:
    source = Path("app/widgets/reconstruction_orientation_gizmo.py").read_text(encoding="utf-8")

    assert "WA_TransparentForMouseEvents" in source
    assert "drawRoundedRect" not in source
    assert "QRectF" not in source
    assert "self.height() * 0.67" in source
    assert "from app.theme import PALETTE" in source
    assert "finally:" in source
    assert "painter.end()" in source


def test_windowed_toolbar_controls_are_shrinkable_not_fixed_minimums() -> None:
    source_bar = Path("app/widgets/source_bar.py").read_text(encoding="utf-8")
    runtime = Path("app/widgets/runtime_toolbar.py").read_text(encoding="utf-8")
    snapshot = Path("app/widgets/snapshot_page.py").read_text(encoding="utf-8")

    assert "QSizePolicy.Policy.Ignored" in source_bar
    assert "setMinimumWidth(200)" not in source_bar
    assert "setMinimumWidth(140)" not in runtime
    assert "QSizePolicy.Policy.Expanding" in runtime
    assert "QSizePolicy.Policy.Ignored" in snapshot


def test_reconstruction_idle_placeholder_has_no_decorative_diamond() -> None:
    source = Path("app/widgets/reconstruction_page.py").read_text(encoding="utf-8")
    block = source.split("self._viewport_placeholder = QWidget()", 1)[1].split(
        "grid.addWidget(self._viewport_placeholder, 0, 0)", 1
    )[0]
    assert 'QLabel("◇")' not in block
    assert "reconstructionViewportGlyph" not in block
    assert 'QLabel("Reconstruction viewport")' in block


def test_source_bar_declares_explicit_icon_only_compact_mode() -> None:
    source = Path("app/widgets/source_bar.py").read_text(encoding="utf-8")
    assert "_SOURCE_TYPE_TEXT_MIN_WIDTH" in source
    assert 'button.setText("" if compact else f"  {text}")' in source
    assert "compact_source_types" in source


def test_stable_video_has_non_collapsing_compact_runtime_actions() -> None:
    source = Path("app/widgets/stable_video_page.py").read_text(encoding="utf-8")
    assert "self.width() < 1180" in source
    assert "def _sync_runtime_action_presentation" in source
    assert "button.setFixedWidth(34)" in source
    assert 'self._start.setText("")' in source
    assert 'self._stop.setText("")' in source
    assert 'self._start.setText("Start Stable Video")' in source
    assert 'self._stop.setText("Stop & Finalize")' in source


def test_shared_viewport_reflows_tools_away_from_header_and_axis_when_short() -> None:
    source = Path("app/widgets/viewport/viewport_panel.py").read_text(encoding="utf-8")
    assert "QBoxLayout.Direction.LeftToRight" in source
    assert "def set_insets(" in source
    assert source.count("top=OVERLAY_MARGIN + 28") >= 2
    assert "Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop" in source
    assert "Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter" not in source


def test_shared_viewport_clips_scene_to_rounded_canvas_surface() -> None:
    source = Path("app/widgets/viewport/viewport_panel.py").read_text(encoding="utf-8")
    assert "CANVAS_CLIP_RADIUS" in source
    assert "QPainterPath" in source
    assert "def _sync_clip_mask" in source
    assert "path.addRoundedRect" in source
    assert "self.setMask(QRegion" in source


def test_shared_viewport_final_microfix_preserves_compact_containment() -> None:
    source = Path("app/widgets/viewport/viewport_panel.py").read_text(encoding="utf-8")
    assert "self._canvas.setMinimumHeight(180 if compact else 260)" in source
    assert "left=OVERLAY_MARGIN + 12" in source
    assert "bottom=OVERLAY_MARGIN + 16" in source
    assert "self._title.setVisible(not compact)" in source
    assert "self._reset_btn.setFixedHeight(24)" in source
