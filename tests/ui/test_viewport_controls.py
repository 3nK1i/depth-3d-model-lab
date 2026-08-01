"""Regression tests for the functional Step 3.1 viewport controls."""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import Qt

from app.main_window import MainWindow
from core.capture.image_source import ImageSource
from core.models.catalog import MOGE2_MODEL_ID
from core.runtime.messages import InferenceResult


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    win.resize(1536, 864)
    win.show()
    qtbot.waitExposed(win)
    yield win


def _left_button():
    return Qt.MouseButton.LeftButton


def _direct_result() -> tuple[InferenceResult, np.ndarray]:
    depth = np.full((4, 4), 2.0, dtype=np.float32)
    yy, xx = np.mgrid[0:4, 0:4].astype(np.float32)
    points = np.stack((xx * 0.1, yy * 0.1, depth), axis=-1)
    normals = np.zeros_like(points)
    normals[..., 0] = 1.0
    normals[..., 2] = -1.0
    valid = np.ones((4, 4), dtype=bool)
    bgr = np.full((4, 4, 3), (10, 20, 30), dtype=np.uint8)
    return (
        InferenceResult(
            session_id=1,
            frame_id=1,
            model_id=MOGE2_MODEL_ID,
            depth=depth,
            points=points,
            normals=normals,
            valid_mask=valid,
            point_stride=1,
            latency_ms=1.0,
        ),
        bgr,
    )


def test_left_toolbar_switches_real_camera_modes(window, qtbot):
    panel = window._viewport_panel
    canvas = window._inference_controller._point_cloud_canvas

    assert panel.active_tool == "orbit"
    assert canvas.interaction_mode == "orbit"

    for mode in ("select", "pan", "zoom", "orbit"):
        qtbot.mouseClick(panel._tools._tool_buttons[mode], _left_button())
        assert panel.active_tool == mode
        assert canvas.interaction_mode == mode


def test_point_size_and_marker_smoothing_update_immediately(window):
    panel = window._viewport_panel
    canvas = window._inference_controller._point_cloud_canvas

    panel._render_card._point_size.slider.setValue(35)
    assert canvas.point_size == pytest.approx(3.5)

    panel._render_card._smoothing.slider.setValue(80)
    assert canvas.smoothing == pytest.approx(0.8)


def test_unimplemented_mesh_and_runtime_vsync_are_honestly_disabled(window):
    assert not window._viewport_panel._render_card.mesh_enabled
    assert not window._settings_panel._vsync_toggle.isEnabled()
    assert window._settings_panel._vsync_toggle.isChecked()
    assert (
        "enabled when the VisPy canvas is created"
        in window._settings_panel._vsync_toggle.toolTip()
    )


def test_sidebar_background_grid_and_axes_apply_immediately(window, qtbot):
    settings = window._settings_panel
    canvas = window._inference_controller._point_cloud_canvas

    settings._background.setCurrentText("White")
    assert canvas.background_name == "white"
    assert window._viewport_panel._header._light_background is True

    qtbot.mouseClick(settings._grid_toggle, _left_button())
    assert canvas.grid_visible is False

    qtbot.mouseClick(settings._axes_toggle, _left_button())
    assert window._viewport_panel._axis_host.isHidden()


def test_viewport_focus_mode_hides_and_restores_application_chrome(window, qtbot):
    button = window._viewport_panel._tools._fullscreen_btn

    qtbot.mouseClick(button, _left_button())
    assert window._viewport_focus_mode is True
    assert window._header.isHidden()
    assert window._mode_tabs.isHidden()
    assert window._control_row.isHidden()
    assert window._top_splitter.isHidden()
    assert window._settings_panel.isHidden()
    assert window._performance_bar.isHidden()
    assert button.isChecked()

    qtbot.keyClick(
        window._inference_controller._point_cloud_canvas.native,
        Qt.Key.Key_Escape,
    )
    assert window._viewport_focus_mode is False
    assert not window._header.isHidden()
    assert not window._mode_tabs.isHidden()
    assert not window._control_row.isHidden()
    assert not window._top_splitter.isHidden()
    assert not window._settings_panel.isHidden()
    assert not window._performance_bar.isHidden()
    assert not button.isChecked()


def test_render_modes_recolor_cached_static_geometry_without_ai_rerun(
    window, monkeypatch
):
    controller = window._inference_controller
    panel = window._viewport_panel
    result, bgr = _direct_result()
    controller._app_controller._active_source = ImageSource("unused.png")
    controller._cached_raw_result = result
    controller._cached_render_result = result
    controller._cached_rgb = bgr
    panel.set_surface_normals_available(True)

    captured: list[np.ndarray] = []
    monkeypatch.setattr(
        controller._point_cloud_canvas,
        "update_points",
        lambda _points, colors: captured.append(colors.copy()),
    )

    panel._render_card._mode_buttons["RGB Points"].setChecked(True)
    controller._rerender_cached_result()
    rgb_colors = captured[-1]
    np.testing.assert_array_equal(rgb_colors[0], np.array([30, 20, 10], dtype=np.uint8))

    panel._render_card._mode_buttons["Depth Colors"].setChecked(True)
    controller._rerender_cached_result()
    depth_colors = captured[-1]
    assert not np.array_equal(depth_colors[0], rgb_colors[0])

    panel._render_card._mode_buttons["Surface Normals"].setChecked(True)
    controller._rerender_cached_result()
    normal_colors = captured[-1]
    np.testing.assert_array_equal(
        normal_colors[0],
        np.array([255, 127, 0], dtype=np.uint8),
    )


def test_render_mode_and_density_changes_schedule_cached_refresh(window):
    controller = window._inference_controller
    panel = window._viewport_panel
    result, bgr = _direct_result()
    controller._cached_render_result = result
    controller._cached_rgb = bgr

    panel._render_card._mode_buttons["Depth Colors"].setChecked(True)
    assert controller._settings_refresh_timer.isActive()

    controller._settings_refresh_timer.stop()
    panel._render_card._density.slider.setValue(50)
    assert controller._settings_refresh_timer.isActive()


def test_background_removal_changes_geometry_mask(window, monkeypatch):
    controller = window._inference_controller
    settings = window._settings_panel
    result, bgr = _direct_result()
    depth = result.depth.copy()
    depth[0, 0] = 10.0
    valid = result.valid_mask.copy()
    valid[0, 1] = False
    result = InferenceResult(
        session_id=result.session_id,
        frame_id=result.frame_id,
        model_id=result.model_id,
        depth=depth,
        points=result.points,
        normals=result.normals,
        valid_mask=valid,
        point_stride=1,
        latency_ms=1.0,
    )

    counts: list[int] = []
    monkeypatch.setattr(
        controller._point_cloud_canvas,
        "update_points",
        lambda points, _colors: counts.append(points.shape[0]),
    )

    settings._edge_filter_slider.setValue(0)
    settings._bg_removal.setChecked(True)
    controller._update_geometry(result, bgr, apply_temporal=False)
    with_removal = counts[-1]

    settings._bg_removal.setChecked(False)
    controller._update_geometry(result, bgr, apply_temporal=False)
    without_removal = counts[-1]

    assert without_removal == with_removal + 2


def test_density_resamples_the_cached_cloud_proportionally(window, monkeypatch):
    controller = window._inference_controller
    panel = window._viewport_panel
    result, bgr = _direct_result()
    counts: list[int] = []
    monkeypatch.setattr(
        controller._point_cloud_canvas,
        "update_points",
        lambda points, _colors: counts.append(points.shape[0]),
    )

    panel._render_card._density.slider.setValue(100)
    controller._update_geometry(result, bgr, apply_temporal=False)
    full_count = counts[-1]

    panel._render_card._density.slider.setValue(50)
    controller._update_geometry(result, bgr, apply_temporal=False)
    half_count = counts[-1]

    assert full_count == 16
    assert half_count == 8
