"""Regression tests for Phase 5 Step 3.2 media-view toolbars."""

from __future__ import annotations

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QPixmap

from app.main_window import MainWindow
from app.widgets.depth_view import DepthMapView
from app.widgets.icon_button import IconButton
from app.widgets.original_view import OriginalSourceView


def _pixmap(width: int = 160, height: int = 90) -> QPixmap:
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.GlobalColor.darkGreen)
    return pixmap


def test_source_toolbar_is_floating_inside_media_surface(qtbot) -> None:
    view = OriginalSourceView()
    qtbot.addWidget(view)
    view.resize(640, 360)
    view.show()
    qtbot.wait(20)

    host_rect = view._display.parentWidget().rect()
    toolbar_top_left = view._toolbar_card.mapTo(view._display.parentWidget(), QPoint(0, 0))
    toolbar_right = toolbar_top_left.x() + view._toolbar_card.width()
    toolbar_bottom = toolbar_top_left.y() + view._toolbar_card.height()

    assert toolbar_top_left.x() >= host_rect.left()
    assert toolbar_top_left.y() >= host_rect.top()
    assert toolbar_right <= host_rect.right() + 1
    assert toolbar_bottom <= host_rect.bottom() + 1


def test_source_tools_are_disabled_without_frame_and_enabled_with_frame(qtbot) -> None:
    view = OriginalSourceView()
    qtbot.addWidget(view)
    assert all(not button.isEnabled() for button in view._tool_buttons)

    view.set_frame(_pixmap())
    assert all(button.isEnabled() for button in view._tool_buttons)

    view.clear_frame()
    assert all(not button.isEnabled() for button in view._tool_buttons)


def test_fullscreen_and_fit_are_distinct_functional_controls(qtbot) -> None:
    view = OriginalSourceView()
    qtbot.addWidget(view)
    view.set_frame(_pixmap())

    assert view._fullscreen_btn is not view._fit_btn
    assert view._fullscreen_btn._icon_name == "fullscreen"
    assert view._fit_btn._icon_name == "fit"
    assert view._fullscreen_btn.toolTip() != view._fit_btn.toolTip()

    with qtbot.waitSignal(view.fullscreen_requested, timeout=500):
        qtbot.mouseClick(view._fullscreen_btn, Qt.MouseButton.LeftButton)

    assert view._display.fit_to_view is True
    qtbot.mouseClick(view._fit_btn, Qt.MouseButton.LeftButton)
    assert view._display.fit_to_view is False


def test_source_rotate_fit_and_grid_controls_are_real(qtbot) -> None:
    view = OriginalSourceView()
    qtbot.addWidget(view)
    view.set_frame(_pixmap(120, 80))

    assert view._display.rotation_degrees == 0
    qtbot.mouseClick(view._rotate_btn, Qt.MouseButton.LeftButton)
    assert view._display.rotation_degrees == 90
    assert view._display.current_full_resolution_pixmap().size().width() == 80

    assert view._display.fit_to_view is True
    qtbot.mouseClick(view._fit_btn, Qt.MouseButton.LeftButton)
    assert view._display.fit_to_view is False

    assert view._display.grid_visible is False
    qtbot.mouseClick(view._grid_btn, Qt.MouseButton.LeftButton)
    assert view._display.grid_visible is True


def test_source_save_action_emits_success_signal(qtbot, monkeypatch, tmp_path) -> None:
    view = OriginalSourceView()
    qtbot.addWidget(view)
    view.set_frame(_pixmap())

    output = tmp_path / "captured-source.png"
    monkeypatch.setattr(
        "app.widgets.original_view.QFileDialog.getSaveFileName",
        lambda *_args, **_kwargs: (str(output), "PNG Image (*.png)"),
    )

    with qtbot.waitSignal(view.frame_saved, timeout=1000) as saved:
        qtbot.mouseClick(view._save_btn, Qt.MouseButton.LeftButton)
    assert saved.args == [str(output)]
    assert output.exists()


def test_source_toolbar_has_no_duplicate_expand_style_action(qtbot) -> None:
    view = OriginalSourceView()
    qtbot.addWidget(view)

    assert len(view._tool_buttons) == 5
    assert not hasattr(view, "_copy_btn")
    assert [button._icon_name for button in view._tool_buttons] == [
        "screenshot",
        "refresh",
        "fullscreen",
        "fit",
        "grid",
    ]


def test_depth_toolbar_has_no_redundant_settings_gear(qtbot) -> None:
    view = DepthMapView()
    qtbot.addWidget(view)
    buttons = view.findChildren(IconButton)
    assert buttons == [view._fullscreen_btn]
    assert "Depth settings" not in view._fullscreen_btn.toolTip()


def test_depth_controls_emit_valid_values_and_update_colorbar(qtbot) -> None:
    view = DepthMapView()
    qtbot.addWidget(view)

    with qtbot.waitSignal(view.colormap_changed, timeout=500) as colormap:
        view._colormap_combo.setCurrentText("Viridis")
    assert colormap.args == ["Viridis"]
    assert view._color_bar._bar.colormap_name == "Viridis"

    with qtbot.waitSignal(view.depth_range_changed, timeout=500) as depth_range:
        view._min_depth.setCurrentText("1.0 m")
    assert depth_range.args == [1.0, 8.0]
    assert view._color_bar._near_value.text() == "1.0 m"
    assert view._color_bar._far_value.text() == "8.0 m"

    view.sync_controls("Magma", 0.5, 10.0)
    assert view._colormap_combo.currentText() == "Magma"
    assert view.min_depth_m == 0.5
    assert view.max_depth_m == 10.0


def test_depth_and_source_controls_sync_with_main_settings(qtbot, monkeypatch) -> None:
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    window = MainWindow()
    qtbot.addWidget(window)

    window._depth_view._colormap_combo.setCurrentText("Inferno")
    assert window._settings_panel.colormap_name == "Inferno"

    window._depth_view._min_depth.setCurrentText("0.5 m")
    window._depth_view._max_depth.setCurrentText("10.0 m")
    assert window._settings_panel.min_depth_m == 0.5
    assert window._settings_panel.max_depth_m == 10.0

    window._settings_panel.set_depth_range(1.0, 5.0)
    assert window._depth_view.min_depth_m == 1.0
    assert window._depth_view.max_depth_m == 5.0


def test_source_and_depth_focus_modes_restore_complete_page(qtbot, monkeypatch) -> None:
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1536, 864)
    window.show()
    qtbot.waitExposed(window)

    window._original_view.set_frame(_pixmap())
    qtbot.mouseClick(window._original_view._fullscreen_btn, Qt.MouseButton.LeftButton)
    assert window._focus_target == "source"
    assert not window._original_view.isHidden()
    assert window._depth_view.isHidden()
    assert window._viewport_panel.isHidden()

    qtbot.keyClick(window, Qt.Key.Key_Escape)
    assert window._focus_target is None
    assert not window._depth_view.isHidden()
    assert not window._viewport_panel.isHidden()

    window._depth_view.set_depth_frame(_pixmap())
    qtbot.mouseClick(window._depth_view._fullscreen_btn, Qt.MouseButton.LeftButton)
    assert window._focus_target == "depth"
    assert window._original_view.isHidden()
    assert not window._depth_view.isHidden()
    assert window._viewport_panel.isHidden()

    qtbot.keyClick(window, Qt.Key.Key_Escape)
    assert window._focus_target is None
    assert not window._original_view.isHidden()
    assert not window._viewport_panel.isHidden()
