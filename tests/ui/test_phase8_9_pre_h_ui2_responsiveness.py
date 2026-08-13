from __future__ import annotations

from PyQt6.QtCore import QPoint, QRect
from PyQt6.QtGui import QPixmap

from app.main_window import MainWindow
from app.widgets.reconstruction_orientation_gizmo import ReconstructionOrientationGizmo
from app.widgets.snapshot_page import SnapshotPage
from app.widgets.source_bar import SourceBar
from app.widgets.stable_video_page import StableVideoPage
from app.widgets.viewport.viewport_panel import RENDER_CARD_WIDTH, ViewportPanel


def _mapped_rect(widget, parent) -> QRect:
    return QRect(widget.mapTo(parent, QPoint(0, 0)), widget.size())


def test_viewport_render_card_compacts_in_short_window_and_restores(qtbot) -> None:
    panel = ViewportPanel()
    qtbot.addWidget(panel)
    panel.resize(560, 300)
    panel.show()
    qtbot.wait(30)

    assert panel._render_card.compact is True
    assert panel._render_card.width() < RENDER_CARD_WIDTH
    assert panel._tools.compact is True
    canvas = panel._canvas.geometry()
    card = _mapped_rect(panel._render_card, panel)
    header = _mapped_rect(panel._header, panel)
    tools = _mapped_rect(panel._tools, panel)
    axis = _mapped_rect(panel._axis, panel)
    assert card.top() >= canvas.top()
    assert card.bottom() <= canvas.bottom()
    assert card.right() <= canvas.right()
    assert not header.intersects(tools)
    assert not tools.intersects(axis)
    assert not tools.intersects(card)

    panel.resize(900, 600)
    qtbot.wait(30)
    assert panel._render_card.compact is False
    assert panel._render_card.width() == RENDER_CARD_WIDTH
    assert panel._tools.compact is False


def test_viewport_short_height_reserves_header_and_axis_safe_zones(qtbot) -> None:
    panel = ViewportPanel()
    qtbot.addWidget(panel)
    panel.resize(900, 330)
    panel.show()
    qtbot.wait(30)

    assert panel._tools.compact is True
    header = _mapped_rect(panel._header, panel)
    tools = _mapped_rect(panel._tools, panel)
    axis = _mapped_rect(panel._axis, panel)
    render_card = _mapped_rect(panel._render_card, panel)
    assert not header.intersects(tools)
    assert not tools.intersects(axis)
    assert not tools.intersects(render_card)




def test_viewport_normal_vertical_tools_keep_clear_of_bottom_axis(qtbot) -> None:
    panel = ViewportPanel()
    qtbot.addWidget(panel)
    panel.resize(900, 520)
    panel.show()
    qtbot.wait(30)

    assert panel._tools.compact is False
    header = _mapped_rect(panel._header, panel)
    tools = _mapped_rect(panel._tools, panel)
    axis = _mapped_rect(panel._axis, panel)
    assert not header.intersects(tools)
    assert not tools.intersects(axis)
    assert tools.top() > header.bottom()
    assert tools.bottom() < axis.top()


def test_viewport_canvas_uses_rounded_clip_mask(qtbot) -> None:
    panel = ViewportPanel()
    qtbot.addWidget(panel)
    panel.resize(560, 300)
    panel.show()
    qtbot.wait(30)

    mask = panel._canvas.mask()
    assert not mask.isEmpty()
    assert not mask.contains(QPoint(0, 0))
    assert mask.contains(QPoint(panel._canvas.width() // 2, panel._canvas.height() // 2))


def test_live3d_control_row_can_negotiate_1366_width_without_fixed_minimums(qtbot, monkeypatch) -> None:
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    win.resize(1366, 768)
    win.show()
    qtbot.wait(50)

    source_bar = win._source_bar
    runtime = win._runtime_toolbar
    assert all(button.minimumWidth() == 0 for button in source_bar._type_buttons.values())
    assert source_bar._camera_device_combo.minimumWidth() == 0
    assert source_bar._camera_resolution_combo.minimumWidth() == 0
    assert source_bar._camera_fps_combo.minimumWidth() == 0
    assert runtime._connect_btn.minimumWidth() == 0
    assert runtime._start_btn.minimumWidth() == 0
    assert runtime._stop_btn.minimumWidth() == 0

    control = win._control_row
    assert control.layout().minimumSize().width() <= control.width()
    win.close()


def test_snapshot_controls_and_viewport_remain_bounded_in_windowed_layout(qtbot) -> None:
    page = SnapshotPage()
    qtbot.addWidget(page)
    page.resize(1100, 690)
    page.show()
    qtbot.wait(50)

    controls = page._capture.parentWidget()
    for button in (page._capture, page._run, page._cancel, page._open):
        assert button.minimumWidth() == 0
        assert button.geometry().right() <= controls.contentsRect().right()

    viewport = page._viewport
    card = _mapped_rect(viewport._render_card, viewport)
    canvas = viewport._canvas.geometry()
    assert card.right() <= canvas.right()
    assert card.bottom() <= canvas.bottom()


def test_reconstruction_gizmo_real_paint_path_is_exception_free(qtbot) -> None:
    gizmo = ReconstructionOrientationGizmo(lambda: __import__("numpy").eye(3, 2))
    qtbot.addWidget(gizmo)
    gizmo.show()
    qtbot.wait(20)
    pixmap = QPixmap(gizmo.size())
    gizmo.render(pixmap)
    assert not pixmap.isNull()
    gizmo.shutdown()


def test_source_type_buttons_switch_to_icon_only_compact_mode(qtbot) -> None:
    bar = SourceBar()
    qtbot.addWidget(bar)
    bar.resize(900, 54)
    bar.show()
    qtbot.wait(40)
    assert bar.compact_source_types is False
    assert all(button.text().strip() for button in bar._type_buttons.values())

    bar.resize(520, 54)
    qtbot.wait(40)
    assert bar.compact_source_types is True
    assert all(button.text() == "" for button in bar._type_buttons.values())
    assert {button.toolTip() for button in bar._type_buttons.values()} == {
        "Image", "Video", "Camera", "RTSP"
    }


def test_stable_video_controls_use_compact_copy_in_windowed_mode(qtbot) -> None:
    page = StableVideoPage()
    qtbot.addWidget(page)
    page.resize(1080, 690)
    page.show()
    qtbot.wait(40)
    assert page.compact_controls is True
    assert page._temporal.isHidden()
    assert page._fps.suffix() == " FPS"
    for button in (page._start, page._pause, page._stop, page._open):
        assert button.text() == ""
        assert button.width() == 34
        assert button.geometry().right() <= page.rect().right()

    page.resize(1360, 760)
    qtbot.wait(40)
    assert page.compact_controls is False
    assert page._temporal.isVisible()
    assert page._start.text() == "Start Stable Video"
    assert page._pause.text() == "Pause"
    assert page._stop.text() == "Stop & Finalize"
    assert page._open.text() == "Open Export"
    assert all(button.width() > 0 for button in (page._start, page._pause, page._stop, page._open))


def test_compact_viewport_relaxes_canvas_minimum_and_keeps_overlays_inside(qtbot) -> None:
    panel = ViewportPanel()
    qtbot.addWidget(panel)
    panel.resize(560, 300)
    panel.show()
    qtbot.wait(30)

    assert panel._canvas.minimumHeight() == 180
    canvas = panel._canvas.geometry()
    axis = _mapped_rect(panel._axis, panel)
    card = _mapped_rect(panel._render_card, panel)
    # Final gizmo nudge reserves visible breathing room from the lower-left
    # viewport edges, especially important when the orientation rotates.
    assert axis.left() >= canvas.left() + 26
    assert axis.bottom() <= canvas.bottom() - 30
    assert card.bottom() <= canvas.bottom()
    assert panel._render_card._reset_btn.isVisible()
    assert panel._render_card._reset_btn.height() == 24

    panel.resize(900, 600)
    qtbot.wait(30)
    assert panel._canvas.minimumHeight() == 260
    assert panel._render_card._title.isVisible()
