"""Original/depth view state tests for static and continuous sources."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

from app.widgets.depth_view import DepthMapView
from app.widgets.original_view import OriginalSourceView


def test_original_source_view_switches_between_idle_ready_and_live(qtbot) -> None:
    view = OriginalSourceView()
    qtbot.addWidget(view)

    view.set_ready()
    assert view.activity_text == "READY"
    assert "READY" in view._live_label.text()

    view.set_live(True)
    assert view.activity_text == "LIVE"
    assert "LIVE" in view._live_label.text()

    view.set_live(False)
    assert view.activity_text == "IDLE"
    assert "IDLE" in view._live_label.text()


def test_depth_view_accepts_pixmap_and_updates_range(qtbot) -> None:
    view = DepthMapView()
    qtbot.addWidget(view)
    view.resize(400, 300)
    view.show()

    view._color_bar.set_range(0.5, 8.0)
    assert view._color_bar._near_value.text() == "0.5 m"
    assert view._color_bar._far_value.text() == "8.0 m"

    pixmap = QPixmap(12, 8)
    pixmap.fill(Qt.GlobalColor.red)
    view.set_depth_frame(pixmap)
    displayed = view._display.pixmap()
    assert displayed is not None and not displayed.isNull()
