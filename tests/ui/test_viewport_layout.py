"""Structural tests for the 3D viewport overlay composition.

These lock in the fix for the "small canvas with a large unused region
below it" problem: the canvas must fill the card, and every control must
float *inside* the canvas rather than sit beside it in a sibling column.
"""

from __future__ import annotations

import pytest

from app.main_window import MainWindow
from app.widgets.viewport.viewport_panel import (
    RENDER_CARD_WIDTH,
    TOOLBAR_WIDTH,
    ViewportPanel,
)


@pytest.fixture
def panel(qtbot):
    # Yield (not return) and keep `win` alive for the test's duration —
    # otherwise the window is garbage-collected and Qt deletes its
    # children out from under the assertions.
    win = MainWindow()
    qtbot.addWidget(win)
    win.resize(1536, 864)
    win.show()
    qtbot.waitExposed(win)
    yield win._viewport_panel


def test_canvas_fills_card(panel: ViewportPanel):
    card = panel.geometry()
    canvas = panel._canvas.geometry()
    assert canvas.width() / card.width() > 0.90
    assert canvas.height() / card.height() > 0.85


def test_render_card_is_compact_not_full_height(panel: ViewportPanel):
    card_h = panel.geometry().height()
    assert panel._render_card.height() < card_h
    assert panel._render_card.width() == RENDER_CARD_WIDTH


def test_toolbar_width_within_spec(panel: ViewportPanel):
    assert 38 <= TOOLBAR_WIDTH <= 44
    assert panel._tools.width() == TOOLBAR_WIDTH


def test_overlays_sit_inside_the_canvas(panel: ViewportPanel):
    canvas = panel._canvas.geometry()
    for overlay in (panel._tools, panel._render_card, panel._header, panel._axis):
        geo = overlay.geometry()
        assert geo.left() >= canvas.left()
        assert geo.right() <= canvas.right()


def test_camera_is_fitted_once_not_per_frame(panel: ViewportPanel):
    assert panel.camera_initialized is False
    panel.mark_camera_initialized()
    assert panel.camera_initialized is True
    # Only an explicit invalidation (new source / session / Reset View)
    # may allow another fit.
    panel.invalidate_camera()
    assert panel.camera_initialized is False


def test_live_status_toggles(panel: ViewportPanel):
    panel.set_live(True)
    assert panel._header._status_text.text() == "LIVE"
    panel.set_live(False)
    assert panel._header._status_text.text() == "IDLE"