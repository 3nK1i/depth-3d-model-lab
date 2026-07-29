"""Tests for the reusable polish widgets: icons, IconButton, ToggleSwitch."""

from __future__ import annotations

import pytest

from app.icons import has_icon, render_pixmap
from app.widgets.icon_button import IconButton
from app.widgets.toggle_switch import ToggleSwitch


@pytest.mark.parametrize("name", ["camera", "play", "pause", "stop", "settings", "grid"])
def test_icons_render_non_empty(qtbot, name):
    assert has_icon(name)
    px = render_pixmap(name, "#e0293e", 18)
    assert not px.isNull()


def test_icon_button_color_changes_when_checked(qtbot):
    btn = IconButton("cursor", checkable=True)
    qtbot.addWidget(btn)
    assert btn._current_color() == btn._color
    btn.setChecked(True)
    assert btn._current_color() == btn._checked_color


def test_icon_button_disabled_color(qtbot):
    btn = IconButton("cursor")
    qtbot.addWidget(btn)
    btn.setEnabled(False)
    assert btn._current_color() == btn._disabled_color


def test_toggle_switch_emits_and_moves_knob(qtbot):
    toggle = ToggleSwitch(checked=False)
    qtbot.addWidget(toggle)
    with qtbot.waitSignal(toggle.toggled_changed, timeout=500) as blocker:
        toggle.setChecked(True)
    assert blocker.args == [True]
    # Knob target is the "on" extreme after the animation completes.
    qtbot.wait(200)
    assert toggle.get_knob_position() == pytest.approx(toggle._knob_max, abs=1.0)