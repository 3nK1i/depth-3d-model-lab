"""UI boundary gate before Phase 8.8 enables Reconstruction."""
from __future__ import annotations

import pytest
from PyQt6.QtCore import Qt

from app.widgets.mode_tabs import AppMode, ModeTabs


@pytest.fixture
def tabs(qtbot):
    widget = ModeTabs()
    qtbot.addWidget(widget)
    return widget


def test_reconstruction_tab_is_visible_but_disabled(tabs) -> None:
    button = tabs._buttons[AppMode.RECONSTRUCTION]
    assert button.isVisible() or not tabs.isVisible()  # widget need not be shown in offscreen tests
    assert button.isEnabled() is False
    assert "Not yet implemented" in button.toolTip()


def test_existing_operational_modes_remain_enabled(tabs) -> None:
    enabled = {AppMode.LIVE_3D, AppMode.INSPECTION, AppMode.STABLE_VIDEO, AppMode.SNAPSHOT, AppMode.BENCHMARK}
    for mode, button in tabs._buttons.items():
        assert button.isEnabled() is (mode in enabled)


def test_programmatic_reconstruction_activation_is_rejected(tabs) -> None:
    with pytest.raises(ValueError, match="not enabled"):
        tabs.set_active_mode(AppMode.RECONSTRUCTION)
    assert tabs.active_mode is AppMode.LIVE_3D


def test_navigation_lock_never_enables_reconstruction(tabs, qtbot) -> None:
    qtbot.mouseClick(tabs._buttons[AppMode.SNAPSHOT], Qt.MouseButton.LeftButton)
    assert tabs.active_mode is AppMode.SNAPSHOT
    tabs.set_navigation_locked(True)
    assert tabs._buttons[AppMode.SNAPSHOT].isEnabled()
    assert not tabs._buttons[AppMode.RECONSTRUCTION].isEnabled()
    tabs.set_navigation_locked(False)
    assert not tabs._buttons[AppMode.RECONSTRUCTION].isEnabled()
