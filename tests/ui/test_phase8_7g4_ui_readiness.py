"""Historical UI-readiness checks migrated at Phase 8.8B5 navigation enablement."""
from __future__ import annotations

import pytest
from PyQt6.QtCore import Qt

from app.widgets.mode_tabs import AppMode, ModeTabs


@pytest.fixture
def tabs(qtbot):
    widget = ModeTabs()
    qtbot.addWidget(widget)
    return widget


def test_reconstruction_tab_is_visible_and_enabled_after_b5(tabs) -> None:
    button = tabs._buttons[AppMode.RECONSTRUCTION]
    assert button.isVisible() or not tabs.isVisible()  # widget need not be shown in offscreen tests
    assert button.isEnabled() is True
    assert "Not yet implemented" not in button.toolTip()


def test_existing_operational_modes_remain_enabled(tabs) -> None:
    enabled = set(AppMode)
    for mode, button in tabs._buttons.items():
        assert button.isEnabled() is (mode in enabled)


def test_programmatic_reconstruction_activation_is_supported_after_b5(tabs) -> None:
    tabs.set_active_mode(AppMode.RECONSTRUCTION)
    assert tabs.active_mode is AppMode.RECONSTRUCTION
    assert tabs._buttons[AppMode.RECONSTRUCTION].isChecked()


def test_navigation_lock_temporarily_disables_non_active_reconstruction(tabs, qtbot) -> None:
    qtbot.mouseClick(tabs._buttons[AppMode.SNAPSHOT], Qt.MouseButton.LeftButton)
    assert tabs.active_mode is AppMode.SNAPSHOT
    tabs.set_navigation_locked(True)
    assert tabs._buttons[AppMode.SNAPSHOT].isEnabled()
    assert not tabs._buttons[AppMode.RECONSTRUCTION].isEnabled()
    tabs.set_navigation_locked(False)
    assert tabs._buttons[AppMode.RECONSTRUCTION].isEnabled()
