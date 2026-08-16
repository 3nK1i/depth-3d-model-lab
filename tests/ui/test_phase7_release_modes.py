"""Visible Phase 7 release identity and operational-mode navigation."""

from __future__ import annotations

from app.widgets.header_widget import APP_VERSION, HeaderWidget
from app.widgets.mode_tabs import AppMode, ModeTabs
from core.version import RELEASE_NAME, __version__


def test_phase7_header_uses_central_version(qtbot) -> None:
    header = HeaderWidget()
    qtbot.addWidget(header)
    assert APP_VERSION == f"v{__version__}" == "v0.9.0"
    assert RELEASE_NAME == "Phase 9 Digital Twin Stable"


def test_phase7_operational_modes_remain_enabled_with_b5_reconstruction(qtbot) -> None:
    tabs = ModeTabs()
    qtbot.addWidget(tabs)
    expected = set(AppMode)
    for mode, button in tabs._buttons.items():
        assert button.isEnabled() is (mode in expected)


def test_repeated_mode_navigation_remains_consistent(qtbot) -> None:
    tabs = ModeTabs()
    qtbot.addWidget(tabs)
    sequence = (
        AppMode.LIVE_3D,
        AppMode.INSPECTION,
        AppMode.BENCHMARK,
        AppMode.SNAPSHOT,
        AppMode.STABLE_VIDEO,
        AppMode.RECONSTRUCTION,
    )
    for _ in range(25):
        for mode in sequence:
            tabs.set_active_mode(mode)
            assert tabs.active_mode is mode
            assert tabs._buttons[mode].isChecked()
