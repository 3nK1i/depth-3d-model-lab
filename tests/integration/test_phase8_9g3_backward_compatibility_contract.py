from __future__ import annotations

import pytest

try:
    from app.main_window import MainWindow
    from app.widgets.mode_tabs import AppMode
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("PyQt6"):
        MainWindow = None  # type: ignore[assignment]
        AppMode = None  # type: ignore[assignment]
    else:
        raise

pytestmark = pytest.mark.skipif(MainWindow is None, reason="PyQt6 is required for the actual MainWindow G3 gate")


@pytest.fixture
def window(qtbot, monkeypatch):
    # Do not start a real camera enumeration thread during the bounded release gate.
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    win.resize(1366, 768)
    yield win
    if win.isVisible():
        win.close()


def test_g3_legacy_operational_modes_remain_additive_and_route_to_existing_pages(window, qtbot) -> None:
    legacy_pages = {
        AppMode.LIVE_3D: window._page,
        AppMode.INSPECTION: window._inspection_page,
        AppMode.STABLE_VIDEO: window._stable_video_page,
        AppMode.SNAPSHOT: window._snapshot_page,
        AppMode.BENCHMARK: window._benchmark_page,
    }
    assert set(legacy_pages) == set(AppMode) - {AppMode.RECONSTRUCTION}
    assert all(window._mode_tabs._buttons[mode].isEnabled() for mode in AppMode)

    # Startup remains the historical Live 3D surface.
    assert window._mode_tabs.active_mode is AppMode.LIVE_3D
    assert window._mode_stack.currentWidget() is window._page

    # Reconstruction is additive: every legacy page remains the exact existing page.
    for mode, expected_page in legacy_pages.items():
        window._mode_tabs.set_active_mode(mode)
        qtbot.wait(5)
        assert window._mode_tabs.active_mode is mode
        assert window._mode_stack.currentWidget() is expected_page

    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    qtbot.wait(5)
    assert window._mode_stack.currentWidget() is window._reconstruction_page
    assert window._reconstruction_controller.is_active is False

    window._mode_tabs.set_active_mode(AppMode.LIVE_3D)
    qtbot.wait(5)
    assert window._mode_stack.currentWidget() is window._page


def test_g3_actual_main_window_shutdown_is_clean_without_starting_reconstruction(window, qtbot) -> None:
    reconstruction = window._reconstruction_controller
    application = window._controller

    assert reconstruction.is_active is False
    assert reconstruction.is_shutdown is False
    assert application._shutdown_started is False

    window.show()
    qtbot.wait(25)
    window.close()
    qtbot.wait(25)

    assert window.isVisible() is False
    assert reconstruction.is_shutdown is True
    assert reconstruction.is_active is False
    assert application._shutdown_started is True
