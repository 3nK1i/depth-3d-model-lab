from __future__ import annotations

from PyQt6.QtCore import Qt

from app.main_window import MainWindow
from app.widgets.mode_tabs import AppMode


def _window(qtbot, monkeypatch) -> MainWindow:
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1366, 768)
    window.show()
    qtbot.wait(50)
    return window


def test_combined_mode_cycle_preserves_ui2_layout_and_ui3_focus_contract(
    qtbot, monkeypatch
) -> None:
    window = _window(qtbot, monkeypatch)

    expected_pages = {
        AppMode.LIVE_3D: window._page,
        AppMode.INSPECTION: window._inspection_page,
        AppMode.STABLE_VIDEO: window._stable_video_page,
        AppMode.SNAPSHOT: window._snapshot_page,
        AppMode.RECONSTRUCTION: window._reconstruction_page,
        AppMode.BENCHMARK: window._benchmark_page,
    }

    for mode in (
        AppMode.LIVE_3D,
        AppMode.SNAPSHOT,
        AppMode.STABLE_VIDEO,
        AppMode.INSPECTION,
        AppMode.RECONSTRUCTION,
        AppMode.BENCHMARK,
        AppMode.LIVE_3D,
    ):
        window._mode_tabs.set_active_mode(mode)
        qtbot.wait(15)
        assert window._mode_tabs.active_mode is mode
        assert window._mode_stack.currentWidget() is expected_pages[mode]
        assert window._focus_target is None

    window.close()


def test_reconstruction_escape_after_canvas_focus_then_live3d_fullscreen_still_works(
    qtbot, monkeypatch
) -> None:
    window = _window(qtbot, monkeypatch)
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    qtbot.wait(20)

    page = window._reconstruction_page
    reconstruction_native = page._reconstruction_canvas.native
    reconstruction_fullscreen = page._viewport_tool_buttons["fullscreen"]

    qtbot.mouseClick(reconstruction_fullscreen, Qt.MouseButton.LeftButton)
    qtbot.wait(15)
    assert window._focus_target == "reconstruction"
    assert page._reconstruction_canvas.native is reconstruction_native

    # Match the real failure path found during UI-3 acceptance: orbiting/panning
    # moves keyboard focus into the native VisPy widget before Escape is pressed.
    reconstruction_native.setFocus(Qt.FocusReason.MouseFocusReason)
    qtbot.keyClick(reconstruction_native, Qt.Key.Key_Escape)
    qtbot.wait(20)

    assert window._focus_target is None
    assert page.viewport_focus_mode is False
    assert not reconstruction_fullscreen.isChecked()

    window._mode_tabs.set_active_mode(AppMode.LIVE_3D)
    qtbot.wait(20)
    live_canvas = window._inference_controller._point_cloud_canvas
    live_native = live_canvas.native
    live_fullscreen = window._viewport_panel._tools._fullscreen_btn
    qtbot.mouseClick(live_fullscreen, Qt.MouseButton.LeftButton)
    qtbot.wait(15)

    assert window._focus_target == "viewport"
    assert live_canvas.native is live_native
    assert live_fullscreen.isChecked()

    qtbot.keyClick(window, Qt.Key.Key_Escape)
    qtbot.wait(15)
    assert window._focus_target is None
    assert window._viewport_focus_mode is False
    assert not live_fullscreen.isChecked()

    window.close()


def test_combined_shared_viewports_remain_compact_safe_after_mode_switching(
    qtbot, monkeypatch
) -> None:
    window = _window(qtbot, monkeypatch)
    window.resize(1100, 690)
    qtbot.wait(40)

    # UI-2 contract: every shared viewport must negotiate a short/windowed area
    # without losing controls or forcing the canvas back to the legacy 260 px
    # minimum. Exercise them after switching modes so UI-3 focus wiring cannot
    # leave stale state behind.
    for mode, viewport in (
        (AppMode.LIVE_3D, window._viewport_panel),
        (AppMode.SNAPSHOT, window._snapshot_page._viewport),
        (AppMode.STABLE_VIDEO, window._stable_video_page._viewport),
    ):
        window._mode_tabs.set_active_mode(mode)
        viewport.resize(560, 300)
        qtbot.wait(25)
        assert viewport._tools.compact is True
        assert viewport._render_card.compact is True
        assert viewport._canvas.minimumHeight() == 180
        assert viewport._render_card._reset_btn.isVisible()

    window.close()


def test_reconstruction_toolbar_actions_remain_real_after_combined_polish(
    qtbot, monkeypatch
) -> None:
    window = _window(qtbot, monkeypatch)
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    qtbot.wait(20)
    page = window._reconstruction_page

    expected = {
        "orbit",
        "pan",
        "zoom",
        "fullscreen",
        "grid",
        "trajectory",
        "landmarks",
        "keyframes",
        "offline_envelope",
        "semantic_objects",
        "diagnostics",
    }
    assert expected.issubset(page._viewport_tool_buttons)

    for mode in ("orbit", "pan", "zoom"):
        qtbot.mouseClick(page._viewport_tool_buttons[mode], Qt.MouseButton.LeftButton)
        assert page._reconstruction_canvas.interaction_mode == mode

    for name in ("grid", "trajectory", "landmarks", "keyframes", "offline_envelope"):
        button = page._viewport_tool_buttons[name]
        before = button.isChecked()
        qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
        assert button.isChecked() is (not before)

    window.close()
