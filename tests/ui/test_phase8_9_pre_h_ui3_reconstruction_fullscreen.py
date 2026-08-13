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
    qtbot.wait(40)
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    qtbot.wait(30)
    return window


def test_reconstruction_fullscreen_button_focuses_existing_canvas_and_restores(
    qtbot, monkeypatch
) -> None:
    window = _window(qtbot, monkeypatch)
    page = window._reconstruction_page
    button = page._viewport_tool_buttons["fullscreen"]
    native = page._reconstruction_canvas.native

    assert page.viewport_focus_mode is False
    assert window._focus_target is None
    assert not button.isChecked()

    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    qtbot.wait(20)

    assert window._focus_target == "reconstruction"
    assert page.viewport_focus_mode is True
    assert window._header.isHidden()
    assert window._mode_tabs.isHidden()
    assert page._header_card.isHidden()
    assert page._source_card.isHidden()
    assert page._runtime_card.isHidden()
    assert page._settings_panel.isHidden()
    assert not page._viewport_card.isHidden()
    assert page._reconstruction_canvas.native is native
    assert button.isChecked()
    assert "Restore" in button.toolTip()

    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    qtbot.wait(20)

    assert window._focus_target is None
    assert page.viewport_focus_mode is False
    assert not window._header.isHidden()
    assert not window._mode_tabs.isHidden()
    assert not page._header_card.isHidden()
    assert not page._source_card.isHidden()
    assert not page._runtime_card.isHidden()
    assert not page._settings_panel.isHidden()
    assert page._reconstruction_canvas.native is native
    assert not button.isChecked()

    window.close()


def test_escape_restores_reconstruction_workspace_from_fullscreen(qtbot, monkeypatch) -> None:
    window = _window(qtbot, monkeypatch)
    page = window._reconstruction_page
    button = page._viewport_tool_buttons["fullscreen"]

    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    assert window._focus_target == "reconstruction"

    qtbot.keyClick(window, Qt.Key.Key_Escape)
    qtbot.wait(20)

    assert window._focus_target is None
    assert page.viewport_focus_mode is False
    assert not button.isChecked()
    assert not window._header.isHidden()
    assert not window._mode_tabs.isHidden()

    window.close()


def test_reconstruction_fullscreen_does_not_break_live3d_focus_contract(qtbot, monkeypatch) -> None:
    window = _window(qtbot, monkeypatch)
    page = window._reconstruction_page

    qtbot.mouseClick(page._viewport_tool_buttons["fullscreen"], Qt.MouseButton.LeftButton)
    qtbot.keyClick(window, Qt.Key.Key_Escape)
    assert window._focus_target is None

    window._mode_tabs.set_active_mode(AppMode.LIVE_3D)
    qtbot.wait(20)
    live_button = window._viewport_panel._tools._fullscreen_btn
    qtbot.mouseClick(live_button, Qt.MouseButton.LeftButton)

    assert window._focus_target == "viewport"
    assert window._viewport_focus_mode is True
    assert live_button.isChecked()

    qtbot.keyClick(window, Qt.Key.Key_Escape)
    assert window._focus_target is None
    assert window._viewport_focus_mode is False
    assert not live_button.isChecked()

    window.close()



def test_escape_restores_reconstruction_workspace_after_canvas_takes_focus(
    qtbot, monkeypatch
) -> None:
    """Regression: orbit/pan interaction moves keyboard focus into the VisPy native."""

    window = _window(qtbot, monkeypatch)
    page = window._reconstruction_page
    button = page._viewport_tool_buttons["fullscreen"]
    native = page._reconstruction_canvas.native

    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    assert window._focus_target == "reconstruction"

    # This is the focus state produced by interacting with/orbiting the canvas.
    native.setFocus(Qt.FocusReason.MouseFocusReason)
    qtbot.wait(10)
    assert native.hasFocus()

    qtbot.keyClick(native, Qt.Key.Key_Escape)
    qtbot.wait(20)

    assert window._focus_target is None
    assert page.viewport_focus_mode is False
    assert not button.isChecked()
    assert not window._header.isHidden()
    assert not window._mode_tabs.isHidden()

    window.close()
