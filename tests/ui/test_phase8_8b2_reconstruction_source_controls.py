from __future__ import annotations

from pathlib import Path

import pytest
from PyQt6.QtWidgets import QPushButton

from app.main_window import MainWindow
from app.reconstruction_controller import ReconstructionWorkflow
from app.widgets.mode_tabs import AppMode


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    return win


def test_b2_source_controls_remain_safe_after_b5_navigation_enablement(window) -> None:
    page = window._reconstruction_page

    assert len(page._source_buttons) == 3
    assert all(button.isEnabled() for button in page._source_buttons.values())
    runtime = [
        button
        for button in page.findChildren(QPushButton)
        if button.objectName() == "reconstructionRuntimeAction"
    ]
    # B2 verifies the original Start/Pause/Stop controls. G2-G4 add separate
    # persistence/resume/recovery actions that intentionally reuse this style.
    original_runtime = (
        page._start_button,
        page._pause_button,
        page._stop_button,
    )
    assert all(button in runtime for button in original_runtime)
    assert all(not button.isEnabled() for button in original_runtime)
    assert window._mode_tabs._buttons[AppMode.RECONSTRUCTION].isEnabled() is True
    assert window._reconstruction_controller.is_active is False


def test_b2_image_sequence_dialog_updates_controller_and_page(
    window, qtbot, monkeypatch, tmp_path: Path
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    for index in range(4):
        (images / f"frame-{index:02d}.jpg").write_bytes(b"image")
    monkeypatch.setattr(
        "app.widgets.reconstruction_page.QFileDialog.getExistingDirectory",
        lambda *_args, **_kwargs: str(images),
    )

    page = window._reconstruction_page
    page._source_buttons[ReconstructionWorkflow.IMAGE_SEQUENCE].click()

    selection = window._reconstruction_controller.source_selection
    assert selection.workflow is ReconstructionWorkflow.IMAGE_SEQUENCE
    assert selection.ready is True
    assert selection.item_count == 4
    assert page._source_value.text() == "images"
    assert page._workflow_value.text() == "Image sequence"
    assert page._source_readiness_value.text() == "Ready"
    assert page._workflow_input_value.text() == "4 images"
    assert window._reconstruction_controller.is_active is False


def test_b2_recorded_video_dialog_updates_controller_without_starting_worker(
    window, qtbot, monkeypatch, tmp_path: Path
) -> None:
    video = tmp_path / "capture.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(
        "app.widgets.reconstruction_page.QFileDialog.getOpenFileName",
        lambda *_args, **_kwargs: (str(video), "Video files"),
    )

    page = window._reconstruction_page
    page._source_buttons[ReconstructionWorkflow.RECORDED_VIDEO].click()

    selection = window._reconstruction_controller.source_selection
    assert selection.workflow is ReconstructionWorkflow.RECORDED_VIDEO
    assert selection.ready is True
    assert page._source_value.text() == video.name
    assert page._source_readiness_value.text() == "Ready"
    assert window._reconstruction_controller.offline_is_running is False


def test_b2_live_source_without_live3d_capture_is_selected_but_not_ready(window, qtbot) -> None:
    page = window._reconstruction_page
    page._source_buttons[ReconstructionWorkflow.LIVE_SOURCE].click()

    selection = window._reconstruction_controller.source_selection
    assert selection.workflow is ReconstructionWorkflow.LIVE_SOURCE
    assert selection.ready is False
    assert page._workflow_value.text() == "Live source"
    assert page._source_readiness_value.text() == "Needs attention"
    assert "Live 3D" in selection.detail
    assert window._reconstruction_controller.is_active is False


def test_b2_cancelled_file_dialog_preserves_previous_selection(
    window, qtbot, monkeypatch, tmp_path: Path
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    for index in range(3):
        (images / f"{index}.png").write_bytes(b"x")
    controller = window._reconstruction_controller
    previous = controller.select_image_sequence(images)
    monkeypatch.setattr(
        "app.widgets.reconstruction_page.QFileDialog.getOpenFileName",
        lambda *_args, **_kwargs: ("", ""),
    )

    page = window._reconstruction_page
    page._source_buttons[ReconstructionWorkflow.RECORDED_VIDEO].click()

    assert controller.source_selection == previous
    assert page._source_buttons[ReconstructionWorkflow.IMAGE_SEQUENCE].isChecked() is True
    assert page._source_buttons[ReconstructionWorkflow.RECORDED_VIDEO].isChecked() is False
