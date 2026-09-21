from __future__ import annotations

from pathlib import Path

import pytest

from app.main_window import MainWindow
from app.offline_reconstruction_controller import OfflineReconstructionRunState
from app.reconstruction_controller import ReconstructionWorkflow


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    return win


def test_i6_1c_initial_workspace_binding_is_passive_and_explicit(window) -> None:
    page = window._reconstruction_page

    assert page._workspace_value.text() == "Offline idle"
    assert page._workspace_value.property("statusKind") == "neutral"
    assert page._workspace_authority_value.text() == "PyCOLMAP • scale unqualified"
    assert "Production Auto: deferred" in page._workspace_authority_value.toolTip()
    assert window._reconstruction_controller.is_active is False


def test_i6_1c_image_sequence_selection_updates_workspace_without_starting_worker(
    window, tmp_path: Path
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    for index in range(4):
        (images / f"frame-{index:02d}.jpg").write_bytes(b"image")

    controller = window._reconstruction_controller
    controller.select_image_sequence(images)

    page = window._reconstruction_page
    assert controller.source_selection.workflow is ReconstructionWorkflow.IMAGE_SEQUENCE
    assert page._workspace_value.text() == "Ready to reconstruct"
    assert page._workspace_value.property("statusKind") == "ready"
    assert controller.offline_is_running is False


def test_i6_1c_progress_event_updates_workspace_summary_but_preserves_runtime_message(window) -> None:
    controller = window._reconstruction_controller
    controller._offline_controller._state = OfflineReconstructionRunState.RUNNING
    controller._on_offline_state_changed(OfflineReconstructionRunState.RUNNING)
    controller._on_offline_event_received(
        {
            "event": "progress",
            "stage": "camera_pose",
            "message": "Registering sparse cameras.",
            "progress_fraction": 0.5,
        }
    )

    page = window._reconstruction_page
    assert page._workspace_value.text() == "Reconstructing"
    assert page._workspace_value.property("statusKind") == "running"
    assert "Camera pose" in page._workspace_value.toolTip()
    assert page._runtime_message.text() == "Registering sparse cameras."


def test_i6_1c_live_workflow_does_not_misrepresent_offline_workspace(window) -> None:
    page = window._reconstruction_page
    page._source_buttons[ReconstructionWorkflow.LIVE_SOURCE].click()

    assert page._workspace_value.text() == "Offline idle"
    assert "image sequence or recorded video" in page._workspace_value.toolTip().lower()
