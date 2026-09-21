from __future__ import annotations

from pathlib import Path

import pytest

from app.main_window import MainWindow
from app.offline_reconstruction_controller import OfflineReconstructionRunState
from app.reconstruction_offline_result_handoff import (
    OfflineReconstructionResultHandoff,
    OfflineResultHandoffAvailability,
)


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    yield win
    win.close()


def _ready_source(window, tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    for index in range(4):
        (images / f"frame-{index:02d}.jpg").write_bytes(b"image")
    window._reconstruction_controller.select_image_sequence(images)


def _handoff(tmp_path: Path) -> OfflineReconstructionResultHandoff:
    root = (tmp_path / "run").resolve()
    root.mkdir(exist_ok=True)
    return OfflineReconstructionResultHandoff(
        session_id="11111111-1111-4111-8111-111111111111",
        output_dir=str(root),
        result_path=str(root / "offline_result.json"),
        status="completed",
        availability=OfflineResultHandoffAvailability.PARTIAL,
        camera_pose_available=True,
        canonical_geometry_authoritative=False,
        persistent_scene_ready=False,
        certified_metrology=False,
        scale_status="arbitrary",
        visualization_path=str(root / "offline_visualization.json"),
        sparse_reconstruction_path=str(root / "sparse_reconstruction.ply"),
        viewer_ready=True,
        sparse_export_ready=True,
        warnings=(),
    )


def test_ready_offline_source_explains_one_start_scope(window, tmp_path: Path) -> None:
    _ready_source(window, tmp_path)
    page = window._reconstruction_page

    assert page._start_button.isEnabled() is True
    assert page._start_button.text() == "Start"
    assert "sparse SfM" in page._start_button.toolTip()
    assert "dense/canonical" in page._start_button.toolTip()


def test_running_offline_job_blocks_duplicate_start_and_explains_controls(window, tmp_path: Path) -> None:
    _ready_source(window, tmp_path)
    controller = window._reconstruction_controller
    page = window._reconstruction_page
    controller._offline_controller._state = OfflineReconstructionRunState.RUNNING

    page.set_offline_state(OfflineReconstructionRunState.RUNNING)

    assert page._start_button.isEnabled() is False
    assert "Duplicate Start" in page._start_button.toolTip()
    assert page._pause_button.isEnabled() is False
    assert "not available" in page._pause_button.toolTip()
    assert page._stop_button.isEnabled() is True
    assert page._stop_button.text() == "Cancel"


def test_admitted_partial_completion_is_explicitly_not_dense_final(window, tmp_path: Path) -> None:
    _ready_source(window, tmp_path)
    page = window._reconstruction_page
    page.set_offline_result_handoff(_handoff(tmp_path))

    page.set_offline_state(OfflineReconstructionRunState.COMPLETED)

    assert page._stage_value.text() == "Sparse reconstruction complete"
    assert page._stage_value.property("statusKind") == "warning"
    assert "partial result" in page._runtime_message.text().lower()
    assert "dense/canonical reconstruction has not run" in page._runtime_message.text().lower()
    assert page._start_button.text() == "Run again"
    assert page._start_button.isEnabled() is True


def test_completed_worker_with_rejected_product_handoff_keeps_error_visible(window, tmp_path: Path) -> None:
    _ready_source(window, tmp_path)
    page = window._reconstruction_page
    page.set_error("Offline reconstruction result handoff rejected result: session mismatch")

    page.set_offline_state(OfflineReconstructionRunState.COMPLETED)

    assert page._stage_value.text() == "Result handoff rejected"
    assert page._stage_value.property("statusKind") == "error"
    assert "session mismatch" in page._runtime_message.text().lower()
    assert page._start_button.text() == "Run again"


def test_preflight_error_remains_user_facing_after_control_refresh(window, tmp_path: Path) -> None:
    _ready_source(window, tmp_path)
    page = window._reconstruction_page
    page.set_error(
        "Offline One Start preflight blocked: output directory became non-empty before launch."
    )

    page._refresh_runtime_controls()

    assert page._runtime_message.property("statusKind") == "error"
    assert "preflight blocked" in page._runtime_message.text().lower()
    assert "output directory" in page._runtime_message.text().lower()
