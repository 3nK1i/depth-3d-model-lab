from __future__ import annotations

import pytest

from app.main_window import MainWindow
from app.offline_reconstruction_controller import OfflineReconstructionRunState
from core.reconstruction.reconstruction_progressive_sfm_contract import ProgressiveSfmStage
from tests.i6_6d_helpers import make_progressive_view_snapshot


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    yield win
    win.close()


def test_i6_6e_initial_progressive_panel_is_waiting_and_non_authoritative(window) -> None:
    page = window._reconstruction_page
    assert page._progressive_sfm_status_value.text() == "Waiting for reconstruction"
    assert page._progressive_sfm_stage_value.text() == "Not active"
    assert "provisional" in page._progressive_sfm_authority_value.text().lower()
    assert "not final" in page._progressive_sfm_boundary_value.text().lower()


def test_i6_6e_mapping_snapshot_updates_professional_counts_and_performance(window) -> None:
    page = window._reconstruction_page
    snapshot = make_progressive_view_snapshot(
        revision=4,
        camera_sample_count=3,
        landmark_sample_count=5,
        registered_camera_total=18,
        landmark_total=1200,
    )
    page.set_offline_state(OfflineReconstructionRunState.RUNNING)
    page.set_progressive_sparse_snapshot(snapshot)
    assert page._progressive_sfm_status_value.text() == "Live provisional"
    assert page._progressive_sfm_stage_value.text() == "Mapping"
    assert page._progressive_sfm_cameras_value.text() == "18 registered • 3 preview"
    assert page._progressive_sfm_landmarks_value.text() == "1,200 sparse • 5 preview"
    assert page._progressive_sfm_revision_value.text().startswith("r4")
    assert "ms latest" in page._progressive_sfm_viewer_value.text()
    payload_text = page._progressive_sfm_payload_value.text()
    assert payload_text != "No preview payload yet"
    assert "latest" in payload_text
    assert "max" in payload_text
    assert "above 128 KiB budget" not in payload_text


def test_i6_6e_finalizing_is_visible_but_remains_provisional(window) -> None:
    page = window._reconstruction_page
    page.set_offline_state(OfflineReconstructionRunState.RUNNING)
    page.set_progressive_sparse_snapshot(
        make_progressive_view_snapshot(revision=1)
    )
    page.set_progressive_sparse_snapshot(
        make_progressive_view_snapshot(revision=2, stage=ProgressiveSfmStage.FINALIZING)
    )
    assert page._progressive_sfm_status_value.text() == "Finalizing provisional"
    assert page._progressive_sfm_stage_value.text() == "Finalizing"
    assert "not final" in page._progressive_sfm_boundary_value.text().lower()


def test_i6_6e_completed_state_clears_active_preview_and_labels_final_admission_separately(window) -> None:
    page = window._reconstruction_page
    page.set_offline_state(OfflineReconstructionRunState.RUNNING)
    page.set_progressive_sparse_snapshot(make_progressive_view_snapshot(revision=1))
    page.set_progressive_sparse_snapshot(None)
    page.set_offline_state(OfflineReconstructionRunState.COMPLETED)
    assert page._reconstruction_canvas.progressive_sfm_active is False
    assert page._progressive_sfm_status_value.text() == "Progressive phase complete"
    assert page._progressive_sfm_stage_value.text() == "Final sparse result admitted separately"
    assert page._progressive_sfm_cameras_value.text().startswith("Last provisional: ")


def test_i6_6e_new_start_resets_old_progressive_performance_and_snapshot_context(window) -> None:
    page = window._reconstruction_page
    page.set_offline_state(OfflineReconstructionRunState.RUNNING)
    page.set_progressive_sparse_snapshot(make_progressive_view_snapshot(revision=1))
    assert page._progressive_sfm_viewer_performance.applied_updates == 1
    page.set_offline_state(OfflineReconstructionRunState.STARTING)
    assert page._progressive_sfm_last_snapshot is None
    assert page._progressive_sfm_viewer_performance.applied_updates == 0
    assert page._progressive_sfm_status_value.text() == "Starting"


def test_i6_6e_cancelled_and_failed_terminal_states_keep_provisional_authority_cleared(window) -> None:
    page = window._reconstruction_page
    for state, expected in (
        (OfflineReconstructionRunState.CANCELLED, "Cancelled"),
        (OfflineReconstructionRunState.FAILED, "Stopped without admission"),
    ):
        page.set_offline_state(OfflineReconstructionRunState.STARTING)
        page.set_offline_state(OfflineReconstructionRunState.RUNNING)
        page.set_progressive_sparse_snapshot(make_progressive_view_snapshot(revision=1))
        page.set_progressive_sparse_snapshot(None)
        page.set_offline_state(state)
        assert page._reconstruction_canvas.progressive_sfm_active is False
        assert page._progressive_sfm_status_value.text() == expected
        assert "provisional preview cleared" in page._progressive_sfm_stage_value.text().lower()
        assert "not final" in page._progressive_sfm_boundary_value.text().lower()
