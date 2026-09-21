from __future__ import annotations

import pytest

from app.main_window import MainWindow
from core.reconstruction.reconstruction_progressive_sfm_contract import ProgressiveSfmStage
from tests.i6_6d_helpers import make_progressive_view_snapshot


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    yield win
    win.close()


def test_i6_6d_first_progressive_snapshot_populates_and_fits_once(window) -> None:
    page = window._reconstruction_page
    canvas = page._reconstruction_canvas
    page.set_progressive_sparse_snapshot(make_progressive_view_snapshot(revision=1))
    assert canvas.progressive_sfm_active is True
    assert canvas.progressive_sfm_revision == 1
    assert canvas.progressive_sfm_stage == "mapping"
    assert canvas.keyframe_count == 7
    assert canvas.sparse_landmark_count == 101
    assert canvas.progressive_camera_path_display_count == 3
    assert canvas.fit_calls == 1
    assert page._viewport_placeholder.isHidden() is True
    assert "provisional" in page._viewport_state.text().lower()


def test_i6_6d_new_revision_replaces_preview_without_refitting(window) -> None:
    page = window._reconstruction_page
    canvas = page._reconstruction_canvas
    page.set_progressive_sparse_snapshot(
        make_progressive_view_snapshot(revision=1, camera_sample_count=3, landmark_sample_count=5)
    )
    page.set_progressive_sparse_snapshot(
        make_progressive_view_snapshot(
            revision=2,
            camera_sample_count=4,
            landmark_sample_count=2,
            registered_camera_total=9,
            landmark_total=88,
        )
    )
    assert canvas.progressive_sfm_revision == 2
    assert canvas.keyframe_display_count == 4
    assert canvas.keyframe_count == 9
    assert canvas.sparse_landmark_display_count == 2
    assert canvas.sparse_landmark_count == 88
    assert canvas.fit_calls == 1


def test_i6_6d_finalizing_revision_is_still_provisional(window) -> None:
    page = window._reconstruction_page
    canvas = page._reconstruction_canvas
    page.set_progressive_sparse_snapshot(make_progressive_view_snapshot(revision=1))
    page.set_progressive_sparse_snapshot(
        make_progressive_view_snapshot(revision=2, stage=ProgressiveSfmStage.FINALIZING)
    )
    assert canvas.progressive_sfm_stage == "finalizing"
    assert "finalizing" in page._viewport_state.text().lower()
    assert "provisional" in page._viewport_state.text().lower()
    assert canvas.offline_visualization_loaded is False


def test_i6_6d_none_clears_only_provisional_sparse_preview(window) -> None:
    page = window._reconstruction_page
    canvas = page._reconstruction_canvas
    page.set_progressive_sparse_snapshot(make_progressive_view_snapshot(revision=1))
    page.set_progressive_sparse_snapshot(None)
    assert canvas.progressive_sfm_active is False
    assert canvas.sparse_landmark_count == 0
    assert canvas.keyframe_count == 0
    assert page._viewport_placeholder.isHidden() is False


def test_i6_6d_completed_offline_visualization_supersedes_provisional_buffers(window) -> None:
    page = window._reconstruction_page
    canvas = page._reconstruction_canvas
    snapshot = make_progressive_view_snapshot(revision=1)
    page.set_progressive_sparse_snapshot(snapshot)
    frame = snapshot.geometry
    assert frame is not None
    import numpy as np

    matrices = np.asarray(
        [sample.T_reconstruction_from_camera for sample in frame.camera_samples],
        dtype=np.float64,
    )
    points = np.asarray(
        [sample.position_reconstruction for sample in frame.landmark_samples],
        dtype=np.float64,
    )
    canvas.show_offline_sparse_reconstruction(
        session_id=snapshot.session_id,
        scale_status="arbitrary",
        points_reconstruction=points,
        camera_matrices_reconstruction=matrices,
        camera_ids=tuple(sample.source_image_name for sample in frame.camera_samples),
        landmark_total=snapshot.statistics.sparse_landmarks,
        registered_camera_total=snapshot.statistics.registered_cameras,
    )
    assert canvas.progressive_sfm_active is False
    assert canvas.offline_visualization_loaded is True


def test_i6_6d_bound_controller_signal_reaches_viewer(window, qtbot) -> None:
    controller = window._reconstruction_controller
    page = window._reconstruction_page
    snapshot = make_progressive_view_snapshot(revision=1)
    controller.progressive_sparse_snapshot_changed.emit(snapshot)
    qtbot.waitUntil(lambda: page._reconstruction_canvas.progressive_sfm_revision == 1)
    assert page._reconstruction_canvas.progressive_sfm_active is True
    assert page._reconstruction_canvas.keyframe_display_count == 3
