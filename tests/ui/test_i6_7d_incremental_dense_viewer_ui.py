"""I6.7D UI qualification; requires the workstation's PyQt6+VisPy+OpenGL."""
from __future__ import annotations

import pytest

from app.main_window import MainWindow
from core.reconstruction.reconstruction_progressive_dense_contract import ProgressiveDenseStage
from core.reconstruction.reconstruction_progressive_dense_transport import (
    ProgressiveDenseTransportBinding, ProgressiveDenseOfferDisposition,
    progressive_dense_transport_line,
)
from tests.i6_7c_helpers import make_snapshot


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    yield win
    win.close()


def test_i67d_real_transport_line_renders_bounded_independent_provisional_gpu_layer(window, qtbot):
    page = window._reconstruction_page
    canvas = page._reconstruction_canvas
    first = make_snapshot(revision=1)
    geometry = make_snapshot(revision=2, geometry_points=32)
    page.begin_progressive_dense_preview(ProgressiveDenseTransportBinding.from_snapshot(first))
    assert page.offer_progressive_dense_transport_line(progressive_dense_transport_line(first)) is ProgressiveDenseOfferDisposition.ACCEPTED
    assert page.offer_progressive_dense_transport_line(progressive_dense_transport_line(geometry)) is ProgressiveDenseOfferDisposition.COALESCED
    qtbot.waitUntil(lambda: canvas.progressive_dense_revision == 2, timeout=3000)
    assert canvas.progressive_dense_active
    assert canvas.progressive_dense_display_count == 32
    assert canvas.progressive_dense_gpu_uploads >= 1
    assert page._progressive_dense_geometry_uploads == 1
    assert canvas.dense_rgb_points_loaded is False  # never promotes persistent geometry
    assert "provisional" in page._viewport_state.text().lower()
    assert "non-metric" in page._viewport_state.text().lower()


def test_i67d_status_only_frame_avoids_gpu_reupload_and_clear_isolated(window, qtbot):
    page = window._reconstruction_page
    canvas = page._reconstruction_canvas
    snapshot = make_snapshot(revision=1, geometry_points=6)
    page.begin_progressive_dense_preview(ProgressiveDenseTransportBinding.from_snapshot(snapshot))
    page.offer_progressive_dense_transport_line(progressive_dense_transport_line(snapshot))
    qtbot.waitUntil(lambda: canvas.progressive_dense_revision == 1, timeout=3000)
    baseline_uploads = canvas.progressive_dense_gpu_uploads
    next_snapshot = make_snapshot(revision=2, stage=ProgressiveDenseStage.FINALIZATION)
    page.offer_progressive_dense_transport_line(progressive_dense_transport_line(next_snapshot))
    qtbot.waitUntil(lambda: canvas.progressive_dense_revision == 2, timeout=3000)
    assert canvas.progressive_dense_gpu_uploads == baseline_uploads
    assert canvas.progressive_dense_display_count == 6
    page.clear_progressive_dense_preview()
    assert page._viewport_placeholder.isHidden() is False
    assert canvas.progressive_dense_active is False
    assert canvas.progressive_dense_display_count == 0
    assert canvas.dense_rgb_points_loaded is False


def test_i67d_rejects_foreign_run_without_mutating_viewer(window, qtbot):
    page = window._reconstruction_page
    canvas = page._reconstruction_canvas
    first = make_snapshot(revision=1, geometry_points=4)
    page.begin_progressive_dense_preview(ProgressiveDenseTransportBinding.from_snapshot(first))
    page.offer_progressive_dense_transport_line(progressive_dense_transport_line(first))
    qtbot.waitUntil(lambda: canvas.progressive_dense_revision == 1, timeout=3000)
    foreign = make_snapshot(
        revision=2, dense_run_id="bbbbbbbb-cccc-4ddd-8eee-ffffffffffff",
        geometry_points=3,
    )
    before_uploads = canvas.progressive_dense_gpu_uploads
    assert page.offer_progressive_dense_transport_line(progressive_dense_transport_line(foreign)) is ProgressiveDenseOfferDisposition.REJECTED
    assert canvas.progressive_dense_revision == 1
    assert canvas.progressive_dense_display_count == 4
    assert canvas.progressive_dense_gpu_uploads == before_uploads


def test_i67d_queued_parsed_snapshot_path_and_completed_preview_remain_provisional(window, qtbot):
    from app.offline_reconstruction_controller import OfflineReconstructionRunState

    page = window._reconstruction_page
    canvas = page._reconstruction_canvas
    first = make_snapshot(revision=1, geometry_points=8)
    page.begin_progressive_dense_preview(ProgressiveDenseTransportBinding.from_snapshot(first))
    page.progressive_dense_snapshot_received.emit(first)
    qtbot.waitUntil(lambda: canvas.progressive_dense_revision == 1, timeout=3000)
    page.set_offline_state(OfflineReconstructionRunState.COMPLETED)
    assert canvas.progressive_dense_active is True
    assert canvas.progressive_dense_display_count == 8
    assert 'provisional' in page._viewport_state.text().lower()
    assert canvas.dense_rgb_points_loaded is False


def test_i67d_new_run_clears_previous_provisional_gpu_visual(window, qtbot):
    from app.offline_reconstruction_controller import OfflineReconstructionRunState

    page = window._reconstruction_page
    canvas = page._reconstruction_canvas
    snapshot = make_snapshot(revision=1, geometry_points=3)
    page.begin_progressive_dense_preview(ProgressiveDenseTransportBinding.from_snapshot(snapshot))
    page.offer_progressive_dense_transport_line(progressive_dense_transport_line(snapshot))
    qtbot.waitUntil(lambda: canvas.progressive_dense_revision == 1, timeout=3000)
    page.set_offline_state(OfflineReconstructionRunState.STARTING)
    assert canvas.progressive_dense_revision is None
    assert canvas.progressive_dense_display_count == 0
    assert page._progressive_dense_buffer.binding is None
