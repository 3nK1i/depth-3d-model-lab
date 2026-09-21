from __future__ import annotations

from app.reconstruction_offline_result_handoff import (
    OfflineReconstructionResultHandoff,
    OfflineResultHandoffAvailability,
)
from app.reconstruction_one_start_ux import present_offline_one_start_ux


def _handoff(*, viewer_ready: bool = True) -> OfflineReconstructionResultHandoff:
    return OfflineReconstructionResultHandoff(
        session_id="11111111-1111-4111-8111-111111111111",
        output_dir="/tmp/i6-2d",
        result_path="/tmp/i6-2d/offline_result.json",
        status="completed",
        availability=OfflineResultHandoffAvailability.PARTIAL,
        camera_pose_available=True,
        canonical_geometry_authoritative=False,
        persistent_scene_ready=False,
        certified_metrology=False,
        scale_status="arbitrary",
        visualization_path="/tmp/i6-2d/offline_visualization.json" if viewer_ready else "",
        sparse_reconstruction_path="/tmp/i6-2d/sparse_reconstruction.ply",
        viewer_ready=viewer_ready,
        sparse_export_ready=True,
        warnings=() if viewer_ready else ("Sparse viewer artifact unavailable.",),
    )


def test_ready_source_explains_sparse_one_start_scope() -> None:
    ux = present_offline_one_start_ux("idle", source_ready=True)
    assert ux.start_text == "Start"
    assert "sparse sfm" in ux.start_tooltip.lower()
    assert "dense" in ux.start_tooltip.lower()
    assert ux.terminal_stage_label == ""


def test_running_state_qualifies_duplicate_start_pause_and_cancel() -> None:
    ux = present_offline_one_start_ux("running", source_ready=True)
    assert "already active" in ux.start_tooltip.lower()
    assert "duplicate" in ux.start_tooltip.lower()
    assert "not available" in ux.pause_tooltip.lower()
    assert ux.stop_text == "Cancel"
    assert "cancel" in ux.stop_tooltip.lower()


def test_completed_partial_result_is_not_presented_as_final_dense_scene() -> None:
    ux = present_offline_one_start_ux(
        "completed",
        source_ready=True,
        handoff=_handoff(),
    )
    assert ux.start_text == "Run again"
    assert ux.terminal_stage_label == "Sparse reconstruction complete"
    assert ux.terminal_status_kind == "warning"
    assert ux.terminal_progress_fraction == 1.0
    assert "partial result" in ux.terminal_message.lower()
    assert "dense/canonical reconstruction has not run" in ux.terminal_message.lower()
    assert "arbitrary" in ux.terminal_message.lower()


def test_completed_result_without_viewer_artifact_remains_admitted_but_warns() -> None:
    ux = present_offline_one_start_ux(
        "completed",
        source_ready=True,
        handoff=_handoff(viewer_ready=False),
    )
    assert ux.terminal_status_kind == "warning"
    assert "viewer artifact" in ux.terminal_message.lower()
    assert "result remains admitted" in ux.terminal_message.lower()


def test_completed_worker_without_admitted_handoff_fails_closed_in_product_ui() -> None:
    ux = present_offline_one_start_ux(
        "completed",
        source_ready=True,
        handoff=None,
        last_error="Offline reconstruction result handoff rejected result: session mismatch",
    )
    assert ux.terminal_stage_label == "Result handoff rejected"
    assert ux.terminal_status_kind == "error"
    assert "session mismatch" in ux.terminal_message.lower()
    assert ux.start_text == "Run again"


def test_cancelled_run_offers_new_run_without_claiming_result() -> None:
    ux = present_offline_one_start_ux("cancelled", source_ready=True)
    assert ux.start_text == "Start again"
    assert "new offline sparse reconstruction" in ux.start_tooltip.lower()
