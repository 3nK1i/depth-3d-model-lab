from __future__ import annotations

from app.reconstruction_progressive_sfm_ux import (
    I66E_PREVIEW_PAYLOAD_BUDGET_BYTES,
    I66E_VIEWER_UPDATE_TARGET_MS,
    ProgressiveSfmViewerPerformance,
    build_progressive_sfm_ux_presentation,
    progressive_viewer_payload_bytes,
)
from app.reconstruction_progressive_sfm_viewer import build_progressive_sfm_viewer_frame
from core.reconstruction.reconstruction_progressive_sfm_contract import ProgressiveSfmStage
from tests.i6_6d_helpers import make_progressive_view_snapshot


def test_i6_6e_waiting_presentation_is_explicit_and_non_authoritative() -> None:
    presentation = build_progressive_sfm_ux_presentation(
        active_snapshot=None,
        last_snapshot=None,
        run_state="idle",
    )
    assert presentation.status_text == "Waiting for reconstruction"
    assert presentation.stage_text == "Not active"
    assert presentation.status_kind == "neutral"
    assert presentation.authority_text == "PyCOLMAP provisional • arbitrary scale"
    assert presentation.boundary_text == "Not final • not persistent • not metric"
    assert presentation.active is False


def test_i6_6e_mapping_presentation_distinguishes_totals_from_bounded_preview() -> None:
    snapshot = make_progressive_view_snapshot(
        revision=7,
        camera_sample_count=3,
        landmark_sample_count=5,
        registered_camera_total=22,
        landmark_total=12345,
    )
    presentation = build_progressive_sfm_ux_presentation(
        active_snapshot=snapshot,
        last_snapshot=snapshot,
        run_state="running",
        transport_stats={"accepted": 7, "coalesced": 2, "rejected": 0, "delivered": 5},
    )
    assert presentation.status_text == "Live provisional"
    assert presentation.stage_text == "Mapping"
    assert presentation.cameras_text == "22 registered • 3 preview"
    assert presentation.landmarks_text == "12,345 sparse • 5 preview"
    assert presentation.revision_text == "r7 • 70%"
    assert presentation.quality_text == "0.450 px reprojection"
    assert presentation.transport_text == "5 delivered • 2 coalesced • 0 rejected"
    assert presentation.active is True


def test_i6_6e_finalizing_remains_provisional() -> None:
    snapshot = make_progressive_view_snapshot(
        revision=8,
        stage=ProgressiveSfmStage.FINALIZING,
    )
    presentation = build_progressive_sfm_ux_presentation(
        active_snapshot=snapshot,
        last_snapshot=snapshot,
        run_state="running",
    )
    assert presentation.status_text == "Finalizing provisional"
    assert presentation.status_kind == "warning"
    assert presentation.stage_text == "Finalizing"
    assert "Not final" in presentation.boundary_text


def test_i6_6e_completed_state_keeps_only_last_provisional_telemetry() -> None:
    snapshot = make_progressive_view_snapshot(revision=8)
    presentation = build_progressive_sfm_ux_presentation(
        active_snapshot=None,
        last_snapshot=snapshot,
        run_state="completed",
    )
    assert presentation.status_text == "Progressive phase complete"
    assert presentation.status_kind == "ready"
    assert presentation.stage_text == "Final sparse result admitted separately"
    assert presentation.cameras_text.startswith("Last provisional: ")
    assert presentation.landmarks_text.startswith("Last provisional: ")
    assert presentation.active is False


def test_i6_6e_failed_and_cancelled_states_never_imply_admission() -> None:
    failed = build_progressive_sfm_ux_presentation(
        active_snapshot=None,
        last_snapshot=None,
        run_state="failed",
    )
    cancelled = build_progressive_sfm_ux_presentation(
        active_snapshot=None,
        last_snapshot=None,
        run_state="cancelled",
    )
    assert failed.status_text == "Stopped without admission"
    assert failed.status_kind == "error"
    assert cancelled.status_text == "Cancelled"
    assert "Provisional preview cleared" in cancelled.stage_text


def test_i6_6e_performance_tracker_is_aggregate_only_and_nonnegative() -> None:
    perf = ProgressiveSfmViewerPerformance()
    perf = perf.record_success(update_ms=4.25, payload_bytes=4096)
    perf = perf.record_success(update_ms=7.5, payload_bytes=8192)
    perf = perf.record_rejection()
    assert perf.applied_updates == 2
    assert perf.rejected_updates == 1
    assert perf.last_update_ms == 7.5
    assert perf.max_update_ms == 7.5
    assert perf.last_payload_bytes == 8192
    assert perf.max_payload_bytes == 8192
    assert not hasattr(perf, "history")


def test_i6_6e_bounded_preview_fits_declared_payload_budget() -> None:
    theoretical_max = (64 * 4 * 4 * 8) + (4096 * 3 * 8) + (4096 * 3)
    assert theoretical_max <= I66E_PREVIEW_PAYLOAD_BUDGET_BYTES

    snapshot = make_progressive_view_snapshot(
        camera_sample_count=48,
        landmark_sample_count=100,
        registered_camera_total=48,
        landmark_total=4096,
    )
    frame = build_progressive_sfm_viewer_frame(snapshot)
    payload_bytes = progressive_viewer_payload_bytes(frame)
    assert payload_bytes <= theoretical_max
    assert payload_bytes > 0


def test_i6_6e_viewer_target_is_presentation_only_not_reconstruction_gate() -> None:
    perf = ProgressiveSfmViewerPerformance().record_success(
        update_ms=I66E_VIEWER_UPDATE_TARGET_MS + 1.0,
        payload_bytes=1024,
    )
    presentation = build_progressive_sfm_ux_presentation(
        active_snapshot=make_progressive_view_snapshot(),
        last_snapshot=make_progressive_view_snapshot(),
        run_state="running",
        performance=perf,
    )
    assert perf.update_target_met is False
    assert "above 50 ms target" in presentation.viewer_text
    assert presentation.status_text == "Live provisional"
