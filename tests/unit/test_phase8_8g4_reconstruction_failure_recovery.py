from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID

from app.reconstruction_checkpoint_resume import (
    ReconstructionCheckpointResumePresentation,
    ReconstructionCheckpointResumeState,
)
from app.reconstruction_failure_recovery import (
    ReconstructionFailureRecoveryState,
    build_reconstruction_failure_recovery_presentation,
)
from app.reconstruction_session_model import build_reconstruction_session_model
from core.reconstruction import LiveCaptureRuntimeState, LiveCaptureRuntimeStats

SESSION_A = "77777777-7777-4777-8777-777777777777"
SESSION_B = "88888888-8888-4888-8888-888888888888"


def _source(*, ready: bool = True):
    return SimpleNamespace(
        workflow=SimpleNamespace(value="image_sequence"),
        ready=ready,
        display_name="generic-sequence",
        detail="generic source",
        path=None,
    )


def _model(*, scene_session: str = "", checkpoint_session: str = ""):
    scene = None
    if scene_session:
        scene = SimpleNamespace(
            scene_id="scene-1",
            session_id=scene_session,
            state=SimpleNamespace(value="ready"),
            artifacts=(),
            provider_provenance=(),
            canonical_geometry_authoritative=True,
            certified_metrology=False,
            created_at_utc="",
        )
    checkpoint = None
    if checkpoint_session:
        checkpoint = SimpleNamespace(
            checkpoint_id="checkpoint-1",
            session_id=checkpoint_session,
            next_stage=SimpleNamespace(value="dense_geometry"),
            resume_allowed=True,
            provider_versions=(),
            stage_results=(),
            created_at_utc="",
        )
    return build_reconstruction_session_model(
        source_selection=_source(),
        live_session_id="",
        offline_session_id=scene_session or checkpoint_session or SESSION_A,
        live_state=LiveCaptureRuntimeState.IDLE,
        offline_state=SimpleNamespace(value="failed"),
        scene_availability=SimpleNamespace(value="ready" if scene else "none"),
        scene=scene,
        scene_manifest="/tmp/scene.json" if scene else None,
        checkpoint=checkpoint,
        checkpoint_manifest="/tmp/checkpoint.json" if checkpoint else None,
    )


def _stats(*, restarts: int = 0, last_error: str = "") -> LiveCaptureRuntimeStats:
    return LiveCaptureRuntimeStats(
        offered_frames=10,
        processed_frames=8,
        replaced_frames=1,
        rejected_frames=0,
        observations=8,
        last_sequence_index=9,
        latest_tracking_state="tracking",
        last_error=last_error,
        state=LiveCaptureRuntimeState.RECOVERING,
        worker_restarts=restarts,
    )


def _compatible_checkpoint_presentation(*, prepared: bool = False):
    return ReconstructionCheckpointResumePresentation(
        state=(
            ReconstructionCheckpointResumeState.PREPARED
            if prepared
            else ReconstructionCheckpointResumeState.COMPATIBLE
        ),
        can_load_checkpoint=True,
        can_prepare_resume=not prepared,
        text="Resume prepared • next dense_geometry" if prepared else "Checkpoint compatible • next dense_geometry",
        details="Validated existing checkpoint compatibility.",
        next_stage_text="dense_geometry",
        reused_stage_count=3,
        canonical_geometry_authoritative=True,
    )


def test_g4_stable_state_requires_no_recovery_action() -> None:
    result = build_reconstruction_failure_recovery_presentation(
        None,
        live_state=LiveCaptureRuntimeState.IDLE,
        offline_state=SimpleNamespace(value="idle"),
        live_stats=None,
        source_ready=True,
        last_error="",
        checkpoint_resume=None,
        controller_active=False,
        controller_shutdown=False,
    )
    assert result.state is ReconstructionFailureRecoveryState.STABLE
    assert not result.can_retry_from_source
    assert result.text == "No recovery action required"
    assert not result.certified_metrology


def test_g4_observes_existing_bounded_live_auto_recovery_without_manual_restart() -> None:
    result = build_reconstruction_failure_recovery_presentation(
        None,
        live_state=LiveCaptureRuntimeState.RECOVERING,
        offline_state=SimpleNamespace(value="idle"),
        live_stats=_stats(restarts=2, last_error="worker protocol timeout"),
        source_ready=True,
        last_error="worker protocol timeout",
        checkpoint_resume=None,
        controller_active=True,
        controller_shutdown=False,
    )
    assert result.state is ReconstructionFailureRecoveryState.AUTO_RECOVERING
    assert result.automatic_recovery
    assert not result.can_retry_from_source
    assert "2 restarts" in result.text
    assert "existing live runtime owns bounded" in result.details


def test_g4_terminal_failure_allows_only_fresh_source_retry_when_no_persisted_evidence() -> None:
    result = build_reconstruction_failure_recovery_presentation(
        None,
        live_state=LiveCaptureRuntimeState.IDLE,
        offline_state=SimpleNamespace(value="failed"),
        live_stats=None,
        source_ready=True,
        last_error="deterministic worker failure",
        checkpoint_resume=None,
        controller_active=False,
        controller_shutdown=False,
    )
    assert result.state is ReconstructionFailureRecoveryState.RETRY_AVAILABLE
    assert result.can_retry_from_source
    assert result.retry_starts_new_session
    assert "fresh reconstruction session" in result.details
    assert "checkpoint" in result.details.lower()


def test_g4_cancelled_attempt_can_be_retried_as_fresh_source_attempt() -> None:
    result = build_reconstruction_failure_recovery_presentation(
        None,
        live_state=LiveCaptureRuntimeState.IDLE,
        offline_state=SimpleNamespace(value="cancelled"),
        live_stats=None,
        source_ready=True,
        last_error="",
        checkpoint_resume=None,
        controller_active=False,
        controller_shutdown=False,
    )
    assert result.state is ReconstructionFailureRecoveryState.CANCELLED_RETRY_AVAILABLE
    assert result.can_retry_from_source
    assert "Cancelled safely" in result.text


def test_g4_failure_without_ready_source_fails_closed() -> None:
    result = build_reconstruction_failure_recovery_presentation(
        None,
        live_state=LiveCaptureRuntimeState.FAILED,
        offline_state=SimpleNamespace(value="idle"),
        live_stats=None,
        source_ready=False,
        last_error="source disconnected",
        checkpoint_resume=None,
        controller_active=False,
        controller_shutdown=False,
    )
    assert result.state is ReconstructionFailureRecoveryState.SOURCE_REQUIRED
    assert not result.can_retry_from_source
    assert "source" in result.text.lower()


def test_g4_preserved_checkpoint_disables_ambiguous_fresh_retry_and_points_to_g3() -> None:
    model = _model(checkpoint_session=SESSION_A)
    result = build_reconstruction_failure_recovery_presentation(
        model,
        live_state=LiveCaptureRuntimeState.IDLE,
        offline_state=SimpleNamespace(value="failed"),
        live_stats=None,
        source_ready=True,
        last_error="worker failed",
        checkpoint_resume=_compatible_checkpoint_presentation(),
        controller_active=False,
        controller_shutdown=False,
    )
    assert result.state is ReconstructionFailureRecoveryState.PRESERVED_EVIDENCE
    assert not result.can_retry_from_source
    assert result.preserved_checkpoint
    assert "G3 Prepare Resume" in result.details


def test_g4_prepared_checkpoint_handoff_remains_explicit_and_is_not_executed_by_g4() -> None:
    model = _model(checkpoint_session=SESSION_A)
    result = build_reconstruction_failure_recovery_presentation(
        model,
        live_state=LiveCaptureRuntimeState.IDLE,
        offline_state=SimpleNamespace(value="failed"),
        live_stats=None,
        source_ready=True,
        last_error="worker failed",
        checkpoint_resume=_compatible_checkpoint_presentation(prepared=True),
        controller_active=False,
        controller_shutdown=False,
    )
    assert result.state is ReconstructionFailureRecoveryState.PRESERVED_EVIDENCE
    assert "resumed execution remains explicit outside G4" in result.details
    assert not result.can_retry_from_source


def test_g4_identity_conflict_blocks_recovery_attempt() -> None:
    model = _model(scene_session=SESSION_A, checkpoint_session=SESSION_B)
    assert model.session_id == str(UUID(SESSION_A))
    result = build_reconstruction_failure_recovery_presentation(
        model,
        live_state=LiveCaptureRuntimeState.IDLE,
        offline_state=SimpleNamespace(value="failed"),
        live_stats=None,
        source_ready=True,
        last_error="failure",
        checkpoint_resume=None,
        controller_active=False,
        controller_shutdown=False,
    )
    assert result.state is ReconstructionFailureRecoveryState.IDENTITY_CONFLICT
    assert not result.can_retry_from_source
    assert "identity conflict" in result.text.lower()
