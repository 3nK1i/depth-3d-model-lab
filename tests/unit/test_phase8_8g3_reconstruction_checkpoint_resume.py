from __future__ import annotations

from pathlib import Path

import pytest

from app.reconstruction_checkpoint_resume import (
    ReconstructionCheckpointResumeState,
    ReconstructionResumeContext,
    build_reconstruction_checkpoint_resume_presentation,
)
from app.reconstruction_session_model import build_reconstruction_session_model
from core.reconstruction import (
    ReconstructionResumeCompatibilityStatus,
    check_reconstruction_engine_resume_compatibility,
)
from tests.reconstruction_g3_helpers import write_checkpoint_fixture


def _loaded_fixture(tmp_path: Path):
    providers, request, plan, checkpoint, manifest = write_checkpoint_fixture(tmp_path / "checkpoint")
    model = build_reconstruction_session_model(
        checkpoint=checkpoint,
        checkpoint_manifest=str(manifest),
    )
    return providers, request, plan, checkpoint, manifest, model


def test_g3_resume_context_normalizes_generic_provider_payload_identity(tmp_path: Path) -> None:
    providers, _request, plan, checkpoint, _manifest, _model = _loaded_fixture(tmp_path)
    context = ReconstructionResumeContext(
        plan=plan,
        providers=providers,
        available_payload_refs=[*checkpoint.payload_refs, "", "  "],
    )
    assert context.plan is plan
    assert context.providers == providers
    assert context.available_payload_refs == frozenset(checkpoint.payload_refs)


def test_g3_without_checkpoint_is_load_only(tmp_path: Path) -> None:
    presentation = build_reconstruction_checkpoint_resume_presentation(
        None,
        checkpoint=None,
        compatibility=None,
        context_available=False,
        prepared=False,
        controller_active=False,
        controller_shutdown=False,
    )
    assert presentation.state is ReconstructionCheckpointResumeState.LOAD_ONLY
    assert presentation.can_load_checkpoint
    assert not presentation.can_prepare_resume
    assert presentation.certified_metrology is False


def test_g3_loaded_checkpoint_requires_exact_current_context(tmp_path: Path) -> None:
    _providers, _request, _plan, checkpoint, _manifest, model = _loaded_fixture(tmp_path)
    presentation = build_reconstruction_checkpoint_resume_presentation(
        model,
        checkpoint=checkpoint,
        compatibility=None,
        context_available=False,
        prepared=False,
        controller_active=False,
        controller_shutdown=False,
    )
    assert presentation.state is ReconstructionCheckpointResumeState.CONTEXT_REQUIRED
    assert not presentation.can_prepare_resume
    assert "will not guess compatibility" in presentation.details


def test_g3_compatible_context_enables_prepare_only(tmp_path: Path) -> None:
    providers, _request, plan, checkpoint, _manifest, model = _loaded_fixture(tmp_path)
    compatibility = check_reconstruction_engine_resume_compatibility(
        checkpoint,
        plan=plan,
        providers=providers,
        available_payload_refs=checkpoint.payload_refs,
    )
    assert compatibility.status is ReconstructionResumeCompatibilityStatus.COMPATIBLE
    presentation = build_reconstruction_checkpoint_resume_presentation(
        model,
        checkpoint=checkpoint,
        compatibility=compatibility,
        context_available=True,
        prepared=False,
        controller_active=False,
        controller_shutdown=False,
    )
    assert presentation.state is ReconstructionCheckpointResumeState.COMPATIBLE
    assert presentation.can_prepare_resume
    assert presentation.next_stage_text == checkpoint.next_stage.value
    assert presentation.reused_stage_count == checkpoint.completed_stage_count


def test_g3_missing_payload_ref_fails_closed(tmp_path: Path) -> None:
    providers, _request, plan, checkpoint, _manifest, model = _loaded_fixture(tmp_path)
    compatibility = check_reconstruction_engine_resume_compatibility(
        checkpoint,
        plan=plan,
        providers=providers,
        available_payload_refs=checkpoint.payload_refs[:-1],
    )
    presentation = build_reconstruction_checkpoint_resume_presentation(
        model,
        checkpoint=checkpoint,
        compatibility=compatibility,
        context_available=True,
        prepared=False,
        controller_active=False,
        controller_shutdown=False,
    )
    assert presentation.state is ReconstructionCheckpointResumeState.INCOMPATIBLE
    assert not presentation.can_prepare_resume
    assert presentation.status_kind == "warning"


def test_g3_prepared_state_is_handoff_not_execution(tmp_path: Path) -> None:
    providers, _request, plan, checkpoint, _manifest, model = _loaded_fixture(tmp_path)
    compatibility = check_reconstruction_engine_resume_compatibility(
        checkpoint,
        plan=plan,
        providers=providers,
        available_payload_refs=checkpoint.payload_refs,
    )
    presentation = build_reconstruction_checkpoint_resume_presentation(
        model,
        checkpoint=checkpoint,
        compatibility=compatibility,
        context_available=True,
        prepared=True,
        controller_active=False,
        controller_shutdown=False,
    )
    assert presentation.state is ReconstructionCheckpointResumeState.PREPARED
    assert not presentation.can_prepare_resume
    assert "does not launch the resumed engine automatically" in presentation.details


def test_g3_runtime_active_disables_checkpoint_actions(tmp_path: Path) -> None:
    _providers, _request, _plan, checkpoint, _manifest, model = _loaded_fixture(tmp_path)
    presentation = build_reconstruction_checkpoint_resume_presentation(
        model,
        checkpoint=checkpoint,
        compatibility=None,
        context_available=True,
        prepared=False,
        controller_active=True,
        controller_shutdown=False,
    )
    assert presentation.state is ReconstructionCheckpointResumeState.RUNTIME_ACTIVE
    assert not presentation.can_load_checkpoint
    assert not presentation.can_prepare_resume


def test_g3_context_rejects_duplicate_provider_ids(tmp_path: Path) -> None:
    providers, _request, plan, checkpoint, _manifest, _model = _loaded_fixture(tmp_path)
    with pytest.raises(ValueError, match="unique provider IDs"):
        ReconstructionResumeContext(
            plan=plan,
            providers=(*providers, providers[0]),
            available_payload_refs=checkpoint.payload_refs,
        )
