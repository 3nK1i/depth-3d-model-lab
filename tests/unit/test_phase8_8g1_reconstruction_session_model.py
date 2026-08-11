from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.reconstruction_session_model import (
    ReconstructionSessionFocus,
    ReconstructionSessionIdentityStatus,
    ReconstructionSessionLifecycle,
    ReconstructionSessionModel,
    build_reconstruction_session_model,
)

SESSION = "11111111-1111-4111-8111-111111111111"
OTHER = "22222222-2222-4222-8222-222222222222"


def _selection(workflow="image_sequence", ready=True):
    return SimpleNamespace(
        workflow=SimpleNamespace(value=workflow),
        ready=ready,
        display_name="Factory bay capture",
        detail="24 overlapping images",
        path=Path("/tmp/g1/source"),
    )


def _artifact(index: int, *, kind: str, authority: str, provider: str):
    return SimpleNamespace(
        artifact_id=f"artifact-{index}",
        stage=SimpleNamespace(value=f"stage_{index}"),
        kind=SimpleNamespace(value=kind),
        authority=SimpleNamespace(value=authority),
        provider_id=provider,
    )


def _provider(provider_id: str, *stages: str):
    return SimpleNamespace(
        provider_id=provider_id,
        provider_version="1.0",
        execution=SimpleNamespace(value="external_process"),
        stages=tuple(SimpleNamespace(value=value) for value in stages),
    )


def _scene(session_id=SESSION, state="ready"):
    return SimpleNamespace(
        scene_id="prs-g1-generic",
        session_id=session_id,
        state=SimpleNamespace(value=state),
        artifacts=(
            _artifact(1, kind="surface_mesh", authority="canonical_geometry", provider="mesh-provider"),
            _artifact(2, kind="semantic_scene", authority="semantic_context", provider="semantic-provider"),
            _artifact(3, kind="gaussian_scene", authority="appearance_only", provider="appearance-provider"),
            _artifact(4, kind="future_appearance", authority="appearance_only", provider="future-provider"),
        ),
        provider_provenance=(
            _provider("mesh-provider", "surface_mesh"),
            _provider("semantic-provider", "semantic_geometry"),
            _provider("appearance-provider", "gaussian_appearance"),
            _provider("future-provider", "future_stage"),
        ),
        canonical_geometry_authoritative=True,
        certified_metrology=False,
        created_at_utc="2026-08-11T12:00:00+00:00",
    )


def _checkpoint(session_id=SESSION):
    return SimpleNamespace(
        checkpoint_id="33333333-3333-4333-8333-333333333333",
        session_id=session_id,
        next_stage=SimpleNamespace(value="dense_geometry"),
        resume_allowed=True,
        created_at_utc="2026-08-11T11:00:00+00:00",
        provider_versions=(("mesh-provider", "1.0", "adapter", "external_process"),),
        stage_results=(),
    )


def test_g1_empty_model_is_explicit_and_does_not_invent_identity() -> None:
    model = build_reconstruction_session_model(source_selection=_selection("none", ready=False))
    assert model.session_id == ""
    assert model.focus is ReconstructionSessionFocus.NONE
    assert model.identity_status is ReconstructionSessionIdentityStatus.NONE
    assert model.lifecycle is ReconstructionSessionLifecycle.IDLE
    assert model.context_text == "No session context • Idle"
    assert model.evidence_text == "Runtime/session metadata only"


def test_g1_persistent_scene_is_generic_session_anchor() -> None:
    model = build_reconstruction_session_model(
        source_selection=_selection(),
        scene=_scene(),
        scene_manifest="/tmp/g1/persistent_reconstruction_scene.json",
        scene_availability=SimpleNamespace(value="ready"),
    )
    assert model.session_id == SESSION
    assert model.focus is ReconstructionSessionFocus.PERSISTENT_SCENE
    assert model.identity_status is ReconstructionSessionIdentityStatus.CONSISTENT
    assert model.lifecycle is ReconstructionSessionLifecycle.READY
    assert len(model.artifacts) == 4
    assert len(model.providers) == 4
    assert model.appearance_representation_kinds == ("gaussian_scene", "future_appearance")
    assert model.canonical_geometry_authoritative is True
    assert model.certified_metrology is False
    assert "4 artifacts" in model.evidence_text
    assert "4 providers" in model.evidence_text


def test_g1_degraded_scene_preserves_authority_without_claiming_metrology() -> None:
    model = build_reconstruction_session_model(scene=_scene(state="degraded"))
    assert model.lifecycle is ReconstructionSessionLifecycle.DEGRADED
    assert model.canonical_geometry_authoritative is True
    assert model.certified_metrology is False


def test_g1_checkpoint_only_is_descriptive_not_resume_action() -> None:
    model = build_reconstruction_session_model(
        checkpoint=_checkpoint(),
        checkpoint_manifest="/tmp/g1/reconstruction_engine_checkpoint.json",
    )
    assert model.session_id == SESSION
    assert model.focus is ReconstructionSessionFocus.CHECKPOINT
    assert model.lifecycle is ReconstructionSessionLifecycle.CHECKPOINTED
    assert model.checkpoint_resume_allowed is True
    assert model.checkpoint_next_stage == "dense_geometry"
    assert model.scene_id == ""


def test_g1_active_runtime_takes_focus_and_links_matching_persistence() -> None:
    model = build_reconstruction_session_model(
        source_selection=_selection(),
        offline_session_id=SESSION,
        offline_state=SimpleNamespace(value="running"),
        scene=_scene(),
        checkpoint=_checkpoint(),
    )
    assert model.focus is ReconstructionSessionFocus.RUNTIME
    assert model.lifecycle is ReconstructionSessionLifecycle.RUNNING
    assert model.identity_status is ReconstructionSessionIdentityStatus.CONSISTENT
    assert model.session_id == SESSION
    assert "runtime=" in model.identity_details
    assert "scene=" in model.identity_details
    assert "checkpoint=" in model.identity_details


def test_g1_identity_conflict_is_explicit_and_never_silently_merges_sessions() -> None:
    model = build_reconstruction_session_model(
        source_selection=_selection(),
        offline_session_id=SESSION,
        offline_state=SimpleNamespace(value="running"),
        scene=_scene(session_id=OTHER),
        checkpoint=_checkpoint(session_id=SESSION),
    )
    assert model.session_id == SESSION
    assert model.identity_status is ReconstructionSessionIdentityStatus.CONFLICT
    assert model.context_text == "Session identity conflict"
    assert OTHER in model.identity_details
    assert SESSION in model.identity_details


def test_g1_last_runtime_state_is_preserved_without_persistent_evidence() -> None:
    model = build_reconstruction_session_model(
        source_selection=_selection(),
        offline_session_id=SESSION,
        offline_state=SimpleNamespace(value="completed"),
    )
    assert model.focus is ReconstructionSessionFocus.LAST_RUNTIME
    assert model.lifecycle is ReconstructionSessionLifecycle.COMPLETED
    assert model.session_id == SESSION


def test_g1_selected_source_is_context_not_persisted_session_provenance() -> None:
    model = build_reconstruction_session_model(
        source_selection=_selection("recorded_video"),
        scene=_scene(),
    )
    assert model.focus is ReconstructionSessionFocus.PERSISTENT_SCENE
    assert model.workflow == "recorded_video"
    assert model.selected_source_label == "Factory bay capture"
    assert model.scene_id == "prs-g1-generic"


def test_g1_model_serialization_stays_generic() -> None:
    model = build_reconstruction_session_model(scene=_scene(), checkpoint=_checkpoint())
    payload = model.as_dict()
    assert payload["session_id"] == SESSION
    assert payload["scene"]["scene_id"] == "prs-g1-generic"
    assert payload["checkpoint"]["resume_allowed"] is True
    assert payload["canonical_geometry_authoritative"] is True
    assert payload["certified_metrology"] is False
    assert "future_appearance" in payload["appearance_representation_kinds"]


def test_g1_rejects_invalid_or_promoted_session_contracts() -> None:
    with pytest.raises(ValueError):
        build_reconstruction_session_model(live_session_id="not-a-uuid")

    base = build_reconstruction_session_model(scene=_scene())
    values = {field: getattr(base, field) for field in base.__dataclass_fields__}
    values["certified_metrology"] = True
    with pytest.raises(ValueError):
        ReconstructionSessionModel(**values)
