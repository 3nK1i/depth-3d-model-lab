from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.reconstruction_session_interaction import (
    ReconstructionSceneInteractionState,
    build_reconstruction_scene_interaction_presentation,
)
from app.reconstruction_session_model import build_reconstruction_session_model

SESSION = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
OTHER = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"


def _selection():
    return SimpleNamespace(
        workflow=SimpleNamespace(value="image_sequence"),
        ready=True,
        display_name="Generic reconstruction source",
        detail="Selected source",
        path=Path("/tmp/g2-unit/source"),
    )


def _scene(session_id=SESSION):
    return SimpleNamespace(
        scene_id="prs-g2-unit",
        session_id=session_id,
        state=SimpleNamespace(value="ready"),
        artifacts=(
            SimpleNamespace(
                artifact_id="artifact-1",
                stage=SimpleNamespace(value="surface_mesh"),
                kind=SimpleNamespace(value="surface_mesh"),
                authority=SimpleNamespace(value="canonical_geometry"),
                provider_id="provider-1",
            ),
        ),
        provider_provenance=(
            SimpleNamespace(
                provider_id="provider-1",
                provider_version="1",
                execution=SimpleNamespace(value="in_process"),
                stages=(SimpleNamespace(value="surface_mesh"),),
            ),
        ),
        canonical_geometry_authoritative=True,
        certified_metrology=False,
        created_at_utc="2026-08-11T12:00:00+00:00",
    )


def _checkpoint(session_id=SESSION):
    return SimpleNamespace(
        checkpoint_id="cccccccc-cccc-4ccc-8ccc-cccccccccccc",
        session_id=session_id,
        next_stage=SimpleNamespace(value="dense_geometry"),
        resume_allowed=True,
        provider_versions=(),
        stage_results=(),
        created_at_utc="2026-08-11T11:00:00+00:00",
    )


def test_g2_empty_idle_session_allows_load_only() -> None:
    model = build_reconstruction_session_model(source_selection=_selection())
    presentation = build_reconstruction_scene_interaction_presentation(
        model,
        controller_active=False,
        controller_shutdown=False,
    )
    assert presentation.state is ReconstructionSceneInteractionState.LOAD_ONLY
    assert presentation.can_load is True
    assert presentation.can_save is False
    assert presentation.can_reload is False
    assert "Load persistent scene" in presentation.text
    assert presentation.certified_metrology is False


def test_g2_loaded_scene_allows_save_and_reload() -> None:
    model = build_reconstruction_session_model(
        source_selection=_selection(),
        scene=_scene(),
        scene_manifest="/tmp/g2-unit/persistent_reconstruction_scene.json",
        scene_availability=SimpleNamespace(value="ready"),
    )
    presentation = build_reconstruction_scene_interaction_presentation(
        model,
        controller_active=False,
        controller_shutdown=False,
    )
    assert presentation.state is ReconstructionSceneInteractionState.READY
    assert presentation.can_load is True
    assert presentation.can_save is True
    assert presentation.can_reload is True
    assert presentation.canonical_geometry_authoritative is True
    assert "guarded persistence" in presentation.text
    assert "No checkpoint resume" in presentation.details


def test_g2_loaded_scene_without_manifest_can_save_but_cannot_reload() -> None:
    model = build_reconstruction_session_model(
        source_selection=_selection(),
        scene=_scene(),
        scene_availability=SimpleNamespace(value="ready"),
    )
    presentation = build_reconstruction_scene_interaction_presentation(
        model,
        controller_active=False,
        controller_shutdown=False,
    )
    assert presentation.can_load is True
    assert presentation.can_save is True
    assert presentation.can_reload is False


def test_g2_runtime_activity_disables_all_scene_mutation() -> None:
    model = build_reconstruction_session_model(
        source_selection=_selection(),
        scene=_scene(),
        scene_manifest="/tmp/g2-unit/persistent_reconstruction_scene.json",
        scene_availability=SimpleNamespace(value="ready"),
    )
    presentation = build_reconstruction_scene_interaction_presentation(
        model,
        controller_active=True,
        controller_shutdown=False,
    )
    assert presentation.state is ReconstructionSceneInteractionState.RUNTIME_ACTIVE
    assert presentation.can_load is False
    assert presentation.can_save is False
    assert presentation.can_reload is False


def test_g2_shutdown_disables_all_scene_interaction() -> None:
    model = build_reconstruction_session_model(source_selection=_selection())
    presentation = build_reconstruction_scene_interaction_presentation(
        model,
        controller_active=False,
        controller_shutdown=True,
    )
    assert presentation.state is ReconstructionSceneInteractionState.UNAVAILABLE
    assert not presentation.can_load
    assert not presentation.can_save
    assert not presentation.can_reload


def test_g2_identity_conflict_stays_visible_without_silent_checkpoint_action() -> None:
    model = build_reconstruction_session_model(
        source_selection=_selection(),
        scene=_scene(OTHER),
        scene_manifest="/tmp/g2-unit/persistent_reconstruction_scene.json",
        checkpoint=_checkpoint(SESSION),
        checkpoint_manifest="/tmp/g2-unit/reconstruction_engine_checkpoint.json",
        scene_availability=SimpleNamespace(value="ready"),
    )
    presentation = build_reconstruction_scene_interaction_presentation(
        model,
        controller_active=False,
        controller_shutdown=False,
    )
    assert presentation.state is ReconstructionSceneInteractionState.IDENTITY_CONFLICT
    assert presentation.can_load and presentation.can_save and presentation.can_reload
    assert presentation.status_kind == "warning"
    assert "identity conflict visible" in presentation.text
