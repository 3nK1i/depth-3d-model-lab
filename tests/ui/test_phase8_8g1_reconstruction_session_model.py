from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.main_window import MainWindow
from app.reconstruction_session_model import (
    ReconstructionSessionIdentityStatus,
    ReconstructionSessionLifecycle,
    build_reconstruction_session_model,
)
from core.reconstruction import LiveCaptureRuntimeState

SESSION = "44444444-4444-4444-8444-444444444444"
OTHER = "55555555-5555-4555-8555-555555555555"


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    yield win
    win.close()


def _selection():
    return SimpleNamespace(
        workflow=SimpleNamespace(value="image_sequence"),
        ready=True,
        display_name="Generic industrial capture",
        detail="Selected overlapping image sequence",
        path=Path("/tmp/g1-ui/source"),
    )


def _artifact(index: int, kind: str, authority: str):
    return SimpleNamespace(
        artifact_id=f"g1-ui-artifact-{index}",
        stage=SimpleNamespace(value=f"stage_{index}"),
        kind=SimpleNamespace(value=kind),
        authority=SimpleNamespace(value=authority),
        provider_id=f"provider-{index}",
    )


def _scene(session_id=SESSION):
    return SimpleNamespace(
        scene_id="prs-g1-ui",
        session_id=session_id,
        state=SimpleNamespace(value="ready"),
        artifacts=(
            _artifact(1, "surface_mesh", "canonical_geometry"),
            _artifact(2, "semantic_scene", "semantic_context"),
            _artifact(3, "future_visual_representation", "appearance_only"),
        ),
        provider_provenance=(
            SimpleNamespace(
                provider_id="provider-1",
                provider_version="1",
                execution=SimpleNamespace(value="in_process"),
                stages=(SimpleNamespace(value="surface_mesh"),),
            ),
            SimpleNamespace(
                provider_id="provider-2",
                provider_version="1",
                execution=SimpleNamespace(value="in_process"),
                stages=(SimpleNamespace(value="semantic_geometry"),),
            ),
            SimpleNamespace(
                provider_id="provider-3",
                provider_version="1",
                execution=SimpleNamespace(value="external_process"),
                stages=(SimpleNamespace(value="future_visual"),),
            ),
        ),
        canonical_geometry_authoritative=True,
        certified_metrology=False,
        created_at_utc="2026-08-11T12:00:00+00:00",
    )


def _checkpoint(session_id=SESSION):
    return SimpleNamespace(
        checkpoint_id="66666666-6666-4666-8666-666666666666",
        session_id=session_id,
        next_stage=SimpleNamespace(value="dense_geometry"),
        resume_allowed=True,
        provider_versions=(),
        stage_results=(),
        created_at_utc="2026-08-11T11:00:00+00:00",
    )


def test_g1_existing_session_group_gains_compact_context_and_evidence_rows(window) -> None:
    page = window._reconstruction_page
    assert page._session_context_value.text() in {"No session context", "No session context • Idle"}
    assert page._session_evidence_value.text() == "Runtime/session metadata only"
    assert not hasattr(page, "_session_save_button")
    assert not hasattr(page, "_session_reload_button")
    assert not hasattr(page, "_session_resume_button")


def test_g1_scene_and_checkpoint_render_as_one_consistent_generic_session(window, qtbot) -> None:
    page = window._reconstruction_page
    model = build_reconstruction_session_model(
        source_selection=_selection(),
        scene=_scene(),
        scene_manifest="/tmp/g1-ui/persistent_reconstruction_scene.json",
        checkpoint=_checkpoint(),
        checkpoint_manifest="/tmp/g1-ui/reconstruction_engine_checkpoint.json",
        scene_availability=SimpleNamespace(value="ready"),
    )
    page.set_session_model(model)
    qtbot.wait(10)

    assert page.reconstruction_session_model == model
    assert page._session_value.text() == SESSION[:8]
    assert page._session_context_value.text() == "Persistent scene • Ready"
    assert "Scene" in page._session_evidence_value.text()
    assert "checkpoint" in page._session_evidence_value.text()
    assert "3 artifacts" in page._session_evidence_value.text()
    assert "3 providers" in page._session_evidence_value.text()
    assert "future_visual_representation" in page._session_evidence_value.toolTip()


def test_g1_identity_conflict_is_visible_without_merging_evidence(window, qtbot) -> None:
    page = window._reconstruction_page
    model = build_reconstruction_session_model(
        source_selection=_selection(),
        offline_session_id=SESSION,
        offline_state=SimpleNamespace(value="running"),
        scene=_scene(OTHER),
        checkpoint=_checkpoint(SESSION),
    )
    page.set_session_model(model)
    qtbot.wait(10)

    assert model.identity_status is ReconstructionSessionIdentityStatus.CONFLICT
    assert page._session_value.text() == SESSION[:8]
    assert page._session_context_value.text() == "Session identity conflict"
    assert page._session_context_value.property("statusKind") == "warning"
    assert OTHER in page._session_value.toolTip()


def test_g1_empty_model_returns_session_group_to_neutral(window, qtbot) -> None:
    page = window._reconstruction_page
    model = build_reconstruction_session_model(source_selection=SimpleNamespace(
        workflow=SimpleNamespace(value="none"),
        ready=False,
        display_name="No reconstruction source selected",
        detail="Choose a workflow",
        path=None,
    ))
    page.set_session_model(model)
    qtbot.wait(10)

    assert page._session_value.text() == "No active session"
    assert page._session_context_value.text() == "No session context • Idle"
    assert page._session_value.property("statusKind") == "neutral"


def test_g1_live_state_signal_does_not_require_full_runtime_object(window, qtbot) -> None:
    """Preserve the compact runtime doubles used by earlier B4 UI coverage."""

    page = window._reconstruction_page
    controller = window._reconstruction_controller
    controller.live_slam_controller._runtime = SimpleNamespace(is_active=True, stats=None)
    try:
        controller.live_state_changed.emit(LiveCaptureRuntimeState.RUNNING)
        qtbot.wait(10)

        model = page.reconstruction_session_model
        assert model is not None
        assert model.lifecycle is ReconstructionSessionLifecycle.RUNNING
    finally:
        controller.live_slam_controller._runtime = None
