from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QFileDialog

from app.main_window import MainWindow
from app.reconstruction_checkpoint_resume import ReconstructionCheckpointResumeState
from app.reconstruction_session_model import build_reconstruction_session_model
from core.reconstruction.errors import ReconstructionValidationError
from tests.reconstruction_g3_helpers import write_checkpoint_fixture


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    yield win
    win.close()


def _load_fixture(controller, tmp_path: Path):
    providers, request, plan, checkpoint, manifest = write_checkpoint_fixture(tmp_path / "checkpoint")
    loaded = controller.load_engine_checkpoint(manifest)
    assert loaded.checkpoint_id == checkpoint.checkpoint_id
    return providers, request, plan, checkpoint, manifest


def test_g3_initial_checkpoint_ui_is_load_only(window) -> None:
    page = window._reconstruction_page
    assert page._session_checkpoint_load_button.isEnabled()
    assert not page._session_resume_button.isEnabled()
    assert page.checkpoint_resume_presentation is not None
    assert page.checkpoint_resume_presentation.state is ReconstructionCheckpointResumeState.LOAD_ONLY
    assert page._session_resume_state_value.text() == "Load reconstruction checkpoint"


def test_g3_load_checkpoint_button_verifies_existing_manifest_without_resuming(
    window, qtbot, monkeypatch, tmp_path: Path
) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    _providers, _request, _plan, checkpoint, manifest = write_checkpoint_fixture(tmp_path / "load")
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (str(manifest), "Reconstruction checkpoint"),
    )

    page._session_checkpoint_load_button.click()
    qtbot.wait(10)

    assert controller.engine_checkpoint is not None
    assert controller.engine_checkpoint.checkpoint_id == checkpoint.checkpoint_id
    assert controller.engine_resume_preparation is None
    assert page.checkpoint_resume_presentation is not None
    assert page.checkpoint_resume_presentation.state is ReconstructionCheckpointResumeState.CONTEXT_REQUIRED
    assert not page._session_resume_button.isEnabled()


def test_g3_exact_context_validation_enables_prepare_resume(window, qtbot, tmp_path: Path) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    providers, _request, plan, checkpoint, _manifest = _load_fixture(controller, tmp_path)

    compatibility = controller.set_engine_resume_context(
        plan=plan,
        providers=providers,
        available_payload_refs=checkpoint.payload_refs,
    )
    qtbot.wait(10)

    assert compatibility is not None and compatibility.compatible
    assert page.checkpoint_resume_presentation is not None
    assert page.checkpoint_resume_presentation.state is ReconstructionCheckpointResumeState.COMPATIBLE
    assert page._session_resume_button.isEnabled()
    assert checkpoint.next_stage.value in page._session_resume_state_value.text()


def test_g3_prepare_resume_emits_handoff_without_starting_runtime(
    window, qtbot, monkeypatch, tmp_path: Path
) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    providers, _request, plan, checkpoint, _manifest = _load_fixture(controller, tmp_path)
    controller.set_engine_resume_context(
        plan=plan,
        providers=providers,
        available_payload_refs=checkpoint.payload_refs,
    )
    monkeypatch.setattr(
        controller,
        "start_selected_offline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("G3 must not start offline runtime")),
    )

    page._session_resume_button.click()
    qtbot.wait(10)

    preparation = controller.engine_resume_preparation
    assert preparation is not None
    assert preparation.checkpoint.checkpoint_id == checkpoint.checkpoint_id
    assert preparation.resume_stage_results == checkpoint.stage_results
    assert page.checkpoint_resume_presentation is not None
    assert page.checkpoint_resume_presentation.state is ReconstructionCheckpointResumeState.PREPARED
    assert not page._session_resume_button.isEnabled()


def test_g3_incompatible_payload_context_fails_closed(window, qtbot, tmp_path: Path) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    providers, _request, plan, checkpoint, _manifest = _load_fixture(controller, tmp_path)
    compatibility = controller.set_engine_resume_context(
        plan=plan,
        providers=providers,
        available_payload_refs=checkpoint.payload_refs[:-1],
    )
    qtbot.wait(10)

    assert compatibility is not None and not compatibility.compatible
    assert page.checkpoint_resume_presentation is not None
    assert page.checkpoint_resume_presentation.state is ReconstructionCheckpointResumeState.INCOMPATIBLE
    assert not page._session_resume_button.isEnabled()


def test_g3_clearing_current_context_disables_prepare_even_if_old_compatibility_exists(
    window, qtbot, tmp_path: Path
) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    providers, _request, plan, checkpoint, _manifest = _load_fixture(controller, tmp_path)
    controller.set_engine_resume_context(
        plan=plan,
        providers=providers,
        available_payload_refs=checkpoint.payload_refs,
    )
    assert page._session_resume_button.isEnabled()

    controller.clear_engine_resume_context()
    qtbot.wait(10)

    assert controller.engine_resume_context is None
    assert page.checkpoint_resume_presentation is not None
    assert page.checkpoint_resume_presentation.state is ReconstructionCheckpointResumeState.CONTEXT_REQUIRED
    assert not page._session_resume_button.isEnabled()


def test_g3_runtime_activity_disables_checkpoint_load_and_prepare(window, monkeypatch, tmp_path: Path) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    providers, _request, plan, checkpoint, _manifest = _load_fixture(controller, tmp_path)
    controller.set_engine_resume_context(
        plan=plan,
        providers=providers,
        available_payload_refs=checkpoint.payload_refs,
    )
    monkeypatch.setattr(type(controller), "is_active", property(lambda _self: True))
    page._refresh_checkpoint_resume_controls()

    assert page.checkpoint_resume_presentation is not None
    assert page.checkpoint_resume_presentation.state is ReconstructionCheckpointResumeState.RUNTIME_ACTIVE
    assert not page._session_checkpoint_load_button.isEnabled()
    assert not page._session_resume_button.isEnabled()


def test_g3_session_identity_conflict_blocks_prepare(window, qtbot, tmp_path: Path) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    providers, _request, plan, checkpoint, manifest = _load_fixture(controller, tmp_path)
    controller.set_engine_resume_context(
        plan=plan,
        providers=providers,
        available_payload_refs=checkpoint.payload_refs,
    )
    scene = SimpleNamespace(
        scene_id="prs-conflict",
        session_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        state=SimpleNamespace(value="ready"),
        artifacts=(),
        provider_provenance=(),
        canonical_geometry_authoritative=True,
        certified_metrology=False,
        created_at_utc="2026-08-11T12:00:00+00:00",
    )
    bridge = controller.persistence_controller
    bridge._scene = scene
    bridge._scene_manifest = Path("/tmp/conflict/persistent_reconstruction_scene.json")
    conflict_model = build_reconstruction_session_model(
        scene=scene,
        scene_manifest=str(bridge._scene_manifest),
        checkpoint=checkpoint,
        checkpoint_manifest=str(manifest),
    )
    page.set_session_model(conflict_model)
    qtbot.wait(10)

    assert page.checkpoint_resume_presentation is not None
    assert page.checkpoint_resume_presentation.state is ReconstructionCheckpointResumeState.IDENTITY_CONFLICT
    assert not page._session_resume_button.isEnabled()
    with pytest.raises(ReconstructionValidationError, match="session identity evidence conflicts"):
        controller.prepare_loaded_engine_resume()


def test_g3_controller_load_fails_closed_on_checkpoint_checksum_tamper(window, tmp_path: Path) -> None:
    controller = window._reconstruction_controller
    _providers, _request, _plan, _checkpoint, manifest = write_checkpoint_fixture(tmp_path / "tampered")
    manifest.write_text(manifest.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ReconstructionValidationError, match="SHA-256"):
        controller.load_engine_checkpoint(manifest)
