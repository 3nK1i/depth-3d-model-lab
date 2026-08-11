from __future__ import annotations

from pathlib import Path

import pytest

from app.main_window import MainWindow
from app.offline_reconstruction_controller import OfflineReconstructionRunState
from app.reconstruction_checkpoint_resume import ReconstructionCheckpointResumeState
from app.reconstruction_failure_recovery import ReconstructionFailureRecoveryState
from app.reconstruction_session_model import ReconstructionSessionIdentityStatus
from core.reconstruction import ReconstructionArtifactAuthority
from tests.reconstruction_h1_helpers import (
    create_checkpoint_fixture,
    run_whole_pipeline,
    write_ready_image_sequence,
)


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    yield win
    win.close()


def _adopt_full_scene(window: MainWindow, tmp_path: Path):
    controller = window._reconstruction_controller
    fixture, result = run_whole_pipeline()
    source = write_ready_image_sequence(tmp_path)
    controller.select_image_sequence(source)
    scene = controller.adopt_pipeline_result(result, (tmp_path / "scene").resolve())
    return fixture, result, scene, source


def test_h1_ready_source_and_full_scene_bind_as_one_application_workflow(window, qtbot, tmp_path: Path) -> None:
    page = window._reconstruction_page
    fixture, _result, scene, source = _adopt_full_scene(window, tmp_path)
    qtbot.wait(10)
    assert page._source_value.text() == source.name
    assert page._start_button.isEnabled()
    assert page.reconstruction_session_model is not None
    assert page.reconstruction_session_model.session_id == fixture.request.session_id == scene.session_id
    assert page.reconstruction_session_model.identity_status is ReconstructionSessionIdentityStatus.CONSISTENT
    assert page._scene_manifest_value.text() == "persistent_reconstruction_scene.json"


def test_h1_scene_signal_populates_semantic_and_appearance_context(window, qtbot, tmp_path: Path) -> None:
    page = window._reconstruction_page
    _fixture, _result, _scene, _source = _adopt_full_scene(window, tmp_path)
    qtbot.wait(10)
    assert page._offline_semantics is not None
    assert page._semantic_objects_group.isVisibleTo(page)
    appearance = page.appearance_representation_presentation
    assert appearance is not None
    assert {entry.representation_kind for entry in appearance.entries} == {"gaussian_scene", "nerf_scene"}
    assert all(entry.authority_text == "Appearance only" for entry in appearance.entries)


def test_h1_save_and_reload_keep_scene_session_and_fingerprint(window, qtbot, tmp_path: Path) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    _fixture, _result, scene, _source = _adopt_full_scene(window, tmp_path)
    saved = controller.save_persistent_scene((tmp_path / "saved").resolve())
    reloaded = controller.reload_persistent_scene()
    qtbot.wait(10)
    assert saved.name == "persistent_reconstruction_scene.json"
    assert reloaded.session_id == scene.session_id
    assert reloaded.scene_fingerprint_sha256 == scene.scene_fingerprint_sha256
    assert page._session_reload_button.isEnabled()


def test_h1_same_session_checkpoint_enables_prepare_resume_without_execution(window, qtbot, tmp_path: Path) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    fixture, _result, _scene, _source = _adopt_full_scene(window, tmp_path)
    checkpoint, prefix = create_checkpoint_fixture(fixture)
    created = controller.create_engine_checkpoint(
        plan=fixture.plan,
        providers=fixture.providers,
        stage_results=prefix,
        output_dir=(tmp_path / "checkpoint").resolve(),
    )
    compatibility = controller.set_engine_resume_context(
        plan=fixture.plan,
        providers=fixture.providers,
        available_payload_refs=created.payload_refs,
    )
    qtbot.wait(10)
    assert checkpoint.session_id == created.session_id
    assert compatibility is not None and compatibility.compatible
    assert page.checkpoint_resume_presentation is not None
    assert page.checkpoint_resume_presentation.state is ReconstructionCheckpointResumeState.COMPATIBLE
    assert page._session_resume_button.isEnabled()
    assert controller.engine_resume_preparation is None
    assert not controller.is_active


def test_h1_failure_with_scene_checkpoint_preserves_evidence_and_blocks_fresh_retry(window, qtbot, tmp_path: Path) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    fixture, _result, _scene, _source = _adopt_full_scene(window, tmp_path)
    _checkpoint, prefix = create_checkpoint_fixture(fixture)
    created = controller.create_engine_checkpoint(
        plan=fixture.plan,
        providers=fixture.providers,
        stage_results=prefix,
        output_dir=(tmp_path / "checkpoint").resolve(),
    )
    controller.set_engine_resume_context(
        plan=fixture.plan,
        providers=fixture.providers,
        available_payload_refs=created.payload_refs,
    )
    page.set_error("H1 contained integration failure")
    page.set_offline_state(OfflineReconstructionRunState.FAILED)
    qtbot.wait(10)
    assert page.failure_recovery_presentation is not None
    assert page.failure_recovery_presentation.state is ReconstructionFailureRecoveryState.PRESERVED_EVIDENCE
    assert not page._session_retry_button.isEnabled()
    assert page._session_resume_button.isEnabled()


def test_h1_cleared_evidence_returns_failure_to_fresh_source_retry(window, qtbot, tmp_path: Path) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    controller.select_image_sequence(write_ready_image_sequence(tmp_path))
    controller.clear_engine_checkpoint()
    controller.clear_persistent_scene()
    page.set_error("H1 retryable source failure")
    page.set_offline_state(OfflineReconstructionRunState.FAILED)
    qtbot.wait(10)
    assert page.failure_recovery_presentation is not None
    assert page.failure_recovery_presentation.state is ReconstructionFailureRecoveryState.RETRY_AVAILABLE
    assert page._session_retry_button.isEnabled()
    assert not page._start_button.isEnabled()


def test_h1_authority_chain_remains_visible_and_fail_closed(window, qtbot, tmp_path: Path) -> None:
    page = window._reconstruction_page
    _fixture, _result, scene, _source = _adopt_full_scene(window, tmp_path)
    qtbot.wait(10)
    assert scene.canonical_geometry_authoritative
    assert not scene.certified_metrology
    assert page.reconstruction_session_model is not None
    assert page.reconstruction_session_model.canonical_geometry_authoritative
    assert not page.reconstruction_session_model.certified_metrology
    appearance = page.appearance_representation_presentation
    assert appearance is not None
    assert all(not entry.canonical_geometry_authoritative for entry in appearance.entries)
    assert all(not entry.certified_metrology for entry in appearance.entries)


def test_h1_inspection_artifact_is_not_silently_recomputed_or_promoted(window, qtbot, tmp_path: Path) -> None:
    page = window._reconstruction_page
    _fixture, _result, scene, _source = _adopt_full_scene(window, tmp_path)
    qtbot.wait(10)
    inspection = next(artifact for artifact in scene.artifacts if artifact.kind.value == "inspection_report")
    assert inspection.authority is ReconstructionArtifactAuthority.INSPECTION_DECISION
    # H1 verifies whole-workflow wiring, but E4 remains explicit evidence binding:
    # loading a persistent scene must not manufacture a new QC decision.
    assert page.reconstruction_inspection_qc_presentation is None
    assert "Not evaluated" in page._inspection_qc_decision_value.text()
