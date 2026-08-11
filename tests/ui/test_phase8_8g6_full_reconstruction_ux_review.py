from __future__ import annotations

from pathlib import Path

import pytest

from app.main_window import MainWindow
from app.offline_reconstruction_controller import OfflineReconstructionRunState
from app.reconstruction_checkpoint_resume import ReconstructionCheckpointResumeState
from app.reconstruction_failure_recovery import ReconstructionFailureRecoveryState
from app.reconstruction_session_model import ReconstructionSessionIdentityStatus
from app.widgets.mode_tabs import AppMode
from app.widgets.reconstruction_settings_disclosure import ReconstructionSettingsDisclosureGroup
from tests.reconstruction_g6_helpers import write_full_review_evidence, write_ready_image_sequence


EXPECTED_GROUP_TITLES = (
    "WORKFLOW",
    "SESSION & PERSISTENCE",
    "OFFLINE INSPECTION",
    "SEMANTIC / OBJECTS",
    "INSPECTION TARGET",
    "RECONSTRUCTION MEASUREMENTS",
    "MEASUREMENT VALIDITY",
    "INSPECTION / QC",
    "REFERENCE COMPARISON",
    "APPEARANCE REPRESENTATIONS",
    "GAUSSIAN SPLATTING VISUALIZATION",
    "NERF VISUALIZATION",
    "RESOURCE & FALLBACK",
    "AUTHORITY & SAFETY",
)


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    yield win
    win.close()


def _settings_groups(page):
    body = page._settings_scroll.widget()
    layout = body.layout()
    groups = []
    for index in range(layout.count()):
        widget = layout.itemAt(index).widget()
        if isinstance(widget, ReconstructionSettingsDisclosureGroup):
            groups.append(widget)
    return tuple(groups)


def test_g6_idle_entry_has_clear_safe_action_hierarchy(window) -> None:
    page = window._reconstruction_page
    assert not page._start_button.isEnabled()
    assert not page._pause_button.isEnabled()
    assert not page._stop_button.isEnabled()
    assert page._session_load_button.isEnabled()
    assert page._session_checkpoint_load_button.isEnabled()
    assert not page._session_resume_button.isEnabled()
    assert not page._session_retry_button.isEnabled()
    assert page.failure_recovery_presentation is not None
    assert page.failure_recovery_presentation.state is ReconstructionFailureRecoveryState.STABLE


def test_g6_ready_image_sequence_exposes_one_normal_start_path(window, qtbot, tmp_path: Path) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    source = write_ready_image_sequence(tmp_path)
    controller.select_image_sequence(source)
    qtbot.wait(10)

    assert page._start_button.isEnabled()
    assert not page._session_retry_button.isEnabled()
    assert not page._session_resume_button.isEnabled()
    assert page._workflow_input_value.text() == "3 images"
    assert page._source_value.text() == source.name
    assert "Image sequence" in page._workflow_value.text()


def test_g6_terminal_failure_transfers_fresh_retry_to_recovery_action(window, qtbot, tmp_path: Path) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    controller.select_image_sequence(write_ready_image_sequence(tmp_path))
    page.set_error("G6 contained offline failure")
    page.set_offline_state(OfflineReconstructionRunState.FAILED)
    qtbot.wait(10)

    assert page.failure_recovery_presentation is not None
    assert page.failure_recovery_presentation.state is ReconstructionFailureRecoveryState.RETRY_AVAILABLE
    assert not page._start_button.isEnabled()
    assert page._session_retry_button.isEnabled()
    assert not page._session_resume_button.isEnabled()


def test_g6_compatible_checkpoint_keeps_resume_distinct_from_recovery(window, qtbot, tmp_path: Path) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    fixture = write_full_review_evidence(tmp_path)
    controller.select_image_sequence(fixture.source_dir)
    controller.load_engine_checkpoint(fixture.checkpoint_manifest)
    compatibility = controller.set_engine_resume_context(
        plan=fixture.plan,
        providers=fixture.providers,
        available_payload_refs=fixture.checkpoint.payload_refs,
    )
    page.set_error("G6 failed attempt with checkpoint evidence retained")
    page.set_offline_state(OfflineReconstructionRunState.FAILED)
    qtbot.wait(10)

    assert compatibility is not None and compatibility.compatible
    assert page.checkpoint_resume_presentation is not None
    assert page.checkpoint_resume_presentation.state is ReconstructionCheckpointResumeState.COMPATIBLE
    assert page._session_resume_button.isEnabled()
    assert page.failure_recovery_presentation is not None
    assert page.failure_recovery_presentation.state is ReconstructionFailureRecoveryState.PRESERVED_EVIDENCE
    assert not page._session_retry_button.isEnabled()


def test_g6_same_session_scene_and_checkpoint_present_one_consistent_session(window, qtbot, tmp_path: Path) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    fixture = write_full_review_evidence(tmp_path)
    controller.load_persistent_scene(fixture.scene_dir)
    controller.load_engine_checkpoint(fixture.checkpoint_manifest)
    controller.set_engine_resume_context(
        plan=fixture.plan,
        providers=fixture.providers,
        available_payload_refs=fixture.checkpoint.payload_refs,
    )
    qtbot.wait(10)

    model = page.reconstruction_session_model
    assert model is not None
    assert model.identity_status is ReconstructionSessionIdentityStatus.CONSISTENT
    assert model.session_id == fixture.scene_session_id
    assert model.canonical_geometry_authoritative
    assert not model.certified_metrology
    assert page._session_save_button.isEnabled()
    assert page._session_reload_button.isEnabled()
    assert page._session_resume_button.isEnabled()


def test_g6_session_disclosure_never_changes_action_availability(window, qtbot, tmp_path: Path) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    fixture = write_full_review_evidence(tmp_path)
    controller.load_persistent_scene(fixture.scene_dir)
    controller.load_engine_checkpoint(fixture.checkpoint_manifest)
    controller.set_engine_resume_context(
        plan=fixture.plan,
        providers=fixture.providers,
        available_payload_refs=fixture.checkpoint.payload_refs,
    )
    qtbot.wait(10)
    before = (
        page._session_load_button.isEnabled(),
        page._session_save_button.isEnabled(),
        page._session_reload_button.isEnabled(),
        page._session_checkpoint_load_button.isEnabled(),
        page._session_resume_button.isEnabled(),
        page._session_retry_button.isEnabled(),
    )

    page._session_group.set_expanded(False)
    qtbot.wait(10)
    after = (
        page._session_load_button.isEnabled(),
        page._session_save_button.isEnabled(),
        page._session_reload_button.isEnabled(),
        page._session_checkpoint_load_button.isEnabled(),
        page._session_resume_button.isEnabled(),
        page._session_retry_button.isEnabled(),
    )
    assert before == after
    page._session_group.set_expanded(True)


def test_g6_sidebar_information_architecture_is_complete_and_ordered(window) -> None:
    page = window._reconstruction_page
    groups = _settings_groups(page)
    assert tuple(group.title for group in groups) == EXPECTED_GROUP_TITLES
    assert len({group.title for group in groups}) == len(EXPECTED_GROUP_TITLES)


def test_g6_detail_disclosure_defaults_reduce_density_without_hiding_context(window) -> None:
    page = window._reconstruction_page
    for group in (
        page._workflow_group,
        page._session_group,
        page._inspection_target_group,
        page._appearance_group,
        page._authority_group,
    ):
        assert group.expanded
    for group in (
        page._reconstruction_measurement_group,
        page._measurement_validity_group,
        page._inspection_qc_group,
        page._reference_comparison_group,
        page._gaussian_visualization_group,
        page._nerf_visualization_group,
        page._appearance_resource_group,
    ):
        assert not group.expanded


def test_g6_sidebar_stays_vertical_only_at_target_desktop_size(window, qtbot) -> None:
    # Layout assertions must exercise the visible Reconstruction page. Hidden
    # stacked pages retain stale geometry and can report a tiny scroll viewport
    # that is unrelated to the actual 1366x768 Reconstruction workspace.
    window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)
    page = window._reconstruction_page
    window.resize(1366, 768)
    window.show()
    qtbot.wait(20)
    assert page._settings_scroll.horizontalScrollBarPolicy().name == "ScrollBarAlwaysOff"
    assert page._settings_scroll.horizontalScrollBar().maximum() == 0
    assert page._settings_scroll.verticalScrollBarPolicy().name == "ScrollBarAsNeeded"


def test_g6_authority_safety_remains_explicit_in_final_review(window) -> None:
    page = window._reconstruction_page
    assert page._authority_group.expanded
    assert page._metric_scale_value.text() == "Evidence-driven"
    assert page._metrology_value.text() == "Not certified"
    assert page._appearance_authority_value.text().startswith("Appearance only") or "Canonical" in page._appearance_authority_value.text()
