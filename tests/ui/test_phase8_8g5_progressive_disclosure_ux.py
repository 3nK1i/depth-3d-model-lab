from __future__ import annotations

import pytest

from app.main_window import MainWindow
from app.widgets.reconstruction_settings_disclosure import ReconstructionSettingsDisclosureGroup


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    yield win
    win.close()


def test_g5_primary_groups_are_expanded_by_default(window) -> None:
    page = window._reconstruction_page
    for group in (
        page._workflow_group,
        page._session_group,
        page._inspection_target_group,
        page._appearance_group,
        page._authority_group,
    ):
        assert isinstance(group, ReconstructionSettingsDisclosureGroup)
        assert group.expanded
        assert not group._body.isHidden()


def test_g5_detail_groups_are_collapsed_by_default(window) -> None:
    page = window._reconstruction_page
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
        assert group._body.isHidden()


def test_g5_session_group_can_collapse_without_changing_action_state(window, qtbot) -> None:
    page = window._reconstruction_page
    session = page._session_group
    assert isinstance(session, ReconstructionSettingsDisclosureGroup)
    load_enabled = page._session_load_button.isEnabled()
    resume_enabled = page._session_resume_button.isEnabled()
    retry_enabled = page._session_retry_button.isEnabled()

    session.set_expanded(False)
    qtbot.wait(10)
    assert not session.expanded
    assert session._body.isHidden()
    assert page._session_load_button.isEnabled() == load_enabled
    assert page._session_resume_button.isEnabled() == resume_enabled
    assert page._session_retry_button.isEnabled() == retry_enabled

    session.set_expanded(True)
    qtbot.wait(10)
    assert session.expanded
    assert not session._body.isHidden()


def test_g5_collapsed_measurement_group_keeps_values_and_reveals_them_on_expand(window, qtbot) -> None:
    page = window._reconstruction_page
    group = page._reconstruction_measurement_group
    page._measurement_state_value.setText("Evidence retained while collapsed")
    assert not group.expanded
    assert page._measurement_state_value.text() == "Evidence retained while collapsed"

    group.set_expanded(True)
    qtbot.wait(10)
    assert group.expanded
    assert not group._body.isHidden()
    assert page._measurement_state_value.text() == "Evidence retained while collapsed"


def test_g5_disclosure_chevron_and_tooltip_follow_state(window, qtbot) -> None:
    group = window._reconstruction_page._measurement_validity_group
    assert group._toggle_button.text() == "▸"
    assert "Expand measurement validity details" in group._toggle_button.toolTip()
    group._toggle_button.click()
    qtbot.wait(10)
    assert group.expanded
    assert group._toggle_button.text() == "▾"
    assert "Collapse measurement validity details" in group._toggle_button.toolTip()


def test_g5_dynamic_offline_and_semantic_groups_remain_context_hidden_initially(window) -> None:
    page = window._reconstruction_page
    assert page._offline_inspection_group.expanded
    assert page._semantic_objects_group.expanded
    assert page._offline_inspection_group.isHidden()
    assert page._semantic_objects_group.isHidden()


def test_g5_g3_resume_and_g4_recovery_controls_remain_in_expanded_session(window) -> None:
    page = window._reconstruction_page
    session = page._session_group
    session.set_expanded(True)
    assert page._session_checkpoint_load_button.text() == "Load Checkpoint"
    assert page._session_resume_button.text() == "Prepare Resume"
    assert page._session_retry_button.text() == "Retry From Source"


def test_g5_settings_scroll_remains_vertical_only(window) -> None:
    page = window._reconstruction_page
    assert page._settings_scroll.horizontalScrollBarPolicy().name == "ScrollBarAlwaysOff"
    assert page._settings_scroll.verticalScrollBarPolicy().name == "ScrollBarAsNeeded"
