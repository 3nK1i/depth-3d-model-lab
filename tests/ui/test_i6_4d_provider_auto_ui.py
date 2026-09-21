from __future__ import annotations

from pathlib import Path

import pytest

from app.main_window import MainWindow
from app.reconstruction_provider_selection_state import ManualProviderSelectionStatus
from tests.i6_4d_helpers import binding


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    yield win
    win.close()


def _images(root: Path, name: str = "images") -> Path:
    path = root / name
    path.mkdir()
    for index in range(4):
        (path / f"frame-{index:03d}.jpg").write_bytes(b"fixture")
    return path


def _settings_group_titles(page):
    layout = page._settings_scroll.widget().layout()
    titles = []
    for index in range(layout.count()):
        widget = layout.itemAt(index).widget()
        if hasattr(widget, "title"):
            titles.append(widget.title)
    return tuple(titles)


def test_i6_4d_initial_ui_is_visible_fail_closed_and_auto_disabled(window) -> None:
    page = window._reconstruction_page

    assert page._provider_selection_status_value.text() == "Waiting for source"
    assert page._provider_selection_status_value.property("statusKind") == "neutral"
    assert page._provider_selection_value.text() == "No manual provider"
    assert page._provider_eligibility_value.text() == "0 / 10 selectable"
    assert page._reconstruction_provider_combo.isEnabled() is False
    assert page._provider_auto_status_value.text() == "Not qualified"
    assert page._provider_auto_status_value.property("statusKind") == "warning"
    assert page._provider_auto_button.isEnabled() is False


def test_i6_4d_exact_qualified_binding_enables_only_eligible_manual_choice(
    window, tmp_path: Path
) -> None:
    controller = window._reconstruction_controller
    controller.select_image_sequence(_images(tmp_path))
    controller.set_reconstruction_advisor_evidence_binding(binding())
    page = window._reconstruction_page

    assert page._provider_selection_status_value.text() == "Ready for manual choice"
    assert page._provider_eligibility_value.text() == "1 / 10 selectable"
    assert page._reconstruction_provider_combo.isEnabled() is True
    assert page._reconstruction_provider_combo.count() == 2
    assert page._reconstruction_provider_combo.itemData(1) == "colmap-mvs-external"
    assert page._provider_auto_button.isEnabled() is False


def test_i6_4d_user_choice_records_manual_intent_without_starting_runtime(
    window, tmp_path: Path
) -> None:
    controller = window._reconstruction_controller
    controller.select_image_sequence(_images(tmp_path))
    controller.set_reconstruction_advisor_evidence_binding(binding())
    page = window._reconstruction_page

    index = page._reconstruction_provider_combo.findData("colmap-mvs-external")
    assert index > 0
    page._on_reconstruction_provider_activated(index)

    state = controller.reconstruction_provider_selection_state
    assert state.status is ManualProviderSelectionStatus.SELECTED
    assert state.selected_provider_id == "colmap-mvs-external"
    assert page._provider_selection_status_value.text() == "Manual selection recorded"
    assert page._provider_selection_value.text() == "COLMAP MVS"
    assert page._reconstruction_provider_clear_button.isEnabled() is True
    assert controller.offline_is_running is False


def test_i6_4d_clear_button_clears_intent_only(window, tmp_path: Path) -> None:
    controller = window._reconstruction_controller
    controller.select_image_sequence(_images(tmp_path))
    controller.set_reconstruction_advisor_evidence_binding(binding())
    controller.select_reconstruction_provider("colmap-mvs-external")
    page = window._reconstruction_page

    page._on_reconstruction_provider_clear_clicked()

    state = controller.reconstruction_provider_selection_state
    assert state.status is ManualProviderSelectionStatus.NONE
    assert state.selected_provider_id == ""
    assert page._provider_selection_value.text() == "No manual provider"
    assert page._reconstruction_provider_clear_button.isEnabled() is False
    assert controller.offline_is_running is False


def test_i6_4d_source_change_visibly_clears_stale_manual_intent(
    window, tmp_path: Path
) -> None:
    controller = window._reconstruction_controller
    controller.select_image_sequence(_images(tmp_path, "images-a"))
    controller.set_reconstruction_advisor_evidence_binding(binding())
    controller.select_reconstruction_provider("colmap-mvs-external")

    controller.select_image_sequence(_images(tmp_path, "images-b"))
    page = window._reconstruction_page

    assert controller.reconstruction_provider_selection_state.selected_provider_id == ""
    assert page._provider_selection_status_value.text() == "Selection cleared"
    assert page._provider_selection_status_value.property("statusKind") == "warning"
    assert page._provider_selection_value.text() == "No manual provider"
    assert controller.offline_is_running is False


def test_i6_4d_auto_never_becomes_selectable_after_manual_provider_is_available(
    window, tmp_path: Path
) -> None:
    controller = window._reconstruction_controller
    controller.select_image_sequence(_images(tmp_path))
    controller.set_reconstruction_advisor_evidence_binding(binding())
    controller.select_reconstruction_provider("colmap-mvs-external")
    page = window._reconstruction_page

    gate = controller.reconstruction_production_auto_gate
    assert gate.production_auto_qualified is False
    assert gate.production_auto_authorized is False
    assert gate.selectable is False
    assert page._provider_auto_status_value.text() == "Not qualified"
    assert page._provider_auto_button.isEnabled() is False
    assert controller.offline_is_running is False


def test_i6_4d_new_group_is_added_after_frozen_i6_3d_primary_stack(window) -> None:
    titles = _settings_group_titles(window._reconstruction_page)
    assert titles[:7] == (
        "DIGITAL TWIN WORKSPACE",
        "BUILD & PROCESSING",
        "CANONICAL SCENE",
        "RULER METRIC ESTIMATE",
        "RECONSTRUCTION ADVISOR",
        "DENSE RGB POINTS",
        "PROVIDER & AUTO",
    )
