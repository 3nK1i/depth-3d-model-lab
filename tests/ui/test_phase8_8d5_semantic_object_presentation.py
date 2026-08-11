from __future__ import annotations

import pytest

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QSizePolicy

from app.main_window import MainWindow
from app.reconstruction_controller import ReconstructionWorkflow
from core.reconstruction import LiveCaptureRuntimeState


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    yield win
    win.close()


def _object(index: int, *, class_name: str, role: str, x: float) -> dict[str, object]:
    return {
        "object_id": f"semantic-{index:04d}",
        "class_name": class_name,
        "semantic_role": role,
        "confidence": 0.91 + index * 0.01,
        "vertex_count": 200 + index * 10,
        "face_count": 120 + index * 5,
        "centroid_reconstruction": [x, 0.0, 2.0],
        "bounds_min_reconstruction": [x - 0.12, -0.16, 1.90],
        "bounds_max_reconstruction": [x + 0.12, 0.16, 2.10],
        "orientation_reliable": True,
        "observation_ids": [f"obs-{index}-0", f"obs-{index}-1", f"obs-{index}-2"],
        "mean_vertex_quality": 0.86 + index * 0.01,
        "median_vertex_uncertainty_reconstruction_units": 0.05,
        "maximum_vertex_uncertainty_reconstruction_units": 0.08,
        "metric_geometry_status": "arbitrary",
        "certified_metrology": False,
    }


def _payload() -> dict[str, object]:
    objects = [
        _object(0, class_name="connector_body", role="connector_body", x=-0.55),
        _object(1, class_name="pin", role="connector_pin", x=0.10),
        _object(2, class_name="pin", role="connector_pin", x=0.42),
    ]
    return {
        "session_id": "phase8-8d5-ui",
        "metric_geometry_status": "arbitrary",
        "certified_metrology": False,
        "statistics": {
            "observations_total": 9,
            "observations_accepted": 9,
            "persistent_object_count": 3,
        },
        "objects": objects,
    }


def _load(window):
    page = window._reconstruction_page
    page.set_semantic_geometry(_payload(), source="phase8-8d5-ui-test")
    return page


def test_d5_semantic_payload_populates_object_group_and_bounded_canvas(window) -> None:
    page = _load(window)
    canvas = page._reconstruction_canvas

    assert page._semantic_objects_group.isHidden() is False
    assert page._semantic_object_combo.isEnabled() is True
    assert page._semantic_object_combo.count() == 4
    assert "3/3 objects shown" in page._semantic_structure_value.text()
    assert "9/9 observations accepted" in page._semantic_structure_value.text()
    assert "Arbitrary reconstruction gauge" in page._semantic_scale_value.text()
    assert "Semantic context only" in page._semantic_authority_value.text()
    assert canvas.semantic_objects_available is True
    assert canvas.semantic_objects_visible is True
    assert canvas.semantic_object_count == 3
    assert canvas.semantic_object_display_count == 3
    assert canvas.semantic_object_ids == ("semantic-0000", "semantic-0001", "semantic-0002")


def test_d5_object_selection_highlights_semantic_context_without_qc_or_measurement_claim(window) -> None:
    page = _load(window)
    page._semantic_object_combo.setCurrentIndex(2)
    canvas = page._reconstruction_canvas

    assert canvas.selected_semantic_object_index == 1
    assert canvas.selected_semantic_object_id == "semantic-0001"
    assert canvas.selected_semantic_object_visible is True
    assert page._semantic_role_value.text() == "Connector Pin"
    assert "3 obs" in page._semantic_support_value.text()
    assert "Semantic 92.0%" in page._semantic_quality_value.text()
    assert "Arbitrary reconstruction gauge" in page._semantic_scale_value.text()
    assert "not inspection/QC decision" in page._semantic_authority_value.text()
    assert "measurement accuracy" in page._semantic_quality_value.toolTip()


def test_d5_navigation_focus_and_clear_are_presentation_only(window) -> None:
    page = _load(window)
    canvas = page._reconstruction_canvas
    page._semantic_object_combo.setCurrentIndex(2)

    page._semantic_next_button.click()
    assert canvas.selected_semantic_object_id == "semantic-0002"
    page._semantic_prev_button.click()
    assert canvas.selected_semantic_object_id == "semantic-0001"
    before = canvas.focus_calls
    page._semantic_focus_button.click()
    assert canvas.focus_calls == before + 1
    page._semantic_clear_button.click()
    assert canvas.selected_semantic_object_id is None
    assert page._semantic_role_value.text() == "—"


def test_d5_semantic_toolbar_toggle_changes_presentation_only(window, qtbot) -> None:
    page = _load(window)
    canvas = page._reconstruction_canvas
    page._semantic_object_combo.setCurrentIndex(1)
    button = page._viewport_tool_buttons["semantic_objects"]

    assert button.isEnabled() is True
    assert button.isChecked() is True
    assert canvas.semantic_objects_visible is True
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    assert canvas.semantic_objects_visible is False
    assert page._offline_semantics is not None
    assert page._semantic_object_combo.currentIndex() == 1
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    assert canvas.semantic_objects_visible is True
    assert canvas.selected_semantic_object_visible is True


def test_d5_new_offline_or_live_session_clears_stale_semantic_context(window) -> None:
    page = _load(window)
    assert page._reconstruction_canvas.semantic_objects_available is True

    page._prepare_runtime_presentation_for_start(ReconstructionWorkflow.IMAGE_SEQUENCE)
    assert page._offline_semantics is None
    assert page._semantic_objects_group.isHidden() is True
    assert page._reconstruction_canvas.semantic_objects_available is False

    page.set_semantic_geometry(_payload())
    page.set_live_state(LiveCaptureRuntimeState.STARTING)
    assert page._offline_semantics is None
    assert page._semantic_objects_group.isHidden() is True
    assert page._reconstruction_canvas.semantic_objects_available is False


def test_d5_semantic_controls_preserve_scrollable_sidebar_contract(window, qtbot) -> None:
    page = _load(window)
    window.resize(1366, 768)
    window.show()
    qtbot.wait(30)

    assert page._settings_scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert page._settings_scroll.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    assert page._semantic_object_combo.minimumWidth() == 0
    assert page._semantic_object_combo.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Ignored
    for value in (
        page._semantic_structure_value,
        page._semantic_role_value,
        page._semantic_support_value,
        page._semantic_quality_value,
        page._semantic_scale_value,
        page._semantic_authority_value,
    ):
        assert value.minimumWidth() == 0
        assert value.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Ignored
    for button in (
        page._semantic_prev_button,
        page._semantic_focus_button,
        page._semantic_next_button,
        page._semantic_clear_button,
    ):
        assert button.minimumWidth() == 0
        assert button.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Ignored
