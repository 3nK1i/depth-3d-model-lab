from __future__ import annotations

import pytest

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QSizePolicy

from app.main_window import MainWindow


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    yield win
    win.close()


def _object(index: int, class_name: str, role: str, x: float) -> dict[str, object]:
    return {
        "object_id": f"e2-ui-{index:04d}",
        "class_name": class_name,
        "semantic_role": role,
        "confidence": 0.92,
        "vertex_count": 320 + index * 20,
        "face_count": 190 + index * 10,
        "observation_ids": [f"e2-ui-obs-{index}-{j}" for j in range(3)],
        "mean_vertex_quality": 0.87,
        "median_vertex_uncertainty_reconstruction_units": 0.05,
        "maximum_vertex_uncertainty_reconstruction_units": 0.09,
        "centroid_reconstruction": [x, 0.0, 2.0],
        "bounds_min_reconstruction": [x - 0.18, -0.20, 1.82],
        "bounds_max_reconstruction": [x + 0.18, 0.20, 2.18],
        "orientation_reliable": True,
        "metric_geometry_status": "arbitrary",
        "certified_metrology": False,
    }


def _semantic_payload() -> dict[str, object]:
    return {
        "session_id": "22222222-2222-4222-8222-222222222221",
        "metric_geometry_status": "arbitrary",
        "certified_metrology": False,
        "statistics": {
            "observations_total": 6,
            "observations_accepted": 6,
            "persistent_object_count": 2,
        },
        "objects": [
            _object(0, "gear", "rotating_component", -0.25),
            _object(1, "shaft", "supporting_component", 0.35),
        ],
    }


def _measurement_payload() -> dict[str, object]:
    return {
        "observation_id": "phase75-e2-ui-gear",
        "measurements": [
            {
                "kind": "width",
                "status": "available",
                "value_si": 0.0421,
                "unit_si": "m",
                "method": "preferred_box_extent",
                "expanded_uncertainty_si": 0.0006,
                "reliability": "medium",
            },
            {
                "kind": "height",
                "status": "available",
                "value_si": 0.0182,
                "unit_si": "m",
                "method": "preferred_box_extent",
                "expanded_uncertainty_si": 0.0005,
                "reliability": "high",
            },
            {
                "kind": "depth",
                "status": "rejected",
                "value_si": None,
                "unit_si": "m",
                "method": "preferred_box_extent",
                "rejection_reasons": ["excessive_uncertainty"],
                "reliability": "rejected",
            },
        ],
    }


def test_e2_initial_measurement_group_is_explicitly_empty(window) -> None:
    page = window._reconstruction_page
    assert page._reconstruction_measurement_group.isHidden() is False
    assert page._measurement_target_value.text() == "No target"
    assert page._measurement_values_value.text() == "—"
    assert page.reconstruction_measurement_presentation is None
    assert "does not compute measurements" in page._measurement_authority_value.text()


def test_e2_selected_object_waits_for_existing_measurement_evidence(window) -> None:
    page = window._reconstruction_page
    page.set_semantic_geometry(_semantic_payload(), source="e2-ui-generic")
    page._semantic_object_combo.setCurrentIndex(1)

    assert page._measurement_target_value.text() == "gear • Rotating Component"
    assert page._measurement_source_value.text() == "No measurement evidence attached"
    assert page._measurement_values_value.text() == "—"
    assert page._measurement_state_value.text() == "No values for selected object"
    assert page.reconstruction_measurement_presentation is None


def test_e2_attaches_phase75_shaped_measurement_evidence_to_active_e1_target(window, qtbot) -> None:
    page = window._reconstruction_page
    page.set_semantic_geometry(_semantic_payload(), source="e2-ui-generic")
    page._semantic_object_combo.setCurrentIndex(1)

    with qtbot.waitSignal(page.reconstruction_measurement_changed, timeout=1000) as blocker:
        page.set_reconstruction_measurement_evidence(
            _measurement_payload(), source="phase7_5.inspection_bundle"
        )

    presentation = blocker.args[0]
    assert presentation is page.reconstruction_measurement_presentation
    assert presentation.target_object_id == "e2-ui-0000"
    assert page._measurement_target_value.text() == "gear • Rotating Component"
    assert page._measurement_source_value.text() == "phase7_5.inspection_bundle"
    assert "Width: 42.1 mm" in page._measurement_values_value.text()
    assert "Height: 18.2 mm" in page._measurement_values_value.text()
    assert "Depth" not in page._measurement_values_value.text()
    assert "±" not in page._measurement_values_value.text()
    assert page._measurement_state_value.text() == "2/3 values presented"
    assert "validity/confidence deferred to E3" in page._measurement_authority_value.text()
    assert "phase75-e2-ui-gear" in page._measurement_source_value.toolTip()


def test_e2_never_reassigns_evidence_by_class_when_d5_selection_changes(window) -> None:
    page = window._reconstruction_page
    page.set_semantic_geometry(_semantic_payload(), source="e2-ui-generic")
    page._semantic_object_combo.setCurrentIndex(1)
    page.set_reconstruction_measurement_evidence(_measurement_payload())
    assert page.reconstruction_measurement_presentation is not None

    page._semantic_object_combo.setCurrentIndex(2)
    assert page.inspection_selection.object_id == "e2-ui-0001"
    assert page.reconstruction_measurement_presentation is None
    assert page._measurement_source_value.text() == "No measurement evidence attached"

    page._semantic_object_combo.setCurrentIndex(1)
    assert page.reconstruction_measurement_presentation is not None
    assert page.reconstruction_measurement_presentation.target_object_id == "e2-ui-0000"


def test_e2_clear_semantic_selection_returns_measurement_panel_to_scene_state(window) -> None:
    page = window._reconstruction_page
    page.set_semantic_geometry(_semantic_payload(), source="e2-ui-generic")
    page._semantic_object_combo.setCurrentIndex(1)
    page.set_reconstruction_measurement_evidence(_measurement_payload())

    page._semantic_clear_button.click()
    assert page.reconstruction_measurement_presentation is None
    assert page._measurement_target_value.text() == "Scene"
    assert page._measurement_state_value.text() == "Select a semantic object"


def test_e2_measurement_rows_preserve_scrollable_sidebar_contract(window, qtbot) -> None:
    page = window._reconstruction_page
    page.set_semantic_geometry(_semantic_payload(), source="e2-ui-generic")
    page._semantic_object_combo.setCurrentIndex(1)
    page.set_reconstruction_measurement_evidence(_measurement_payload())
    window.resize(1366, 768)
    window.show()
    qtbot.wait(30)

    assert page._settings_scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert page._settings_scroll.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    for value in (
        page._measurement_target_value,
        page._measurement_source_value,
        page._measurement_values_value,
        page._measurement_state_value,
        page._measurement_authority_value,
    ):
        assert value.minimumWidth() == 0
        assert value.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Ignored
    assert page._settings_scroll.verticalScrollBar().maximum() > 0
