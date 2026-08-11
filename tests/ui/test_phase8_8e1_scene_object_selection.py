from __future__ import annotations

from types import SimpleNamespace

import pytest

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QSizePolicy

from app.main_window import MainWindow
from app.reconstruction_inspection_selection import ReconstructionInspectionSelectionScope


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    yield win
    win.close()


def _object(index: int, *, class_name: str, role: str, x: float) -> dict[str, object]:
    return {
        "object_id": f"generic-object-{index:04d}",
        "class_name": class_name,
        "semantic_role": role,
        "confidence": 0.90 + index * 0.02,
        "vertex_count": 300 + index * 20,
        "face_count": 180 + index * 10,
        "centroid_reconstruction": [x, 0.0, 2.0],
        "bounds_min_reconstruction": [x - 0.15, -0.18, 1.85],
        "bounds_max_reconstruction": [x + 0.15, 0.18, 2.15],
        "orientation_reliable": True,
        "observation_ids": [f"obs-{index}-0", f"obs-{index}-1", f"obs-{index}-2"],
        "mean_vertex_quality": 0.86 + index * 0.01,
        "median_vertex_uncertainty_reconstruction_units": 0.05,
        "maximum_vertex_uncertainty_reconstruction_units": 0.08,
        "metric_geometry_status": "arbitrary",
        "certified_metrology": False,
    }


def _payload(session_id: str = "e1-ui-session") -> dict[str, object]:
    objects = [
        _object(0, class_name="gear", role="rotating_component", x=-0.35),
        _object(1, class_name="shaft", role="supporting_component", x=0.35),
    ]
    return {
        "session_id": session_id,
        "metric_geometry_status": "arbitrary",
        "certified_metrology": False,
        "statistics": {
            "observations_total": 6,
            "observations_accepted": 6,
            "persistent_object_count": 2,
        },
        "objects": objects,
    }


def test_e1_initial_target_group_is_explicitly_empty(window) -> None:
    page = window._reconstruction_page
    selection = page.inspection_selection

    assert page._inspection_target_group.isHidden() is False
    assert selection.scope is ReconstructionInspectionSelectionScope.NONE
    assert page._inspection_scope_value.text() == "—"
    assert page._inspection_scene_value.text() == "No reconstruction scene"
    assert page._inspection_object_value.text() == "—"
    assert page._inspection_geometry_value.text() == "No geometry target"
    assert page._inspection_state_value.text() == "No inspection target"
    assert "measurement/QC not evaluated" in page._inspection_authority_value.text()


def test_e1_d5_semantic_context_selects_scene_without_duplicate_object_picker(window) -> None:
    page = window._reconstruction_page
    page.set_semantic_geometry(_payload(), source="e1-ui-generic")

    selection = page.inspection_selection
    assert selection.scope is ReconstructionInspectionSelectionScope.SCENE
    assert selection.session_id == "e1-ui-session"
    assert page._inspection_scope_value.text() == "Scene"
    assert "Session" in page._inspection_scene_value.text()
    assert page._inspection_object_value.text() == "—"
    assert page._inspection_state_value.text() == "Scene target selected"
    assert page._semantic_object_combo.count() == 3


def test_e1_existing_d5_object_selection_updates_same_inspection_target(window, qtbot) -> None:
    page = window._reconstruction_page
    page.set_semantic_geometry(_payload(), source="e1-ui-generic")

    with qtbot.waitSignal(page.inspection_target_changed, timeout=1000) as blocker:
        page._semantic_object_combo.setCurrentIndex(2)

    selection = blocker.args[0]
    assert selection is page.inspection_selection
    assert selection.scope is ReconstructionInspectionSelectionScope.OBJECT
    assert selection.object_id == "generic-object-0001"
    assert selection.class_name == "shaft"
    assert selection.semantic_role == "supporting_component"
    assert page._inspection_scope_value.text() == "Object"
    assert page._inspection_object_value.text() == "shaft • Supporting Component"
    assert page._inspection_geometry_value.text() == "Semantic support geometry available"
    assert page._inspection_state_value.text() == "Object target selected"
    assert "measurement/QC not evaluated" in page._inspection_authority_value.text()
    assert "generic-object-0001" in page._inspection_object_value.toolTip()


def test_e1_clearing_d5_object_returns_to_scene_target(window) -> None:
    page = window._reconstruction_page
    page.set_semantic_geometry(_payload(), source="e1-ui-generic")
    page._semantic_object_combo.setCurrentIndex(1)
    assert page.inspection_selection.has_object

    page._semantic_clear_button.click()
    assert page.inspection_selection.scope is ReconstructionInspectionSelectionScope.SCENE
    assert page.inspection_selection.has_object is False
    assert page._inspection_scope_value.text() == "Scene"
    assert page._inspection_state_value.text() == "Scene target selected"


def test_e1_persistent_scene_becomes_canonical_scene_target_without_semantics(window) -> None:
    page = window._reconstruction_page
    scene = SimpleNamespace(
        scene_id="prs-0123456789abcdef01234567",
        session_id="77777777-7777-4777-8777-777777777771",
        canonical_geometry_authoritative=True,
        artifacts=(),
    )
    page.set_persistent_scene(scene, "/tmp/persistent_reconstruction_scene.json")

    selection = page.inspection_selection
    assert selection.scope is ReconstructionInspectionSelectionScope.SCENE
    assert selection.scene_id == scene.scene_id
    assert selection.canonical_geometry_authoritative is True
    assert page._inspection_geometry_value.text() == "Canonical scene geometry available"
    assert scene.scene_id in page._inspection_scene_value.toolTip()
    assert selection.certified_metrology is False


def test_e1_selection_rows_preserve_scrollable_sidebar_contract(window, qtbot) -> None:
    page = window._reconstruction_page
    page.set_semantic_geometry(_payload(), source="e1-ui-generic")
    page._semantic_object_combo.setCurrentIndex(1)
    window.resize(1366, 768)
    window.show()
    qtbot.wait(30)

    assert page._settings_scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert page._settings_scroll.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    for value in (
        page._inspection_scope_value,
        page._inspection_scene_value,
        page._inspection_object_value,
        page._inspection_geometry_value,
        page._inspection_state_value,
        page._inspection_authority_value,
    ):
        assert value.minimumWidth() == 0
        assert value.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Ignored
    assert page._settings_scroll.verticalScrollBar().maximum() > 0
