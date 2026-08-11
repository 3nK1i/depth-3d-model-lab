from __future__ import annotations

import pytest

from app.main_window import MainWindow
from app.reconstruction_persistence_controller import ReconstructionSceneAvailability


SESSION = "99999999-9999-4999-8999-999999999991"


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    yield win
    win.close()


def _object(index: int, class_name: str, role: str, x: float) -> dict[str, object]:
    return {
        "object_id": f"e4-ui-{index:04d}",
        "class_name": class_name,
        "semantic_role": role,
        "confidence": 0.97 - 0.05 * index,
        "vertex_count": 360 + 20 * index,
        "face_count": 210 + 10 * index,
        "observation_ids": [f"e4-ui-obs-{index}-{j}" for j in range(3)],
        "mean_vertex_quality": 0.90,
        "median_vertex_uncertainty_reconstruction_units": 0.04,
        "maximum_vertex_uncertainty_reconstruction_units": 0.08,
        "centroid_reconstruction": [x, 0.0, 2.0],
        "bounds_min_reconstruction": [x - 0.18, -0.20, 1.82],
        "bounds_max_reconstruction": [x + 0.18, 0.20, 2.18],
        "orientation_reliable": True,
        "metric_geometry_status": "arbitrary",
        "certified_metrology": False,
    }


def _semantic_payload() -> dict[str, object]:
    return {
        "session_id": SESSION,
        "metric_geometry_status": "arbitrary",
        "certified_metrology": False,
        "statistics": {"observations_total": 9, "observations_accepted": 9, "persistent_object_count": 3},
        "objects": [
            _object(0, "housing", "structural_component", -0.45),
            _object(1, "gear", "rotating_component", 0.0),
            _object(2, "shaft", "supporting_component", 0.45),
        ],
    }


def _report() -> dict[str, object]:
    return {
        "schema_version": 1,
        "session_id": SESSION,
        "reference_id": "known-good-assembly",
        "candidate_metric_geometry_status": "arbitrary",
        "reference_metric_geometry_status": "arbitrary",
        "comparison_metric_available": False,
        "registration": {
            "status": "success",
            "method": "multi_anchor_similarity",
            "confidence": 0.91,
            "reason": "registration residual accepted",
        },
        "feature_results": [
            {
                "reference_feature_id": "housing-ref",
                "candidate_object_id": "e4-ui-0000",
                "semantic_role": "structural_component",
                "decision": "pass",
                "findings": ["within_tolerance"],
            },
            {
                "reference_feature_id": "gear-ref",
                "candidate_object_id": "e4-ui-0001",
                "semantic_role": "rotating_component",
                "decision": "reject",
                "findings": ["position_deviation", "orientation_deviation"],
            },
        ],
        "extra_candidate_object_ids": ["e4-ui-0002"],
        "decision": "reject",
        "statistics": {
            "reference_feature_count": 2,
            "matched_feature_count": 2,
            "missing_required_count": 0,
            "extra_candidate_count": 1,
            "pass_count": 1,
            "warn_count": 1,
            "reject_count": 1,
            "insufficient_evidence_count": 0,
        },
        "certified_metrology": False,
        "inspection_qc_enabled": True,
    }


def _prepare(window):
    page = window._reconstruction_page
    page.set_scene_availability(ReconstructionSceneAvailability.PARTIAL)
    page.set_semantic_geometry(_semantic_payload(), source="e4-ui-generic")
    page._semantic_object_combo.setCurrentIndex(2)  # gear
    return page


def test_e4_initial_group_is_explicitly_unevaluated(window) -> None:
    page = window._reconstruction_page
    assert page._inspection_qc_group.isHidden() is False
    assert page._inspection_qc_decision_value.text() == "Not evaluated"
    assert page.reconstruction_inspection_qc_presentation is None
    assert "does not run inspection" in page._inspection_qc_authority_value.text()


def test_e4_presents_existing_qc_for_exact_selected_object(window, qtbot) -> None:
    page = _prepare(window)
    with qtbot.waitSignal(page.reconstruction_inspection_qc_changed, timeout=1000) as blocker:
        page.set_reconstruction_inspection_qc_evidence(_report(), source="phase8_7c.inspection_report")

    presentation = blocker.args[0]
    assert presentation is page.reconstruction_inspection_qc_presentation
    assert presentation.target_object_id == "e4-ui-0001"
    assert page._inspection_qc_target_value.text() == "gear • Rotating Component"
    assert page._inspection_qc_decision_value.text() == "REJECT • Object"
    assert "Position deviation" in page._inspection_qc_findings_value.text()
    assert "Orientation deviation" in page._inspection_qc_findings_value.text()
    assert page._inspection_qc_coverage_value.text() == "2/2 matched • 0 missing • 1 extra"
    assert "1 pass" in page._inspection_qc_decisions_value.text()
    assert "1 reject" in page._inspection_qc_decisions_value.text()
    assert page._inspection_qc_registration_value.text().startswith("Success • 91%")
    assert page._inspection_qc_comparison_value.text().startswith("Relative-only comparison")
    assert "reference workflow deferred to E5" in page._inspection_qc_authority_value.text()


def test_e4_selection_change_never_reuses_scene_decision_as_unmatched_object_qc(window) -> None:
    page = _prepare(window)
    page.set_reconstruction_inspection_qc_evidence(_report())
    assert page._inspection_qc_decision_value.text() == "REJECT • Object"

    page._semantic_object_combo.setCurrentIndex(3)  # shaft is extra / has no feature result
    assert page.inspection_selection.object_id == "e4-ui-0002"
    assert page.reconstruction_inspection_qc_presentation is not None
    assert page.reconstruction_inspection_qc_presentation.target_result_available is False
    assert page._inspection_qc_decision_value.text() == "NOT EVALUATED • Object"
    assert page._inspection_qc_findings_value.text() == "No selected-object result in attached report"

    page._semantic_object_combo.setCurrentIndex(2)
    assert page._inspection_qc_decision_value.text() == "REJECT • Object"


def test_e4_clear_semantic_selection_exposes_source_scene_decision_only(window) -> None:
    page = _prepare(window)
    page.set_reconstruction_inspection_qc_evidence(_report())
    page._semantic_object_combo.setCurrentIndex(0)
    assert page.inspection_selection.has_object is False
    assert page._inspection_qc_decision_value.text() == "REJECT • Scene"
    assert page._inspection_qc_findings_value.text() == "See source report statistics"


def test_e4_rejects_mismatched_report_session(window) -> None:
    page = _prepare(window)
    report = _report()
    report["session_id"] = "different-session"
    with pytest.raises(ValueError, match="session does not match"):
        page.set_reconstruction_inspection_qc_evidence(report)


def test_e4_new_runtime_or_scene_clear_drops_stale_qc(window) -> None:
    page = _prepare(window)
    page.set_reconstruction_inspection_qc_evidence(_report())
    assert page.reconstruction_inspection_qc_presentation is not None

    page.clear_reconstruction_inspection_qc_evidence()
    assert page.reconstruction_inspection_qc_presentation is None
    assert page._inspection_qc_decision_value.text() == "Not evaluated"
