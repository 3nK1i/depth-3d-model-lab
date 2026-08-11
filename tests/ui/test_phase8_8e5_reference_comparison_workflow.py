from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.main_window import MainWindow
from app.reconstruction_persistence_controller import ReconstructionSceneAvailability

SESSION = "77777777-7777-4777-8777-777777777771"


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    yield win
    win.close()


def _object(index: int, class_name: str, role: str, x: float) -> dict[str, object]:
    return {
        "object_id": f"e5-ui-{index:04d}",
        "class_name": class_name,
        "semantic_role": role,
        "confidence": 0.95 - 0.03 * index,
        "vertex_count": 400 + index * 30,
        "face_count": 230 + index * 20,
        "observation_ids": [f"e5-ui-obs-{index}-{j}" for j in range(3)],
        "mean_vertex_quality": 0.90,
        "median_vertex_uncertainty_reconstruction_units": 0.03,
        "maximum_vertex_uncertainty_reconstruction_units": 0.07,
        "centroid_reconstruction": [x, 0.0, 2.0],
        "bounds_min_reconstruction": [x - 0.20, -0.20, 1.80],
        "bounds_max_reconstruction": [x + 0.20, 0.20, 2.20],
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


def _reference(reference_id: str = "e5-reference-a") -> dict[str, object]:
    return {
        "schema_version": 1,
        "reference_id": reference_id,
        "source": "golden_reconstruction",
        "scene_profile": "object_scan",
        "reference_frame_id": "e5-reference-frame",
        "metric_geometry_status": "arbitrary",
        "unit_to_metre": None,
        "relative_scale_uncertainty": None,
        "certified_metrology": False,
        "features": [
            {"feature_id": "housing-ref", "association_key": "housing-key", "registration_anchor": True},
            {"feature_id": "gear-ref", "association_key": "gear-key", "registration_anchor": False},
        ],
    }


def _report(reference_id: str = "e5-reference-a") -> dict[str, object]:
    return {
        "schema_version": 1,
        "session_id": SESSION,
        "reference_id": reference_id,
        "scene_profile": "object_scan",
        "pose_convention": "T_reconstruction_from_camera",
        "candidate_frame_id": "e5-candidate-frame",
        "reference_frame_id": "e5-reference-frame",
        "candidate_metric_geometry_status": "arbitrary",
        "reference_metric_geometry_status": "arbitrary",
        "comparison_metric_available": False,
        "reference_unit_to_metre": None,
        "relative_scale_uncertainty": None,
        "registration": {
            "status": "success",
            "method": "multi_anchor_similarity",
            "scale_candidate_to_reference": 1.0,
            "anchor_feature_ids": ["housing-ref"],
            "rms_residual_reference_units": 0.03,
            "maximum_residual_reference_units": 0.06,
            "rms_residual_m": None,
            "maximum_residual_m": None,
            "normalized_rms_fraction": 0.015,
            "confidence": 0.93,
            "reason": "registration residual accepted",
        },
        "feature_results": [
            {
                "reference_feature_id": "housing-ref",
                "candidate_object_id": "e5-ui-0000",
                "semantic_role": "structural_component",
                "decision": "pass",
                "findings": ["within_tolerance"],
                "position_error_reference_units": 0.02,
                "position_error_m": None,
                "orientation_error_degrees": 0.7,
                "extent_relative_error": 0.02,
                "geometry_quality": 0.91,
                "uncertainty_reference_units": 0.02,
                "uncertainty_m": None,
            },
            {
                "reference_feature_id": "gear-ref",
                "candidate_object_id": "e5-ui-0001",
                "semantic_role": "rotating_component",
                "decision": "reject",
                "findings": ["position_deviation", "orientation_deviation"],
                "position_error_reference_units": 0.18,
                "position_error_m": None,
                "orientation_error_degrees": 14.0,
                "extent_relative_error": 0.09,
                "geometry_quality": 0.86,
                "uncertainty_reference_units": 0.04,
                "uncertainty_m": None,
            },
        ],
        "extra_candidate_object_ids": ["e5-ui-0002"],
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
    page.set_semantic_geometry(_semantic_payload(), source="phase8_7b.semantic_geometry")
    page._semantic_object_combo.setCurrentIndex(2)  # gear
    return page


def test_e5_initial_group_requires_explicit_reference(window) -> None:
    page = window._reconstruction_page
    assert page._reference_comparison_group.isHidden() is False
    assert page._reference_comparison_reference_value.text() == "No reference attached"
    assert page.reconstruction_reference_comparison_presentation is None
    assert page._reference_comparison_load_button.isEnabled() is False


def test_e5_reference_plus_matching_e4_report_exposes_exact_object_deviations(window, qtbot) -> None:
    page = _prepare(window)
    page.set_reconstruction_inspection_qc_evidence(_report())
    with qtbot.waitSignal(page.reconstruction_reference_comparison_changed, timeout=1000) as blocker:
        page.set_reconstruction_reference_comparison_reference(_reference())

    presentation = blocker.args[0]
    assert presentation is page.reconstruction_reference_comparison_presentation
    assert presentation.report_reference_match is True
    assert presentation.target_object_id == "e5-ui-0001"
    assert page._reference_comparison_target_value.text() == "gear • Rotating Component"
    assert "e5-reference-a" in page._reference_comparison_reference_value.text()
    assert page._reference_comparison_registration_value.text().startswith("Success • 93%")
    assert "Position 0.18 reference units" in page._reference_comparison_deviations_value.text()
    assert "Orientation 14.00°" in page._reference_comparison_deviations_value.text()
    assert "Reference feature gear-ref" in page._reference_comparison_evidence_value.text()
    assert page._reference_comparison_scale_value.text().startswith("Relative-only comparison")
    assert page._reference_comparison_state_value.text() == "Comparison evidence available"
    assert "does not register/apply tolerances" in page._reference_comparison_authority_value.text()


def test_e5_switch_to_unmatched_object_never_reuses_other_object_deviation(window) -> None:
    page = _prepare(window)
    page.set_reconstruction_inspection_qc_evidence(_report())
    page.set_reconstruction_reference_comparison_reference(_reference())
    assert "Position 0.18" in page._reference_comparison_deviations_value.text()

    page._semantic_object_combo.setCurrentIndex(3)  # shaft is extra, no feature result
    assert page.inspection_selection.object_id == "e5-ui-0002"
    assert page._reference_comparison_deviations_value.text() == "No exact selected-object deviation result"
    assert "selected object not evaluated" in page._reference_comparison_state_value.text()


def test_e5_reference_switch_does_not_reuse_report_from_different_reference(window) -> None:
    page = _prepare(window)
    page.set_reconstruction_inspection_qc_evidence(_report(reference_id="reference-a"))
    page.set_reconstruction_reference_comparison_reference(_reference(reference_id="reference-b"))
    presentation = page.reconstruction_reference_comparison_presentation
    assert presentation is not None
    assert presentation.report_reference_match is False
    assert page._reference_comparison_deviations_value.text() == "—"
    assert page._reference_comparison_state_value.text() == "Reference/report mismatch • comparison unavailable"


def test_e5_load_json_and_clear_reference_workflow(window, tmp_path: Path) -> None:
    page = _prepare(window)
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(_reference()), encoding="utf-8")
    page.load_reconstruction_reference_comparison_json(path)
    assert page.reconstruction_reference_comparison_presentation is not None
    assert page._reference_comparison_clear_button.isEnabled() is True
    assert str(path) in page._reference_comparison_reference_value.toolTip()

    page._clear_active_reconstruction_reference()
    assert page.reconstruction_reference_comparison_presentation is None
    assert page._reference_comparison_reference_value.text() == "No reference attached"
    assert page._reference_comparison_clear_button.isEnabled() is False


def test_e5_new_scene_clear_drops_stale_reference(window) -> None:
    page = _prepare(window)
    page.set_reconstruction_reference_comparison_reference(_reference())
    assert page.reconstruction_reference_comparison_presentation is not None
    page.clear_reconstruction_reference_comparison()
    assert page.reconstruction_reference_comparison_presentation is None
