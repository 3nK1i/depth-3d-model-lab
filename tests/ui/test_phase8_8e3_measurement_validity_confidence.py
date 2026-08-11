from __future__ import annotations

import pytest

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
        "object_id": f"e3-ui-{index:04d}",
        "class_name": class_name,
        "semantic_role": role,
        "confidence": 0.99 if index == 0 else 0.91,
        "vertex_count": 360 + index * 20,
        "face_count": 210 + index * 10,
        "observation_ids": [f"e3-ui-obs-{index}-{j}" for j in range(3)],
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
        "session_id": "55555555-5555-4555-8555-555555555551",
        "metric_geometry_status": "arbitrary",
        "certified_metrology": False,
        "statistics": {"observations_total": 6, "observations_accepted": 6, "persistent_object_count": 2},
        "objects": [
            _object(0, "gear", "rotating_component", -0.25),
            _object(1, "shaft", "supporting_component", 0.35),
        ],
    }


def _measurement_payload(*, observation_id: str = "phase75-e3-ui-gear") -> dict[str, object]:
    return {
        "observation_id": observation_id,
        "measurements": [
            {
                "kind": "width", "status": "available", "value_si": 0.0421, "unit_si": "m",
                "method": "preferred_box_extent", "expanded_uncertainty_si": 0.0006,
                "lower_bound_si": 0.0415, "upper_bound_si": 0.0427,
                "reliability": "medium", "reliability_score": 0.78, "valid_observation_count": 1,
                "uncertainty_basis": "heuristic_perturbation",
                "uncertainty_qualification": "Estimated Uncertainty — Not Calibrated",
                "rejection_reasons": [], "warnings": [],
            },
            {
                "kind": "height", "status": "available", "value_si": 0.0182, "unit_si": "m",
                "method": "preferred_box_extent", "expanded_uncertainty_si": 0.0005,
                "lower_bound_si": 0.0177, "upper_bound_si": 0.0187,
                "reliability": "high", "reliability_score": 0.94, "valid_observation_count": 1,
                "uncertainty_basis": "heuristic_perturbation",
                "uncertainty_qualification": "Estimated Uncertainty — Not Calibrated",
                "rejection_reasons": [], "warnings": [],
            },
            {
                "kind": "depth", "status": "rejected", "value_si": None, "unit_si": "m",
                "method": "preferred_box_extent", "expanded_uncertainty_si": None,
                "lower_bound_si": None, "upper_bound_si": None,
                "reliability": "rejected", "valid_observation_count": 0,
                "uncertainty_basis": "unknown", "uncertainty_qualification": "",
                "rejection_reasons": ["excessive_uncertainty"],
                "warnings": ["uncertainty exceeds configured threshold"],
            },
        ],
    }


def _context() -> dict[str, object]:
    return {
        "depth_scale": "metric",
        "intrinsics_source": "model_predicted",
        "calibration_status": "not_calibrated",
        "confidence_kind": "uncertainty",
    }


def _prepare_target(window):
    page = window._reconstruction_page
    page.set_semantic_geometry(_semantic_payload(), source="e3-ui-generic")
    page._semantic_object_combo.setCurrentIndex(1)
    return page


def test_e3_initial_group_is_explicitly_empty(window) -> None:
    page = window._reconstruction_page
    assert page._measurement_validity_group.isHidden() is False
    assert page._measurement_validity_qualified_value.text() == "—"
    assert page._measurement_validity_status_value.text() == "Waiting for reconstruction target"
    assert page.reconstruction_measurement_validity_presentation is None
    assert "does not compute confidence/uncertainty" in page._measurement_validity_authority_value.text()


def test_e3_requires_existing_e2_evidence(window) -> None:
    page = _prepare_target(window)
    with pytest.raises(ValueError, match="existing E2 measurement evidence"):
        page.set_reconstruction_measurement_validity_evidence(
            _measurement_payload(), source="phase7_5.inspection_bundle", context=_context()
        )


def test_e3_presents_source_authored_qualification_for_exact_e2_target(window, qtbot) -> None:
    page = _prepare_target(window)
    payload = _measurement_payload()
    page.set_reconstruction_measurement_evidence(payload, source="phase7_5.inspection_bundle")

    with qtbot.waitSignal(page.reconstruction_measurement_validity_changed, timeout=1000) as blocker:
        page.set_reconstruction_measurement_validity_evidence(
            payload, source="phase7_5.inspection_bundle", context=_context()
        )

    presentation = blocker.args[0]
    assert presentation is page.reconstruction_measurement_validity_presentation
    assert presentation.target_object_id == "e3-ui-0000"
    assert "Width: 42.1 mm ± 0.6 mm" in page._measurement_validity_qualified_value.text()
    assert page._measurement_validity_status_value.text() == "2 available • 1 rejected"
    assert "Width: Medium 78%" in page._measurement_validity_reliability_value.text()
    assert "Depth: Rejected" in page._measurement_validity_reliability_value.text()
    assert "Estimated Uncertainty — Not Calibrated" in page._measurement_validity_uncertainty_value.text()
    assert "Measurement: Metric depth" in page._measurement_validity_scale_value.text()
    assert "scene: Arbitrary gauge" in page._measurement_validity_scale_value.text()
    assert page._measurement_validity_calibration_value.text() == "Model-predicted intrinsics • Not Calibrated"
    assert "Depth: Excessive uncertainty" in page._measurement_validity_reasons_value.text()
    assert "scene scale unchanged" in page._measurement_validity_authority_value.text()
    # The semantic object confidence is 99%, but E3 shows the source measurement
    # reliability (78% medium); it does not copy semantic confidence.
    assert "99%" not in page._measurement_validity_reliability_value.text()


def test_e3_rejects_source_or_observation_mismatch_against_e2(window) -> None:
    page = _prepare_target(window)
    payload = _measurement_payload()
    page.set_reconstruction_measurement_evidence(payload, source="phase7_5.inspection_bundle")
    with pytest.raises(ValueError, match="source must match"):
        page.set_reconstruction_measurement_validity_evidence(
            payload, source="different-source", context=_context()
        )
    with pytest.raises(ValueError, match="observation must match"):
        page.set_reconstruction_measurement_validity_evidence(
            _measurement_payload(observation_id="different-observation"),
            source="phase7_5.inspection_bundle",
            context=_context(),
        )


def test_e3_selection_change_hides_and_return_restores_exact_bound_validity(window) -> None:
    page = _prepare_target(window)
    payload = _measurement_payload()
    page.set_reconstruction_measurement_evidence(payload, source="phase7_5.inspection_bundle")
    page.set_reconstruction_measurement_validity_evidence(
        payload, source="phase7_5.inspection_bundle", context=_context()
    )
    saved = page.reconstruction_measurement_validity_presentation

    page._semantic_object_combo.setCurrentIndex(2)
    assert page.reconstruction_measurement_validity_presentation is None
    assert page._measurement_validity_status_value.text() == "Measurement evidence required"

    page._semantic_object_combo.setCurrentIndex(1)
    assert page.reconstruction_measurement_validity_presentation is saved


def test_e3_replacing_e2_evidence_invalidates_previous_validity(window) -> None:
    page = _prepare_target(window)
    payload = _measurement_payload()
    page.set_reconstruction_measurement_evidence(payload, source="phase7_5.inspection_bundle")
    page.set_reconstruction_measurement_validity_evidence(
        payload, source="phase7_5.inspection_bundle", context=_context()
    )
    assert page.reconstruction_measurement_validity_presentation is not None

    page.set_reconstruction_measurement_evidence(payload, source="phase7_5.inspection_bundle")
    assert page.reconstruction_measurement_validity_presentation is None
    assert page._measurement_validity_status_value.text() == "Validity evidence not attached"
