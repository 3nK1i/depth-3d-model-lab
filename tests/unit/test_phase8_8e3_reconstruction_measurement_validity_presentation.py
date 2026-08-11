from __future__ import annotations

import pytest

from core.inspection import load_inspection_bundle
from tests.inspection_helpers import write_inspection_bundle

from app.reconstruction_inspection_selection import (
    ReconstructionInspectionSelection,
    ReconstructionInspectionSelectionScope,
)
from app.reconstruction_measurement_validity_presentation import (
    ReconstructionMeasurementValidityContext,
    ReconstructionMeasurementValidityEntry,
    build_reconstruction_measurement_validity_presentation,
    empty_reconstruction_measurement_validity_text,
)


def _selection(*, metric_status: str = "arbitrary") -> ReconstructionInspectionSelection:
    return ReconstructionInspectionSelection(
        scope=ReconstructionInspectionSelectionScope.OBJECT,
        session_id="e3-session",
        scene_source="semantic-smoke",
        scene_availability="partial",
        object_id="generic-object-0001",
        class_name="gear",
        semantic_role="rotating_component",
        semantic_source="phase8_7b.semantic_geometry",
        metric_geometry_status=metric_status,
        certified_metrology=False,
    )


def _payload() -> dict[str, object]:
    return {
        "observation_id": "phase75-e3-observation",
        "measurements": [
            {
                "kind": "width",
                "status": "available",
                "value_si": 0.0421,
                "unit_si": "m",
                "method": "preferred_box_extent",
                "expanded_uncertainty_si": 0.0006,
                "lower_bound_si": 0.0415,
                "upper_bound_si": 0.0427,
                "reliability": "medium",
                "reliability_score": 0.78,
                "valid_observation_count": 1,
                "uncertainty_basis": "heuristic_perturbation",
                "uncertainty_qualification": "Estimated Uncertainty — Not Calibrated",
                "rejection_reasons": [],
                "warnings": [],
            },
            {
                "kind": "height",
                "status": "available",
                "value_si": 0.0182,
                "unit_si": "m",
                "method": "preferred_box_extent",
                "expanded_uncertainty_si": 0.0005,
                "lower_bound_si": 0.0177,
                "upper_bound_si": 0.0187,
                "reliability": "high",
                "reliability_score": 0.94,
                "valid_observation_count": 1,
                "uncertainty_basis": "heuristic_perturbation",
                "uncertainty_qualification": "Estimated Uncertainty — Not Calibrated",
                "rejection_reasons": [],
                "warnings": [],
            },
            {
                "kind": "depth",
                "status": "rejected",
                "value_si": None,
                "unit_si": "m",
                "method": "preferred_box_extent",
                "expanded_uncertainty_si": None,
                "lower_bound_si": None,
                "upper_bound_si": None,
                "reliability": "rejected",
                "valid_observation_count": 0,
                "uncertainty_basis": "unknown",
                "uncertainty_qualification": "",
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


def test_e3_presents_exact_source_uncertainty_bounds_reliability_and_reasons() -> None:
    presentation = build_reconstruction_measurement_validity_presentation(
        _selection(),
        _payload(),
        source="phase7_5.inspection_bundle",
        context=_context(),
    )

    assert presentation.target_object_id == "generic-object-0001"
    assert presentation.source_observation_id == "phase75-e3-observation"
    assert "Width: 42.1 mm ± 0.6 mm [41.5 mm–42.7 mm]" in presentation.qualified_values_text
    assert "Height: 18.2 mm ± 0.5 mm [17.7 mm–18.7 mm]" in presentation.qualified_values_text
    assert "Depth" not in presentation.qualified_values_text
    assert presentation.validity_text == "2 available • 1 rejected"
    assert "Width: Medium 78%" in presentation.reliability_text
    assert "Height: High 94%" in presentation.reliability_text
    assert "Depth: Rejected" in presentation.reliability_text
    assert "Estimated Uncertainty — Not Calibrated" in presentation.uncertainty_text
    assert "depth uncertainty channel" in presentation.uncertainty_text
    assert "Measurement: Metric depth" in presentation.scale_text
    assert "scene: Arbitrary gauge" in presentation.scale_text
    assert presentation.calibration_text == "Model-predicted intrinsics • Not Calibrated"
    assert "Width: 1 obs" in presentation.support_text
    assert "Depth: 0 obs" in presentation.support_text
    assert "Depth: Excessive uncertainty" in presentation.reasons_text
    assert "Uncertainty exceeds configured threshold" in presentation.reasons_text
    assert presentation.certified_metrology is False


def test_e3_preserves_scene_scale_authority_when_measurement_source_is_metric() -> None:
    presentation = build_reconstruction_measurement_validity_presentation(
        _selection(metric_status="arbitrary"), _payload(), context=_context()
    )
    assert "Measurement: Metric depth" in presentation.scale_text
    assert "scene: Arbitrary gauge" in presentation.scale_text
    assert "scene scale unchanged" in presentation.authority_text
    assert "QC decision" in presentation.authority_text


def test_e3_does_not_accept_available_measurement_without_uncertainty() -> None:
    payload = _payload()
    payload["measurements"][0]["expanded_uncertainty_si"] = None  # type: ignore[index]
    with pytest.raises(ValueError, match="source value and uncertainty"):
        build_reconstruction_measurement_validity_presentation(_selection(), payload)


def test_e3_enforces_source_status_reliability_consistency() -> None:
    payload = _payload()
    payload["measurements"][2]["reliability"] = "high"  # type: ignore[index]
    with pytest.raises(ValueError, match="rejected.*source reliability"):
        build_reconstruction_measurement_validity_presentation(_selection(), payload)


def test_e3_rejects_numeric_values_on_rejected_measurements() -> None:
    payload = _payload()
    payload["measurements"][2]["value_si"] = 0.011  # type: ignore[index]
    with pytest.raises(ValueError, match="must not expose numeric values"):
        build_reconstruction_measurement_validity_presentation(_selection(), payload)


def test_e3_supports_export_schema_uncertainty_qualification_key() -> None:
    payload = {
        "observation_id": "export-schema-observation",
        "measurements": [
            {
                "kind": "width",
                "status": "available",
                "value_si": 0.01,
                "unit_si": "m",
                "method": "test",
                "expanded_uncertainty_si": 0.001,
                "lower_bound_si": 0.009,
                "upper_bound_si": 0.011,
                "reliability": "medium",
                "uncertainty_display_qualification": "Estimated Uncertainty — Not Calibrated",
            }
        ],
    }
    presentation = build_reconstruction_measurement_validity_presentation(
        _selection(), payload, context=_context()
    )
    assert "Estimated Uncertainty — Not Calibrated" in presentation.uncertainty_text


def test_e3_formats_phase75_volume_unit_m3_as_litres() -> None:
    entry = ReconstructionMeasurementValidityEntry(
        kind="volume",
        status="available",
        unit_si="m3",
        method="shape_volume",
        reliability="medium",
        value_si=0.002,
        expanded_uncertainty_si=0.0001,
        lower_bound_si=0.0019,
        upper_bound_si=0.0021,
    )
    assert entry.qualified_value_text == "Volume: 2.00 L ± 0.10 L [1.90 L–2.10 L]"



def test_e3_accepts_real_phase75_inspection_bundle_summary_context(tmp_path) -> None:
    summary, _arrays = load_inspection_bundle(write_inspection_bundle(tmp_path / "bundle"))
    presentation = build_reconstruction_measurement_validity_presentation(
        _selection(),
        summary.objects[0],
        source="phase7_5.inspection_bundle",
        context=summary,
    )
    assert presentation.source_observation_id == "object-0"
    assert presentation.context.depth_scale == "metric"
    assert presentation.context.intrinsics_source == "model_predicted"
    assert presentation.context.calibration_status == "not_calibrated"
    assert "Width: 400.0 mm ± 10.0 mm" in presentation.qualified_values_text
    assert presentation.reliability_text == "Width: High 90%"
    # Explicit binding means the source object's class does not need to equal
    # the persistent reconstruction class; E3 never performs class matching.
    assert presentation.target_label == "gear • Rotating Component"

def test_e3_context_rejects_unknown_confidence_semantics() -> None:
    with pytest.raises(ValueError, match="confidence kind"):
        ReconstructionMeasurementValidityContext(confidence_kind="probability_magic")


def test_e3_empty_state_distinguishes_no_target_no_e2_and_no_e3() -> None:
    no_target = ReconstructionInspectionSelection(scope=ReconstructionInspectionSelectionScope.NONE)
    assert empty_reconstruction_measurement_validity_text(no_target)[1] == "Waiting for reconstruction target"
    assert empty_reconstruction_measurement_validity_text(_selection(), has_e2_evidence=False)[1] == "Measurement evidence required"
    assert empty_reconstruction_measurement_validity_text(_selection(), has_e2_evidence=True)[1] == "Validity evidence not attached"
