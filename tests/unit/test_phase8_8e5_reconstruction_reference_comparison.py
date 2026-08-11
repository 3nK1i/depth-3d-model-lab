from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.reconstruction_inspection_selection import (
    ReconstructionInspectionSelection,
    ReconstructionInspectionSelectionScope,
)
from app.reconstruction_reference_comparison import (
    build_reconstruction_reference_comparison_presentation,
    empty_reference_comparison_text,
    load_reference_descriptor_json,
    reference_descriptor_from_payload,
)

SESSION = "55555555-5555-4555-8555-555555555551"


def _selection(object_id: str = "gear-01") -> ReconstructionInspectionSelection:
    if object_id:
        return ReconstructionInspectionSelection(
            scope=ReconstructionInspectionSelectionScope.OBJECT,
            scene_id="scene-e5",
            session_id=SESSION,
            scene_source="phase8_7b.semantic_geometry",
            scene_availability="partial",
            object_id=object_id,
            class_name="gear",
            semantic_role="rotating_component",
            semantic_source="phase8_7b.semantic_geometry",
            metric_geometry_status="arbitrary",
            certified_metrology=False,
        )
    return ReconstructionInspectionSelection(
        scope=ReconstructionInspectionSelectionScope.SCENE,
        scene_id="scene-e5",
        session_id=SESSION,
        scene_source="phase8_7b.semantic_geometry",
        scene_availability="partial",
        metric_geometry_status="arbitrary",
        certified_metrology=False,
    )


def _reference(*, reference_id: str = "golden-assembly-a", metric: bool = False) -> dict[str, object]:
    return {
        "schema_version": 1,
        "reference_id": reference_id,
        "source": "golden_reconstruction",
        "scene_profile": "object_scan",
        "reference_frame_id": "reference-frame",
        "metric_geometry_status": "calibrated_metric" if metric else "arbitrary",
        "unit_to_metre": 0.001 if metric else None,
        "relative_scale_uncertainty": 0.01 if metric else None,
        "certified_metrology": False,
        "features": [
            {
                "feature_id": "housing-ref",
                "association_key": "housing-key",
                "registration_anchor": True,
            },
            {
                "feature_id": "gear-ref",
                "association_key": "gear-key",
                "registration_anchor": False,
            },
        ],
    }


def _report(*, reference_id: str = "golden-assembly-a", metric: bool = False) -> dict[str, object]:
    return {
        "schema_version": 1,
        "session_id": SESSION,
        "reference_id": reference_id,
        "scene_profile": "object_scan",
        "pose_convention": "T_reconstruction_from_camera",
        "candidate_frame_id": "candidate-frame",
        "reference_frame_id": "reference-frame",
        "candidate_metric_geometry_status": "calibrated_metric" if metric else "arbitrary",
        "reference_metric_geometry_status": "calibrated_metric" if metric else "arbitrary",
        "comparison_metric_available": metric,
        "reference_unit_to_metre": 0.001 if metric else None,
        "relative_scale_uncertainty": 0.02 if metric else None,
        "registration": {
            "status": "success",
            "method": "multi_anchor_similarity",
            "scale_candidate_to_reference": 1.0,
            "anchor_feature_ids": ["housing-ref"],
            "rms_residual_reference_units": 0.04,
            "maximum_residual_reference_units": 0.08,
            "rms_residual_m": 0.00004 if metric else None,
            "maximum_residual_m": 0.00008 if metric else None,
            "normalized_rms_fraction": 0.02,
            "confidence": 0.94,
            "reason": "registration residual accepted",
        },
        "feature_results": [
            {
                "reference_feature_id": "gear-ref",
                "candidate_object_id": "gear-01",
                "semantic_role": "rotating_component",
                "decision": "warn",
                "findings": ["position_deviation"],
                "position_error_reference_units": 0.12,
                "position_error_m": 0.00012 if metric else None,
                "orientation_error_degrees": 3.5,
                "extent_relative_error": 0.06,
                "geometry_quality": 0.88,
                "uncertainty_reference_units": 0.03,
                "uncertainty_m": 0.00003 if metric else None,
            }
        ],
        "extra_candidate_object_ids": [],
        "decision": "warn",
        "statistics": {
            "reference_feature_count": 2,
            "matched_feature_count": 2,
            "missing_required_count": 0,
            "extra_candidate_count": 0,
            "pass_count": 1,
            "warn_count": 1,
            "reject_count": 0,
            "insufficient_evidence_count": 0,
        },
        "certified_metrology": False,
        "inspection_qc_enabled": True,
    }


def test_e5_reference_descriptor_preserves_reference_identity_and_scale_semantics() -> None:
    descriptor = reference_descriptor_from_payload(_reference(), provenance="fixture.json")
    assert descriptor.reference_id == "golden-assembly-a"
    assert descriptor.source == "golden_reconstruction"
    assert descriptor.feature_count == 2
    assert descriptor.registration_anchor_count == 1
    assert descriptor.metric_geometry_status == "arbitrary"
    assert descriptor.unit_to_metre is None
    assert "Arbitrary reference gauge" in descriptor.scale_text
    assert descriptor.certified_metrology is False


def test_e5_reference_json_loader_reads_phase87c_reference_artifact(tmp_path: Path) -> None:
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(_reference()), encoding="utf-8")
    descriptor = load_reference_descriptor_json(path)
    assert descriptor.reference_id == "golden-assembly-a"
    assert descriptor.provenance == str(path.resolve())


def test_e5_reference_without_report_is_explicitly_awaiting_source_comparison() -> None:
    presentation = build_reconstruction_reference_comparison_presentation(
        _selection(), _reference(), None
    )
    assert presentation.report_attached is False
    assert presentation.comparison_available is False
    assert presentation.state_text == "Reference attached • comparison report required"
    assert presentation.registration_text == "—"
    assert presentation.deviations_text == "—"
    assert "does not register/apply tolerances" in presentation.authority_text


def test_e5_pairs_exact_reference_and_object_with_existing_report_deviations() -> None:
    presentation = build_reconstruction_reference_comparison_presentation(
        _selection(), _reference(), _report(), report_source="phase8_7c.inspection_report"
    )
    assert presentation.report_attached is True
    assert presentation.report_reference_match is True
    assert presentation.comparison_available is True
    assert presentation.target_result_available is True
    assert presentation.matched_reference_feature_id == "gear-ref"
    assert presentation.registration_status == "success"
    assert presentation.registration_method == "multi_anchor_similarity"
    assert presentation.registration_text.startswith("Success • 94% • Multi anchor similarity")
    assert "Position 0.12 reference units" in presentation.deviations_text
    assert "Orientation 3.50°" in presentation.deviations_text
    assert "Extent 6.00%" in presentation.deviations_text
    assert "Reference feature gear-ref" in presentation.evidence_text
    assert "geometry quality 88.0%" in presentation.evidence_text
    assert presentation.scale_text.startswith("Relative-only comparison")
    assert presentation.certified_metrology is False


def test_e5_reference_mismatch_never_reuses_deviations_from_other_reference() -> None:
    presentation = build_reconstruction_reference_comparison_presentation(
        _selection(), _reference(reference_id="reference-b"), _report(reference_id="reference-a")
    )
    assert presentation.report_attached is True
    assert presentation.report_reference_match is False
    assert presentation.comparison_available is False
    assert presentation.state_text == "Reference/report mismatch • comparison unavailable"
    assert presentation.registration_text == "—"
    assert presentation.deviations_text == "—"


def test_e5_object_without_exact_source_result_never_inherits_scene_deviation() -> None:
    presentation = build_reconstruction_reference_comparison_presentation(
        _selection("shaft-99"), _reference(), _report()
    )
    assert presentation.comparison_available is True
    assert presentation.target_result_available is False
    assert presentation.state_text == "Matching scene comparison • selected object not evaluated"
    assert presentation.deviations_text == "No exact selected-object deviation result"


def test_e5_scene_scope_uses_registration_residual_not_object_result() -> None:
    presentation = build_reconstruction_reference_comparison_presentation(
        _selection(""), _reference(), _report()
    )
    assert presentation.target_scope == "scene"
    assert presentation.target_result_available is False
    assert "Registration RMS 0.04 reference units" in presentation.deviations_text
    assert "normalized 2.00%" in presentation.deviations_text


def test_e5_metric_comparison_shows_physical_deviation_without_certification() -> None:
    presentation = build_reconstruction_reference_comparison_presentation(
        _selection(), _reference(metric=True), _report(metric=True)
    )
    assert presentation.comparison_metric_available is True
    assert presentation.scale_text.startswith("Metric-compatible comparison")
    assert "Position 0.120 mm" in presentation.deviations_text
    assert "uncertainty 0.030 mm" in presentation.evidence_text
    assert presentation.certified_metrology is False
    assert "not certified metrology" in presentation.authority_text


def test_e5_rejects_session_mismatch_certified_claim_and_invalid_reference() -> None:
    report = _report()
    report["session_id"] = "other-session"
    with pytest.raises(ValueError, match="session does not match"):
        build_reconstruction_reference_comparison_presentation(_selection(), _reference(), report)

    certified = _reference()
    certified["certified_metrology"] = True
    with pytest.raises(ValueError, match="certified metrology"):
        reference_descriptor_from_payload(certified)

    no_anchor = _reference()
    no_anchor["features"] = [
        {"feature_id": "one", "association_key": "one", "registration_anchor": False}
    ]
    with pytest.raises(ValueError, match="registration anchor"):
        reference_descriptor_from_payload(no_anchor)


def test_e5_matching_reference_metric_status_must_agree_with_report() -> None:
    report = _report(metric=False)
    report["reference_metric_geometry_status"] = "calibrated_metric"
    with pytest.raises(ValueError, match="disagree"):
        build_reconstruction_reference_comparison_presentation(_selection(), _reference(), report)


def test_e5_empty_state_never_manufactures_reference_or_deviation() -> None:
    none = ReconstructionInspectionSelection(scope=ReconstructionInspectionSelectionScope.NONE)
    rows = empty_reference_comparison_text(none)
    assert rows[0] == "No reconstruction target"
    assert rows[1] == "No reference attached"
    assert rows[4] == "—"
    assert rows[6] == "Waiting for reconstruction target"

    rows = empty_reference_comparison_text(_selection())
    assert rows[6] == "No reference attached"

def test_e5_reference_can_be_attached_before_scene_scale_is_known() -> None:
    selection = ReconstructionInspectionSelection(
        scope=ReconstructionInspectionSelectionScope.SCENE,
        scene_id="scene-unknown-scale",
        session_id=SESSION,
        scene_source="persistent_scene",
        scene_availability="partial",
        metric_geometry_status="unknown",
        certified_metrology=False,
    )
    presentation = build_reconstruction_reference_comparison_presentation(
        selection, _reference(), None
    )
    assert presentation.comparison_available is False
    assert presentation.state_text == "Reference attached • comparison report required"
