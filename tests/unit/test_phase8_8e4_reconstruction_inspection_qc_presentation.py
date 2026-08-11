from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.reconstruction_inspection_qc_presentation import (
    build_reconstruction_inspection_qc_presentation,
    empty_reconstruction_inspection_qc_text,
)
from app.reconstruction_inspection_selection import (
    ReconstructionInspectionSelection,
    ReconstructionInspectionSelectionScope,
)


SESSION = "88888888-8888-4888-8888-888888888881"


def _selection(*, object_id: str = "gear-01") -> ReconstructionInspectionSelection:
    return ReconstructionInspectionSelection(
        scope=ReconstructionInspectionSelectionScope.OBJECT if object_id else ReconstructionInspectionSelectionScope.SCENE,
        scene_id="scene-e4",
        session_id=SESSION,
        scene_source="semantic-test",
        scene_availability="partial",
        canonical_geometry_authoritative=False,
        object_id=object_id,
        class_name="gear" if object_id else "",
        semantic_role="rotating_component" if object_id else "",
        semantic_source="phase8_7b.semantic_geometry" if object_id else "",
        metric_geometry_status="arbitrary",
        certified_metrology=False,
    )


def _report(*, session_id: str = SESSION, selected_id: str = "gear-01", decision: str = "reject") -> dict[str, object]:
    return {
        "schema_version": 1,
        "session_id": session_id,
        "reference_id": "golden-assembly-a",
        "candidate_metric_geometry_status": "arbitrary",
        "reference_metric_geometry_status": "arbitrary",
        "comparison_metric_available": False,
        "registration": {
            "status": "success",
            "method": "multi_anchor_similarity",
            "confidence": 0.93,
            "reason": "registration residual accepted",
        },
        "feature_results": [
            {
                "reference_feature_id": "feature-gear",
                "candidate_object_id": selected_id,
                "semantic_role": "rotating_component",
                "decision": decision,
                "findings": ["position_deviation", "orientation_deviation"],
            },
            {
                "reference_feature_id": "feature-shaft",
                "candidate_object_id": "shaft-01",
                "semantic_role": "supporting_component",
                "decision": "pass",
                "findings": ["within_tolerance"],
            },
        ],
        "decision": decision,
        "statistics": {
            "reference_feature_count": 2,
            "matched_feature_count": 2,
            "missing_required_count": 0,
            "extra_candidate_count": 0,
            "pass_count": 1,
            "warn_count": 0,
            "reject_count": 1 if decision == "reject" else 0,
            "insufficient_evidence_count": 0,
        },
        "certified_metrology": False,
        "inspection_qc_enabled": True,
    }


def test_e4_presents_exact_selected_object_result_without_recomputing() -> None:
    presentation = build_reconstruction_inspection_qc_presentation(
        _selection(), _report(), source="phase8_7c.inspection_report"
    )
    assert presentation.target_object_id == "gear-01"
    assert presentation.target_result_available is True
    assert presentation.target_decision == "reject"
    assert presentation.decision_text == "REJECT • Object"
    assert "Position deviation" in presentation.findings_text
    assert "Orientation deviation" in presentation.findings_text
    assert presentation.coverage_text == "2/2 matched • 0 missing • 0 extra"
    assert "1 pass" in presentation.decisions_text
    assert "1 reject" in presentation.decisions_text
    assert presentation.registration_text.startswith("Success • 93%")
    assert presentation.comparison_text.startswith("Relative-only comparison")
    assert "reference workflow deferred to E5" in presentation.authority_text
    assert presentation.certified_metrology is False


def test_e4_scene_scope_uses_only_source_overall_decision() -> None:
    presentation = build_reconstruction_inspection_qc_presentation(_selection(object_id=""), _report())
    assert presentation.target_object_id == ""
    assert presentation.target_result_available is False
    assert presentation.decision_text == "REJECT • Scene"
    assert presentation.findings_text == "See source report statistics"


def test_e4_does_not_guess_qc_for_unmatched_selected_object() -> None:
    selection = ReconstructionInspectionSelection(
        scope=ReconstructionInspectionSelectionScope.OBJECT,
        scene_id="scene-e4",
        session_id=SESSION,
        scene_source="semantic-test",
        scene_availability="partial",
        object_id="unrelated-object-99",
        class_name="gear",
        semantic_role="rotating_component",
        semantic_source="phase8_7b.semantic_geometry",
        metric_geometry_status="arbitrary",
        certified_metrology=False,
    )
    presentation = build_reconstruction_inspection_qc_presentation(selection, _report())
    assert presentation.target_result_available is False
    assert presentation.target_decision == ""
    assert presentation.decision_text == "NOT EVALUATED • Object"
    assert presentation.findings_text == "No selected-object result in attached report"


def test_e4_rejects_session_mismatch_duplicate_object_result_and_certified_claim() -> None:
    with pytest.raises(ValueError, match="session does not match"):
        build_reconstruction_inspection_qc_presentation(_selection(), _report(session_id="other-session"))

    duplicate = _report()
    duplicate["feature_results"] = list(duplicate["feature_results"]) + [dict(duplicate["feature_results"][0])]
    with pytest.raises(ValueError, match="duplicate results"):
        build_reconstruction_inspection_qc_presentation(_selection(), duplicate)

    certified = _report()
    certified["certified_metrology"] = True
    with pytest.raises(ValueError, match="certified metrology"):
        build_reconstruction_inspection_qc_presentation(_selection(), certified)


def test_e4_accepts_report_object_with_as_dict_without_core_import() -> None:
    @dataclass
    class Report:
        payload: dict[str, object]

        def as_dict(self):
            return self.payload

    presentation = build_reconstruction_inspection_qc_presentation(_selection(), Report(_report()))
    assert presentation.overall_decision == "reject"


def test_e4_empty_state_never_manufactures_pass_fail() -> None:
    no_scene = ReconstructionInspectionSelection(scope=ReconstructionInspectionSelectionScope.NONE)
    rows = empty_reconstruction_inspection_qc_text(no_scene)
    assert rows[1] == "Not evaluated"
    assert "does not run inspection" in rows[-1]

    rows = empty_reconstruction_inspection_qc_text(_selection())
    assert rows[1] == "Not evaluated"
    assert "No attached inspection/QC result" in rows[2]
