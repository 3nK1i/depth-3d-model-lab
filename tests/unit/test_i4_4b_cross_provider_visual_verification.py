from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_cross_provider_visual_verification import (
    I44B_PHASE,
    VisualVerificationEvidenceBinding,
    build_i4_4b_cross_provider_visual_verification,
    validate_g2_visual_closure,
    validate_g3_visual_closure,
    validate_g4_visual_closure,
    validate_i4_4a_result_for_i4_4b,
    validate_i4_4b_visual_receipt,
)
from tests.i4_4b_helpers import clone, passing_g2, passing_g3, passing_g4, passing_h2, passing_i4_4a, passing_receipt


def binding(path: str, char: str) -> VisualVerificationEvidenceBinding:
    return VisualVerificationEvidenceBinding(path=path, sha256=char * 64)


def build(**kwargs) -> dict:
    args = {
        "i4_3h2_closure": passing_h2(),
        "i4_3h2_binding": binding("/evidence/h2.json", "1"),
        "i4_4a_result": passing_i4_4a(),
        "i4_4a_binding": binding("/evidence/i44a.json", "2"),
        "i4_3g2_closure": passing_g2(),
        "i4_3g2_binding": binding("/evidence/g2.json", "3"),
        "i4_3g3c_closure": passing_g3(),
        "i4_3g3c_binding": binding("/evidence/g3.json", "4"),
        "i4_3g4c_closure": passing_g4(),
        "i4_3g4c_binding": binding("/evidence/g4.json", "5"),
        "visual_receipt": passing_receipt(),
        "visual_receipt_binding": binding("/evidence/receipt.json", "6"),
    }
    args.update(kwargs)
    return build_i4_4b_cross_provider_visual_verification(**args)


def test_i44b_builds_passing_cross_provider_visual_verification() -> None:
    result = build()
    assert result["phase"] == I44B_PHASE == "I4.4B"
    assert result["status"] == "visual_verification_complete"
    assert result["pass"] is True
    assert result["decision"]["i4_4b_complete"] is True
    assert result["decision"]["i4_4c_authorized"] is True
    assert result["decision"]["i4_4c_started"] is False


def test_i44b_scope_is_evidence_only_and_does_not_start_later_boundaries() -> None:
    assert build()["scope"] == {
        "evidence_only": True,
        "cross_provider_visual_verification_only": True,
        "provider_execution_started": False,
        "provider_runtime_probe_started": False,
        "sfm_rerun": False,
        "matched_view_rerun": False,
        "manual_review_rerun": False,
        "provenance_authority_final_audit_started": False,
        "final_i4_closure_started": False,
    }


def test_i44b_coverage_and_visual_family_inventory_are_complete() -> None:
    result = build()
    assert result["coverage"] == {
        "qualification_family_count": 3,
        "scenario_count": 11,
        "provider_implementation_count": 10,
        "fresh_visual_review_scenario_count": 10,
        "historical_visual_review_scenario_count": 1,
        "review_dimension_total": 21,
        "all_visual_review_families_accounted": True,
    }
    assert tuple(result["family_visual_verification"]) == ("mvs_da3", "temporal_depth", "single_image_depth")


def test_i44b_preserves_terminal_quality_events_without_scoring_or_ranking() -> None:
    summary = build()["cross_family_visual_verification"]
    assert len(summary["terminal_quality_events"]) == 5
    assert summary["quality_event_counts_are_not_provider_scores"] is True
    assert summary["manual_preferences_aggregated_into_score"] is False
    assert summary["direct_cross_family_provider_ranking_valid"] is False
    assert summary["universal_provider_winner_declared"] is False


def test_i44b_validators_accept_passing_family_closures() -> None:
    assert validate_g2_visual_closure(passing_g2())["manual_review_dimension_count"] == 8
    assert validate_g3_visual_closure(passing_g3())["manual_review_dimension_count"] == 6
    assert validate_g4_visual_closure(passing_g4())["manual_review_dimension_count"] == 7


@pytest.mark.parametrize("field", ["i4_4a_complete", "i4_4b_authorized"])
def test_i44b_rejects_i44a_decision_drift(field: str) -> None:
    i44a = clone(passing_i4_4a())
    i44a["decision"][field] = False
    with pytest.raises(ReconstructionValidationError):
        validate_i4_4a_result_for_i4_4b(i44a)


@pytest.mark.parametrize("field", ["provider_execution_started", "visual_review_rerun", "manual_review_rerun"])
def test_i44b_rejects_visual_receipt_that_started_forbidden_execution(field: str) -> None:
    receipt = clone(passing_receipt())
    receipt["scope"][field] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i4_4b_visual_receipt(receipt)


def test_i44b_rejects_g3_visual_closure_if_production_rejection_overridden() -> None:
    g3 = clone(passing_g3())
    g3["scenario_synthesis"]["g3_temporal_motion"]["production_rejection_overridden"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_g3_visual_closure(g3)


def test_i44b_rejects_g4_visual_closure_if_provider_ranking_created() -> None:
    g4 = clone(passing_g4())
    g4["scenario_synthesis"]["g4_single_image_complete_room"]["provider_ranking_created"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_g4_visual_closure(g4)


def test_i44b_rejects_g2_visual_closure_if_historical_house_was_reexecuted() -> None:
    g2 = clone(passing_g2())
    g2["scenario_synthesis"]["g2_house_mvs_da3_historical"]["provider_reexecution_performed"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_g2_visual_closure(g2)


def test_i44b_rejects_terminal_event_drift_between_visual_closures_and_h2() -> None:
    g3 = clone(passing_g3())
    g3["scenario_synthesis"]["g3_temporal_motion"]["production_statuses"]["video_depth_anything"] = "pass"
    with pytest.raises(ReconstructionValidationError):
        build(i4_3g3c_closure=g3)


def test_i44b_does_not_mutate_frozen_inputs() -> None:
    h2 = passing_h2(); i44a = passing_i4_4a(); g2 = passing_g2(); g3 = passing_g3(); g4 = passing_g4(); receipt = passing_receipt()
    before = [clone(item) for item in (h2, i44a, g2, g3, g4, receipt)]
    build(i4_3h2_closure=h2, i4_4a_result=i44a, i4_3g2_closure=g2, i4_3g3c_closure=g3, i4_3g4c_closure=g4, visual_receipt=receipt)
    after = [h2, i44a, g2, g3, g4, receipt]
    assert after == before
