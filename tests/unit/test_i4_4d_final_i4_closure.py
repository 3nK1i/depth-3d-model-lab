from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_final_i4_closure import (
    I44D_PHASE,
    FinalI4EvidenceBinding,
    build_i4_4d_final_i4_closure,
    validate_i4_4c_result_for_i4_4d,
    validate_i4_4d_closure_receipt,
)
from tests.i4_4c_helpers import build_passing_i4_4c
from tests.i4_4d_helpers import clone, passing_receipt


def binding(path: str, char: str) -> FinalI4EvidenceBinding:
    return FinalI4EvidenceBinding(path, char * 64)


def build(**kwargs) -> dict:
    args = {
        "i4_4c_result": build_passing_i4_4c(),
        "i4_4c_binding": binding("/evidence/i44c.json", "1"),
        "closure_receipt": passing_receipt(),
        "closure_receipt_binding": binding("/evidence/i44d_receipt.json", "2"),
        "i4_4c_bound_artifacts_reverified": True,
        "current_repository_authority_matches_i4_4c": True,
    }
    args.update(kwargs)
    return build_i4_4d_final_i4_closure(**args)


def test_i44d_formally_closes_i4_and_freezes_offline_dense_foundation() -> None:
    result = build()
    assert result["phase"] == I44D_PHASE == "I4.4D"
    assert result["status"] == "formally_closed"
    assert result["pass"] is True
    assert result["decision"]["i4_4d_final_closure_complete"] is True
    assert result["decision"]["i4_4_offline_dense_foundation_frozen"] is True
    assert result["decision"]["i4_closed"] is True


def test_i44d_authorizes_i5_without_starting_it() -> None:
    result = build()
    assert result["decision"]["i5_authorized"] is True
    assert result["decision"]["i5_started"] is False
    assert result["handoff_to_i5"]["first_boundary"] == "I5.1 difficult handheld captures"
    assert result["next"] == "I5.1 difficult handheld captures."


def test_i44d_scope_is_formal_closure_only_and_execution_free() -> None:
    assert build()["scope"] == {
        "evidence_only": True,
        "formal_i4_closure_performed": True,
        "offline_dense_foundation_freeze_performed": True,
        "provider_execution_started": False,
        "provider_runtime_probe_started": False,
        "sfm_rerun": False,
        "matched_view_rerun": False,
        "manual_review_rerun": False,
        "new_provider_comparison_started": False,
        "i5_work_started": False,
    }


def test_i44d_coverage_is_frozen_to_complete_i4_inventory() -> None:
    assert build()["coverage"] == {
        "provider_family_count": 5,
        "executable_provider_family_count": 4,
        "provider_implementation_count": 10,
        "qualification_family_count": 3,
        "qualification_scenario_count": 11,
        "terminal_quality_event_count": 5,
        "i4_4_boundary_count": 4,
        "i4_4_completed_boundary_count": 4,
    }


def test_i44d_preserves_pycolmap_arbitrary_scale_and_no_metric_authority() -> None:
    result = build()
    authority = result["authority"]
    assert authority["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert authority["permanent_global_view_cap"] is None
    assert result["authority_audit"]["default_reconstruction_scale_status"] == "arbitrary"
    for key in (
        "provider_selected", "provider_promoted", "auto_policy_implemented", "hybrid_authorized",
        "metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated",
    ):
        assert authority[key] is False


def test_i44d_preserves_terminal_quality_events_without_score_or_winner() -> None:
    q = build()["qualification_closure"]
    assert len(q["terminal_quality_events"]) == 5
    assert q["quality_event_counts_are_not_provider_scores"] is True
    assert q["direct_cross_family_provider_ranking_valid"] is False
    assert q["universal_provider_conclusion_drawn"] is False


def test_i44d_provenance_does_not_inflate_raw_artifact_rehash_claim() -> None:
    p = build()["provenance_closure"]
    assert p["raw_nested_provider_artifact_tree_recursively_rehashed_by_i4_4d"] is False
    assert p["no_stronger_raw_artifact_claim_created"] is True


def test_i44d_preserves_deferred_auto_and_hybrid_roadmap() -> None:
    handoff = build()["handoff_to_i5"]
    assert handoff["evidence_driven_auto_policy_deferred_to_i5_13"] is True
    assert handoff["hybrid_experiment_deferred_to_i5_12_if_justified"] is True


def test_i44d_validates_passing_i44c_predecessor() -> None:
    validated = validate_i4_4c_result_for_i4_4d(build_passing_i4_4c())
    assert validated["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert len(validated["terminal_quality_events"]) == 5


@pytest.mark.parametrize("field", ["i4_4c_complete", "i4_4d_authorized"])
def test_i44d_rejects_i44c_decision_drift(field: str) -> None:
    c = clone(build_passing_i4_4c())
    c["decision"][field] = False
    with pytest.raises(ReconstructionValidationError):
        validate_i4_4c_result_for_i4_4d(c)


@pytest.mark.parametrize("field", [
    "h2_h1_chain_reverified",
    "g5b_predecessor_artifacts_reverified",
    "i4_4a_chain_reverified",
    "i4_4b_chain_reverified",
    "formal_i4_3_evidence_graph_rehashed",
    "i4_4_predecessor_chain_rehashed",
    "current_repository_authority_matches_frozen_h1",
])
def test_i44d_rejects_i44c_provenance_gate_drift(field: str) -> None:
    c = clone(build_passing_i4_4c())
    c["provenance_chain_audit"][field] = False
    with pytest.raises(ReconstructionValidationError):
        validate_i4_4c_result_for_i4_4d(c)


@pytest.mark.parametrize("field", ["metric_scale_claimed", "measurement_authoritative", "certified_metrology", "hybrid_authorized"])
def test_i44d_rejects_i44c_authority_promotion(field: str) -> None:
    c = clone(build_passing_i4_4c())
    c["authority"][field] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i4_4c_result_for_i4_4d(c)


def test_i44d_rejects_inflated_recursive_raw_artifact_claim() -> None:
    c = clone(build_passing_i4_4c())
    c["provenance_chain_audit"]["raw_nested_provider_artifact_tree_recursively_rehashed_by_i4_4c"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i4_4c_result_for_i4_4d(c)


def test_i44d_requires_reverification_of_i44c_bound_artifacts() -> None:
    with pytest.raises(ReconstructionValidationError):
        build(i4_4c_bound_artifacts_reverified=False)


def test_i44d_requires_current_repository_authority_match() -> None:
    with pytest.raises(ReconstructionValidationError):
        build(current_repository_authority_matches_i4_4c=False)


def test_i44d_rejects_too_narrow_predecessor_regression() -> None:
    receipt = clone(passing_receipt())
    receipt["counts"]["predecessor_total_tests"] = 20
    receipt["counts"]["combined_exercised_tests"] = receipt["counts"]["focused_tests"] + 20
    with pytest.raises(ReconstructionValidationError):
        validate_i4_4d_closure_receipt(receipt)


def test_i44d_rejects_receipt_that_started_i5_work() -> None:
    receipt = clone(passing_receipt())
    receipt["scope"]["i5_work_started"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i4_4d_closure_receipt(receipt)


def test_i44d_does_not_mutate_frozen_inputs() -> None:
    c, receipt = build_passing_i4_4c(), passing_receipt()
    before_c, before_receipt = clone(c), clone(receipt)
    build(i4_4c_result=c, closure_receipt=receipt)
    assert c == before_c
    assert receipt == before_receipt
