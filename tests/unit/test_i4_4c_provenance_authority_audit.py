from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_provenance_authority_audit import (
    I44C_PHASE,
    ProvenanceAuthorityEvidenceBinding,
    audit_current_offline_dense_authority_contracts,
    build_i4_4c_provenance_authority_audit,
    validate_i4_4b_result_for_i4_4c,
    validate_i4_4c_audit_receipt,
)
from tests.i4_4a_helpers import passing_h2
from tests.i4_4b_helpers import passing_i4_4a
from tests.i4_4c_helpers import clone, passing_i4_4b, passing_receipt


def binding(path: str, char: str) -> ProvenanceAuthorityEvidenceBinding:
    return ProvenanceAuthorityEvidenceBinding(path=path, sha256=char * 64)


def build(**kwargs) -> dict:
    args = {
        "i4_3h2_closure": passing_h2(),
        "i4_3h2_binding": binding("/evidence/h2.json", "1"),
        "i4_4a_result": passing_i4_4a(),
        "i4_4a_binding": binding("/evidence/i44a.json", "2"),
        "i4_4b_result": passing_i4_4b(),
        "i4_4b_binding": binding("/evidence/i44b.json", "3"),
        "audit_receipt": passing_receipt(),
        "audit_receipt_binding": binding("/evidence/receipt.json", "4"),
        "h2_h1_chain_reverified": True,
        "h1_bound_artifacts_reverified": True,
        "g5b_predecessor_artifacts_reverified": True,
        "i4_4a_chain_reverified": True,
        "i4_4b_chain_reverified": True,
        "current_repository_authority_matches_frozen_h1": True,
    }
    args.update(kwargs)
    return build_i4_4c_provenance_authority_audit(**args)


def test_i44c_builds_passing_provenance_authority_audit_and_authorizes_only_d() -> None:
    result = build()
    assert result["phase"] == I44C_PHASE == "I4.4C"
    assert result["status"] == "provenance_authority_audit_complete"
    assert result["pass"] is True
    assert result["decision"]["i4_4c_complete"] is True
    assert result["decision"]["i4_4d_authorized"] is True
    assert result["decision"]["i4_4d_started"] is False


def test_i44c_scope_is_execution_free_and_does_not_close_i4() -> None:
    assert build()["scope"] == {
        "evidence_and_repository_contract_audit": True,
        "provider_execution_started": False,
        "provider_runtime_probe_started": False,
        "sfm_rerun": False,
        "matched_view_rerun": False,
        "manual_review_rerun": False,
        "new_provider_comparison_started": False,
        "final_i4_closure_started": False,
    }


def test_i44c_current_authority_audit_preserves_pycolmap_and_arbitrary_scale() -> None:
    audit = audit_current_offline_dense_authority_contracts()
    assert audit["provider_architecture"]["provider_family_count"] == 5
    assert audit["provider_architecture"]["executable_provider_family_count"] == 4
    assert audit["provider_architecture"]["provider_implementation_count"] == 10
    assert audit["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert audit["default_reconstruction_scale_status"] == "arbitrary"
    assert audit["permanent_global_view_cap"] is None
    assert audit["dense_provider_metric_scale_claimed"] is False
    assert audit["dense_provider_measurement_authoritative"] is False
    assert audit["dense_provider_certified_metrology"] is False
    assert audit["dense_provider_surface_mesh_authority"] is False


def test_i44c_provenance_audit_is_explicitly_not_a_recursive_raw_artifact_rehash() -> None:
    provenance = build()["provenance_chain_audit"]
    assert provenance["formal_i4_3_evidence_graph_rehashed"] is True
    assert provenance["i4_4_predecessor_chain_rehashed"] is True
    assert provenance["raw_nested_provider_artifact_tree_recursively_rehashed_by_i4_4c"] is False
    assert provenance["no_stronger_raw_artifact_claim_created"] is True


def test_i44c_preserves_five_terminal_quality_events_without_scoring() -> None:
    guard = build()["qualification_guard"]
    assert guard["terminal_quality_event_count"] == 5
    assert guard["quality_event_counts_are_not_provider_scores"] is True
    assert guard["provider_ranking_created"] is False
    assert guard["universal_provider_winner_declared"] is False
    assert guard["global_scalar_score_created"] is False


def test_i44c_validates_passing_i44b_predecessor() -> None:
    validated = validate_i4_4b_result_for_i4_4c(passing_i4_4b())
    assert validated["coverage"]["scenario_count"] == 11
    assert len(validated["terminal_quality_events"]) == 5


@pytest.mark.parametrize("field", ["i4_4b_complete", "i4_4c_authorized"])
def test_i44c_rejects_i44b_decision_drift(field: str) -> None:
    i44b = clone(passing_i4_4b())
    i44b["decision"][field] = False
    with pytest.raises(ReconstructionValidationError):
        validate_i4_4b_result_for_i4_4c(i44b)


@pytest.mark.parametrize("field", ["metric_scale_claimed", "measurement_authoritative", "certified_metrology", "hybrid_authorized"])
def test_i44c_rejects_i44b_authority_promotion(field: str) -> None:
    i44b = clone(passing_i4_4b())
    i44b["authority"][field] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i4_4b_result_for_i4_4c(i44b)


@pytest.mark.parametrize("gate", [
    "h2_h1_chain_reverified",
    "h1_bound_artifacts_reverified",
    "g5b_predecessor_artifacts_reverified",
    "i4_4a_chain_reverified",
    "i4_4b_chain_reverified",
    "current_repository_authority_matches_frozen_h1",
])
def test_i44c_fails_closed_when_any_provenance_authority_gate_is_false(gate: str) -> None:
    with pytest.raises(ReconstructionValidationError):
        build(**{gate: False})


def test_i44c_rejects_too_narrow_audit_receipt() -> None:
    receipt = clone(passing_receipt())
    receipt["counts"]["affected_total_tests"] = 50
    with pytest.raises(ReconstructionValidationError):
        validate_i4_4c_audit_receipt(receipt)


def test_i44c_rejects_receipt_that_started_provider_execution() -> None:
    receipt = clone(passing_receipt())
    receipt["scope"]["provider_execution_started"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i4_4c_audit_receipt(receipt)


def test_i44c_does_not_mutate_frozen_inputs() -> None:
    h2 = passing_h2(); i44a = passing_i4_4a(); i44b = passing_i4_4b(); receipt = passing_receipt()
    before = [clone(item) for item in (h2, i44a, i44b, receipt)]
    build(i4_3h2_closure=h2, i4_4a_result=i44a, i4_4b_result=i44b, audit_receipt=receipt)
    assert [h2, i44a, i44b, receipt] == before
