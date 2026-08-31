from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i4_3_final_closure import (
    H2EvidenceBinding,
    build_i4_3h2_final_closure,
    validate_i4_3h1_qualification_audit,
)
from tests.i4_3h2_helpers import clone, passing_h1


def binding() -> H2EvidenceBinding:
    return H2EvidenceBinding(path="/evidence/h1.json", sha256="c" * 64)


def build(**kwargs) -> dict:
    args = {
        "i4_3h1_audit": passing_h1(),
        "evidence_binding": binding(),
        "h1_bound_artifacts_reverified": True,
        "g5b_predecessor_artifacts_reverified": True,
        "current_architecture_matches_h1": True,
    }
    args.update(kwargs)
    return build_i4_3h2_final_closure(**args)


def test_h2_accepts_passing_h1_and_formally_closes_i43() -> None:
    closure = build()
    assert closure["phase"] == "I4.3H2"
    assert closure["status"] == "formally_closed"
    assert closure["pass"] is True
    assert closure["decision"]["i4_3_closed"] is True
    assert closure["decision"]["i4_4_authorized"] is True


def test_h2_scope_is_strictly_evidence_only_and_does_not_start_i44() -> None:
    scope = build()["scope"]
    assert scope == {
        "evidence_only": True,
        "provider_execution_started": False,
        "provider_runtime_probe_started": False,
        "sfm_rerun": False,
        "matched_view_rerun": False,
        "manual_review_rerun": False,
        "formal_i4_3_closure_performed": True,
        "i4_4_work_started": False,
    }


def test_h2_preserves_exact_i43_architecture_summary() -> None:
    coverage = build()["coverage"]
    assert coverage["provider_family_count"] == 5
    assert coverage["executable_provider_family_count"] == 4
    assert coverage["provider_implementation_count"] == 10
    assert coverage["qualification_family_count"] == 3
    assert coverage["qualification_scenario_count"] == 11
    assert coverage["terminal_quality_event_count"] == 5


def test_h2_preserves_full_h1_regression_evidence_without_rerunning_backends() -> None:
    regression = build()["regression_summary"]
    assert regression["predecessor_total_tests"] == 668
    assert regression["focused_h1_tests"] == 40
    assert regression["all_required_suites_passed"] is True
    assert regression["h2_does_not_reinterpret_test_counts_as_runtime_quality"] is True


def test_h2_closure_chain_requires_h1_g5b_and_nested_predecessor_reverification() -> None:
    chain = build()["closure_chain"]
    assert all(chain.values())


@pytest.mark.parametrize(
    "field",
    [
        "provider_selected",
        "provider_promoted",
        "provider_ranking_created",
        "universal_provider_winner_declared",
        "global_scalar_score_created",
        "auto_policy_implemented",
        "hybrid_authorized",
    ],
)
def test_h2_keeps_forbidden_decisions_false(field: str) -> None:
    assert build()["decision"][field] is False


@pytest.mark.parametrize(
    "field",
    [
        "authoritative_sfm_modified",
        "provider_selected",
        "provider_promoted",
        "auto_policy_implemented",
        "hybrid_authorized",
        "metric_scale_claimed",
        "measurement_authoritative",
        "certified_metrology",
        "surface_mesh_generated",
    ],
)
def test_h2_keeps_authority_promotions_false(field: str) -> None:
    assert build()["authority"][field] is False


def test_h2_preserves_pycolmap_and_no_permanent_global_view_cap() -> None:
    authority = build()["authority"]
    assert authority["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert authority["permanent_global_view_cap"] is None


def test_h2_next_boundary_is_i44a_focused_regression() -> None:
    assert build()["next"] == "I4.4A focused regression."


def test_h2_rejects_h1_that_is_not_qualified() -> None:
    h1 = clone(passing_h1())
    h1["pass"] = False
    with pytest.raises(ReconstructionValidationError):
        validate_i4_3h1_qualification_audit(h1)


def test_h2_rejects_h1_if_h2_was_not_authorized() -> None:
    h1 = clone(passing_h1())
    h1["decision"]["i4_3h2_authorized"] = False
    with pytest.raises(ReconstructionValidationError):
        validate_i4_3h1_qualification_audit(h1)


def test_h2_rejects_h1_if_i43_was_prematurely_closed() -> None:
    h1 = clone(passing_h1())
    h1["decision"]["i4_3_closed"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i4_3h1_qualification_audit(h1)


def test_h2_rejects_h1_if_i44_was_prematurely_authorized() -> None:
    h1 = clone(passing_h1())
    h1["decision"]["i4_4_authorized"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i4_3h1_qualification_audit(h1)


def test_h2_rejects_h1_quality_event_drift() -> None:
    h1 = clone(passing_h1())
    h1["qualification_audit"]["terminal_quality_events"].pop()
    with pytest.raises(ReconstructionValidationError):
        validate_i4_3h1_qualification_audit(h1)


def test_h2_rejects_h1_regression_coverage_drift() -> None:
    h1 = clone(passing_h1())
    h1["regression_audit"]["predecessor_total_tests"] = 667
    with pytest.raises(ReconstructionValidationError):
        validate_i4_3h1_qualification_audit(h1)


def test_h2_rejects_h1_authority_drift() -> None:
    h1 = clone(passing_h1())
    h1["authority"]["metric_scale_claimed"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i4_3h1_qualification_audit(h1)


def test_h2_requires_reverification_of_h1_bound_artifacts() -> None:
    with pytest.raises(ReconstructionValidationError):
        build(h1_bound_artifacts_reverified=False)


def test_h2_requires_reverification_of_g5b_predecessors() -> None:
    with pytest.raises(ReconstructionValidationError):
        build(g5b_predecessor_artifacts_reverified=False)


def test_h2_requires_current_architecture_to_match_frozen_h1() -> None:
    with pytest.raises(ReconstructionValidationError):
        build(current_architecture_matches_h1=False)


def test_h2_requires_valid_h1_sha_binding() -> None:
    with pytest.raises(ReconstructionValidationError):
        build(evidence_binding=H2EvidenceBinding(path="", sha256="bad"))


def test_h2_does_not_mutate_h1_input() -> None:
    h1 = passing_h1()
    before = clone(h1)
    build(i4_3h1_audit=h1)
    assert h1 == before
