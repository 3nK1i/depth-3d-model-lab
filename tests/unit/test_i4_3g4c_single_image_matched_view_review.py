from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.single_image_provider_matched_view_review import (
    I43G4C_BLIND_LABELS,
    I43G4C_PROVIDER_IMPLEMENTATIONS,
    I43G4C_REVIEW_DIMENSIONS,
    I43G4C_SCENARIOS,
    assignment_commitment,
    blind_assignment,
    build_controlled_reveal,
    build_i4_3g4c_final_closure,
    build_manual_review_template,
    review_is_complete,
    validate_controlled_reveal,
    verify_g1_single_image_review_requirement,
)
from tests.i4_3g4a_helpers import passing_g1_snapshot
from tests.i4_3g4c_helpers import completed_review


def test_g1_single_image_review_contract_is_exact() -> None:
    g1 = passing_g1_snapshot()
    for scenario_id in I43G4C_SCENARIOS:
        entry = verify_g1_single_image_review_requirement(g1, scenario_id=scenario_id)
        assert tuple(entry["manual_review_dimensions"]) == I43G4C_REVIEW_DIMENSIONS
        assert entry["manual_review_required"] is True


def test_six_way_blind_assignment_is_deterministic_permutation() -> None:
    seed = "a" * 64
    assignment = blind_assignment(seed, scenario_id=I43G4C_SCENARIOS[0])
    assert assignment == blind_assignment(seed, scenario_id=I43G4C_SCENARIOS[0])
    assert tuple(assignment) == I43G4C_BLIND_LABELS
    assert set(assignment.values()) == set(I43G4C_PROVIDER_IMPLEMENTATIONS)
    assert len(assignment_commitment(assignment, seed_sha256=seed, scenario_id=I43G4C_SCENARIOS[0])) == 64


def test_review_requires_all_dimension_confirmations() -> None:
    review = build_manual_review_template(
        scenario_id=I43G4C_SCENARIOS[0],
        assignment_commitment_sha256="b" * 64,
        common_keyframe_ids=("kf-0", "kf-1"),
        canonical_available_labels=I43G4C_BLIND_LABELS,
    )
    assert not review_is_complete(review)
    for dimension in I43G4C_REVIEW_DIMENSIONS:
        review["dimension_reviews"][dimension]["dimension_confirmation"] = True
    review["reviewer_confirmation"] = True
    assert review_is_complete(review)


def test_all_not_assessable_requires_note() -> None:
    review = build_manual_review_template(
        scenario_id=I43G4C_SCENARIOS[0],
        assignment_commitment_sha256="c" * 64,
        common_keyframe_ids=("kf-0",),
        canonical_available_labels=("A",),
    )
    for dimension in I43G4C_REVIEW_DIMENSIONS:
        review["dimension_reviews"][dimension]["dimension_confirmation"] = True
    first = I43G4C_REVIEW_DIMENSIONS[0]
    review["dimension_reviews"][first]["not_assessable_blind_labels"] = list(I43G4C_BLIND_LABELS)
    review["reviewer_confirmation"] = True
    assert not review_is_complete(review)
    review["dimension_reviews"][first]["notes"] = "No frozen evidence supports this dimension."
    assert review_is_complete(review)


def test_controlled_reveal_preserves_production_rejection() -> None:
    scenario = I43G4C_SCENARIOS[1]
    seed = "d" * 64
    assignment = blind_assignment(seed, scenario_id=scenario)
    commitment = assignment_commitment(assignment, seed_sha256=seed, scenario_id=scenario)
    review = completed_review(scenario_id=scenario, commitment=commitment)
    statuses = {name: "pass" for name in I43G4C_PROVIDER_IMPLEMENTATIONS}
    statuses["depth_anything_v2"] = "rejected"
    statuses["marigold"] = "rejected"
    reveal = build_controlled_reveal(
        review=review,
        production_snapshot_sha256=seed,
        production_statuses=statuses,
        prepare_summary_sha256="e" * 64,
    )
    validate_controlled_reveal(reveal)
    assert reveal["production_statuses"]["depth_anything_v2"] == "rejected"
    assert reveal["production_statuses"]["marigold"] == "rejected"
    assert reveal["decision"]["production_rejection_overridden_by_manual_review"] is False
    assert reveal["decision"]["provider_ranking_created"] is False


def _reveal_for(scenario_id: str, seed_char: str) -> dict:
    seed = seed_char * 64
    assignment = blind_assignment(seed, scenario_id=scenario_id)
    commitment = assignment_commitment(assignment, seed_sha256=seed, scenario_id=scenario_id)
    review = completed_review(scenario_id=scenario_id, commitment=commitment)
    statuses = {name: "pass" for name in I43G4C_PROVIDER_IMPLEMENTATIONS}
    if scenario_id == "g4_single_image_low_texture":
        statuses["depth_anything_v2"] = "rejected"
        statuses["marigold"] = "rejected"
    return build_controlled_reveal(
        review=review,
        production_snapshot_sha256=seed,
        production_statuses=statuses,
        prepare_summary_sha256="f" * 64,
    )


def test_final_closure_requires_three_scenarios_and_authorizes_g5() -> None:
    reveals = [_reveal_for(sid, ch) for sid, ch in zip(I43G4C_SCENARIOS, "abc", strict=True)]
    closure = build_i4_3g4c_final_closure(reveals)
    assert closure["status"] == "formally_closed"
    assert closure["i4_3g4_closed"] is True
    assert closure["i4_3g5_authorized"] is True
    assert closure["decision"]["universal_provider_winner_declared"] is False
    assert closure["decision"]["global_scalar_score_created"] is False


def test_final_closure_rejects_duplicate_scenario() -> None:
    reveal = _reveal_for(I43G4C_SCENARIOS[0], "a")
    with pytest.raises(ReconstructionValidationError):
        build_i4_3g4c_final_closure([reveal, reveal, _reveal_for(I43G4C_SCENARIOS[2], "c")])
