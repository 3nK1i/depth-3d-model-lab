from __future__ import annotations

from core.reconstruction.single_image_provider_matched_view_review import (
    I43G4C_BLIND_LABELS,
    I43G4C_PROVIDER_IMPLEMENTATIONS,
    I43G4C_SCENARIOS,
    assignment_commitment,
    blind_assignment,
)


def test_many_blind_seeds_remain_total_six_provider_permutations() -> None:
    seen = set()
    for index in range(128):
        seed = f"{index:064x}"[-64:]
        for scenario in I43G4C_SCENARIOS:
            assignment = blind_assignment(seed, scenario_id=scenario)
            assert tuple(assignment) == I43G4C_BLIND_LABELS
            assert set(assignment.values()) == set(I43G4C_PROVIDER_IMPLEMENTATIONS)
            commitment = assignment_commitment(assignment, seed_sha256=seed, scenario_id=scenario)
            assert len(commitment) == 64
            seen.add(tuple(assignment[label] for label in I43G4C_BLIND_LABELS))
    assert len(seen) > 12
