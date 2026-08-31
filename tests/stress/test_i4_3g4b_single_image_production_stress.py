from __future__ import annotations

from copy import deepcopy

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.single_image_provider_production_qualification import (
    I43G4B_PROVIDER_IMPLEMENTATIONS,
    build_g4b_scenario_snapshot,
    validate_i4_3g4b_predecessor,
)
from tests.i4_3g4b_helpers import g4b_binding, passing_g4a_snapshot, provider_evidence


def test_repeated_predecessor_validation_is_deterministic() -> None:
    original = passing_g4a_snapshot()
    for _ in range(100):
        candidate = deepcopy(original)
        validate_i4_3g4b_predecessor(candidate)
        assert candidate == original


def test_closure_never_turns_provider_quality_rejection_into_hard_scenario_failure() -> None:
    for rejected_provider in I43G4B_PROVIDER_IMPLEMENTATIONS:
        evidence = {name: provider_evidence(name) for name in I43G4B_PROVIDER_IMPLEMENTATIONS}
        evidence[rejected_provider] = provider_evidence(rejected_provider, status="rejected")
        artifacts = {
            name: {"path": f"/evidence/{name}.json", "sha256": "8" * 64}
            for name in I43G4B_PROVIDER_IMPLEMENTATIONS
        }
        snapshot = build_g4b_scenario_snapshot(
            binding=g4b_binding(),
            g4a_snapshot_path="/evidence/g4a.json",
            g4a_snapshot_sha256="7" * 64,
            provider_evidence=evidence,
            provider_artifacts=artifacts,
        )
        assert snapshot["pass"] is True
        assert snapshot["status"] == "production_native_complete_with_provider_rejection"
        assert snapshot["accounting"]["rejected_providers"] == [rejected_provider]


def test_closure_refuses_incomplete_six_provider_matrix() -> None:
    evidence = {name: provider_evidence(name) for name in I43G4B_PROVIDER_IMPLEMENTATIONS[:-1]}
    artifacts = {
        name: {"path": f"/evidence/{name}.json", "sha256": "8" * 64}
        for name in I43G4B_PROVIDER_IMPLEMENTATIONS[:-1]
    }
    with pytest.raises(ReconstructionValidationError, match="all six providers"):
        build_g4b_scenario_snapshot(
            binding=g4b_binding(),
            g4a_snapshot_path="/evidence/g4a.json",
            g4a_snapshot_sha256="7" * 64,
            provider_evidence=evidence,
            provider_artifacts=artifacts,
        )
