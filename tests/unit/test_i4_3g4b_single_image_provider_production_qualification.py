from __future__ import annotations

from copy import deepcopy

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.single_image_provider_production_qualification import (
    I43G4B_PROVIDER_IMPLEMENTATIONS,
    build_g4b_scenario_snapshot,
    recognized_quality_rejection,
    resolve_g4b_scenario_binding,
    validate_current_runtime_against_g4a,
    validate_i4_3g4b_predecessor,
    validate_single_image_production_plan,
    validate_single_image_provider_run_evidence,
)
from tests.i4_3g4b_helpers import (
    current_runtime,
    g4b_binding,
    passing_g4a_snapshot,
    plan_dict,
    provider_evidence,
)


def test_g4a_predecessor_and_all_three_scenarios_are_accepted() -> None:
    snapshot = passing_g4a_snapshot()
    validate_i4_3g4b_predecessor(snapshot)
    for scenario in (
        "g4_single_image_complete_room",
        "g4_single_image_low_texture",
        "g4_single_image_weak_overlap",
    ):
        binding = resolve_g4b_scenario_binding(
            i4_3g4a_snapshot=snapshot,
            scenario_id=scenario,
            verify_files=False,
        )
        assert binding.scenario_id == scenario
        assert binding.registered_view_count == 8


def test_g4b_refuses_g4a_that_did_not_authorize_production() -> None:
    snapshot = passing_g4a_snapshot()
    snapshot["decision"]["i4_3g4b_production_native_execution_authorized"] = False
    with pytest.raises(ReconstructionValidationError, match="not authorized"):
        validate_i4_3g4b_predecessor(snapshot)


@pytest.mark.parametrize("implementation", I43G4B_PROVIDER_IMPLEMENTATIONS)
def test_runtime_identity_must_match_frozen_g4a(implementation: str) -> None:
    frozen = passing_g4a_snapshot()["runtime"][implementation]
    current = current_runtime(implementation)
    validate_current_runtime_against_g4a(
        implementation=implementation,
        frozen_runtime=frozen,
        current_runtime=current,
    )
    drifted = deepcopy(current)
    drifted["project_root"] = "/different-project-root"
    with pytest.raises(ReconstructionValidationError, match="runtime identity drift"):
        validate_current_runtime_against_g4a(
            implementation=implementation,
            frozen_runtime=frozen,
            current_runtime=drifted,
        )


@pytest.mark.parametrize("implementation", I43G4B_PROVIDER_IMPLEMENTATIONS)
def test_single_image_adaptive_plan_contract(implementation: str) -> None:
    plan = plan_dict(implementation)
    validate_single_image_production_plan(plan, implementation=implementation)
    plan["requirements"]["stateful_sequence"] = True
    with pytest.raises(ReconstructionValidationError, match="stateful temporal"):
        validate_single_image_production_plan(plan, implementation=implementation)


@pytest.mark.parametrize("implementation", I43G4B_PROVIDER_IMPLEMENTATIONS)
def test_provider_run_evidence_accepts_pass_and_only_recognized_rejection(implementation: str) -> None:
    validate_single_image_provider_run_evidence(provider_evidence(implementation, status="pass"))
    rejected = provider_evidence(implementation, status="rejected")
    validate_single_image_provider_run_evidence(rejected)
    assert recognized_quality_rejection(
        implementation=implementation,
        error_message=rejected["rejection"]["reason"],
    )
    rejected["rejection"]["reason"] = "CUDA worker crashed"
    with pytest.raises(ReconstructionValidationError, match="not an authorized"):
        validate_single_image_provider_run_evidence(rejected)


def test_rejection_must_preserve_thresholds_and_first_terminal_attempt() -> None:
    rejected = provider_evidence("metric3d_v2", status="rejected")
    rejected["rejection"]["threshold_changed"] = True
    with pytest.raises(ReconstructionValidationError, match="no threshold"):
        validate_single_image_provider_run_evidence(rejected)
    rejected = provider_evidence("metric3d_v2", status="rejected")
    rejected["rejection"]["provider_execution_reran"] = True
    with pytest.raises(ReconstructionValidationError, match="without retrying"):
        validate_single_image_provider_run_evidence(rejected)


def test_scenario_closure_requires_all_six_terminal_provider_results() -> None:
    evidence = {name: provider_evidence(name) for name in I43G4B_PROVIDER_IMPLEMENTATIONS}
    evidence["depth_pro"] = provider_evidence("depth_pro", status="rejected")
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
    assert snapshot["production_statuses"]["depth_pro"] == "rejected"
    assert snapshot["accounting"]["provider_quality_rejection_present"] is True
    assert snapshot["manual_review"]["deferred_to"] == "I4.3G4C"
    assert snapshot["authority"]["provider_selected"] is False
