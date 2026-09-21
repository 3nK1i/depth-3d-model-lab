from __future__ import annotations

from dataclasses import replace

import pytest

from core.reconstruction.dense_provider_architecture import DenseProviderImplementation
from core.reconstruction.reconstruction_production_auto_gate import (
    I64C_I5_AUTHORITY_PHASE,
    I64C_POLICY_PROFILE,
    I64C_REQUIRED_BLOCKING_GAPS,
    I64C_REQUIRED_FUTURE_PREREQUISITES,
    ProductionAutoBoundaryStatus,
    ProductionAutoQualificationBoundary,
    build_production_auto_qualification_boundary,
)
from core.reconstruction.reconstruction_provider_selection_contract import (
    ProductionProviderAvailability,
    ProductionProviderQualification,
    ProductionProviderSelectionMode,
    ProductionProviderSelectionSnapshot,
    canonical_production_provider_catalog,
)


def _none_snapshot() -> ProductionProviderSelectionSnapshot:
    return ProductionProviderSelectionSnapshot(
        providers=canonical_production_provider_catalog(),
    )


def _manual_snapshot() -> ProductionProviderSelectionSnapshot:
    providers = []
    for item in canonical_production_provider_catalog():
        if item.implementation is DenseProviderImplementation.COLMAP_MVS:
            item = replace(
                item,
                availability=ProductionProviderAvailability.AVAILABLE,
                qualification=ProductionProviderQualification.QUALIFIED_FOR_CONTEXT,
                qualification_context_id="ctx-clean",
                qualification_evidence_sha256=("a" * 64,),
                manual_selection_allowed=True,
            )
        providers.append(item)
    return ProductionProviderSelectionSnapshot(
        providers=tuple(providers),
        selection_mode=ProductionProviderSelectionMode.MANUAL,
        selected_provider_id="colmap-mvs-external",
    )


def test_i6_4c_boundary_preserves_i5_authority_and_policy_profile() -> None:
    boundary = build_production_auto_qualification_boundary(_none_snapshot())
    assert boundary.qualification_authority_phase == I64C_I5_AUTHORITY_PHASE == "I5.14B"
    assert boundary.policy_profile == I64C_POLICY_PROFILE == "evidence_bound_conservative_v1"
    assert boundary.status is ProductionAutoBoundaryStatus.DEFERRED_NOT_QUALIFIED


def test_i6_4c_offline_shadow_is_qualified_but_production_auto_is_fail_closed() -> None:
    boundary = build_production_auto_qualification_boundary(_none_snapshot())
    assert boundary.offline_shadow_qualified is True
    assert boundary.offline_replay_qualified is True
    assert boundary.production_auto_qualified is False
    assert boundary.production_auto_authorized is False
    assert boundary.selectable is False
    assert boundary.decision == "defer_production_auto"


def test_i6_4c_preserves_exact_blocking_gaps_and_future_prerequisites() -> None:
    boundary = build_production_auto_qualification_boundary(_none_snapshot())
    assert boundary.blocking_gaps == I64C_REQUIRED_BLOCKING_GAPS
    assert boundary.required_future_prerequisites == I64C_REQUIRED_FUTURE_PREREQUISITES
    assert "live_context_to_evidence_token_mapping_not_qualified" in boundary.blocking_gaps
    assert "production_auto_orchestration_entrypoint_missing" in boundary.blocking_gaps
    assert "qualified_live_context_resolver" in boundary.required_future_prerequisites
    assert "production_shadow_validation_before_live_activation" in boundary.required_future_prerequisites


def test_i6_4c_manual_selection_can_coexist_without_becoming_auto() -> None:
    boundary = build_production_auto_qualification_boundary(_manual_snapshot())
    assert boundary.current_selection_mode is ProductionProviderSelectionMode.MANUAL
    assert boundary.manual_selected_provider_id == "colmap-mvs-external"
    assert boundary.selectable is False
    assert boundary.production_auto_authorized is False


def test_i6_4c_none_selection_remains_none() -> None:
    boundary = build_production_auto_qualification_boundary(_none_snapshot())
    assert boundary.current_selection_mode is ProductionProviderSelectionMode.NONE
    assert boundary.manual_selected_provider_id == ""


def test_i6_4c_boundary_has_no_production_effects() -> None:
    boundary = build_production_auto_qualification_boundary(_manual_snapshot())
    assert boundary.provider_selection_effect is False
    assert boundary.provider_activation_authorized is False
    assert boundary.provider_execution_authorized is False
    assert boundary.settings_mutation_authorized is False
    assert boundary.start_action_authorized is False
    assert boundary.production_default_modified is False


def test_i6_4c_as_dict_keeps_auto_and_manual_selection_separate() -> None:
    payload = build_production_auto_qualification_boundary(_manual_snapshot()).as_dict()
    assert payload["production_auto"]["selectable"] is False
    assert payload["production_auto"]["production_authorized"] is False
    assert payload["current_selection"]["mode"] == "manual"
    assert payload["current_selection"]["manual_selected_provider_id"] == "colmap-mvs-external"
    assert payload["claim_boundaries"]["manual_selection_is_auto"] is False
    assert payload["claim_boundaries"]["auto_gate_is_execution"] is False
    assert payload["claim_boundaries"]["global_provider_ranking"] is False
    assert payload["claim_boundaries"]["global_scalar_score"] is False


def test_i6_4c_requires_i6_4a_selection_snapshot() -> None:
    with pytest.raises(TypeError):
        build_production_auto_qualification_boundary(object())  # type: ignore[arg-type]


def test_i6_4c_boundary_dataclass_rejects_broadened_auto_claims() -> None:
    boundary = build_production_auto_qualification_boundary(_none_snapshot())
    with pytest.raises(ValueError):
        ProductionAutoQualificationBoundary(
            gate=boundary.gate,
            current_selection_mode=boundary.current_selection_mode,
            manual_selected_provider_id=boundary.manual_selected_provider_id,
            production_auto_authorized=True,
        )
