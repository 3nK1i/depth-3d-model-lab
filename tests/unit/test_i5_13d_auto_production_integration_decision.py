from __future__ import annotations

import copy
from pathlib import Path

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from tests.i5_13d_helpers import (
    make_binding,
    make_i5_13c_qualification,
    make_i5_13d_receipt,
    make_integration_inventory,
)
from core.reconstruction.i5_auto_policy_production_qualification import (
    I513D_PHASE,
    I513D_POLICY_PROFILE,
    build_i5_13d_production_integration_decision,
    inspect_i5_13d_production_integration_surface,
    qualify_i5_13d_production_integration,
    validate_i5_13c_for_i5_13d,
    validate_i5_13d_receipt,
)


def test_constants():
    assert I513D_PHASE == "I5.13D"
    assert I513D_POLICY_PROFILE == "evidence_bound_conservative_v1"


def test_c_qualification_is_admitted_for_d_review():
    c = validate_i5_13c_for_i5_13d(make_i5_13c_qualification())
    assert c["decision"]["i5_13d_authorized"] is True
    assert c["decision"]["production_auto_authorized"] is False


def test_c_rejects_if_production_auto_was_already_authorized():
    c = make_i5_13c_qualification()
    c["decision"]["production_auto_authorized"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_13c_for_i5_13d(c)


def test_inventory_qualifies_offline_but_not_production_auto():
    result = qualify_i5_13d_production_integration(
        i5_13c_qualification=make_i5_13c_qualification(),
        integration_inventory=make_integration_inventory(),
    )
    assert result["offline_policy_qualified"] is True
    assert result["offline_replay_qualified"] is True
    assert result["production_auto_qualified"] is False
    assert result["production_auto_authorized"] is False
    assert result["decision"] == "defer_production_auto"


def test_missing_live_context_resolution_is_blocking_gap():
    result = qualify_i5_13d_production_integration(
        i5_13c_qualification=make_i5_13c_qualification(),
        integration_inventory=make_integration_inventory(),
    )
    assert "live_context_to_evidence_token_mapping_not_qualified" in result["blocking_gaps"]
    assert "production_auto_orchestration_entrypoint_missing" in result["blocking_gaps"]


def test_explicit_planning_and_inspection_only_ui_are_preserved():
    result = qualify_i5_13d_production_integration(
        i5_13c_qualification=make_i5_13c_qualification(),
        integration_inventory=make_integration_inventory(),
    )
    assert result["integration_inventory"]["adaptive_production_planning_mode"] == "explicit_provider_only"
    assert result["integration_inventory"]["production_ui_provider_selection_mode"] == "inspection_only"


def test_scene_classifier_is_not_invented_by_d():
    result = qualify_i5_13d_production_integration(
        i5_13c_qualification=make_i5_13c_qualification(),
        integration_inventory=make_integration_inventory(),
    )
    assert result["integration_inventory"]["production_scene_classifier_implemented"] is False
    assert result["required_future_prerequisites"]["qualified_live_context_resolver"] is True


def test_d_rejects_unexpected_existing_production_auto_entrypoint():
    inventory = make_integration_inventory()
    inventory["production_auto_orchestration_entrypoint_implemented"] = True
    with pytest.raises(ReconstructionValidationError):
        qualify_i5_13d_production_integration(
            i5_13c_qualification=make_i5_13c_qualification(),
            integration_inventory=inventory,
        )


def test_d_rejects_production_default_drift():
    inventory = make_integration_inventory()
    inventory["production_default_modified"] = True
    with pytest.raises(ReconstructionValidationError):
        qualify_i5_13d_production_integration(
            i5_13c_qualification=make_i5_13c_qualification(),
            integration_inventory=inventory,
        )


def test_receipt_validation():
    assert validate_i5_13d_receipt(make_i5_13d_receipt())["pass"] is True


def test_receipt_rejects_execution():
    receipt = make_i5_13d_receipt()
    receipt["scope"]["provider_execution"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_13d_receipt(receipt)


def test_formal_decision_closes_i5_13_and_authorizes_i5_14_only():
    out = build_i5_13d_production_integration_decision(
        i5_13c_qualification=make_i5_13c_qualification(),
        i5_13c_binding=make_binding("c.json", "1"),
        post_i5_13c_checkpoint_binding=make_binding("post_c.tar.gz", "2"),
        integration_inventory=make_integration_inventory(),
        integration_anchor_bindings={
            "dense_provider_architecture": make_binding("dense_provider_architecture.py", "3"),
            "adaptive_production_planning": make_binding("adaptive_production_planning.py", "4"),
            "reconstruction_page": make_binding("reconstruction_page.py", "5"),
            "i5_13b_engine_module": make_binding("i5_auto_policy_engine.py", "6"),
        },
        decision_receipt=make_i5_13d_receipt(),
        decision_receipt_binding=make_binding("receipt.json", "7"),
    )
    assert out["pass"] is True
    assert out["quality_status"] == "offline_auto_qualified_production_integration_deferred"
    assert out["decision"]["i5_13_complete"] is True
    assert out["decision"]["i5_13d_complete"] is True
    assert out["decision"]["production_auto_authorized"] is False
    assert out["decision"]["i5_14_authorized"] is True
    assert out["decision"]["i5_14_started"] is False


def test_formal_decision_does_not_create_provider_rank_or_score():
    out = build_i5_13d_production_integration_decision(
        i5_13c_qualification=make_i5_13c_qualification(),
        i5_13c_binding=make_binding("c.json", "1"),
        post_i5_13c_checkpoint_binding=make_binding("post_c.tar.gz", "2"),
        integration_inventory=make_integration_inventory(),
        integration_anchor_bindings={
            "dense_provider_architecture": make_binding("dense_provider_architecture.py", "3"),
            "adaptive_production_planning": make_binding("adaptive_production_planning.py", "4"),
            "reconstruction_page": make_binding("reconstruction_page.py", "5"),
            "i5_13b_engine_module": make_binding("i5_auto_policy_engine.py", "6"),
        },
        decision_receipt=make_i5_13d_receipt(),
        decision_receipt_binding=make_binding("receipt.json", "7"),
    )
    assert out["claim_boundaries"]["global_provider_ranking"] is False
    assert out["claim_boundaries"]["global_scalar_score"] is False
    assert out["decision"]["provider_ranking_created"] is False


def test_actual_repository_inventory_reports_current_nonproduction_surface():
    root = Path(__file__).resolve().parents[2]
    inventory = inspect_i5_13d_production_integration_surface(root)
    assert inventory["auto_selection_mode_contract_exists"] is True
    assert inventory["offline_engine_requires_explicit_frozen_context_tokens"] is True
    assert inventory["adaptive_production_planning_mode"] == "explicit_provider_only"
    assert inventory["adaptive_planning_provider_selected_by_auto"] is False
    assert inventory["production_ui_provider_selection_mode"] == "inspection_only"
    assert inventory["production_ui_provider_selection_activates_provider"] is False
    assert inventory["production_auto_engine_referenced_by_app"] is False
    assert inventory["production_auto_orchestration_entrypoint_implemented"] is False
