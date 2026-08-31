from __future__ import annotations

from copy import deepcopy

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i4_3_final_qualification import (
    I43H1_EXPECTED_PREDECESSOR_TOTAL_TESTS,
    audit_i4_3_provider_architecture,
    build_i4_3h1_final_qualification_audit,
    validate_i4_3g5b_closure,
    validate_i4_3h1_regression_receipt,
)
from tests.i4_3h1_helpers import passing_g5b, passing_h1_bindings, passing_regression_receipt


def build(*, g5b=None, receipt=None, verified=True):
    return build_i4_3h1_final_qualification_audit(
        i4_3g5b_closure=passing_g5b() if g5b is None else g5b,
        regression_receipt=passing_regression_receipt() if receipt is None else receipt,
        evidence_bindings=passing_h1_bindings(),
        g5b_predecessor_hashes_reverified=verified,
    )


def test_h1_architecture_audit_covers_exact_five_families_and_ten_implementations() -> None:
    audit = audit_i4_3_provider_architecture()
    assert audit["provider_family_count"] == 5
    assert audit["executable_provider_family_count"] == 4
    assert audit["provider_implementation_count"] == 10
    assert audit["hybrid_family_reserved"] is True
    assert audit["hybrid_implementation_authorized"] is False
    assert len(audit["provider_inventory"]) == 10


def test_h1_architecture_audit_imports_every_provider_contract_surface_without_runtime_probe() -> None:
    audit = audit_i4_3_provider_architecture()
    for item in audit["provider_inventory"].values():
        assert item["module_contract_imported"] is True
        assert item["backend_runtime_probe_started"] is False
        assert item["adapter_module"].startswith("core.reconstruction.")
        assert item["globally_promoted"] is False


def test_h1_adaptive_requirements_remain_per_provider_and_non_empirical() -> None:
    audit = audit_i4_3_provider_architecture()
    assert audit["permanent_global_view_cap"] is None
    assert audit["adaptive_planning_coefficients_are_empirical_runtime_claims"] is False
    for item in audit["provider_inventory"].values():
        planning = item["adaptive_planning"]
        assert planning["minimum_views"] >= 2
        assert planning["relative_cost_per_megapixel"] > 0
        assert planning["relative_cost_is_empirical_runtime_claim"] is False


def test_h1_temporal_adaptive_contracts_remain_stateful_and_source_ordered() -> None:
    audit = audit_i4_3_provider_architecture()["provider_inventory"]
    for provider in ("rolling_depth", "video_depth_anything"):
        assert audit[provider]["adaptive_planning"]["stateful_sequence"] is True
        assert audit[provider]["adaptive_planning"]["source_order_required"] is True


def test_h1_accepts_formally_closed_g5b_and_preserves_five_quality_events() -> None:
    validated = validate_i4_3g5b_closure(passing_g5b())
    assert validated["coverage"]["scenario_count"] == 11
    assert len(validated["terminal_quality_events"]) == 5
    assert len(validated["predecessor_bindings"]) == 5


def test_h1_builds_qualification_without_premature_i43_or_i44_closure() -> None:
    result = build()
    assert result["phase"] == "I4.3H1"
    assert result["status"] == "qualification_complete"
    assert result["pass"] is True
    assert result["decision"]["i4_3h1_qualification_complete"] is True
    assert result["decision"]["i4_3h2_authorized"] is True
    assert result["decision"]["i4_3_closed"] is False
    assert result["decision"]["i4_4_authorized"] is False
    assert result["scope"]["formal_i4_3_closure_performed"] is False


def test_h1_groups_terminal_quality_events_into_provider_inventory_without_score() -> None:
    inventory = build()["architecture_audit"]["provider_inventory"]
    assert len(inventory["rolling_depth"]["terminal_quality_events"]) == 2
    assert len(inventory["video_depth_anything"]["terminal_quality_events"]) == 1
    assert len(inventory["depth_anything_v2"]["terminal_quality_events"]) == 1
    assert len(inventory["marigold"]["terminal_quality_events"]) == 1
    assert inventory["moge_2"]["terminal_quality_events"] == []
    assert all(item["no_terminal_quality_event_implies_universal_robustness"] is False for item in inventory.values())


def test_h1_regression_receipt_requires_full_668_predecessor_tests() -> None:
    receipt = validate_i4_3h1_regression_receipt(passing_regression_receipt())
    assert receipt["counts"]["predecessor_total_tests"] == I43H1_EXPECTED_PREDECESSOR_TOTAL_TESTS == 668


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("predecessor_unit_tests", 443),
        ("predecessor_integration_tests", 85),
        ("predecessor_stress_tests", 137),
        ("predecessor_total_tests", 667),
        ("predecessor_test_file_count", 90),
    ],
)
def test_h1_regression_receipt_fails_closed_on_coverage_drift(field: str, value: int) -> None:
    receipt = passing_regression_receipt()
    receipt["counts"][field] = value
    with pytest.raises(ReconstructionValidationError, match="contract drift"):
        validate_i4_3h1_regression_receipt(receipt)


def test_h1_regression_receipt_refuses_tiny_focused_suite() -> None:
    receipt = passing_regression_receipt(focused_tests=3)
    with pytest.raises(ReconstructionValidationError, match="unexpectedly small"):
        validate_i4_3h1_regression_receipt(receipt)


def test_h1_rejects_g5b_phase_or_status_drift() -> None:
    g5b = passing_g5b(); g5b["status"] = "pass"
    with pytest.raises(ReconstructionValidationError, match="formal I4.3G5B closure"):
        validate_i4_3g5b_closure(g5b)


def test_h1_rejects_g5b_chain_drift() -> None:
    g5b = passing_g5b(); g5b["closure_chain"]["g5a_deterministically_rebuilt_from_predecessors"] = False
    with pytest.raises(ReconstructionValidationError, match="closure chain"):
        build(g5b=g5b)


def test_h1_rejects_g5b_coverage_drift() -> None:
    g5b = passing_g5b(); g5b["coverage"]["scenario_count"] = 10
    with pytest.raises(ReconstructionValidationError, match="coverage.scenario_count"):
        build(g5b=g5b)


def test_h1_rejects_g5b_quality_event_drift() -> None:
    g5b = passing_g5b(); g5b["operating_envelope_summary"]["terminal_quality_events"].pop()
    g5b["operating_envelope_summary"]["terminal_quality_event_count"] = 4
    with pytest.raises(ReconstructionValidationError):
        build(g5b=g5b)


def test_h1_rejects_g5b_authority_drift() -> None:
    g5b = passing_g5b(); g5b["authority"]["metric_scale_claimed"] = True
    with pytest.raises(ReconstructionValidationError, match="metric_scale_claimed"):
        build(g5b=g5b)


def test_h1_rejects_g5b_selection_or_auto_drift() -> None:
    for field in ("provider_selected", "auto_policy_implemented", "hybrid_authorized"):
        g5b = passing_g5b(); g5b["decision"][field] = True
        with pytest.raises(ReconstructionValidationError, match=field):
            build(g5b=g5b)


def test_h1_requires_independent_predecessor_hash_reverification() -> None:
    with pytest.raises(ReconstructionValidationError, match="independent re-verification"):
        build(verified=False)


def test_h1_requires_exact_two_final_evidence_bindings() -> None:
    bindings = passing_h1_bindings(); del bindings["i4_3h1_regression_receipt"]
    with pytest.raises(ReconstructionValidationError, match="evidence-binding set"):
        build_i4_3h1_final_qualification_audit(
            i4_3g5b_closure=passing_g5b(),
            regression_receipt=passing_regression_receipt(),
            evidence_bindings=bindings,
            g5b_predecessor_hashes_reverified=True,
        )


def test_h1_keeps_all_forbidden_decisions_and_authorities_false() -> None:
    result = build()
    for field in (
        "provider_selected", "provider_promoted", "provider_ranking_created",
        "universal_provider_winner_declared", "global_scalar_score_created",
        "auto_policy_implemented", "hybrid_authorized",
    ):
        assert result["decision"][field] is False
    assert result["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert result["authority"]["permanent_global_view_cap"] is None
    for field in (
        "authoritative_sfm_modified", "provider_selected", "provider_promoted",
        "auto_policy_implemented", "hybrid_authorized", "metric_scale_claimed",
        "measurement_authoritative", "certified_metrology", "surface_mesh_generated",
    ):
        assert result["authority"][field] is False


def test_h1_does_not_mutate_frozen_inputs() -> None:
    g5b = passing_g5b(); receipt = passing_regression_receipt()
    before_g5b = deepcopy(g5b); before_receipt = deepcopy(receipt)
    build(g5b=g5b, receipt=receipt)
    assert g5b == before_g5b
    assert receipt == before_receipt
