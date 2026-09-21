from __future__ import annotations

from core.reconstruction.dense_provider_architecture import DenseProviderSelectionMode, I43A_AUTO_POLICY_ID
from core.reconstruction.i5_auto_policy_admission import build_i5_13a_auto_policy_admission
from tests.i5_13a_helpers import build_args


def build():
    return build_i5_13a_auto_policy_admission(**build_args())


def test_canonical_auto_contract_is_reused_not_redesigned():
    out = build()
    assert DenseProviderSelectionMode.AUTO.value == I43A_AUTO_POLICY_ID == "auto"
    assert out["architecture_contract"]["selection_mode"] == "auto"


def test_cross_provider_family_boundaries_are_preserved():
    out = build()
    assert out["evidence_coverage"]["cross_provider_family_count"] == 3
    assert out["evidence_coverage"]["provider_implementation_count"] == 10
    assert out["evidence_synthesis"]["provider_operating_envelope"]["cross_family_direct_ranking_valid"] is False


def test_sparse_failure_precedes_dense_provider_selection():
    out = build()
    assert out["evidence_synthesis"]["sparse_sfm_operating_envelope"]["forward_motion"]["hard_auto_abstention_required"] is True
    assert out["auto_policy_v1_design"]["hard_gates"]["authoritative_sfm_must_be_production_usable"] is True


def test_historical_quality_events_are_evidence_not_scores():
    out = build()
    env = out["evidence_synthesis"]["provider_operating_envelope"]
    assert len(env["cross_provider_terminal_quality_events"]) == 5
    assert env["terminal_event_counts_are_not_scores"] is True


def test_no_execution_or_default_change_is_introduced():
    out = build()
    assert out["scope"]["provider_execution"] is False
    assert out["scope"]["production_default_modified"] is False
    assert out["decision"]["production_auto_authorized"] is False


def test_b_authorization_does_not_authorize_c_or_d():
    d = build()["decision"]
    assert d["i5_13b_authorized"] is True
    assert d["i5_13c_authorized"] is False
    assert d["i5_13d_authorized"] is False
