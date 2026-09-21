from __future__ import annotations

from core.reconstruction.dense_provider_architecture import DenseProviderFamily
from core.reconstruction.i5_auto_policy_engine import compile_i5_13b_policy, evaluate_i5_13b_auto_policy
from tests.i5_13b_helpers import admission, context, request


def test_shadow_engine_returns_existing_dense_selection_contract():
    out = evaluate_i5_13b_auto_policy(compile_i5_13b_policy(admission()), request(), context())
    payload = out.decision.as_dict()
    assert payload["request_mode"] == "auto"
    assert payload["selected_provider_id"] == "colmap-mvs-external"
    assert payload["automatic"] is True
    assert payload["fallback_used"] is False


def test_shadow_engine_abstention_is_valid_existing_selection_contract():
    ctx = context(sparse_context_id="forward_motion")
    out = evaluate_i5_13b_auto_policy(compile_i5_13b_policy(admission()), request(), ctx)
    payload = out.decision.as_dict()
    assert payload["selected_provider_id"] == ""
    assert payload["selected_implementation"] == ""
    assert payload["reason_codes"]


def test_hybrid_family_never_becomes_selected_provider():
    ctx = context(family=DenseProviderFamily.HYBRID, evidence_context_id="hybrid")
    out = evaluate_i5_13b_auto_policy(compile_i5_13b_policy(admission()), request(), ctx)
    assert out.decision.selected_provider_id == ""
