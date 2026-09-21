from __future__ import annotations

from core.reconstruction.i5_final_qualification_readiness import build_i5_14a_final_qualification_readiness
from tests.i5_14a_helpers import make_binding, make_bindings, make_closures, make_receipt


def test_full_contract_closes_readiness_only_not_i5():
    out = build_i5_14a_final_qualification_readiness(
        closures=make_closures(), closure_bindings=make_bindings(),
        post_i5_13d_checkpoint_binding=make_binding("post.tar.gz", "f"),
        readiness_receipt=make_receipt(), readiness_receipt_binding=make_binding("receipt.json", "e"),
    )
    assert out["decision"]["i5_14a_complete"] is True
    assert out["decision"]["i5_14b_authorized"] is True
    assert out["decision"]["i5_14b_started"] is False
    assert out["decision"]["i5_complete"] is False
    assert out["decision"]["production_auto_authorized"] is False


def test_output_has_no_global_provider_ranking():
    out = build_i5_14a_final_qualification_readiness(
        closures=make_closures(), closure_bindings=make_bindings(),
        post_i5_13d_checkpoint_binding=make_binding("post.tar.gz", "f"),
        readiness_receipt=make_receipt(), readiness_receipt_binding=make_binding("receipt.json", "e"),
    )
    assert out["claim_boundaries"]["global_provider_ranking"] is False
    assert out["claim_boundaries"]["universal_provider_winner"] is False
    assert out["scope"]["production_default_modified"] is False
