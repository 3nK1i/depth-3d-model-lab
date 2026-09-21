from __future__ import annotations

from core.reconstruction.i5_final_closure import build_i5_14b_final_closure
from tests.i5_14b_helpers import make_binding, make_closure_receipt, make_readiness


def _build():
    return build_i5_14b_final_closure(
        readiness=make_readiness(),
        readiness_binding=make_binding("i5_14a.json", "a"),
        post_i5_14a_checkpoint_binding=make_binding("post_i5_14a.tar.gz", "b"),
        closure_receipt=make_closure_receipt(),
        closure_receipt_binding=make_binding("receipt.json", "c"),
    )


def test_final_contract_closes_i5_without_production_auto_activation():
    out = _build()
    assert out["decision"]["i5_closed"] is True
    assert out["decision"]["production_auto_authorized"] is False
    assert out["scope"]["production_auto_integration"] is False
    assert out["immutable_handoff_contract"]["full_post_i5_source_checkpoint_required"] is True


def test_final_contract_preserves_pycolmap_and_no_global_ranking_claims():
    out = _build()
    assert out["architecture_invariants"]["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert out["claim_boundaries"]["global_provider_ranking"] is False
    assert out["claim_boundaries"]["universal_provider_winner"] is False
    assert out["claim_boundaries"]["certified_metric_scale"] is False
