from __future__ import annotations

from core.reconstruction.dense_provider_architecture import DenseProviderFamily
from core.reconstruction.i5_auto_policy_engine import compile_i5_13b_policy, evaluate_i5_13b_auto_policy
from tests.i5_13b_helpers import admission, context, request


def test_repeated_shadow_decisions_are_byte_stable():
    policy = compile_i5_13b_policy(admission())
    req = request()
    ctx = context()
    first = evaluate_i5_13b_auto_policy(policy, req, ctx).as_dict()
    for _ in range(250):
        assert evaluate_i5_13b_auto_policy(policy, req, ctx).as_dict() == first


def test_repeated_forward_motion_never_selects_provider():
    policy = compile_i5_13b_policy(admission())
    ctx = context(sparse_context_id="forward_motion")
    for _ in range(250):
        assert evaluate_i5_13b_auto_policy(policy, request(), ctx).decision.selected_provider_id == ""


def test_repeated_ambiguous_single_image_never_invents_tiebreak():
    policy = compile_i5_13b_policy(admission())
    ctx = context(
        evidence_context_id="g4_single_image_complete_room",
        family=DenseProviderFamily.SINGLE_IMAGE_DEPTH,
        available_provider_ids=("metric3d-v2-single-image",),
    )
    for _ in range(250):
        out = evaluate_i5_13b_auto_policy(policy, request(), ctx)
        assert out.decision.selected_provider_id == ""
        assert "evidence_not_unique_even_after_availability" in out.decision.reason_codes
