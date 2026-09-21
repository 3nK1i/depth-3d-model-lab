from __future__ import annotations

from core.reconstruction.i5_failure_robustness import build_i5_11a_failure_mode_admission
from tests.i5_11a_helpers import I510D_SHA, POST_I510D_DELTA_SHA, binding, synthetic_i510d_closure, synthetic_i511a_receipt


def _artifact() -> dict:
    return build_i5_11a_failure_mode_admission(
        i5_10d_closure=synthetic_i510d_closure(),
        i5_10d_binding=binding("/evidence/i5_10d.json", I510D_SHA),
        post_i5_10d_delta_binding=binding("/checkpoints/post_i5_10d_delta.tar.gz", POST_I510D_DELTA_SHA),
        admission_receipt=synthetic_i511a_receipt(),
        admission_receipt_binding=binding("/evidence/i5_11a_receipt.json", "a" * 64),
        repository_anchors={"anchor": {"verified": True, "sha256": "b" * 64, "path": "/repo/anchor"}},
    )


def test_i511a_contract_preserves_no_git_checkpoint_chain() -> None:
    x = _artifact()
    assert x["source_control"]["mode"] == "NO_GIT_GITEA"
    assert x["source_control"]["post_i5_10d_delta_checkpoint_bound"] is True


def test_i511a_contract_separates_injected_fault_from_unexpected_infrastructure_failure() -> None:
    x = _artifact()
    semantics = x["outcome_semantics"]
    assert semantics["admitted_injected_fault_is_expected_event"] is True
    assert semantics["unexpected_infrastructure_failure_is_experiment_incomplete"] is True
    assert semantics["failed_recovery_is_robustness_failure"] is True


def test_i511a_contract_requires_immutable_fault_and_recovery_workspaces() -> None:
    x = _artifact()
    c = x["i5_11c_campaign_contract"]
    assert c["fault_attempts_total"] == 3
    assert c["recovery_attempts_total"] == 3
    assert c["attempt_directories_new_and_immutable"] is True
    assert c["retry_until_pass"] is False


def test_i511a_contract_does_not_authorize_hybrid_or_auto() -> None:
    x = _artifact()
    assert x["decision"]["i5_12_started"] is False
    assert x["decision"]["i5_13_started"] is False
    assert x["decision"]["hybrid_authorized"] is False
    assert x["decision"]["auto_policy_implemented"] is False
