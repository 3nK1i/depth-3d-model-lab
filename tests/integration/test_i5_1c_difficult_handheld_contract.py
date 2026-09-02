from __future__ import annotations

from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding, build_i5_1c_difficult_handheld_closure
from tests.i5_1b_helpers import passing_i5_1a
from tests.i5_1c_helpers import I51A_SHA, I51B_SHA, OOM_SHA, RECEIPT_SHA, degraded_i5_1b, oom_diagnostic_text, passing_i5_1c_receipt


def _binding(path: str, sha: str) -> I51EvidenceBinding:
    return I51EvidenceBinding(path, sha)


def result() -> dict:
    return build_i5_1c_difficult_handheld_closure(
        i5_1a_result=passing_i5_1a(), i5_1a_binding=_binding("/a.json", I51A_SHA),
        i5_1b_result=degraded_i5_1b(), i5_1b_binding=_binding("/b.json", I51B_SHA),
        oom_diagnostic=oom_diagnostic_text(), oom_diagnostic_binding=_binding("/oom.txt", OOM_SHA),
        qualification_receipt=passing_i5_1c_receipt(), qualification_receipt_binding=_binding("/receipt.json", RECEIPT_SHA),
    )


def test_i51c_evidence_inventory_and_provenance_are_explicit() -> None:
    payload = result()
    assert set(payload["evidence_bindings"]) == {"i5_1a_baseline", "i5_1b_replay", "i5_1b_failed_attempt_diagnostic", "i5_1c_qualification_receipt"}
    provenance = payload["provenance"]
    assert provenance["i5_1a_reverified"] is True
    assert provenance["i5_1b_reverified"] is True
    assert provenance["i5_1b_i5_1a_binding_reverified"] is True
    assert provenance["fresh_source_identity_preserved"] is True
    assert provenance["fresh_vs_historical_content_hash_equality_claimed"] is False


def test_i51c_qualification_is_envelope_evidence_not_ranking() -> None:
    q = result()["qualification"]
    assert q["material_improvement_over_historical_baseline"] is True
    assert q["production_dense_gate_reached"] is False
    assert q["counts_are_not_provider_scores"] is True
    assert q["recommended_plan_is_scenario_experiment_result_not_production_default"] is True


def test_i51c_scope_performs_no_new_reconstruction_work() -> None:
    scope = result()["scope"]
    for key in ("provider_execution_started", "sfm_rerun", "dense_pair_rerun", "quality_replay_rerun", "manual_review_rerun"):
        assert scope[key] is False
    assert scope["evidence_only_qualification_and_closure"] is True
