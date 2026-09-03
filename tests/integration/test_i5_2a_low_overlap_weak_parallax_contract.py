from __future__ import annotations

from pathlib import Path

from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_low_overlap_weak_parallax import (
    build_i5_2a_low_overlap_weak_parallax_admission,
    build_ordered_source_identity,
)
from tests.i5_2a_helpers import (
    G2A3_DOC_SHA,
    G2_FINAL_SHA,
    G2B_DOC_SHA,
    G2C_DOC_SHA,
    I51C_SHA,
    RECEIPT_52A_SHA,
    g2a3_doc,
    g2_final,
    g2b_doc,
    g2c_doc,
    make_ordered_dataset,
    passing_i5_1c,
    passing_receipt,
)


def _b(path: str, sha: str) -> I51EvidenceBinding:
    return I51EvidenceBinding(path, sha)


def test_i52a_contract_carries_exact_source_names_into_both_controlled_slices(tmp_path: Path) -> None:
    dataset = make_ordered_dataset(tmp_path / "scene")
    identity = build_ordered_source_identity(dataset)
    result = build_i5_2a_low_overlap_weak_parallax_admission(
        i5_1c_result=passing_i5_1c(), i5_1c_binding=_b("/e/i51c.json", I51C_SHA),
        i4_3g2_final_result=g2_final(), i4_3g2_final_binding=_b("/e/g2.json", G2_FINAL_SHA),
        i4_3g2a3_doc=g2a3_doc(), i4_3g2a3_binding=_b("/r/g2a3.md", G2A3_DOC_SHA),
        i4_3g2b_doc=g2b_doc(), i4_3g2b_binding=_b("/r/g2b.md", G2B_DOC_SHA),
        i4_3g2c_doc=g2c_doc(), i4_3g2c_binding=_b("/r/g2c.md", G2C_DOC_SHA),
        source_identity=identity, semantic_confirmation=True,
        admission_receipt=passing_receipt(), admission_receipt_binding=_b("/e/r.json", RECEIPT_52A_SHA),
    )
    all_names = set(identity["ordered_names"])
    plan = result["stress_plan"]
    assert set(plan["control"]["names"]) == all_names
    assert set(plan["low_overlap_proxy"]["names"]) <= all_names
    assert set(plan["weak_parallax_proxy"]["names"]) <= all_names
    assert plan["execution_contract"]["matching_strategy_locked_for_like_for_like_replay"] == "exhaustive"
    assert plan["execution_contract"]["production_sampling_policy_changed"] is False
