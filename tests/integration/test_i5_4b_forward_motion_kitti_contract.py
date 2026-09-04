from __future__ import annotations

from pathlib import Path

from core.reconstruction.i5_forward_motion_kitti import analyze_i5_4b_mapping_no_sparse_model_rejection, analyze_i5_4b_sfm_report, build_i5_4b_forward_motion_kitti_replay
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from tests.i5_4b_helpers import I54A_SHA, RECEIPT_54B_SHA, passing_i5_4a, passing_i5_4b_receipt, synthetic_i54b_sfm_report


def test_i54b_contract_end_to_end(tmp_path: Path) -> None:
    admission = passing_i5_4a(tmp_path)
    analysis = analyze_i5_4b_sfm_report(synthetic_i54b_sfm_report(admission), scenario_plan=admission["stress_plan"]["forward_motion_window"])
    result = build_i5_4b_forward_motion_kitti_replay(
        i5_4a_result=admission,
        i5_4a_binding=I51EvidenceBinding("/evidence/i5_4a.json", I54A_SHA),
        replay_receipt=passing_i5_4b_receipt(),
        replay_receipt_binding=I51EvidenceBinding("/evidence/i5_4b_receipt.json", RECEIPT_54B_SHA),
        sfm_analysis=analysis,
        source_reverified_before=True,
        source_window_reverified_after=True,
        staged_inputs_reverified=True,
    )
    assert result["status"] == "current_frozen_foundation_forward_motion_kitti_replayed"
    assert result["forward_motion_revalidation"]["authoritative_forward_motion_geometry_proven"] is True
    assert result["next"] == "I5.4C forward-motion / KITTI robustness qualification and closure."


def test_i54b_keeps_arbitrary_gauge(tmp_path: Path) -> None:
    admission = passing_i5_4a(tmp_path)
    analysis = analyze_i5_4b_sfm_report(synthetic_i54b_sfm_report(admission), scenario_plan=admission["stress_plan"]["forward_motion_window"])
    assert analysis["authority"]["metric_scale_claimed"] is False
    assert analysis["geometry"]["parallax"]["scale_invariant"] is True


def test_i54b_mapping_no_model_rejection_closes_as_rejected_evidence(tmp_path: Path) -> None:
    admission = passing_i5_4a(tmp_path)
    plan = admission["stress_plan"]["forward_motion_window"]
    analysis = analyze_i5_4b_mapping_no_sparse_model_rejection(
        scenario_plan=plan,
        staged_names=list(plan["names"]),
        reason="RuntimeError: PyCOLMAP incremental mapping produced no sparse model.",
    )
    result = build_i5_4b_forward_motion_kitti_replay(
        i5_4a_result=admission,
        i5_4a_binding=I51EvidenceBinding("/evidence/i5_4a.json", I54A_SHA),
        replay_receipt=passing_i5_4b_receipt(),
        replay_receipt_binding=I51EvidenceBinding("/evidence/i5_4b_receipt.json", RECEIPT_54B_SHA),
        sfm_analysis=analysis,
        source_reverified_before=True,
        source_window_reverified_after=True,
        staged_inputs_reverified=True,
    )
    assert result["quality_status"] == "rejected"
    assert result["replay"]["rejection_stage"] == "incremental_mapping"
    assert result["qualification"]["forward_motion_robustness_conclusion"] == "rejected_before_forward_motion_geometry_revalidation"
    assert result["decision"]["i5_4c_qualification_closure_authorized"] is True
