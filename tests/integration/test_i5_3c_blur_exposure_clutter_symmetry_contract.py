from __future__ import annotations

from pathlib import Path

from core.reconstruction.i5_blur_exposure_clutter_symmetry import build_i5_3c_blur_exposure_clutter_symmetry_closure
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from tests.i5_3b_helpers import I53A_SHA, passing_i5_3a
from tests.i5_3c_helpers import I53B_SHA, RECEIPT_53C_SHA, degraded_i5_3b, passing_i5_3c_receipt


def _binding(path: str, sha: str) -> I51EvidenceBinding:
    return I51EvidenceBinding(path, sha)


def test_i53c_contract_closes_degraded_visual_stress_envelope(tmp_path: Path) -> None:
    result = build_i5_3c_blur_exposure_clutter_symmetry_closure(
        i5_3a_result=passing_i5_3a(tmp_path),
        i5_3a_binding=_binding("/evidence/i5_3a.json", I53A_SHA),
        i5_3b_result=degraded_i5_3b(tmp_path),
        i5_3b_binding=_binding("/evidence/i5_3b.json", I53B_SHA),
        qualification_receipt=passing_i5_3c_receipt(),
        qualification_receipt_binding=_binding("/evidence/i5_3c_receipt.json", RECEIPT_53C_SHA),
    )
    assert result["status"] == "blur_exposure_clutter_symmetry_robustness_qualified"
    assert result["qualification"]["operating_envelope"] == "degraded_on_admitted_castle_p19_visual_stress_cases_with_clutter_inconclusive"
    assert result["scope"]["evidence_only_qualification_and_closure"] is True
    assert result["scope"]["sfm_rerun"] is False


def test_i53c_contract_does_not_start_later_roadmap_work(tmp_path: Path) -> None:
    result = build_i5_3c_blur_exposure_clutter_symmetry_closure(
        i5_3a_result=passing_i5_3a(tmp_path),
        i5_3a_binding=_binding("/evidence/i5_3a.json", I53A_SHA),
        i5_3b_result=degraded_i5_3b(tmp_path),
        i5_3b_binding=_binding("/evidence/i5_3b.json", I53B_SHA),
        qualification_receipt=passing_i5_3c_receipt(),
        qualification_receipt_binding=_binding("/evidence/i5_3c_receipt.json", RECEIPT_53C_SHA),
    )
    assert result["decision"]["provider_specific_robustness_started"] is False
    assert result["decision"]["mp_sfm_authorized"] is False
    assert result["decision"]["auto_policy_implemented"] is False
    assert result["decision"]["hybrid_authorized"] is False
