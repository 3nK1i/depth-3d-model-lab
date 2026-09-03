from __future__ import annotations

from pathlib import Path

from core.reconstruction.i5_blur_exposure_clutter_symmetry import build_i5_3c_blur_exposure_clutter_symmetry_closure
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from tests.i5_3b_helpers import I53A_SHA, passing_i5_3a
from tests.i5_3c_helpers import I53B_SHA, RECEIPT_53C_SHA, degraded_i5_3b, passing_i5_3c_receipt


def _binding(path: str, sha: str) -> I51EvidenceBinding:
    return I51EvidenceBinding(path, sha)


def test_i53c_repeated_closure_is_deterministic(tmp_path: Path) -> None:
    kwargs = dict(
        i5_3a_result=passing_i5_3a(tmp_path),
        i5_3a_binding=_binding("/evidence/i5_3a.json", I53A_SHA),
        i5_3b_result=degraded_i5_3b(tmp_path),
        i5_3b_binding=_binding("/evidence/i5_3b.json", I53B_SHA),
        qualification_receipt=passing_i5_3c_receipt(),
        qualification_receipt_binding=_binding("/evidence/i5_3c_receipt.json", RECEIPT_53C_SHA),
    )
    a = build_i5_3c_blur_exposure_clutter_symmetry_closure(**kwargs)
    b = build_i5_3c_blur_exposure_clutter_symmetry_closure(**kwargs)
    assert a == b


def test_i53c_inconclusive_clutter_never_becomes_positive_claim(tmp_path: Path) -> None:
    result = build_i5_3c_blur_exposure_clutter_symmetry_closure(
        i5_3a_result=passing_i5_3a(tmp_path),
        i5_3a_binding=_binding("/evidence/i5_3a.json", I53A_SHA),
        i5_3b_result=degraded_i5_3b(tmp_path),
        i5_3b_binding=_binding("/evidence/i5_3b.json", I53B_SHA),
        qualification_receipt=passing_i5_3c_receipt(),
        qualification_receipt_binding=_binding("/evidence/i5_3c_receipt.json", RECEIPT_53C_SHA),
    )
    clutter = result["stress_conclusions"]["clutter"]
    assert clutter["robustness_qualified"] is False
    assert result["qualification"]["clutter_robustness_claimed"] is False
