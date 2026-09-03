from __future__ import annotations

from pathlib import Path

from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_low_overlap_weak_parallax import build_i5_2c_low_overlap_weak_parallax_closure
from tests.i5_2b_helpers import I52A_SHA, passing_i5_2a
from tests.i5_2c_helpers import I52B_SHA, RECEIPT_52C_SHA, passing_i5_2b, passing_i5_2c_receipt


def _binding(path: str, sha: str) -> I51EvidenceBinding:
    return I51EvidenceBinding(path, sha)


def test_i52c_closure_preserves_sparse_sfm_authority_and_roadmap_boundary(tmp_path: Path) -> None:
    result = build_i5_2c_low_overlap_weak_parallax_closure(
        i5_2a_result=passing_i5_2a(tmp_path),
        i5_2a_binding=_binding("/evidence/i5_2a.json", I52A_SHA),
        i5_2b_result=passing_i5_2b(tmp_path),
        i5_2b_binding=_binding("/evidence/i5_2b.json", I52B_SHA),
        qualification_receipt=passing_i5_2c_receipt(),
        qualification_receipt_binding=_binding("/evidence/i5_2c_receipt.json", RECEIPT_52C_SHA),
    )
    assert result["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert result["authority"]["permanent_global_view_cap"] is None
    assert result["decision"]["threshold_relaxation_authorized"] is False
    assert result["decision"]["production_default_modified"] is False
    assert result["decision"]["provider_selected"] is False
    assert result["decision"]["auto_policy_implemented"] is False
    assert result["decision"]["hybrid_authorized"] is False
    assert result["decision"]["i5_3_authorized"] is True


def test_i52c_closure_retains_authoritative_directional_geometry(tmp_path: Path) -> None:
    result = build_i5_2c_low_overlap_weak_parallax_closure(
        i5_2a_result=passing_i5_2a(tmp_path),
        i5_2a_binding=_binding("/evidence/i5_2a.json", I52A_SHA),
        i5_2b_result=passing_i5_2b(tmp_path),
        i5_2b_binding=_binding("/evidence/i5_2b.json", I52B_SHA),
        qualification_receipt=passing_i5_2c_receipt(),
        qualification_receipt_binding=_binding("/evidence/i5_2c_receipt.json", RECEIPT_52C_SHA),
    )
    stress = result["stress_evidence"]
    assert stress["low_overlap"]["authoritative_covisibility_reduction_confirmed"] is True
    assert stress["weak_parallax"]["authoritative_parallax_reduction_confirmed"] is True
    assert stress["weak_parallax"]["authoritative_baseline_to_range_reduction_confirmed"] is True
