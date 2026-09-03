from __future__ import annotations

from pathlib import Path

from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_blur_exposure_clutter_symmetry import build_i5_3a_blur_exposure_clutter_symmetry_admission
from core.reconstruction.i5_low_overlap_weak_parallax import build_ordered_source_identity
from tests.i5_3a_helpers import I52C_SHA, RECEIPT_53A_SHA, passing_i5_2c, passing_i5_3a_receipt, source_dir


def _binding(path: str, sha: str) -> I51EvidenceBinding:
    return I51EvidenceBinding(path, sha)


def _result(tmp_path: Path) -> dict:
    return build_i5_3a_blur_exposure_clutter_symmetry_admission(
        i5_2c_result=passing_i5_2c(tmp_path),
        i5_2c_binding=_binding("/evidence/i5_2c.json", I52C_SHA),
        source_identity=build_ordered_source_identity(source_dir(tmp_path)),
        controlled_source_confirmation=True,
        admission_receipt=passing_i5_3a_receipt(),
        admission_receipt_binding=_binding("/evidence/i5_3a_receipt.json", RECEIPT_53A_SHA),
    )


def test_i53a_contract_preserves_authority_and_production_defaults(tmp_path: Path) -> None:
    result = _result(tmp_path)
    assert result["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert result["authority"]["permanent_global_view_cap"] is None
    assert result["scope"]["production_default_modified"] is False
    assert result["scope"]["thresholds_modified"] is False
    assert result["scope"]["sfm_rerun"] is False
    assert result["scope"]["stress_images_materialized"] is False


def test_i53a_contract_freezes_all_five_stress_recipes(tmp_path: Path) -> None:
    result = _result(tmp_path)
    plan = result["stress_plan"]
    assert len(plan) == 6
    assert result["stress_design"]["stress_recipe_count"] == 5
    assert result["stress_design"]["result_driven_recipe_tuning_authorized"] is False
    assert result["stress_design"]["semantic_stress_labels_require_i5_3b_revalidation"] is True


def test_i53a_contract_defers_later_roadmap_work(tmp_path: Path) -> None:
    result = _result(tmp_path)
    assert result["roadmap_guard"]["i5_4_forward_motion_kitti_not_started"] is True
    assert result["roadmap_guard"]["provider_specific_robustness_not_started"] is True
    assert result["roadmap_guard"]["mp_sfm_deferred_to_i5_10"] is True
    assert result["authority"]["auto_policy_implemented"] is False
    assert result["authority"]["hybrid_authorized"] is False
