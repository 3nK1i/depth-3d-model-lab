from __future__ import annotations

from pathlib import Path

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_blur_exposure_clutter_symmetry import (
    I53C_PHASE,
    build_i5_3c_blur_exposure_clutter_symmetry_closure,
    validate_i5_3b_for_i5_3c,
    validate_i5_3c_receipt,
)
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from tests.i5_3b_helpers import I53A_SHA, passing_i5_3a
from tests.i5_3c_helpers import I53B_SHA, RECEIPT_53C_SHA, clone, degraded_i5_3b, passing_i5_3c_receipt


def _binding(path: str, sha: str) -> I51EvidenceBinding:
    return I51EvidenceBinding(path, sha)


def _build(tmp_path: Path) -> dict:
    return build_i5_3c_blur_exposure_clutter_symmetry_closure(
        i5_3a_result=passing_i5_3a(tmp_path),
        i5_3a_binding=_binding("/evidence/i5_3a.json", I53A_SHA),
        i5_3b_result=degraded_i5_3b(tmp_path),
        i5_3b_binding=_binding("/evidence/i5_3b.json", I53B_SHA),
        qualification_receipt=passing_i5_3c_receipt(),
        qualification_receipt_binding=_binding("/evidence/i5_3c_receipt.json", RECEIPT_53C_SHA),
    )


def test_i53c_accepts_degraded_evidence_complete_replay(tmp_path: Path) -> None:
    replay = validate_i5_3b_for_i5_3c(degraded_i5_3b(tmp_path))
    assert replay["quality_status"] == "degraded"
    assert replay["scenarios"]["blur"]["quality_status"] == "rejected"
    assert replay["scenarios"]["symmetry"]["registered_images"] == 12


def test_i53c_preserves_clutter_as_inconclusive(tmp_path: Path) -> None:
    result = _build(tmp_path)
    clutter = result["stress_conclusions"]["clutter"]
    assert clutter["semantic_stress_revalidated"] is False
    assert clutter["robustness_conclusion"] == "inconclusive_proxy_not_revalidated"
    assert clutter["robustness_qualified"] is False


def test_i53c_classifies_confirmed_stress_outcomes(tmp_path: Path) -> None:
    result = _build(tmp_path)
    conclusions = result["stress_conclusions"]
    assert conclusions["blur"]["robustness_conclusion"] == "outside_current_sparse_sfm_ingestion_envelope"
    assert conclusions["underexposure"]["robustness_conclusion"] == "outside_current_sparse_sfm_ingestion_envelope"
    assert conclusions["overexposure"]["robustness_conclusion"] == "survived_admitted_stress_case"
    assert conclusions["symmetry"]["robustness_conclusion"] == "degraded_partial_registration"


def test_i53c_builds_formal_degraded_closure_and_authorizes_only_i54(tmp_path: Path) -> None:
    result = _build(tmp_path)
    assert result["phase"] == I53C_PHASE == "I5.3C"
    assert result["pass"] is True
    assert result["qualification"]["quality_status"] == "degraded"
    assert result["decision"]["i5_3_complete"] is True
    assert result["decision"]["i5_4_authorized"] is True
    assert result["decision"]["i5_5_authorized"] is False
    assert result["next"] == "I5.4 forward-motion / KITTI robustness."


def test_i53c_preserves_registration_counts(tmp_path: Path) -> None:
    q = _build(tmp_path)["qualification"]
    assert q["control_registration"] == "19/19"
    assert q["blur_registration"] == "0/19"
    assert q["underexposure_registration"] == "0/19"
    assert q["overexposure_registration"] == "19/19"
    assert q["clutter_registration"] == "19/19"
    assert q["symmetry_registration"] == "12/19"


def test_i53c_requires_evidence_only_green_receipt() -> None:
    receipt = validate_i5_3c_receipt(passing_i5_3c_receipt())
    assert receipt["scope"]["sfm_rerun"] is False
    bad = clone(passing_i5_3c_receipt())
    bad["scope"]["stress_images_rematerialized"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_3c_receipt(bad)


def test_i53c_rejects_attempt_to_claim_clutter_robustness(tmp_path: Path) -> None:
    replay = clone(degraded_i5_3b(tmp_path))
    replay["image_domain_revalidation"]["clutter"]["semantic_stress_revalidated"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_3b_for_i5_3c(replay)


def test_i53c_rejects_pre_sfm_rejection_drift(tmp_path: Path) -> None:
    replay = clone(degraded_i5_3b(tmp_path))
    replay["scenarios"]["blur"]["sfm_executed"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_3b_for_i5_3c(replay)


def test_i53c_independently_binds_i53a_predecessor(tmp_path: Path) -> None:
    replay = clone(degraded_i5_3b(tmp_path))
    replay["evidence_bindings"]["i5_3a_admission"]["sha256"] = "0" * 64
    with pytest.raises(ReconstructionValidationError):
        build_i5_3c_blur_exposure_clutter_symmetry_closure(
            i5_3a_result=passing_i5_3a(tmp_path),
            i5_3a_binding=_binding("/evidence/i5_3a.json", I53A_SHA),
            i5_3b_result=replay,
            i5_3b_binding=_binding("/evidence/i5_3b.json", I53B_SHA),
            qualification_receipt=passing_i5_3c_receipt(),
            qualification_receipt_binding=_binding("/evidence/i5_3c_receipt.json", RECEIPT_53C_SHA),
        )


def test_i53c_preserves_authority_and_no_overclaims(tmp_path: Path) -> None:
    result = _build(tmp_path)
    assert result["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert result["authority"]["permanent_global_view_cap"] is None
    q = result["qualification"]
    assert q["universal_visual_stress_robustness_claimed"] is False
    assert q["clutter_robustness_claimed"] is False
    assert q["dense_reconstruction_robustness_claimed"] is False
    assert q["provider_robustness_claimed"] is False
