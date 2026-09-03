from __future__ import annotations

from pathlib import Path

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_low_overlap_weak_parallax import (
    I52C_PHASE,
    build_i5_2c_low_overlap_weak_parallax_closure,
    validate_i5_2b_for_i5_2c,
    validate_i5_2c_receipt,
)
from tests.i5_2b_helpers import I52A_SHA, passing_i5_2a
from tests.i5_2c_helpers import I52B_SHA, RECEIPT_52C_SHA, clone, passing_i5_2b, passing_i5_2c_receipt


def _binding(path: str, sha: str) -> I51EvidenceBinding:
    return I51EvidenceBinding(path, sha)


def test_i52c_accepts_passing_frozen_i52b_and_preserves_partial_registration(tmp_path: Path) -> None:
    replay = validate_i5_2b_for_i5_2c(passing_i5_2b(tmp_path))
    assert replay["quality_status"] == "pass"
    assert replay["scenarios"]["low_overlap"]["registered_images"] == 6
    assert replay["scenarios"]["low_overlap"]["input_images"] == 7
    assert replay["decision"]["i5_2c_qualification_closure_authorized"] is True


def test_i52c_rejects_unconfirmed_stress_revalidation(tmp_path: Path) -> None:
    replay = clone(passing_i5_2b(tmp_path))
    replay["comparison"]["semantic_stress_revalidation_complete"] = False
    with pytest.raises(ReconstructionValidationError):
        validate_i5_2b_for_i5_2c(replay)


def test_i52c_rejects_production_or_authority_drift(tmp_path: Path) -> None:
    replay = clone(passing_i5_2b(tmp_path))
    replay["scope"]["production_default_modified"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_2b_for_i5_2c(replay)

    replay = clone(passing_i5_2b(tmp_path))
    replay["authority"]["permanent_global_view_cap"] = 7
    with pytest.raises(ReconstructionValidationError):
        validate_i5_2b_for_i5_2c(replay)


def test_i52c_requires_green_evidence_only_receipt() -> None:
    receipt = validate_i5_2c_receipt(passing_i5_2c_receipt())
    assert receipt["scope"]["sfm_rerun"] is False
    assert receipt["scope"]["quality_replay_rerun"] is False

    bad = clone(passing_i5_2c_receipt())
    bad["scope"]["mp_sfm_started"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_2c_receipt(bad)


def test_i52c_builds_formal_pass_closure_and_authorizes_only_i53(tmp_path: Path) -> None:
    admission = passing_i5_2a(tmp_path)
    replay = passing_i5_2b(tmp_path)
    result = build_i5_2c_low_overlap_weak_parallax_closure(
        i5_2a_result=admission,
        i5_2a_binding=_binding("/evidence/i5_2a.json", I52A_SHA),
        i5_2b_result=replay,
        i5_2b_binding=_binding("/evidence/i5_2b.json", I52B_SHA),
        qualification_receipt=passing_i5_2c_receipt(),
        qualification_receipt_binding=_binding("/evidence/i5_2c_receipt.json", RECEIPT_52C_SHA),
    )
    assert result["phase"] == I52C_PHASE == "I5.2C"
    assert result["status"] == "low_overlap_weak_parallax_robustness_qualified"
    assert result["pass"] is True
    assert result["qualification"]["quality_status"] == "pass"
    assert result["qualification"]["control_registration"] == "19/19"
    assert result["qualification"]["low_overlap_registration"] == "6/7"
    assert result["qualification"]["weak_parallax_registration"] == "7/7"
    assert result["qualification"]["partial_registration_preserved"] is True
    assert result["decision"]["i5_2_complete"] is True
    assert result["decision"]["i5_3_authorized"] is True
    assert result["decision"]["i5_3_started"] is False
    assert result["decision"]["mp_sfm_authorized"] is False
    assert result["next"] == "I5.3 blur / exposure / clutter / symmetry."


def test_i52c_independently_binds_i52a_to_i52b_predecessor_sha(tmp_path: Path) -> None:
    admission = passing_i5_2a(tmp_path)
    replay = clone(passing_i5_2b(tmp_path))
    replay["evidence_bindings"]["i5_2a_admission"]["sha256"] = "f" * 64
    with pytest.raises(ReconstructionValidationError):
        build_i5_2c_low_overlap_weak_parallax_closure(
            i5_2a_result=admission,
            i5_2a_binding=_binding("/evidence/i5_2a.json", I52A_SHA),
            i5_2b_result=replay,
            i5_2b_binding=_binding("/evidence/i5_2b.json", I52B_SHA),
            qualification_receipt=passing_i5_2c_receipt(),
            qualification_receipt_binding=_binding("/evidence/i5_2c_receipt.json", RECEIPT_52C_SHA),
        )


def test_i52c_does_not_overclaim_universal_or_dense_robustness(tmp_path: Path) -> None:
    result = build_i5_2c_low_overlap_weak_parallax_closure(
        i5_2a_result=passing_i5_2a(tmp_path),
        i5_2a_binding=_binding("/evidence/i5_2a.json", I52A_SHA),
        i5_2b_result=passing_i5_2b(tmp_path),
        i5_2b_binding=_binding("/evidence/i5_2b.json", I52B_SHA),
        qualification_receipt=passing_i5_2c_receipt(),
        qualification_receipt_binding=_binding("/evidence/i5_2c_receipt.json", RECEIPT_52C_SHA),
    )
    q = result["qualification"]
    assert q["operating_envelope"] == "qualified_on_admitted_castle_p19_sparse_sfm_stress_cases"
    assert q["universal_low_overlap_robustness_claimed"] is False
    assert q["universal_weak_parallax_robustness_claimed"] is False
    assert q["dense_reconstruction_robustness_claimed"] is False
    assert q["provider_robustness_claimed"] is False
