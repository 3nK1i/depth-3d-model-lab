from __future__ import annotations

from pathlib import Path

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_blur_exposure_clutter_symmetry import (
    build_i5_3b_blur_exposure_clutter_symmetry_replay,
    validate_i5_3a_for_i5_3b,
)
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from tests.i5_3b_helpers import (
    I53A_SHA,
    RECEIPT_53B_SHA,
    clone,
    passing_i5_3a,
    passing_i5_3b_receipt,
    passing_image_domain_comparison,
    synthetic_i53_sfm_analysis,
)


def _binding(path: str, sha: str) -> I51EvidenceBinding:
    return I51EvidenceBinding(path, sha)


def test_i53b_rejects_result_driven_tuning_authorization_drift(tmp_path: Path) -> None:
    admission = clone(passing_i5_3a(tmp_path))
    admission["stress_design"]["result_driven_recipe_tuning_authorized"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_3a_for_i5_3b(admission)


def test_i53b_rejects_source_identity_change_in_handoff(tmp_path: Path) -> None:
    admission = clone(passing_i5_3a(tmp_path))
    admission["source_identity"]["asset_set_sha256"] = "0" * 64
    with pytest.raises(ReconstructionValidationError):
        validate_i5_3a_for_i5_3b(admission)


def test_i53b_preserves_partial_registration_as_evidence(tmp_path: Path) -> None:
    admission = passing_i5_3a(tmp_path)
    analyses = {
        "control": synthetic_i53_sfm_analysis(admission, "control"),
        "blur": synthetic_i53_sfm_analysis(admission, "blur_proxy", registered_count=18),
        "underexposure": synthetic_i53_sfm_analysis(admission, "underexposure_proxy"),
        "overexposure": synthetic_i53_sfm_analysis(admission, "overexposure_proxy"),
        "clutter": synthetic_i53_sfm_analysis(admission, "clutter_proxy"),
        "symmetry": synthetic_i53_sfm_analysis(admission, "symmetry_proxy"),
    }
    result = build_i5_3b_blur_exposure_clutter_symmetry_replay(
        i5_3a_result=admission,
        i5_3a_binding=_binding("/evidence/i5_3a.json", I53A_SHA),
        replay_receipt=passing_i5_3b_receipt(),
        replay_receipt_binding=_binding("/evidence/i5_3b_receipt.json", RECEIPT_53B_SHA),
        image_domain_comparison=passing_image_domain_comparison(),
        scenario_analyses=analyses,
        source_reverified_before=True,
        source_reverified_after=True,
        staged_inputs_reverified=True,
    )
    assert result["scenarios"]["blur"]["registered_images"] == 18
    assert result["scenarios"]["blur"]["partial_registration_preserved"] is True
    assert result["quality_status"] == "pass"
