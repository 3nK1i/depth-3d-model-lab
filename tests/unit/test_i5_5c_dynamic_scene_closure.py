from __future__ import annotations

from pathlib import Path

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_dynamic_scene_handling import (
    I55C_PHASE,
    build_i5_5c_dynamic_scene_closure,
    validate_i5_5b_for_i5_5c,
    validate_i5_5c_receipt,
)
from tests.i5_5b_helpers import I55A_SHA, passing_i5_5a
from tests.i5_5c_helpers import (
    I55B_SHA,
    RECEIPT_55C_SHA,
    clone,
    i55c_binding,
    passing_i5_5b,
    passing_i5_5c_receipt,
)


def build(tmp_path: Path, *, directional_penalty: bool = True) -> dict:
    return build_i5_5c_dynamic_scene_closure(
        i5_5a_result=passing_i5_5a(tmp_path),
        i5_5a_binding=i55c_binding("/evidence/i5_5a.json", I55A_SHA),
        i5_5b_result=passing_i5_5b(tmp_path, directional_penalty=directional_penalty),
        i5_5b_binding=i55c_binding("/evidence/i5_5b.json", I55B_SHA),
        qualification_receipt=passing_i5_5c_receipt(),
        qualification_receipt_binding=i55c_binding("/evidence/i5_5c_receipt.json", RECEIPT_55C_SHA),
    )


def test_phase_constant() -> None:
    assert I55C_PHASE == "I5.5C"


def test_passing_b_is_valid_for_closure(tmp_path: Path) -> None:
    assert validate_i5_5b_for_i5_5c(passing_i5_5b(tmp_path))["quality_status"] == "pass"


def test_receipt_validates() -> None:
    assert validate_i5_5c_receipt(passing_i5_5c_receipt())["pass"] is True


def test_closure_preserves_pass_quality(tmp_path: Path) -> None:
    result = build(tmp_path)
    assert result["quality_status"] == "pass"
    assert result["qualification"]["quality_status"] == "pass"


def test_closure_preserves_dynamic_proxy_conclusion(tmp_path: Path) -> None:
    result = build(tmp_path)
    assert result["qualification"]["dynamic_scene_robustness_conclusion"] == "pass_on_admitted_controlled_dynamic_foreground_proxy"


def test_closure_records_directional_sparse_quality_penalty(tmp_path: Path) -> None:
    penalty = build(tmp_path)["directional_quality_evidence"]
    assert penalty["landmark_support_reduced"] is True
    assert penalty["observation_support_reduced"] is True
    assert penalty["reprojection_error_increased"] is True
    assert penalty["mean_track_length_reduced"] is True
    assert penalty["directional_sparse_quality_penalty_observed"] is True
    assert penalty["classification_threshold_used"] is None


def test_closure_operating_envelope_is_narrow(tmp_path: Path) -> None:
    q = build(tmp_path)["qualification"]
    assert q["operating_envelope"] == "qualified_on_admitted_controlled_dynamic_foreground_proxy_with_observed_sparse_quality_penalty"
    assert q["natural_dynamic_scene_ground_truth"] is False
    assert q["universal_dynamic_scene_robustness_claimed"] is False


def test_closure_without_penalty_does_not_invent_one(tmp_path: Path) -> None:
    result = build(tmp_path, directional_penalty=False)
    assert result["directional_quality_evidence"]["directional_sparse_quality_penalty_observed"] is False
    assert result["qualification"]["operating_envelope"] == "qualified_on_admitted_controlled_dynamic_foreground_proxy"


def test_closure_preserves_registration_and_connectivity(tmp_path: Path) -> None:
    q = build(tmp_path)["qualification"]
    assert q["control_registration"] == "19/19"
    assert q["dynamic_proxy_registration"] == "19/19"
    assert q["control_sparse_component_count"] == 1
    assert q["dynamic_proxy_sparse_component_count"] == 1


def test_closure_is_evidence_only(tmp_path: Path) -> None:
    scope = build(tmp_path)["scope"]
    assert scope["evidence_only_qualification_and_closure"] is True
    assert scope["sfm_rerun"] is False
    assert scope["source_rematerialized"] is False
    assert scope["stress_images_rematerialized"] is False


def test_closure_preserves_authority_guards(tmp_path: Path) -> None:
    authority = build(tmp_path)["authority"]
    assert authority["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert authority["permanent_global_view_cap"] is None
    assert authority["metric_scale_claimed"] is False


def test_closure_completes_i55_and_authorizes_i56(tmp_path: Path) -> None:
    decision = build(tmp_path)["decision"]
    assert decision["i5_5_complete"] is True
    assert decision["i5_6_authorized"] is True
    assert decision["i5_6_started"] is False
    assert decision["mp_sfm_authorized"] is False
    assert decision["auto_policy_implemented"] is False
    assert decision["hybrid_authorized"] is False


def test_validator_rejects_quality_upgrade(tmp_path: Path) -> None:
    payload = clone(passing_i5_5b(tmp_path))
    payload["quality_status"] = "degraded"
    with pytest.raises(ReconstructionValidationError):
        validate_i5_5b_for_i5_5c(payload)


def test_validator_rejects_false_natural_ground_truth(tmp_path: Path) -> None:
    payload = clone(passing_i5_5b(tmp_path))
    payload["qualification"]["natural_dynamic_scene_ground_truth"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_5b_for_i5_5c(payload)


def test_validator_rejects_threshold_relaxation(tmp_path: Path) -> None:
    payload = clone(passing_i5_5b(tmp_path))
    payload["scope"]["thresholds_modified"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_5b_for_i5_5c(payload)


def test_validator_rejects_early_i56_authorization(tmp_path: Path) -> None:
    payload = clone(passing_i5_5b(tmp_path))
    payload["decision"]["i5_6_authorized"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_5b_for_i5_5c(payload)


def test_receipt_rejects_sfm_rerun() -> None:
    payload = clone(passing_i5_5c_receipt())
    payload["scope"]["sfm_rerun"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_5c_receipt(payload)
