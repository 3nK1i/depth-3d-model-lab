from __future__ import annotations

from pathlib import Path

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_low_overlap_weak_parallax import (
    I52B_PHASE,
    analyze_i5_2b_sfm_report,
    build_i5_2b_low_overlap_weak_parallax_replay,
    compare_i5_2b_stress_geometry,
    validate_i5_2a_for_i5_2b,
    validate_i5_2b_receipt,
)
from tests.i5_2b_helpers import (
    I52A_SHA,
    RECEIPT_52B_SHA,
    clone,
    passing_i5_2a,
    passing_i5_2b_receipt,
    synthetic_sfm_report,
)


def _binding(path: str, sha: str) -> I51EvidenceBinding:
    return I51EvidenceBinding(path, sha)


def _analyses(tmp_path: Path):
    admission = passing_i5_2a(tmp_path)
    control = analyze_i5_2b_sfm_report(
        synthetic_sfm_report(admission, "control", camera_step=1.0, common_landmarks_per_adjacent_pair=40),
        scenario_plan=admission["stress_plan"]["control"],
    )
    low = analyze_i5_2b_sfm_report(
        synthetic_sfm_report(admission, "low_overlap_proxy", camera_step=2.0, common_landmarks_per_adjacent_pair=12),
        scenario_plan=admission["stress_plan"]["low_overlap_proxy"],
    )
    weak = analyze_i5_2b_sfm_report(
        synthetic_sfm_report(admission, "weak_parallax_proxy", camera_step=0.18, common_landmarks_per_adjacent_pair=35),
        scenario_plan=admission["stress_plan"]["weak_parallax_proxy"],
    )
    return admission, control, low, weak


def test_i52b_validates_frozen_i52a_boundary(tmp_path: Path) -> None:
    admission = validate_i5_2a_for_i5_2b(passing_i5_2a(tmp_path))
    assert admission["decision"]["i5_2a_complete"] is True
    assert admission["decision"]["i5_2b_real_replay_authorized"] is True
    assert admission["decision"]["i5_2_complete"] is False


def test_i52b_rejects_drifted_i52a_authority(tmp_path: Path) -> None:
    payload = clone(passing_i5_2a(tmp_path))
    payload["authority"]["permanent_global_view_cap"] = 19
    with pytest.raises(ReconstructionValidationError):
        validate_i5_2a_for_i5_2b(payload)


def test_i52b_sfm_analysis_uses_only_real_arbitrary_scale_pycolmap_evidence(tmp_path: Path) -> None:
    admission, control, _, _ = _analyses(tmp_path)
    assert control["input_images"] == admission["stress_plan"]["control"]["image_count"]
    assert control["registered_images"] == control["input_images"]
    assert control["quality_status"] == "accepted"
    assert control["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert control["authority"]["metric_scale_claimed"] is False
    assert control["geometry"]["parallax"]["scale_invariant"] is True
    assert control["geometry"]["covisibility"]["registered_graph_connected"] is True


@pytest.mark.parametrize("field,value", [("real_sfm_executed", False), ("phase", "mock")])
def test_i52b_rejects_non_real_sfm_report(tmp_path: Path, field: str, value) -> None:
    admission = passing_i5_2a(tmp_path)
    report = synthetic_sfm_report(admission, "control", camera_step=1.0, common_landmarks_per_adjacent_pair=20)
    report[field] = value
    with pytest.raises(ReconstructionValidationError):
        analyze_i5_2b_sfm_report(report, scenario_plan=admission["stress_plan"]["control"])


def test_i52b_rejects_non_exhaustive_matching(tmp_path: Path) -> None:
    admission = passing_i5_2a(tmp_path)
    report = synthetic_sfm_report(admission, "control", camera_step=1.0, common_landmarks_per_adjacent_pair=20)
    report["sfm_result"]["metadata"]["matching_strategy"] = "sequential"
    with pytest.raises(ReconstructionValidationError):
        analyze_i5_2b_sfm_report(report, scenario_plan=admission["stress_plan"]["control"])


def test_i52b_authoritatively_confirms_directional_stress_effects(tmp_path: Path) -> None:
    _, control, low, weak = _analyses(tmp_path)
    comparison = compare_i5_2b_stress_geometry(control=control, low_overlap=low, weak_parallax=weak)
    assert comparison["low_overlap"]["authoritative_covisibility_reduction_confirmed"] is True
    assert comparison["weak_parallax"]["authoritative_parallax_reduction_confirmed"] is True
    assert comparison["weak_parallax"]["authoritative_baseline_to_range_reduction_confirmed"] is True
    assert comparison["semantic_stress_revalidation_complete"] is True


def test_i52b_does_not_force_stress_label_when_geometry_does_not_support_it(tmp_path: Path) -> None:
    admission, control, _, _ = _analyses(tmp_path)
    not_low = analyze_i5_2b_sfm_report(
        synthetic_sfm_report(admission, "low_overlap_proxy", camera_step=1.0, common_landmarks_per_adjacent_pair=50),
        scenario_plan=admission["stress_plan"]["low_overlap_proxy"],
    )
    not_weak = analyze_i5_2b_sfm_report(
        synthetic_sfm_report(admission, "weak_parallax_proxy", camera_step=1.5, common_landmarks_per_adjacent_pair=40),
        scenario_plan=admission["stress_plan"]["weak_parallax_proxy"],
    )
    comparison = compare_i5_2b_stress_geometry(control=control, low_overlap=not_low, weak_parallax=not_weak)
    assert comparison["low_overlap"]["authoritative_covisibility_reduction_confirmed"] is False
    assert comparison["weak_parallax"]["authoritative_parallax_reduction_confirmed"] is False
    assert comparison["semantic_stress_revalidation_complete"] is False


def test_i52b_receipt_must_be_green_without_real_sfm_claim() -> None:
    receipt = validate_i5_2b_receipt(passing_i5_2b_receipt())
    assert receipt["scope"]["real_sfm_executed_by_test_receipt"] is False
    bad = clone(passing_i5_2b_receipt())
    bad["scope"]["thresholds_modified"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_2b_receipt(bad)


def test_i52b_builds_evidence_complete_replay_and_only_authorizes_i52c(tmp_path: Path) -> None:
    admission, control, low, weak = _analyses(tmp_path)
    result = build_i5_2b_low_overlap_weak_parallax_replay(
        i5_2a_result=admission,
        i5_2a_binding=_binding("/evidence/i5_2a.json", I52A_SHA),
        replay_receipt=passing_i5_2b_receipt(),
        replay_receipt_binding=_binding("/evidence/i5_2b_receipt.json", RECEIPT_52B_SHA),
        control_analysis=control,
        low_overlap_analysis=low,
        weak_parallax_analysis=weak,
        source_reverified_before=True,
        source_reverified_after=True,
    )
    assert result["phase"] == I52B_PHASE == "I5.2B"
    assert result["status"] == "low_overlap_weak_parallax_replay_evidence_complete"
    assert result["pass"] is True
    assert result["quality_status"] == "pass"
    assert result["decision"]["i5_2c_qualification_closure_authorized"] is True
    assert result["decision"]["i5_2_complete"] is False
    assert result["decision"]["i5_3_authorized"] is False
    assert result["next"] == "I5.2C low-overlap / weak-parallax robustness qualification and closure."


def test_i52b_legitimate_unconfirmed_stressor_is_degraded_not_retried_until_pass(tmp_path: Path) -> None:
    admission, control, low, _ = _analyses(tmp_path)
    weak = analyze_i5_2b_sfm_report(
        synthetic_sfm_report(admission, "weak_parallax_proxy", camera_step=1.5, common_landmarks_per_adjacent_pair=35),
        scenario_plan=admission["stress_plan"]["weak_parallax_proxy"],
    )
    result = build_i5_2b_low_overlap_weak_parallax_replay(
        i5_2a_result=admission,
        i5_2a_binding=_binding("/evidence/i5_2a.json", I52A_SHA),
        replay_receipt=passing_i5_2b_receipt(),
        replay_receipt_binding=_binding("/evidence/i5_2b_receipt.json", RECEIPT_52B_SHA),
        control_analysis=control,
        low_overlap_analysis=low,
        weak_parallax_analysis=weak,
        source_reverified_before=True,
        source_reverified_after=True,
    )
    assert result["quality_status"] == "degraded"
    assert result["provenance"]["retry_until_quality_pass"] is False
    assert result["decision"]["i5_2c_qualification_closure_authorized"] is True
