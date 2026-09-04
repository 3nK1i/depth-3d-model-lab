from __future__ import annotations

from pathlib import Path

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_forward_motion_kitti import (
    I54B_PHASE,
    analyze_i5_4b_pre_sfm_ingestion_rejection,
    analyze_i5_4b_mapping_no_sparse_model_rejection,
    analyze_i5_4b_sfm_report,
    build_i5_4b_forward_motion_kitti_replay,
    validate_i5_4a_for_i5_4b,
    validate_i5_4b_receipt,
)
from tests.i5_4b_helpers import (
    I54A_SHA,
    RECEIPT_54B_SHA,
    clone,
    passing_i5_4a,
    passing_i5_4b_receipt,
    synthetic_i54b_sfm_report,
)


def _binding(path: str, sha: str) -> I51EvidenceBinding:
    return I51EvidenceBinding(path, sha)


def _build(tmp_path: Path, *, analysis: dict | None = None) -> dict:
    admission = passing_i5_4a(tmp_path)
    if analysis is None:
        analysis = analyze_i5_4b_sfm_report(synthetic_i54b_sfm_report(admission), scenario_plan=admission["stress_plan"]["forward_motion_window"])
    return build_i5_4b_forward_motion_kitti_replay(
        i5_4a_result=admission,
        i5_4a_binding=_binding("/evidence/i5_4a.json", I54A_SHA),
        replay_receipt=passing_i5_4b_receipt(),
        replay_receipt_binding=_binding("/evidence/i5_4b_receipt.json", RECEIPT_54B_SHA),
        sfm_analysis=analysis,
        source_reverified_before=True,
        source_window_reverified_after=True,
        staged_inputs_reverified=True,
    )


def test_i54b_accepts_frozen_i54a_admission(tmp_path: Path) -> None:
    result = validate_i5_4a_for_i5_4b(passing_i5_4a(tmp_path))
    assert result["decision"]["i5_4b_real_replay_authorized"] is True
    assert result["stress_plan"]["forward_motion_window"]["image_count"] == 60


def test_i54b_rejects_result_driven_window_tuning_drift(tmp_path: Path) -> None:
    admission = clone(passing_i5_4a(tmp_path))
    admission["stress_plan"]["result_driven_window_tuning"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_4a_for_i5_4b(admission)


def test_i54b_receipt_requires_execution_free_test_gate() -> None:
    receipt = validate_i5_4b_receipt(passing_i5_4b_receipt())
    assert receipt["scope"]["real_sfm_executed_by_test_receipt"] is False
    bad = clone(passing_i5_4b_receipt())
    bad["scope"]["thresholds_modified"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_4b_receipt(bad)


def test_i54b_analysis_requires_sequential_matching(tmp_path: Path) -> None:
    admission = passing_i5_4a(tmp_path)
    report = synthetic_i54b_sfm_report(admission, matching_strategy="exhaustive")
    with pytest.raises(ReconstructionValidationError):
        analyze_i5_4b_sfm_report(report, scenario_plan=admission["stress_plan"]["forward_motion_window"])


def test_i54b_analysis_revalidates_forward_axis_motion(tmp_path: Path) -> None:
    admission = passing_i5_4a(tmp_path)
    analysis = analyze_i5_4b_sfm_report(
        synthetic_i54b_sfm_report(admission, forward_step=0.25, lateral_step=0.02),
        scenario_plan=admission["stress_plan"]["forward_motion_window"],
    )
    motion = analysis["geometry"]["camera_motion"]
    assert motion["authoritative_forward_motion_geometry_proven"] is True
    assert motion["median_forward_component_fraction"] > motion["median_lateral_component_fraction"]
    assert motion["adjacent_registered_motion_pairs"] == 59


def test_i54b_analysis_preserves_nonforward_proxy_as_inconclusive(tmp_path: Path) -> None:
    admission = passing_i5_4a(tmp_path)
    analysis = analyze_i5_4b_sfm_report(
        synthetic_i54b_sfm_report(admission, forward_step=0.02, lateral_step=0.25),
        scenario_plan=admission["stress_plan"]["forward_motion_window"],
    )
    motion = analysis["geometry"]["camera_motion"]
    assert motion["authoritative_forward_motion_geometry_proven"] is False
    assert motion["forward_motion_label"] == "inconclusive_forward_motion_proxy_not_revalidated"


def test_i54b_analysis_reports_scale_invariant_i24_geometry(tmp_path: Path) -> None:
    admission = passing_i5_4a(tmp_path)
    analysis = analyze_i5_4b_sfm_report(synthetic_i54b_sfm_report(admission), scenario_plan=admission["stress_plan"]["forward_motion_window"])
    parallax = analysis["geometry"]["parallax"]
    assert parallax["scale_invariant"] is True
    assert parallax["absolute_baseline_cross_scenario_comparison_permitted"] is False
    assert parallax["maximum_pair_parallax_degrees_quantiles"]["p50"] is not None
    assert parallax["maximum_baseline_to_range_ratio_quantiles"]["p50"] is not None


def test_i54b_analysis_preserves_partial_registration(tmp_path: Path) -> None:
    admission = passing_i5_4a(tmp_path)
    analysis = analyze_i5_4b_sfm_report(synthetic_i54b_sfm_report(admission, registered_count=52), scenario_plan=admission["stress_plan"]["forward_motion_window"])
    assert analysis["registered_images"] == 52
    assert analysis["partial_registration_preserved"] is True


def test_i54b_pre_sfm_ingestion_rejection_is_structured(tmp_path: Path) -> None:
    admission = passing_i5_4a(tmp_path)
    plan = admission["stress_plan"]["forward_motion_window"]
    analysis = analyze_i5_4b_pre_sfm_ingestion_rejection(
        scenario_plan=plan,
        staged_names=list(plan["names"]),
        reason="Fewer than three valid keyframes survived ingestion. Use a sharper, better exposed, overlapping sequence.",
    )
    assert analysis["quality_status"] == "rejected"
    assert analysis["registered_images"] == 0
    assert analysis["pre_sfm_ingestion_rejected"] is True
    assert analysis["geometry"]["camera_motion"]["authoritative_forward_motion_geometry_proven"] is False


def test_i54b_incremental_mapping_no_model_rejection_is_structured(tmp_path: Path) -> None:
    admission = passing_i5_4a(tmp_path)
    plan = admission["stress_plan"]["forward_motion_window"]
    analysis = analyze_i5_4b_mapping_no_sparse_model_rejection(
        scenario_plan=plan,
        staged_names=list(plan["names"]),
        reason="RuntimeError: PyCOLMAP incremental mapping produced no sparse model.",
    )
    assert analysis["quality_status"] == "rejected"
    assert analysis["registered_images"] == 0
    assert analysis["pre_sfm_ingestion_rejected"] is False
    assert analysis["rejection_stage"] == "incremental_mapping"
    assert analysis["mapping_produced_sparse_model"] is False
    assert analysis["registration_attempted"] is True
    assert analysis["sfm_executed"] is True
    assert analysis["geometry"]["camera_motion"]["authoritative_forward_motion_geometry_proven"] is False


def test_i54b_builds_pass_for_full_registered_revalidated_forward_motion(tmp_path: Path) -> None:
    result = _build(tmp_path)
    assert result["phase"] == I54B_PHASE == "I5.4B"
    assert result["pass"] is True
    assert result["quality_status"] == "pass"
    assert result["forward_motion_revalidation"]["authoritative_forward_motion_geometry_proven"] is True
    assert result["decision"]["i5_4c_qualification_closure_authorized"] is True


def test_i54b_degrades_partial_registration(tmp_path: Path) -> None:
    admission = passing_i5_4a(tmp_path)
    analysis = analyze_i5_4b_sfm_report(synthetic_i54b_sfm_report(admission, registered_count=52), scenario_plan=admission["stress_plan"]["forward_motion_window"])
    result = _build(tmp_path, analysis=analysis)
    assert result["quality_status"] == "degraded"
    assert result["replay"]["partial_registration_preserved"] is True


def test_i54b_degrades_when_forward_geometry_does_not_revalidate(tmp_path: Path) -> None:
    admission = passing_i5_4a(tmp_path)
    analysis = analyze_i5_4b_sfm_report(synthetic_i54b_sfm_report(admission, forward_step=0.01, lateral_step=0.3), scenario_plan=admission["stress_plan"]["forward_motion_window"])
    result = _build(tmp_path, analysis=analysis)
    assert result["quality_status"] == "degraded"
    assert result["qualification"]["forward_motion_robustness_conclusion"] == "inconclusive_forward_motion_proxy_not_revalidated"


def test_i54b_rejects_quality_when_sfm_report_is_rejected(tmp_path: Path) -> None:
    admission = passing_i5_4a(tmp_path)
    analysis = analyze_i5_4b_sfm_report(synthetic_i54b_sfm_report(admission, registered_count=10, quality_status="rejected"), scenario_plan=admission["stress_plan"]["forward_motion_window"])
    result = _build(tmp_path, analysis=analysis)
    assert result["quality_status"] == "rejected"


def test_i54b_requires_all_reverification_guards(tmp_path: Path) -> None:
    admission = passing_i5_4a(tmp_path)
    analysis = analyze_i5_4b_sfm_report(synthetic_i54b_sfm_report(admission), scenario_plan=admission["stress_plan"]["forward_motion_window"])
    with pytest.raises(ReconstructionValidationError):
        build_i5_4b_forward_motion_kitti_replay(
            i5_4a_result=admission,
            i5_4a_binding=_binding("/evidence/i5_4a.json", I54A_SHA),
            replay_receipt=passing_i5_4b_receipt(),
            replay_receipt_binding=_binding("/evidence/i5_4b_receipt.json", RECEIPT_54B_SHA),
            sfm_analysis=analysis,
            source_reverified_before=True,
            source_window_reverified_after=True,
            staged_inputs_reverified=False,
        )


def test_i54b_preserves_authority_and_roadmap_boundaries(tmp_path: Path) -> None:
    result = _build(tmp_path)
    assert result["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert result["authority"]["permanent_global_view_cap"] is None
    assert result["decision"]["i5_5_authorized"] is False
    assert result["decision"]["mp_sfm_authorized"] is False
    assert result["decision"]["auto_policy_implemented"] is False
    assert result["decision"]["hybrid_authorized"] is False
    assert result["scope"]["dense_provider_rerun"] is False
