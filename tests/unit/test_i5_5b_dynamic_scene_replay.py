from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_dynamic_scene_handling import (
    I55B_PHASE,
    analyze_i5_5b_mapping_no_sparse_model_rejection,
    analyze_i5_5b_materialized_pair,
    analyze_i5_5b_pre_sfm_ingestion_rejection,
    analyze_i5_5b_sfm_report,
    build_i5_5b_dynamic_scene_replay,
    compare_i5_5b_sfm_outcomes,
    materialize_i5_5b_pair,
    validate_i5_5a_for_i5_5b,
    validate_i5_5b_receipt,
)
from tests.i5_5b_helpers import (
    I55A_SHA,
    RECEIPT_55B_SHA,
    binding,
    clone,
    image_source_dir,
    passing_i5_5a,
    passing_i5_5b_receipt,
    passing_materialization_revalidation,
    synthetic_i55b_analysis,
    synthetic_i55b_sfm_report,
)


def _build(tmp_path: Path, *, control: dict | None = None, stress: dict | None = None, materialization: dict | None = None) -> dict:
    admission = passing_i5_5a(tmp_path)
    control = control or synthetic_i55b_analysis(admission, "static_control")
    stress = stress or synthetic_i55b_analysis(admission, "dynamic_foreground_proxy")
    materialization = materialization or passing_materialization_revalidation()
    comparison = compare_i5_5b_sfm_outcomes(control=control, dynamic_proxy=stress, materialization_revalidation=materialization)
    return build_i5_5b_dynamic_scene_replay(
        i5_5a_result=admission,
        i5_5a_binding=binding("/evidence/i5_5a.json", I55A_SHA),
        replay_receipt=passing_i5_5b_receipt(),
        replay_receipt_binding=binding("/evidence/i5_5b_receipt.json", RECEIPT_55B_SHA),
        materialization_revalidation=materialization,
        control_analysis=control,
        dynamic_proxy_analysis=stress,
        paired_comparison=comparison,
        source_reverified_before=True,
        source_reverified_after=True,
        staged_inputs_reverified=True,
    )


def test_i55b_accepts_frozen_i55a_admission(tmp_path: Path) -> None:
    result = validate_i5_5a_for_i5_5b(passing_i5_5a(tmp_path))
    assert result["decision"]["i5_5b_real_replay_authorized"] is True
    assert result["stress_plan"]["dynamic_foreground_proxy"]["image_count"] == 19


def test_i55b_rejects_recipe_hash_drift(tmp_path: Path) -> None:
    admission = clone(passing_i5_5a(tmp_path))
    admission["stress_plan"]["dynamic_foreground_proxy"]["recipe_sha256"] = "0" * 64
    with pytest.raises(ReconstructionValidationError):
        validate_i5_5a_for_i5_5b(admission)


def test_i55b_receipt_requires_execution_free_test_gate() -> None:
    receipt = validate_i5_5b_receipt(passing_i5_5b_receipt())
    assert receipt["scope"]["real_sfm_executed_by_test_receipt"] is False
    bad = clone(passing_i5_5b_receipt())
    bad["scope"]["thresholds_modified"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_5b_receipt(bad)


def test_i55b_materialization_preserves_control_and_moves_exact_patch(tmp_path: Path) -> None:
    admission = passing_i5_5a(tmp_path)
    source = Path(admission["source_identity"]["directory"])
    out = tmp_path / "pair"
    evidence = materialize_i5_5b_pair(dataset_dir=source, admission=admission, output_dir=out)
    assert evidence["control"]["image_count"] == 19
    assert evidence["dynamic_proxy"]["image_count"] == 19
    diag = analyze_i5_5b_materialized_pair(evidence)
    assert diag["same_decoded_source_pixels_for_control_and_stress"] is True
    assert diag["control_pixel_values_preserved"] is True
    assert diag["donor_patch_pixel_identity_preserved"] is True
    assert diag["target_centroid_trajectory_matches_frozen_recipe"] is True
    assert diag["changed_pixels_confined_to_frozen_target_boxes"] is True
    assert diag["controlled_independent_foreground_motion_proxy"] is True


def test_i55b_materialization_is_feature_bearing_on_textured_proxy(tmp_path: Path) -> None:
    admission = passing_i5_5a(tmp_path)
    evidence = materialize_i5_5b_pair(
        dataset_dir=Path(admission["source_identity"]["directory"]),
        admission=admission,
        output_dir=tmp_path / "pair",
    )
    diag = analyze_i5_5b_materialized_pair(evidence)
    assert diag["donor_patch_orb_keypoints"] > 0
    assert diag["dynamic_region_feature_bearing_revalidated"] is True


def test_i55b_materialization_detects_outside_box_tamper(tmp_path: Path) -> None:
    admission = passing_i5_5a(tmp_path)
    evidence = materialize_i5_5b_pair(
        dataset_dir=Path(admission["source_identity"]["directory"]),
        admission=admission,
        output_dir=tmp_path / "pair",
    )
    first = Path(evidence["dynamic_proxy"]["directory"]) / evidence["dynamic_proxy"]["files"][0]["staged_name"]
    image = cv2.imread(str(first), cv2.IMREAD_COLOR)
    assert image is not None
    image[0, 0] = (image[0, 0].astype(np.uint16) + 1).astype(np.uint8)
    assert cv2.imwrite(str(first), image)
    diag = analyze_i5_5b_materialized_pair(evidence)
    assert diag["changed_pixels_confined_to_frozen_target_boxes"] is False


def test_i55b_sfm_analysis_preserves_component_count_and_authority(tmp_path: Path) -> None:
    admission = passing_i5_5a(tmp_path)
    plan = admission["stress_plan"]["static_control"]
    report = synthetic_i55b_sfm_report(admission, "static_control", registered_count=18)
    analysis = analyze_i5_5b_sfm_report(report, scenario_plan=plan, staged_names=[f"{i:04d}.png" for i in range(19)], sparse_component_count=2)
    assert analysis["registered_images"] == 18
    assert analysis["sparse_component_count"] == 2
    assert analysis["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"


def test_i55b_pre_sfm_ingestion_rejection_is_structured(tmp_path: Path) -> None:
    admission = passing_i5_5a(tmp_path)
    plan = admission["stress_plan"]["dynamic_foreground_proxy"]
    analysis = analyze_i5_5b_pre_sfm_ingestion_rejection(
        scenario_plan=plan,
        staged_names=[f"{i:04d}.png" for i in range(19)],
        reason="Fewer than three valid keyframes survived ingestion. Use a sharper, better exposed, overlapping sequence.",
    )
    assert analysis["quality_status"] == "rejected"
    assert analysis["rejection_stage"] == "ingestion"
    assert analysis["sfm_executed"] is False
    assert analysis["sparse_component_count"] == 0


def test_i55b_mapping_no_model_rejection_is_structured(tmp_path: Path) -> None:
    admission = passing_i5_5a(tmp_path)
    plan = admission["stress_plan"]["dynamic_foreground_proxy"]
    analysis = analyze_i5_5b_mapping_no_sparse_model_rejection(
        scenario_plan=plan,
        staged_names=[f"{i:04d}.png" for i in range(19)],
        reason="RuntimeError: PyCOLMAP incremental mapping produced no sparse model.",
    )
    assert analysis["quality_status"] == "rejected"
    assert analysis["rejection_stage"] == "incremental_mapping"
    assert analysis["mapping_produced_sparse_model"] is False
    assert analysis["sfm_executed"] is True


def test_i55b_comparison_is_scale_gauge_safe(tmp_path: Path) -> None:
    admission = passing_i5_5a(tmp_path)
    control = synthetic_i55b_analysis(admission, "static_control")
    stress = synthetic_i55b_analysis(admission, "dynamic_foreground_proxy", registered_count=15, quality_status="degraded", sparse_component_count=2)
    result = compare_i5_5b_sfm_outcomes(control=control, dynamic_proxy=stress, materialization_revalidation=passing_materialization_revalidation())
    assert result["cross_run_absolute_geometry_comparison_permitted"] is False
    assert result["registration_delta"] == -4
    assert result["sparse_component_count_delta"] == 1


def test_i55b_builds_pass_when_control_and_dynamic_proxy_are_accepted(tmp_path: Path) -> None:
    result = _build(tmp_path)
    assert result["phase"] == I55B_PHASE == "I5.5B"
    assert result["pass"] is True
    assert result["quality_status"] == "pass"
    assert result["qualification"]["dynamic_scene_robustness_conclusion"] == "pass_on_admitted_controlled_dynamic_foreground_proxy"
    assert result["decision"]["i5_5c_qualification_closure_authorized"] is True


def test_i55b_degrades_when_dynamic_proxy_quality_degrades(tmp_path: Path) -> None:
    admission = passing_i5_5a(tmp_path)
    stress = synthetic_i55b_analysis(admission, "dynamic_foreground_proxy", registered_count=14, quality_status="degraded", sparse_component_count=2)
    result = _build(tmp_path, stress=stress)
    assert result["quality_status"] == "degraded"
    assert result["qualification"]["dynamic_scene_robustness_conclusion"] == "degraded_on_admitted_controlled_dynamic_foreground_proxy"


def test_i55b_rejects_when_dynamic_proxy_is_rejected(tmp_path: Path) -> None:
    admission = passing_i5_5a(tmp_path)
    stress = synthetic_i55b_analysis(admission, "dynamic_foreground_proxy", registered_count=4, quality_status="rejected", sparse_component_count=3)
    result = _build(tmp_path, stress=stress)
    assert result["quality_status"] == "rejected"
    assert result["qualification"]["dynamic_scene_robustness_conclusion"] == "rejected_on_admitted_controlled_dynamic_foreground_proxy"


def test_i55b_rejects_conclusion_when_control_is_not_accepted(tmp_path: Path) -> None:
    admission = passing_i5_5a(tmp_path)
    control = synthetic_i55b_analysis(admission, "static_control", registered_count=10, quality_status="rejected", sparse_component_count=2)
    result = _build(tmp_path, control=control)
    assert result["quality_status"] == "rejected"
    assert result["qualification"]["dynamic_scene_robustness_conclusion"] == "scenario_control_not_qualified_for_dynamic_robustness_conclusion"


def test_i55b_marks_proxy_inconclusive_when_feature_revalidation_fails(tmp_path: Path) -> None:
    materialization = passing_materialization_revalidation()
    materialization["dynamic_region_feature_bearing_revalidated"] = False
    result = _build(tmp_path, materialization=materialization)
    assert result["quality_status"] == "degraded"
    assert result["qualification"]["dynamic_scene_robustness_conclusion"] == "inconclusive_controlled_dynamic_proxy_not_revalidated"


def test_i55b_requires_source_and_staged_reverification(tmp_path: Path) -> None:
    admission = passing_i5_5a(tmp_path)
    control = synthetic_i55b_analysis(admission, "static_control")
    stress = synthetic_i55b_analysis(admission, "dynamic_foreground_proxy")
    materialization = passing_materialization_revalidation()
    comparison = compare_i5_5b_sfm_outcomes(control=control, dynamic_proxy=stress, materialization_revalidation=materialization)
    with pytest.raises(ReconstructionValidationError):
        build_i5_5b_dynamic_scene_replay(
            i5_5a_result=admission,
            i5_5a_binding=binding("/evidence/i5_5a.json", I55A_SHA),
            replay_receipt=passing_i5_5b_receipt(),
            replay_receipt_binding=binding("/evidence/i5_5b_receipt.json", RECEIPT_55B_SHA),
            materialization_revalidation=materialization,
            control_analysis=control,
            dynamic_proxy_analysis=stress,
            paired_comparison=comparison,
            source_reverified_before=True,
            source_reverified_after=True,
            staged_inputs_reverified=False,
        )


def test_i55b_preserves_authority_and_roadmap_boundaries(tmp_path: Path) -> None:
    result = _build(tmp_path)
    assert result["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert result["authority"]["permanent_global_view_cap"] is None
    assert result["scope"]["dynamic_masking_added"] is False
    assert result["scope"]["semantic_segmentation_added"] is False
    assert result["decision"]["i5_6_authorized"] is False
    assert result["decision"]["mp_sfm_authorized"] is False
    assert result["decision"]["auto_policy_implemented"] is False
    assert result["decision"]["hybrid_authorized"] is False
