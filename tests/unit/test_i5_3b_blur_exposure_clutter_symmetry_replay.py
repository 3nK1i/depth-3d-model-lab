from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_blur_exposure_clutter_symmetry import (
    I53B_PHASE,
    analyze_i5_3b_image,
    analyze_i5_3b_image_set,
    analyze_i5_3b_sfm_report,
    analyze_i5_3b_pre_sfm_ingestion_rejection,
    apply_i5_3b_transform,
    build_i5_3b_blur_exposure_clutter_symmetry_replay,
    compare_i5_3b_image_domain,
    validate_i5_3a_for_i5_3b,
    validate_i5_3b_receipt,
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


def _analyses(admission: dict, *, quality: str = "accepted") -> dict[str, dict]:
    return {
        "control": synthetic_i53_sfm_analysis(admission, "control", quality_status=quality),
        "blur": synthetic_i53_sfm_analysis(admission, "blur_proxy"),
        "underexposure": synthetic_i53_sfm_analysis(admission, "underexposure_proxy"),
        "overexposure": synthetic_i53_sfm_analysis(admission, "overexposure_proxy"),
        "clutter": synthetic_i53_sfm_analysis(admission, "clutter_proxy"),
        "symmetry": synthetic_i53_sfm_analysis(admission, "symmetry_proxy"),
    }


def _build(tmp_path: Path, *, comparison: dict | None = None, analyses: dict | None = None) -> dict:
    admission = passing_i5_3a(tmp_path)
    return build_i5_3b_blur_exposure_clutter_symmetry_replay(
        i5_3a_result=admission,
        i5_3a_binding=_binding("/evidence/i5_3a.json", I53A_SHA),
        replay_receipt=passing_i5_3b_receipt(),
        replay_receipt_binding=_binding("/evidence/i5_3b_receipt.json", RECEIPT_53B_SHA),
        image_domain_comparison=comparison or passing_image_domain_comparison(),
        scenario_analyses=analyses or _analyses(admission),
        source_reverified_before=True,
        source_reverified_after=True,
        staged_inputs_reverified=True,
    )


def _pattern() -> np.ndarray:
    image = np.zeros((96, 128, 3), dtype=np.uint8)
    for x in range(0, 128, 8):
        cv2.line(image, (x, 0), (x, 95), (255, 255, 255), 2)
    cv2.putText(image, "I53B", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2, cv2.LINE_AA)
    return image


def test_i53b_accepts_frozen_i53a_handoff(tmp_path: Path) -> None:
    result = validate_i5_3a_for_i5_3b(passing_i5_3a(tmp_path))
    assert result["decision"]["i5_3a_complete"] is True
    assert result["decision"]["i5_3b_real_replay_authorized"] is True
    assert result["scope"]["stress_images_materialized"] is False


def test_i53b_rejects_i53a_semantic_authority_drift(tmp_path: Path) -> None:
    admission = clone(passing_i5_3a(tmp_path))
    admission["stress_plan"]["blur_proxy"]["semantic_stress_authoritative"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_3a_for_i5_3b(admission)


def test_i53b_blur_transform_reduces_laplacian_variance() -> None:
    image = _pattern()
    stress = apply_i5_3b_transform(image, {"algorithm": "gaussian_blur", "kernel_size": 13, "sigma": 3.0, "border_mode": "reflect_101"})
    assert analyze_i5_3b_image(stress, band_fraction=0.18)["blur_variance"] < analyze_i5_3b_image(image, band_fraction=0.18)["blur_variance"]


def test_i53b_exposure_transforms_are_directional() -> None:
    image = _pattern()
    under = apply_i5_3b_transform(image, {"algorithm": "linear_intensity", "gain": 0.35, "bias": 0.0, "clip_uint8": True})
    over = apply_i5_3b_transform(image, {"algorithm": "linear_intensity", "gain": 1.55, "bias": 32.0, "clip_uint8": True})
    base = analyze_i5_3b_image(image, band_fraction=0.18)
    assert analyze_i5_3b_image(under, band_fraction=0.18)["mean_luma"] < base["mean_luma"]
    assert analyze_i5_3b_image(over, band_fraction=0.18)["mean_luma"] > base["mean_luma"]


def test_i53b_clutter_transform_injects_offset_bottom_band() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    distractor = np.full((100, 100, 3), 220, dtype=np.uint8)
    stress = apply_i5_3b_transform(
        image,
        {"algorithm": "source_derived_distractor_band", "band_fraction": 0.18, "band_location": "bottom", "resize_mode": "cover_then_center_crop"},
        distractor_image=distractor,
    )
    assert int(stress[-10:].mean()) > 200
    assert int(stress[:50].mean()) == 0


def test_i53b_symmetry_transform_creates_mirrored_side_bands() -> None:
    image = _pattern()
    stress = apply_i5_3b_transform(image, {"algorithm": "mirrored_repeated_side_bands", "band_fraction": 0.18, "source_region": "center_vertical_band", "placements": ["left", "right"], "right_band_mirrored": True})
    diag = analyze_i5_3b_image(stress, band_fraction=0.18)
    assert diag["mirrored_side_similarity"] > 0.98




def test_i53b_image_set_aggregates_separate_diagnostic_band_fractions(tmp_path: Path) -> None:
    root = tmp_path / "images"
    root.mkdir()
    names = []
    for index in range(3):
        name = f"{index:04d}.png"
        names.append(name)
        assert cv2.imwrite(str(root / name), _pattern())
    result = analyze_i5_3b_image_set(
        root,
        names=names,
        band_fraction=0.18,
        symmetry_band_fraction=0.22,
    )
    assert result["image_count"] == 3
    assert result["diagnostic_band_fraction"] == pytest.approx(0.18)
    assert result["symmetry_diagnostic_band_fraction"] == pytest.approx(0.22)

def test_i53b_image_domain_comparison_revalidates_all_stresses() -> None:
    control = {"blur_variance_p50": 100.0, "mean_luma_p50": 120.0, "underexposed_fraction_p50": 0.01, "overexposed_fraction_p50": 0.01, "quality_score_p50": 0.8, "bottom_band_orb_feature_density_p50": 1000.0, "mirrored_side_similarity_p50": 0.3}
    stresses = {
        "blur": {**control, "blur_variance_p50": 25.0},
        "underexposure": {**control, "mean_luma_p50": 40.0, "underexposed_fraction_p50": 0.2, "quality_score_p50": 0.4},
        "overexposure": {**control, "mean_luma_p50": 210.0, "overexposed_fraction_p50": 0.25, "quality_score_p50": 0.5},
        "clutter": {**control, "bottom_band_orb_feature_density_p50": 1800.0},
        "symmetry": {**control, "mirrored_side_similarity_p50": 0.98},
    }
    result = compare_i5_3b_image_domain(control=control, stresses=stresses)
    assert result["all_semantic_stresses_revalidated"] is True


def test_i53b_image_domain_comparison_preserves_failed_proxy_revalidation() -> None:
    comparison = passing_image_domain_comparison()
    comparison["clutter"]["stress_bottom_band_orb_feature_density_p50"] = 900.0
    # Recompute from aggregates rather than trusting caller-provided booleans.
    control = {"blur_variance_p50": 100.0, "mean_luma_p50": 120.0, "underexposed_fraction_p50": 0.01, "overexposed_fraction_p50": 0.01, "quality_score_p50": 0.8, "bottom_band_orb_feature_density_p50": 1000.0, "mirrored_side_similarity_p50": 0.3}
    stresses = {
        "blur": {**control, "blur_variance_p50": 20.0},
        "underexposure": {**control, "mean_luma_p50": 40.0, "underexposed_fraction_p50": 0.1, "quality_score_p50": 0.5},
        "overexposure": {**control, "mean_luma_p50": 220.0, "overexposed_fraction_p50": 0.2, "quality_score_p50": 0.5},
        "clutter": {**control, "bottom_band_orb_feature_density_p50": 900.0},
        "symmetry": {**control, "mirrored_side_similarity_p50": 0.95},
    }
    result = compare_i5_3b_image_domain(control=control, stresses=stresses)
    assert result["clutter"]["semantic_stress_revalidated"] is False
    assert result["all_semantic_stresses_revalidated"] is False




def test_i53b_preserves_pre_sfm_ingestion_rejection_as_structured_evidence(tmp_path: Path) -> None:
    admission = passing_i5_3a(tmp_path)
    plan = admission["stress_plan"]["blur_proxy"]
    staged = [f"{index:04d}.png" for index in range(plan["image_count"])]
    analysis = analyze_i5_3b_pre_sfm_ingestion_rejection(
        scenario_plan=plan,
        staged_names=staged,
        reason="Fewer than three valid keyframes survived ingestion. Use a sharper, better exposed, overlapping sequence.",
    )
    assert analysis["quality_status"] == "rejected"
    assert analysis["input_images"] == 19
    assert analysis["registered_images"] == 0
    assert analysis["registration_attempted"] is False
    assert analysis["pre_sfm_ingestion_rejected"] is True
    assert analysis["rejection_stage"] == "ingestion"
    assert analysis["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"

def test_i53b_sfm_analysis_reuses_authoritative_pycolmap_contract(tmp_path: Path) -> None:
    admission = passing_i5_3a(tmp_path)
    plan = admission["stress_plan"]["blur_proxy"]
    # Synthetic helper already passes through the I5.2B authoritative analyzer.
    analysis = synthetic_i53_sfm_analysis(admission, "blur_proxy", registered_count=18)
    assert analysis["registered_images"] == 18
    assert analysis["partial_registration_preserved"] is True
    assert analysis["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"


def test_i53b_builds_pass_when_stress_is_revalidated_and_sfm_accepted(tmp_path: Path) -> None:
    result = _build(tmp_path)
    assert result["phase"] == I53B_PHASE == "I5.3B"
    assert result["pass"] is True
    assert result["quality_status"] == "pass"
    assert result["decision"]["i5_3c_qualification_closure_authorized"] is True
    assert result["decision"]["i5_3_complete"] is False


def test_i53b_degrades_when_semantic_proxy_does_not_materialize(tmp_path: Path) -> None:
    comparison = passing_image_domain_comparison()
    comparison["clutter"]["stress_bottom_band_orb_feature_density_p50"] = 900.0
    comparison["clutter"]["stress_to_control_ratio"] = 0.9
    comparison["clutter"]["semantic_stress_revalidated"] = False
    comparison["all_semantic_stresses_revalidated"] = False
    result = _build(tmp_path, comparison=comparison)
    assert result["quality_status"] == "degraded"
    assert result["image_domain_revalidation"]["clutter"]["semantic_stress_revalidated"] is False
    assert result["provenance"]["retry_until_quality_pass"] is False


def test_i53b_degrades_when_one_stress_sfm_is_rejected(tmp_path: Path) -> None:
    admission = passing_i5_3a(tmp_path)
    analyses = _analyses(admission)
    analyses["blur"] = synthetic_i53_sfm_analysis(admission, "blur_proxy", quality_status="rejected", registered_count=5)
    result = _build(tmp_path, analyses=analyses)
    assert result["quality_status"] == "degraded"


def test_i53b_rejects_overall_when_control_sfm_is_rejected(tmp_path: Path) -> None:
    admission = passing_i5_3a(tmp_path)
    analyses = _analyses(admission)
    analyses["control"] = synthetic_i53_sfm_analysis(admission, "control", quality_status="rejected", registered_count=5)
    result = _build(tmp_path, analyses=analyses)
    assert result["quality_status"] == "rejected"


def test_i53b_requires_source_and_staged_input_reverification(tmp_path: Path) -> None:
    admission = passing_i5_3a(tmp_path)
    kwargs = dict(
        i5_3a_result=admission,
        i5_3a_binding=_binding("/evidence/i5_3a.json", I53A_SHA),
        replay_receipt=passing_i5_3b_receipt(),
        replay_receipt_binding=_binding("/evidence/i5_3b_receipt.json", RECEIPT_53B_SHA),
        image_domain_comparison=passing_image_domain_comparison(),
        scenario_analyses=_analyses(admission),
        source_reverified_before=True,
        source_reverified_after=True,
        staged_inputs_reverified=False,
    )
    with pytest.raises(ReconstructionValidationError):
        build_i5_3b_blur_exposure_clutter_symmetry_replay(**kwargs)


def test_i53b_receipt_requires_execution_free_test_gate() -> None:
    receipt = validate_i5_3b_receipt(passing_i5_3b_receipt())
    assert receipt["scope"]["real_sfm_executed_by_test_receipt"] is False
    bad = clone(passing_i5_3b_receipt())
    bad["scope"]["thresholds_modified"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_3b_receipt(bad)


def test_i53b_preserves_authority_and_roadmap_boundaries(tmp_path: Path) -> None:
    result = _build(tmp_path)
    assert result["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert result["authority"]["permanent_global_view_cap"] is None
    assert result["decision"]["i5_4_authorized"] is False
    assert result["decision"]["mp_sfm_authorized"] is False
    assert result["decision"]["auto_policy_implemented"] is False
    assert result["decision"]["hybrid_authorized"] is False
    assert result["scope"]["dense_provider_rerun"] is False
