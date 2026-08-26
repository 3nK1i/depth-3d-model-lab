from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.mvs_visual_ab_closure import (
    MVS_VISUAL_AB_REQUIRED_VIEWS,
    MVS_VISUAL_AB_REVIEW_DIMENSIONS,
    build_a3_1_manual_closure,
    sha256_file,
    validate_a3_manual_review_evidence,
)


def _write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, (dict, list)):
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    else:
        path.write_bytes(bytes(payload))
    return path


def _fixture(tmp_path: Path) -> tuple[Path, dict]:
    a3 = tmp_path / "a3_visual_ab"
    renders = a3 / "renders"
    da3 = tmp_path / "da3"
    a2 = tmp_path / "a2"
    shared = tmp_path / "i4_1_7"

    i2_summary = _write(da3 / "02_visual_i2_quality/i2_summary.json", {"phase": "I2"})
    i4_summary = _write(da3 / "i4_1_9_summary.json", {"phase": "I4.1.9", "artifacts": {"i2_summary": str(i2_summary)}})
    da3_npz = _write(da3 / "02_visual_i2_quality/i2_extended_spatial_diagnostics.npz", b"da3-cloud")
    a2_summary = _write(a2 / "i4_1_9a_a2_summary.json", {"phase": "I4.1.9A-A2", "status": "pass"})
    fused = _write(a2 / "dense/fused.ply", b"ply\nend_header\n")
    a1_manifest = _write(a2 / "evidence/frozen_authority/a1_camera_authority_manifest.json", {"authority": "frozen"})
    shared_summary = _write(shared / "i4_1_7_summary.json", {"phase": "I4.1.7"})

    render_records: dict[str, dict] = {}
    for view in MVS_VISUAL_AB_REQUIRED_VIEWS:
        record: dict[str, object] = {"sha256": {}}
        for kind in ("da3", "mvs", "ab"):
            path = _write(renders / f"a3_{view}_{kind}.png", f"{view}-{kind}".encode())
            record[kind] = str(path)
            record["sha256"][kind] = sha256_file(path)
        render_records[view] = record

    summary = {
        "schema_version": 1,
        "phase": "I4.1.9A-A3",
        "status": "ready_for_manual_visual_ab_review",
        "comparison": {
            "left": "existing_DA3_I4.1.9_extended_RGB_geometry",
            "right": "A2_COLMAP_MVS_geometric_stereo_fusion",
            "same_house_input": True,
            "same_authoritative_i4_1_7_source": True,
            "same_authoritative_sfm_frame": True,
            "independent_auto_fit_forbidden": True,
            "shared_camera_parameters": True,
            "rendered_views": list(MVS_VISUAL_AB_REQUIRED_VIEWS),
            "manual_review_dimensions": list(MVS_VISUAL_AB_REVIEW_DIMENSIONS),
        },
        "authority": {
            "diagnostic_visual_comparison_only": True,
            "camera_geometry_modified": False,
            "sfm_rerun": False,
            "bundle_adjustment_rerun": False,
            "pose_refinement": False,
            "da3_rerun": False,
            "mvs_rerun": False,
            "patch_match_rerun": False,
            "stereo_fusion_rerun": False,
            "house_k_txt_consumed": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "dense_provider_promoted": False,
        },
        "decision": {
            "automatic_winner_selected": False,
            "manual_visual_verdict_required": True,
            "decision_matrix": {
                "mvs_clean_da3_smeared": "learned_branch",
                "mvs_similarly_smeared": "camera_branch",
                "mvs_better_but_imperfect": "characterize_residuals",
            },
        },
        "artifacts": {"renders": render_records},
        "source_artifacts": {
            "da3_i4_1_9_dir": str(da3),
            "da3_spatial_diagnostics_npz": str(da3_npz),
            "mvs_a2_dir": str(a2),
            "mvs_fused_ply": str(fused),
            "shared_i4_1_7_dir": str(shared),
        },
        "frozen_sha256": {
            "da3_i4_1_9_summary": sha256_file(i4_summary),
            "da3_i2_summary": sha256_file(i2_summary),
            "da3_spatial_diagnostics_npz": sha256_file(da3_npz),
            "mvs_a2_summary": sha256_file(a2_summary),
            "mvs_fused_ply": sha256_file(fused),
            "a1_camera_authority_manifest": sha256_file(a1_manifest),
            "shared_i4_1_7_summary": sha256_file(shared_summary),
        },
    }
    _write(a3 / "i4_1_9a_a3_summary.json", summary)
    return a3, summary


def test_a31_revalidates_all_four_views_and_frozen_sources(tmp_path: Path) -> None:
    a3, summary = _fixture(tmp_path)
    evidence = validate_a3_manual_review_evidence(summary, a3_dir=a3)
    assert evidence["controlled_comparison_revalidated"] is True
    assert evidence["required_views"] == list(MVS_VISUAL_AB_REQUIRED_VIEWS)
    assert len(evidence["source_sha256"]) == 7
    assert set(evidence["render_sha256"]) == set(MVS_VISUAL_AB_REQUIRED_VIEWS)


def test_a31_mvs_better_but_imperfect_closes_house_and_only_authorizes_i42(tmp_path: Path) -> None:
    a3, summary = _fixture(tmp_path)
    evidence = validate_a3_manual_review_evidence(summary, a3_dir=a3)
    closure = build_a3_1_manual_closure(
        summary,
        evidence_validation=evidence,
        verdict="mvs_better_but_imperfect",
        confirmed=True,
        a3_summary_sha256="a" * 64,
    )
    assert closure["status"] == "pass"
    assert closure["manual_review"]["verdict"] == "mvs_better_but_imperfect"
    assert closure["scientific_conclusion"]["mvs_materially_better_than_current_da3"] is True
    assert closure["scientific_conclusion"]["mvs_residual_errors_require_characterization"] is True
    assert closure["scientific_conclusion"]["authoritative_pycolmap_camera_solution_completely_exonerated"] is False
    assert closure["decision"]["i4_1_9a_closed"] is True
    assert closure["decision"]["i4_2_detail_preservation_authorized"] is True
    assert closure["decision"]["i4_3_provider_architecture_authorized"] is False
    assert closure["authority"]["dense_provider_promoted"] is False
    assert closure["authority"]["metric_scale_claimed"] is False


def test_a31_requires_explicit_manual_confirmation(tmp_path: Path) -> None:
    a3, summary = _fixture(tmp_path)
    evidence = validate_a3_manual_review_evidence(summary, a3_dir=a3)
    with pytest.raises(ReconstructionValidationError, match="explicit confirmation"):
        build_a3_1_manual_closure(
            summary,
            evidence_validation=evidence,
            verdict="mvs_better_but_imperfect",
            confirmed=False,
            a3_summary_sha256="a" * 64,
        )


def test_a31_rejects_unsupported_manual_verdict(tmp_path: Path) -> None:
    a3, summary = _fixture(tmp_path)
    evidence = validate_a3_manual_review_evidence(summary, a3_dir=a3)
    with pytest.raises(ReconstructionValidationError, match="Unsupported"):
        build_a3_1_manual_closure(
            summary,
            evidence_validation=evidence,
            verdict="mvs_perfect_forever",
            confirmed=True,
            a3_summary_sha256="a" * 64,
        )


@pytest.mark.parametrize("field,value", [("phase", "I4.1.9A-A2"), ("status", "pass")])
def test_a31_rejects_non_reviewable_a3_state(tmp_path: Path, field: str, value: str) -> None:
    a3, summary = _fixture(tmp_path)
    summary[field] = value
    with pytest.raises(ReconstructionValidationError):
        validate_a3_manual_review_evidence(summary, a3_dir=a3)


def test_a31_rejects_uncontrolled_camera_or_frame_comparison(tmp_path: Path) -> None:
    a3, summary = _fixture(tmp_path)
    summary["comparison"]["shared_camera_parameters"] = False
    with pytest.raises(ReconstructionValidationError, match="uncontrolled"):
        validate_a3_manual_review_evidence(summary, a3_dir=a3)


def test_a31_requires_all_four_review_views(tmp_path: Path) -> None:
    a3, summary = _fixture(tmp_path)
    summary["comparison"]["rendered_views"] = ["oblique", "front", "side"]
    with pytest.raises(ReconstructionValidationError, match="four controlled"):
        validate_a3_manual_review_evidence(summary, a3_dir=a3)


def test_a31_rejects_any_dense_provider_or_metric_authority_promotion(tmp_path: Path) -> None:
    a3, summary = _fixture(tmp_path)
    mutated = copy.deepcopy(summary)
    mutated["authority"]["dense_provider_promoted"] = True
    with pytest.raises(ReconstructionValidationError, match="dense_provider_promoted"):
        validate_a3_manual_review_evidence(mutated, a3_dir=a3)
    mutated = copy.deepcopy(summary)
    mutated["authority"]["measurement_authoritative"] = True
    with pytest.raises(ReconstructionValidationError, match="measurement_authoritative"):
        validate_a3_manual_review_evidence(mutated, a3_dir=a3)


def test_a31_similarly_smeared_verdict_blocks_i42_and_requires_camera_followup(tmp_path: Path) -> None:
    a3, summary = _fixture(tmp_path)
    evidence = validate_a3_manual_review_evidence(summary, a3_dir=a3)
    closure = build_a3_1_manual_closure(
        summary,
        evidence_validation=evidence,
        verdict="mvs_similarly_smeared",
        confirmed=True,
        a3_summary_sha256="b" * 64,
    )
    assert closure["decision"]["i4_2_detail_preservation_authorized"] is False
    assert closure["scientific_conclusion"]["camera_intrinsics_projection_frame_assumptions_primary_investigation_required"] is True
    assert closure["decision"]["mvs_static_object_orbit_provider_candidate_supported"] is False
