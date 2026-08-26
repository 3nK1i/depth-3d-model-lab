from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.temple_ring_closure import (
    I42F_REQUIRED_VIEWS,
    I42F_REVIEW_DIMENSIONS,
    build_i4_2f_closure,
    sha256_file,
    validate_i4_2f_evidence,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_bytes(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    c_dir = tmp_path / "c"
    d_dir = tmp_path / "d"
    e_dir = tmp_path / "e"
    p_dir = tmp_path / "preview"

    a_summary = _write_json(a_dir / "i4_2a_summary.json", {"phase": "I4.2A", "status": "pass", "pass": True})
    b_summary_payload = {
        "phase": "I4.2B", "status": "pass", "pass": True,
        "artifacts": {"frozen_i4_2a_summary": str(a_summary)},
    }
    b_summary = _write_json(b_dir / "i4_2b_summary.json", b_summary_payload)
    c_summary_payload = {
        "phase": "I4.2C", "status": "pass", "pass": True,
        "trajectory_quality": {"accepted": True, "status": "pass"},
    }
    c_summary = _write_json(c_dir / "i4_2c_summary.json", c_summary_payload)

    raw = _write_bytes(d_dir / "dense/fused.ply", b"ply-data")
    vis = _write_bytes(d_dir / "dense/fused.ply.vis", b"vis-data")
    d_summary_payload = {
        "phase": "I4.2D", "status": "pass", "pass": True,
        "authority": {
            "single_authoritative_sfm_frame": True,
            "raw_mvs_dense_reconstruction": True,
            "authoritative_camera_geometry_modified": False,
            "bundle_adjustment_rerun": False,
            "certified_metrology": False,
            "da3_started": False,
            "dense_provider_promoted": False,
            "held_out_ground_truth_consumed_by_i4_2d": False,
            "i4_3_provider_architecture_authorized": False,
            "measurement_authoritative": False,
            "metric_scale_claimed": False,
            "pose_refinement": False,
            "production_filter_policy_changed": False,
            "production_manifold_applied": False,
            "sfm_rerun": False,
            "surface_mesh_generated": False,
        },
        "input_authority": {
            "authoritative_sparse_unchanged": True,
            "staged_images_unchanged": True,
            "held_out_trajectory_validation_passed": True,
            "i4_2b_summary_sha256": sha256_file(b_summary),
            "i4_2c_summary_sha256": sha256_file(c_summary),
        },
        "artifacts": {
            "raw_fused_ply": str(raw),
            "frozen_i4_2b_summary": str(b_summary),
            "frozen_i4_2c_summary": str(c_summary),
        },
    }
    d_summary = _write_json(d_dir / "i4_2d_summary.json", d_summary_payload)

    artifacts = {}
    hashes = {}
    for key in ("analysis_npz", "d12_rejected_only_ply", "post_d12_extended_ply", "pre_d12_extended_ply", "support_npz"):
        suffix = ".npz" if key.endswith("npz") else ".ply"
        path = _write_bytes(e_dir / "artifacts" / f"{key}{suffix}", key.encode())
        artifacts[key] = str(path)
        hashes[key] = sha256_file(path)

    sparse_hashes = {"cameras.bin": "a" * 64, "images.bin": "b" * 64, "points3D.bin": "c" * 64}
    e_payload = {
        "phase": "I4.2E", "status": "ready_for_manual_review", "pass": True,
        "execution_integrity_pass": True,
        "artifact_sha256": hashes,
        "artifacts": artifacts,
        "authoritative_sparse_sha256_before": sparse_hashes,
        "authoritative_sparse_sha256_after": dict(sparse_hashes),
        "authority": {
            "d12_counterfactual_only": True,
            "native_i4_2d_fusion_visibility_consumed_for_support_evidence": True,
            "single_authoritative_sfm_frame_preserved": True,
            "authoritative_sparse_modified": False,
            "bundle_adjustment_rerun": False,
            "certified_metrology": False,
            "dense_provider_promoted": False,
            "held_out_ground_truth_consumed_by_filtering": False,
            "i4_3_provider_architecture_authorized": False,
            "measurement_authoritative": False,
            "metric_scale_claimed": False,
            "patch_match_rerun": False,
            "pose_refinement": False,
            "production_filter_policy_changed": False,
            "production_manifold_applied_to_authoritative_scene": False,
            "sfm_rerun": False,
            "stereo_fusion_replayed_for_support_evidence_only": False,
            "surface_mesh_generated": False,
        },
        "manual_review_contract": {
            "required": True,
            "automatic_winner_selected": False,
            "review_dimensions": list(I42F_REVIEW_DIMENSIONS),
        },
        "qualification": {
            "manual_visual_review_required": True,
            "automatic_detail_winner_selected": False,
            "production_policy_change_authorized": False,
            "i4_3_provider_architecture_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "potential_detail_loss_signal": True,
            "counts": {
                "pre_d12_extended_voxels": 1000,
                "post_d12_extended_voxels": 998,
                "d12_rejected_voxels": 2,
                "d12_rejected_fraction": 0.002,
            },
            "policy": {"scenario_name": "d12_sparse", "exact_production_predicate": True, "production_policy_changed": False},
            "populations": {"supported_line_like_proxy": {"population": 100, "rejected": 2}},
            "envelope": {"robust_diagonal_shrink_fraction": 0.001},
        },
        "source": {
            "i4_2d_dir": str(d_dir),
            "native_visibility_consumed": True,
            "native_visibility_full_view_coverage_required": False,
            "raw_fused_points": 2000,
            "visibility_observations": 12000,
            "visibility_support_min": 2,
            "raw_fused_ply": str(raw),
            "native_visibility_path": str(vis),
            "raw_fused_ply_sha256_before": sha256_file(raw),
            "raw_fused_ply_sha256_after": sha256_file(raw),
            "native_visibility_sha256_before": sha256_file(vis),
            "native_visibility_sha256_after": sha256_file(vis),
        },
    }
    e_summary = _write_json(e_dir / "i4_2e_summary.json", e_payload)

    renders = {}
    for view in I42F_REQUIRED_VIEWS:
        record = {"sha256": {}}
        for kind in ("pre_d12", "post_d12", "ab", "loss_overlay"):
            path = _write_bytes(p_dir / "renders" / f"{view}_{kind}.png", f"{view}-{kind}".encode())
            record[kind] = str(path)
            record["sha256"][kind] = sha256_file(path)
        renders[view] = record
    preview_payload = {
        "phase": "I4.2E-preview", "status": "ready_for_manual_review",
        "manual_visual_review_required": True,
        "same_camera_pre_post": True,
        "same_sample_indices_pre_post": True,
        "automatic_winner_selected": False,
        "independent_auto_fit": False,
        "production_filter_policy_changed": False,
        "i4_3_provider_architecture_authorized": False,
        "metric_scale_claimed": False,
        "measurement_authoritative": False,
        "surface_mesh_generated": False,
        "source_i4_2e_summary": str(e_summary),
        "source_i4_2e_summary_sha256": sha256_file(e_summary),
        "sampled_points": 1000,
        "sampled_d12_rejected": 2,
        "shared_framing": {"independent_auto_fit": False, "metric_scale_claimed": False},
        "renders": renders,
    }
    _write_json(p_dir / "i4_2e_preview_summary.json", preview_payload)
    return e_dir, p_dir


def test_i42f_revalidates_complete_evidence_chain_and_four_view_review(tmp_path: Path) -> None:
    e_dir, p_dir = _fixture(tmp_path)
    evidence = validate_i4_2f_evidence(i4_2e_dir=e_dir, preview_dir=p_dir)
    assert evidence["qualification"]["counts"]["d12_rejected_voxels"] == 2
    assert set(evidence["preview"]["render_sha256"]) == set(I42F_REQUIRED_VIEWS)
    assert set(evidence["upstream"]["summary_sha256"]) == {"i4_2a_summary", "i4_2b_summary", "i4_2c_summary", "i4_2d_summary"}


def test_i42f_safe_verdict_closes_i42_and_only_authorizes_i43(tmp_path: Path) -> None:
    e_dir, p_dir = _fixture(tmp_path)
    evidence = validate_i4_2f_evidence(i4_2e_dir=e_dir, preview_dir=p_dir)
    closure = build_i4_2f_closure(evidence_validation=evidence, verdict="d12_safe_on_temple_ring", confirmed=True)
    assert closure["pass"] is True
    assert closure["decision"]["i4_2_closed"] is True
    assert closure["decision"]["i4_3_provider_architecture_authorized"] is True
    assert closure["decision"]["dense_provider_promoted"] is False
    assert closure["decision"]["production_d12_policy_change_authorized"] is False
    assert closure["authority"]["metric_scale_claimed"] is False


def test_i42f_requires_explicit_manual_confirmation(tmp_path: Path) -> None:
    e_dir, p_dir = _fixture(tmp_path)
    evidence = validate_i4_2f_evidence(i4_2e_dir=e_dir, preview_dir=p_dir)
    with pytest.raises(ReconstructionValidationError, match="explicit confirmation"):
        build_i4_2f_closure(evidence_validation=evidence, verdict="d12_safe_on_temple_ring", confirmed=False)


def test_i42f_detail_loss_verdict_blocks_i43_and_i42_closure(tmp_path: Path) -> None:
    e_dir, p_dir = _fixture(tmp_path)
    evidence = validate_i4_2f_evidence(i4_2e_dir=e_dir, preview_dir=p_dir)
    closure = build_i4_2f_closure(evidence_validation=evidence, verdict="d12_detail_loss_observed", confirmed=True)
    assert closure["pass"] is False
    assert closure["decision"]["i4_2_closed"] is False
    assert closure["decision"]["i4_3_provider_architecture_authorized"] is False
    assert closure["scientific_conclusion"]["temple_ring_driven_d12_policy_correction_required"] is True


def test_i42f_inconclusive_verdict_blocks_i43(tmp_path: Path) -> None:
    e_dir, p_dir = _fixture(tmp_path)
    evidence = validate_i4_2f_evidence(i4_2e_dir=e_dir, preview_dir=p_dir)
    closure = build_i4_2f_closure(evidence_validation=evidence, verdict="inconclusive", confirmed=True)
    assert closure["pass"] is False
    assert closure["decision"]["i4_3_provider_architecture_authorized"] is False


def test_i42f_preserves_original_detail_risk_signal_in_provenance(tmp_path: Path) -> None:
    e_dir, p_dir = _fixture(tmp_path)
    evidence = validate_i4_2f_evidence(i4_2e_dir=e_dir, preview_dir=p_dir)
    closure = build_i4_2f_closure(evidence_validation=evidence, verdict="d12_safe_on_temple_ring", confirmed=True)
    assert closure["manual_review"]["original_potential_detail_loss_signal"] is True
    assert closure["manual_review"]["automatic_signal_overwritten"] is False
    assert closure["manual_review"]["observations"]["rejected_geometry_interpretation"].startswith("detached")


def test_i42f_rejects_nonreviewable_e_state(tmp_path: Path) -> None:
    e_dir, p_dir = _fixture(tmp_path)
    path = e_dir / "i4_2e_summary.json"
    payload = json.loads(path.read_text())
    payload["status"] = "pass"
    _write_json(path, payload)
    with pytest.raises(ReconstructionValidationError, match="READY_FOR_MANUAL_REVIEW"):
        validate_i4_2f_evidence(i4_2e_dir=e_dir, preview_dir=p_dir)


def test_i42f_rejects_any_production_policy_change(tmp_path: Path) -> None:
    e_dir, p_dir = _fixture(tmp_path)
    path = e_dir / "i4_2e_summary.json"
    payload = json.loads(path.read_text())
    payload["authority"]["production_filter_policy_changed"] = True
    _write_json(path, payload)
    with pytest.raises(ReconstructionValidationError, match="production_filter_policy_changed"):
        validate_i4_2f_evidence(i4_2e_dir=e_dir, preview_dir=p_dir)


def test_i42f_rejects_mutated_review_render(tmp_path: Path) -> None:
    e_dir, p_dir = _fixture(tmp_path)
    preview = json.loads((p_dir / "i4_2e_preview_summary.json").read_text())
    path = Path(preview["renders"]["oblique"]["loss_overlay"])
    path.write_bytes(path.read_bytes() + b"mutated")
    with pytest.raises(ReconstructionValidationError, match="render hash mismatch"):
        validate_i4_2f_evidence(i4_2e_dir=e_dir, preview_dir=p_dir)


def test_i42f_rejects_preview_to_e_summary_hash_mismatch(tmp_path: Path) -> None:
    e_dir, p_dir = _fixture(tmp_path)
    path = p_dir / "i4_2e_preview_summary.json"
    payload = json.loads(path.read_text())
    payload["source_i4_2e_summary_sha256"] = "0" * 64
    _write_json(path, payload)
    with pytest.raises(ReconstructionValidationError, match="no longer matches"):
        validate_i4_2f_evidence(i4_2e_dir=e_dir, preview_dir=p_dir)


def test_i42f_rejects_mutated_raw_mvs_artifact(tmp_path: Path) -> None:
    e_dir, p_dir = _fixture(tmp_path)
    e = json.loads((e_dir / "i4_2e_summary.json").read_text())
    path = Path(e["source"]["raw_fused_ply"])
    path.write_bytes(path.read_bytes() + b"mutated")
    with pytest.raises(ReconstructionValidationError, match="raw fused.ply hash mismatch"):
        validate_i4_2f_evidence(i4_2e_dir=e_dir, preview_dir=p_dir)


def test_i42f_rejects_authoritative_sparse_hash_change(tmp_path: Path) -> None:
    e_dir, p_dir = _fixture(tmp_path)
    path = e_dir / "i4_2e_summary.json"
    payload = json.loads(path.read_text())
    payload["authoritative_sparse_sha256_after"]["images.bin"] = "f" * 64
    _write_json(path, payload)
    with pytest.raises(ReconstructionValidationError, match="unchanged authoritative sparse"):
        validate_i4_2f_evidence(i4_2e_dir=e_dir, preview_dir=p_dir)
