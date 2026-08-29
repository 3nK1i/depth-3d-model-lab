from __future__ import annotations

from copy import deepcopy

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.temporal_provider_production_qualification import (
    build_g3b_runtime_preflight_snapshot,
    resolve_g3b_scenario_binding,
    validate_g3b_predecessor,
    validate_temporal_production_plan,
    validate_temporal_provider_summary_authority,
    validate_temporal_provider_run_evidence,
)
from tests.i4_3g3b_helpers import (
    current_rolling_runtime,
    current_vda_runtime,
    passing_g3a_snapshot,
    production_plan,
    provider_evidence,
    sfm_runtime,
)


def test_g3b_accepts_passing_g3a_predecessor_and_all_frozen_scenarios() -> None:
    snapshot = passing_g3a_snapshot()
    validate_g3b_predecessor(i4_3g3a_snapshot=snapshot)
    for scenario_id in (
        "g3_temporal_static_sweep",
        "g3_temporal_motion",
        "g3_temporal_low_texture",
    ):
        binding = resolve_g3b_scenario_binding(
            i4_3g3a_snapshot=snapshot,
            scenario_id=scenario_id,
            verify_files=False,
        )
        assert binding.scenario_id == scenario_id
        assert binding.source_file_sha256


def test_g3b_rejects_authority_or_matched_view_contract_drift() -> None:
    snapshot = passing_g3a_snapshot()
    snapshot["authority"]["provider_selected"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_g3b_predecessor(i4_3g3a_snapshot=snapshot)

    snapshot = passing_g3a_snapshot()
    snapshot["matched_view_contract"]["common_contiguous_source_ordered_subsequence_required"] = False
    with pytest.raises(ReconstructionValidationError):
        validate_g3b_predecessor(i4_3g3a_snapshot=snapshot)


def test_runtime_preflight_requires_all_bound_cuda_runtimes() -> None:
    snapshot = passing_g3a_snapshot()
    preflight = build_g3b_runtime_preflight_snapshot(
        i4_3g3a_snapshot=snapshot,
        scenario_id="g3_temporal_static_sweep",
        sfm_runtime=sfm_runtime(),
        video_depth_anything_runtime=current_vda_runtime(snapshot),
        rolling_depth_runtime=current_rolling_runtime(snapshot),
        verify_files=False,
    )
    assert preflight["status"] == "ready"
    assert preflight["pass"] is True
    assert preflight["decision"]["production_native_real_execution_authorized"] is True
    assert preflight["decision"]["matched_view_diagnostic_deferred_to_g3c"] is True


def test_runtime_preflight_fails_closed_on_bound_runtime_identity_drift() -> None:
    snapshot = passing_g3a_snapshot()
    current_vda = current_vda_runtime(snapshot)
    current_vda["source_revision"] = "b" * 40
    preflight = build_g3b_runtime_preflight_snapshot(
        i4_3g3a_snapshot=snapshot,
        scenario_id="g3_temporal_motion",
        sfm_runtime=sfm_runtime(),
        video_depth_anything_runtime=current_vda,
        rolling_depth_runtime=current_rolling_runtime(snapshot),
        verify_files=False,
    )
    assert preflight["pass"] is False
    assert preflight["runtime"]["video_depth_anything_ready_and_bound"] is False


def test_temporal_production_plan_preserves_source_order_without_forcing_contiguity() -> None:
    for implementation in ("rolling_depth", "video_depth_anything"):
        validate_temporal_production_plan(production_plan(implementation), implementation=implementation)
        with pytest.raises(ReconstructionValidationError):
            validate_temporal_production_plan(
                production_plan(implementation, scrambled=True),
                implementation=implementation,
            )


def test_run_evidence_requires_g1_metrics_and_rejects_g3c_masquerade() -> None:
    evidence = provider_evidence()
    validate_temporal_provider_run_evidence(evidence)

    missing = deepcopy(evidence)
    del missing["metrics"]["peak_vram_mib"]
    with pytest.raises(ReconstructionValidationError):
        validate_temporal_provider_run_evidence(missing)

    matched = deepcopy(evidence)
    matched["temporal_sequence"]["matched_view_contiguous_subsequence_used"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_temporal_provider_run_evidence(matched)


def test_rejected_provider_evidence_is_terminal_and_does_not_claim_canonical_completion() -> None:
    evidence = provider_evidence()
    evidence["status"] = "rejected"
    evidence["metrics"]["execution_completed"] = False
    evidence["metrics"]["aligned_view_fraction"] = 0.0
    evidence["rejection"] = {
        "terminal": True,
        "reason": "global_sequence_alignment_rejected:sequence_affine_residual_exceeds_limit",
        "quality_gate": "sequence_affine_residual_exceeds_limit",
        "canonical_geometry_generated": False,
    }
    validate_temporal_provider_run_evidence(evidence)

    missing_reason = deepcopy(evidence)
    missing_reason["rejection"]["reason"] = ""
    with pytest.raises(ReconstructionValidationError):
        validate_temporal_provider_run_evidence(missing_reason)

    false_completion = deepcopy(evidence)
    false_completion["metrics"]["execution_completed"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_temporal_provider_run_evidence(false_completion)


def _provider_summary(implementation: str) -> dict:
    authority = {
        "camera_pose_authority_owner": "pycolmap_sfm",
        "metric_scale_claimed": False,
        "measurement_authoritative": False,
        "certified_metrology": False,
        "surface_mesh_generated": False,
        "provider_promotion_performed": False,
    }
    if implementation == "rolling_depth":
        authority.update({
            "rolling_depth_pose_authority_promoted": False,
            "sequence_affine_establishes_physical_metric_scale": False,
            "per_frame_alignment_performed": False,
        })
    else:
        authority.update({
            "estimated_model_intrinsics_consumed_for_canonical_projection": False,
            "model_metric_depth_promoted_to_canonical_metric_scale": False,
        })
    return {
        "integrity": {
            "authoritative_sfm_unchanged": True,
            "registered_images_unchanged": True,
            "authoritative_sfm_digest": "a" * 64,
            "registered_image_sha256": "b" * 64,
            "canonical_pose_layer": "offline_sfm_bundle_adjusted",
        },
        "authority": authority,
    }


def test_frozen_provider_summary_must_prove_authority_preservation() -> None:
    for implementation in ("rolling_depth", "video_depth_anything"):
        summary = _provider_summary(implementation)
        validate_temporal_provider_summary_authority(summary, implementation=implementation)
        drifted = deepcopy(summary)
        drifted["integrity"]["authoritative_sfm_unchanged"] = False
        with pytest.raises(ReconstructionValidationError):
            validate_temporal_provider_summary_authority(drifted, implementation=implementation)
