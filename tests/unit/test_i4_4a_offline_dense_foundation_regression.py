from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_foundation_regression import (
    I44A_PHASE,
    RegressionEvidenceBinding,
    audit_offline_dense_foundation_contracts,
    build_i4_4a_focused_regression,
    validate_i4_3h2_for_i4_4a,
    validate_i4_4a_regression_receipt,
)
from tests.i4_4a_helpers import clone, passing_h2, passing_receipt


def binding(path: str, char: str) -> RegressionEvidenceBinding:
    return RegressionEvidenceBinding(path=path, sha256=char * 64)


def build(**kwargs) -> dict:
    args = {
        "i4_3h2_closure": passing_h2(),
        "i4_3h2_binding": binding("/evidence/h2.json", "a"),
        "regression_receipt": passing_receipt(),
        "regression_binding": binding("/evidence/i44a-receipt.json", "b"),
    }
    args.update(kwargs)
    return build_i4_4a_focused_regression(**args)


def test_i44a_builds_passing_focused_regression_without_advancing_later_boundaries() -> None:
    result = build()
    assert result["phase"] == I44A_PHASE == "I4.4A"
    assert result["status"] == "focused_regression_passed"
    assert result["pass"] is True
    assert result["decision"]["i4_4a_complete"] is True
    assert result["decision"]["i4_4b_authorized"] is True
    assert result["decision"]["i4_4c_started"] is False
    assert result["decision"]["i4_4d_started"] is False


def test_i44a_scope_is_execution_light_and_does_not_start_visual_audit_or_closure() -> None:
    assert build()["scope"] == {
        "focused_regression_only": True,
        "provider_execution_started": False,
        "provider_runtime_probe_started": False,
        "sfm_rerun": False,
        "matched_view_rerun": False,
        "manual_visual_verification_started": False,
        "provenance_authority_final_audit_started": False,
        "final_i4_closure_started": False,
    }


def test_i44a_contract_audit_covers_locked_foundation_surfaces() -> None:
    audit = audit_offline_dense_foundation_contracts()
    assert audit["provider_architecture"]["provider_family_count"] == 5
    assert audit["provider_architecture"]["provider_implementation_count"] == 10
    assert audit["sfm_authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert audit["sfm_authority"]["default_scale_status"] == "arbitrary"
    assert audit["adaptive_planning"]["permanent_global_view_cap"] is None
    assert audit["dense_alignment_projection"]["alignment_callable"] is True
    assert audit["dense_alignment_projection"]["projection_callable"] is True
    assert audit["fusion_persistence"]["fusion_callable"] is True
    assert audit["fusion_persistence"]["dense_scene_roundtrip_surfaces_importable"] is True
    assert audit["fusion_persistence"]["canonical_scene_roundtrip_surfaces_importable"] is True
    assert audit["heavy_execution_started"] is False


def test_i44a_preserves_h2_authority_and_policy_locks() -> None:
    result = build()
    authority = result["authority"]
    assert authority["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert authority["permanent_global_view_cap"] is None
    for key in (
        "provider_selected", "provider_promoted", "auto_policy_implemented", "hybrid_authorized",
        "metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated",
    ):
        assert authority[key] is False


def test_i44a_preserves_five_terminal_quality_events_without_scoring() -> None:
    result = build()
    q = result["qualification_guard"]
    assert q["terminal_quality_event_count"] == 5
    assert q["quality_event_counts_are_not_provider_scores"] is True
    assert q["provider_ranking_created"] is False
    assert q["universal_provider_winner_declared"] is False


@pytest.mark.parametrize("field", ["i4_3_closed", "i4_4_authorized", "i4_3_production_dense_provider_architecture_frozen"])
def test_i44a_rejects_h2_if_required_closure_decision_drifted(field: str) -> None:
    h2 = clone(passing_h2())
    h2["decision"][field] = False
    with pytest.raises(ReconstructionValidationError):
        validate_i4_3h2_for_i4_4a(h2)


@pytest.mark.parametrize("field", ["metric_scale_claimed", "measurement_authoritative", "certified_metrology", "hybrid_authorized"])
def test_i44a_rejects_h2_authority_promotion(field: str) -> None:
    h2 = clone(passing_h2())
    h2["authority"][field] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i4_3h2_for_i4_4a(h2)


def test_i44a_rejects_regression_receipt_that_started_heavy_execution() -> None:
    receipt = clone(passing_receipt())
    receipt["scope"]["provider_execution_started"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i4_4a_regression_receipt(receipt)


def test_i44a_rejects_incomplete_affected_regression() -> None:
    receipt = clone(passing_receipt())
    receipt["counts"]["affected_total_tests"] = 50
    with pytest.raises(ReconstructionValidationError):
        validate_i4_4a_regression_receipt(receipt)


def test_i44a_does_not_mutate_frozen_inputs() -> None:
    h2, receipt = passing_h2(), passing_receipt()
    h2_before, receipt_before = clone(h2), clone(receipt)
    build(i4_3h2_closure=h2, regression_receipt=receipt)
    assert h2 == h2_before
    assert receipt == receipt_before
