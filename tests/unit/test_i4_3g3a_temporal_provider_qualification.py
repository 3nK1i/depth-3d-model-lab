from copy import deepcopy

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.temporal_provider_qualification import (
    TemporalVideoIdentity,
    build_i4_3g3a_preflight_snapshot,
    validate_i4_3g3_predecessors,
)
from tests.i4_3g3a_helpers import all_bindings, passing_g1_snapshot, passing_g2_closure, runtime_probe


def _build(**kwargs):
    return build_i4_3g3a_preflight_snapshot(
        i4_3g1_snapshot=kwargs.get("g1", passing_g1_snapshot()),
        i4_3g2_closure=kwargs.get("g2", passing_g2_closure()),
        scenario_bindings=kwargs.get("bindings", all_bindings()),
        video_depth_anything_runtime=kwargs.get("vda", runtime_probe("video_depth_anything")),
        rolling_depth_runtime=kwargs.get("rolling", runtime_probe("rolling_depth")),
    )


def test_g3a_ready_preflight_is_execution_free_and_authority_clean() -> None:
    snapshot = _build()
    assert snapshot["status"] == "ready" and snapshot["pass"] is True
    assert snapshot["decision"]["i4_3g3b_production_native_execution_authorized"] is True
    assert snapshot["scope"]["real_provider_execution_started"] is False
    assert snapshot["scope"]["authoritative_sfm_started"] is False
    assert snapshot["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert snapshot["authority"]["metric_scale_claimed"] is False
    assert snapshot["decision"]["provider_selected"] is False


def test_g3a_freezes_primary_vs_matched_temporal_semantics() -> None:
    snapshot = _build()
    production = snapshot["production_native_contract"]
    matched = snapshot["matched_view_contract"]
    assert production["provider_specific_i4_3f_adaptive_plans_required"] is True
    assert production["source_order_preserved"] is True
    assert production["contiguous_subsequence_required"] is False
    assert matched["common_contiguous_source_ordered_subsequence_required"] is True
    assert matched["provider_specific_temporal_semantics_must_be_preserved"] is True
    assert matched["independent_provider_auto_fit_forbidden"] is True


def test_g3a_rejects_g1_noncontiguous_temporal_contract() -> None:
    g1 = deepcopy(passing_g1_snapshot())
    g1["scenario_matrix"][0]["overlap_regime"] = "normal"
    with pytest.raises(ReconstructionValidationError, match="temporal-order contract drift"):
        validate_i4_3g3_predecessors(i4_3g1_snapshot=g1, i4_3g2_closure=passing_g2_closure())



def test_g3a_rejects_g1_temporal_manual_review_dimension_drift() -> None:
    g1 = deepcopy(passing_g1_snapshot())
    g1["scenario_matrix"][1]["manual_review_dimensions"][-1] = "invented_metric"
    with pytest.raises(ReconstructionValidationError, match="manual-review dimensions drift"):
        validate_i4_3g3_predecessors(i4_3g1_snapshot=g1, i4_3g2_closure=passing_g2_closure())


def test_g3a_rejects_g1_matched_view_semantics_drift() -> None:
    g1 = deepcopy(passing_g1_snapshot())
    g1["decision_rules"]["matched_view_mode_must_preserve_provider_semantics"] = False
    with pytest.raises(ReconstructionValidationError, match="decision-rule drift"):
        validate_i4_3g3_predecessors(i4_3g1_snapshot=g1, i4_3g2_closure=passing_g2_closure())

def test_g3a_rejects_missing_g2_authorization() -> None:
    g2 = deepcopy(passing_g2_closure())
    g2["i4_3g3_authorized"] = False
    with pytest.raises(ReconstructionValidationError, match="G3 authorization"):
        _build(g2=g2)


def test_g3a_runtime_unavailable_does_not_authorize_g3b() -> None:
    snapshot = _build(rolling=runtime_probe("rolling_depth", available=False))
    assert snapshot["status"] == "runtime_unavailable" and snapshot["pass"] is False
    assert snapshot["decision"]["i4_3g3b_production_native_execution_authorized"] is False
    assert snapshot["scope"]["rolling_depth_started"] is False


def test_g3a_requires_exact_three_explicit_scenario_bindings() -> None:
    with pytest.raises(ReconstructionValidationError, match="exactly three"):
        _build(bindings=all_bindings()[:2])


def test_temporal_video_identity_rejects_non_content_addressed_single_video_identity() -> None:
    with pytest.raises(ReconstructionValidationError, match="must equal"):
        TemporalVideoIdentity(
            source_path="/tmp/sequence.mp4", source_file_sha256="1" * 64, asset_set_sha256="2" * 64,
            file_size_bytes=1, reader_fingerprint="3" * 64, expected_frame_rate_hz=30.0, reported_frame_count=10,
        )
