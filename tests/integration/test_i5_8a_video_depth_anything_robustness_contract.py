from __future__ import annotations

from core.reconstruction.i5_video_depth_anything_robustness import (
    build_i5_8a_video_depth_anything_robustness_admission,
)
from tests.i5_8a_helpers import build_args


def test_i58a_contract_binds_closed_i57_source_matrix_g3_and_next_boundary() -> None:
    result = build_i5_8a_video_depth_anything_robustness_admission(**build_args())
    assert set(result["evidence_bindings"]) == {
        "i5_7c_rolling_depth_closure",
        "i5_6a_castle_sfm_admission",
        "i4_3g3c_temporal_closure",
        "i5_8a_admission_receipt",
    }
    contract = result["provider_contract"]
    assert contract["provider_id"] == "video-depth-anything-temporal"
    assert contract["temporal_mode"] == "official_streaming_cache"
    assert contract["canonical_alignment"] == "per_registered_frame_affine_to_authoritative_sfm"
    assert contract["sequence_global_alignment_performed"] is False
    assert result["i5_8b_evidence_contract"]["warmup_reset_before_user_sequence_required"] is True
    assert result["i5_8b_evidence_contract"]["resolved_checkpoint_must_match_probed_checkpoint"] is True
    assert result["next"] == "I5.8B real frozen-foundation Video Depth Anything robustness replay."


def test_i58a_preserves_historical_vda_motion_degradation_as_active_evidence() -> None:
    result = build_i5_8a_video_depth_anything_robustness_admission(**build_args())
    anchors = result["historical_temporal_anchors"]
    assert anchors["g3_temporal_static_sweep"]["video_depth_anything_production_status"] == "pass"
    assert anchors["g3_temporal_motion"]["video_depth_anything_production_status"] == "degraded"
    assert anchors["g3_temporal_low_texture"]["video_depth_anything_production_status"] == "pass"
    assert all(node["rerun_authorized_by_i5_8a"] is False for node in anchors.values())
    assert result["known_qualification_gaps"]["historical_g3_motion_degraded_remains_active_evidence"] is True
