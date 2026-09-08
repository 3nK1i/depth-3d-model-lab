from __future__ import annotations

from core.reconstruction.i5_rolling_depth_robustness import build_i5_7a_rolling_depth_robustness_admission
from tests.i5_7a_helpers import build_args


def test_i57a_contract_binds_predecessors_provider_and_next_boundary() -> None:
    result = build_i5_7a_rolling_depth_robustness_admission(**build_args())
    assert set(result["evidence_bindings"]) == {
        "i5_6a_castle_sfm_admission", "i5_6c_mvs_closure", "i4_3g3c_temporal_closure", "i5_7a_admission_receipt"
    }
    assert result["provider_contract"]["provider_id"] == "rolling-depth-temporal"
    assert result["provider_contract"]["sequence_alignment"] == "global_affine_to_authoritative_sfm"
    assert result["provider_contract"]["sequence_affine_establishes_metric_scale"] is False
    assert result["i5_7b_evidence_contract"]["new_timestamped_attempt_directory_per_real_attempt"] is True
    assert result["i5_7b_evidence_contract"]["missing_resource_or_performance_metrics_must_remain_missing"] is True
    assert result["roadmap_guard"]["i5_11_oom_disk_crash_recovery_deferred"] is True
    assert result["next"] == "I5.7B real frozen-foundation RollingDepth robustness replay."


def test_i57a_historical_g3_anchor_is_evidence_only() -> None:
    result = build_i5_7a_rolling_depth_robustness_admission(**build_args())
    anchors = result["historical_temporal_anchors"]
    assert anchors["g3_temporal_static_sweep"]["rolling_depth_production_status"] == "degraded"
    assert anchors["g3_temporal_motion"]["rolling_depth_production_status"] == "rejected"
    assert anchors["g3_temporal_low_texture"]["rolling_depth_production_status"] == "pass"
    assert all(node["rerun_authorized_by_i5_7a"] is False for node in anchors.values())
