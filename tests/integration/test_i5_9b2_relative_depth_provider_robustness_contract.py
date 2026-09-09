from __future__ import annotations

from core.reconstruction.i5_relative_depth_provider_robustness_replay import I59B2_PROVIDERS, build_i5_9b2_replay
from tests.i5_9b2_helpers import build_replay_args


def test_i59b2_contract_keeps_both_relative_providers_in_pycolmap_authority():
    result = build_i5_9b2_replay(**build_replay_args())
    assert tuple(result["providers"]) == I59B2_PROVIDERS
    assert result["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert result["authority"]["authoritative_sfm_modified"] is False
    assert result["scope"]["sfm_rerun"] is False
    assert result["scope"]["provider_comparison"] is False


def test_i59b2_contract_preserves_historical_g4_rejections_separately_from_current_results():
    result = build_i5_9b2_replay(**build_replay_args())
    anchor = result["historical_g4_low_texture_anchor"]
    assert anchor == {"depth_anything_v2": "rejected", "marigold": "rejected", "preserved": True}
    assert result["terminal_status_counts"] == {"pass": 12, "degraded": 0, "rejected": 0}


def test_i59b2_contract_authorizes_i59c_but_does_not_start_it_or_i510():
    result = build_i5_9b2_replay(**build_replay_args())
    assert result["decision"]["i5_9c_authorized"] is True
    assert result["decision"]["i5_9c_started"] is False
    assert result["decision"]["i5_10_started"] is False
    assert result["next"].startswith("I5.9C")
