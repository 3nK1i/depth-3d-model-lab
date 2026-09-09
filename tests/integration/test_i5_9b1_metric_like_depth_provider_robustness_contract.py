from __future__ import annotations

from core.reconstruction.i5_remaining_depth_provider_robustness_replay import (
    I59B1_PROVIDERS,
    build_i5_9b1_replay,
)
from tests.i5_9b1_helpers import build_replay_args


def test_i59b1_replay_contract_keeps_all_four_providers_in_same_pycolmap_authority():
    result = build_i5_9b1_replay(**build_replay_args())
    assert tuple(result["providers"]) == I59B1_PROVIDERS
    assert result["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert result["authority"]["authoritative_sfm_modified"] is False
    assert result["scope"]["sfm_rerun"] is False
    assert result["scope"]["provider_comparison"] is False


def test_i59b1_replay_contract_defers_b2_and_i59c():
    result = build_i5_9b1_replay(**build_replay_args())
    assert result["decision"]["i5_9b2_authorized"] is True
    assert result["decision"]["i5_9b2_started"] is False
    assert result["decision"]["i5_9c_started"] is False
    assert result["decision"]["i5_10_started"] is False
