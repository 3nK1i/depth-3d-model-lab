from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_remaining_depth_provider_robustness_qualification import (
    I59C_PHASE,
    build_i5_9c_remaining_depth_provider_robustness_closure,
    validate_i5_9b1_for_i5_9c,
    validate_i5_9b2_for_i5_9c,
)
from tests.i5_9c_helpers import build_args, b1_replay, b2_replay


def _build() -> dict:
    return build_i5_9c_remaining_depth_provider_robustness_closure(**build_args())


def test_i5_9c_closes_remaining_provider_robustness_with_mixed_bounded_envelope() -> None:
    result = _build()
    assert result['phase'] == I59C_PHASE
    assert result['pass'] is True
    assert result['quality_status'] == 'mixed_terminal_quality'
    assert result['qualification']['combined_terminal_runs'] == 36
    assert result['qualification']['combined_status_counts'] == {'pass': 35, 'degraded': 1, 'rejected': 0}
    assert result['decision']['i5_9_complete'] is True
    assert result['decision']['i5_10_authorized'] is True
    assert result['decision']['i5_10_started'] is False


def test_i5_9c_preserves_b1_24_pass_matrix() -> None:
    result = _build()
    assert result['qualification']['b1_status_counts'] == {'pass': 24, 'degraded': 0, 'rejected': 0}
    for provider in ('metric3d_v2', 'unidepth_v2', 'moge_2', 'depth_pro'):
        assert result['provider_qualification'][provider]['current_castle_status_counts']['pass'] == 6
        assert result['provider_qualification'][provider]['current_castle_quality_status'] == 'pass'


def test_i5_9c_preserves_b2_11_pass_one_degraded_matrix() -> None:
    result = _build()
    assert result['qualification']['b2_status_counts'] == {'pass': 11, 'degraded': 1, 'rejected': 0}
    marigold = result['provider_qualification']['marigold']
    assert marigold['current_castle_status_counts'] == {'pass': 5, 'degraded': 1, 'rejected': 0}
    assert marigold['current_castle_quality_status'] == 'mixed_terminal_quality'
    assert marigold['current_castle_degraded_scenarios'] == ['castle_dynamic_proxy']


def test_i5_9c_preserves_dynamic_proxy_marigold_alignment_rejection() -> None:
    result = _build()
    observation = result['observed_limitations']['marigold_castle_dynamic_proxy']
    assert observation['runtime_status'] == 'degraded'
    assert observation['selected_views'] == 19
    assert observation['accepted_views'] == 18
    assert observation['alignment_rejected_views'] == 1
    assert observation['fusion_capacity_drops'] == 0
    assert observation['degradation_reasons'] == ['alignment_rejected_views:1']


def test_i5_9c_preserves_historical_g4_low_texture_rejections() -> None:
    result = _build()
    historical = result['historical_g4_anchors']['low_texture']
    assert historical['depth_anything_v2'] == 'rejected'
    assert historical['marigold'] == 'rejected'
    assert result['provenance']['historical_g4_rejections_overwritten'] is False


def test_i5_9c_does_not_turn_current_castle_pass_into_universal_robustness() -> None:
    result = _build()
    q = result['qualification']
    assert q['universal_single_image_provider_robustness_claimed'] is False
    assert q['universal_provider_winner_claimed'] is False
    assert q['provider_ranking_claimed'] is False
    assert q['global_scalar_score_created'] is False


def test_i5_9c_is_evidence_only() -> None:
    result = _build()
    scope = result['scope']
    assert scope['evidence_only_qualification_and_closure'] is True
    for key in ('provider_inference', 'sfm_rerun', 'mvs_rerun', 'provider_comparison_started', 'thresholds_modified', 'fusion_capacity_increased', 'production_default_modified', 'result_driven_retuning', 'retry_until_pass'):
        assert scope[key] is False


def test_i5_9c_preserves_pycolmap_authority_and_no_metric_claim() -> None:
    result = _build()
    authority = result['authority']
    assert authority['camera_pose_authority_owner'] == 'pycolmap_sfm'
    assert authority['single_authoritative_sfm_frame_preserved'] is True
    assert authority['metric_scale_claimed'] is False
    assert authority['measurement_authoritative'] is False
    assert authority['certified_metrology'] is False


def test_i5_9c_rejects_b1_quality_drift() -> None:
    payload = b1_replay()
    payload['quality_status'] = 'mixed_terminal_quality'
    with pytest.raises(ReconstructionValidationError):
        validate_i5_9b1_for_i5_9c(payload)


def test_i5_9c_rejects_b1_hidden_nonpass_record() -> None:
    payload = b1_replay()
    payload['records']['castle_control']['metric3d_v2']['runtime_status'] = 'degraded'
    with pytest.raises(ReconstructionValidationError):
        validate_i5_9b1_for_i5_9c(payload)


def test_i5_9c_rejects_b2_clean_pass_upgrade() -> None:
    payload = b2_replay()
    payload['quality_status'] = 'pass'
    with pytest.raises(ReconstructionValidationError):
        validate_i5_9b2_for_i5_9c(payload)


def test_i5_9c_rejects_marigold_dynamic_proxy_pass_rewrite() -> None:
    payload = b2_replay()
    payload['records']['castle_dynamic_proxy']['marigold']['runtime_status'] = 'pass'
    with pytest.raises(ReconstructionValidationError):
        validate_i5_9b2_for_i5_9c(payload)


def test_i5_9c_rejects_capacity_cause_substitution_for_marigold_degradation() -> None:
    payload = b2_replay()
    payload['records']['castle_dynamic_proxy']['marigold']['fusion_capacity_drops'] = 1
    with pytest.raises(ReconstructionValidationError):
        validate_i5_9b2_for_i5_9c(payload)


def test_i5_9c_rejects_historical_low_texture_rejection_overwrite() -> None:
    payload = b2_replay()
    payload['historical_g4_low_texture_anchor']['marigold'] = 'pass'
    with pytest.raises(ReconstructionValidationError):
        validate_i5_9b2_for_i5_9c(payload)


def test_i5_9c_does_not_start_i5_10() -> None:
    result = _build()
    assert result['decision']['i5_10_authorized'] is True
    assert result['decision']['i5_10_started'] is False
    assert result['roadmap_guard']['i5_10_mp_sfm_authorized_not_started'] is True
