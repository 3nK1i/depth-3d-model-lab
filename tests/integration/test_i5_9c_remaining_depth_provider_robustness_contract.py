from core.reconstruction.i5_remaining_depth_provider_robustness_qualification import (
    build_i5_9c_remaining_depth_provider_robustness_closure,
)
from tests.i5_9c_helpers import build_args


def test_i5_9c_closure_binds_a_b1_b2_and_receipt() -> None:
    result = build_i5_9c_remaining_depth_provider_robustness_closure(**build_args())
    bindings = result['evidence_bindings']
    assert set(bindings) == {'i5_9a_admission', 'i5_9b1_real_replay', 'i5_9b2_real_replay', 'i5_9c_qualification_receipt'}
    assert result['scope']['i5_9a_admission_reused'] is True
    assert result['scope']['i5_9b1_real_replay_reused'] is True
    assert result['scope']['i5_9b2_real_replay_reused'] is True


def test_i5_9c_combines_six_provider_current_castle_evidence_without_ranking() -> None:
    result = build_i5_9c_remaining_depth_provider_robustness_closure(**build_args())
    assert set(result['provider_qualification']) == {
        'metric3d_v2', 'unidepth_v2', 'moge_2', 'depth_pro', 'depth_anything_v2', 'marigold'
    }
    assert result['qualification']['provider_comparison_performed'] is False
    assert result['qualification']['provider_ranking_claimed'] is False


def test_i5_9c_closure_preserves_current_and_historical_evidence_as_distinct_axes() -> None:
    result = build_i5_9c_remaining_depth_provider_robustness_closure(**build_args())
    dav2 = result['provider_qualification']['depth_anything_v2']
    marigold = result['provider_qualification']['marigold']
    assert dav2['current_castle_status_counts']['pass'] == 6
    assert dav2['historical_g4_low_texture_status'] == 'rejected'
    assert marigold['current_castle_status_counts']['degraded'] == 1
    assert marigold['historical_g4_low_texture_status'] == 'rejected'


def test_i5_9c_authorizes_only_next_roadmap_step() -> None:
    result = build_i5_9c_remaining_depth_provider_robustness_closure(**build_args())
    decision = result['decision']
    assert decision['i5_9_complete'] is True
    assert decision['i5_10_authorized'] is True
    assert decision['i5_10_started'] is False
    assert decision['i5_11_started'] is False
    assert decision['i5_12_started'] is False
    assert decision['i5_13_started'] is False
    assert decision['i5_14_started'] is False
