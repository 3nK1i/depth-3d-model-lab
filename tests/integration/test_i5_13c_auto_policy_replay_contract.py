from __future__ import annotations

from tests.i5_13c_helpers import make_i5_13a_admission, make_i5_13b_freeze
from core.reconstruction.i5_auto_policy_replay import replay_i5_13c_frozen_evidence


def test_replay_decisions_remain_offline_shadow_contracts():
    result = replay_i5_13c_frozen_evidence(i5_13a_admission=make_i5_13a_admission(), i5_13b_freeze=make_i5_13b_freeze())
    for case in result["cases"].values():
        assert case["production_effect"] is False
        assert case["provider_execution"] is False
        assert case["sfm_execution"] is False
        assert case["mvs_execution"] is False


def test_replay_never_selects_hybrid_or_da3():
    result = replay_i5_13c_frozen_evidence(i5_13a_admission=make_i5_13a_admission(), i5_13b_freeze=make_i5_13b_freeze())
    selected = {case["actual_selected_provider_id"] for case in result["cases"].values() if case["actual_selected_provider_id"]}
    assert selected == {"colmap-mvs-external", "rolling-depth-temporal"}


def test_replay_summary_has_no_global_score_or_rank():
    result = replay_i5_13c_frozen_evidence(i5_13a_admission=make_i5_13a_admission(), i5_13b_freeze=make_i5_13b_freeze())
    assert result["summary"]["global_scalar_score_created"] is False
    assert result["summary"]["provider_ranking_created"] is False
