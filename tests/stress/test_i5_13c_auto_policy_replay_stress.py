from __future__ import annotations

import json

from tests.i5_13c_helpers import make_i5_13a_admission, make_i5_13b_freeze
from core.reconstruction.i5_auto_policy_replay import replay_i5_13c_frozen_evidence


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def test_replay_is_byte_stable_across_many_runs():
    expected = _canonical(replay_i5_13c_frozen_evidence(i5_13a_admission=make_i5_13a_admission(), i5_13b_freeze=make_i5_13b_freeze()))
    for _ in range(25):
        assert _canonical(replay_i5_13c_frozen_evidence(i5_13a_admission=make_i5_13a_admission(), i5_13b_freeze=make_i5_13b_freeze())) == expected


def test_forward_motion_never_selects_across_replays():
    for _ in range(25):
        result = replay_i5_13c_frozen_evidence(i5_13a_admission=make_i5_13a_admission(), i5_13b_freeze=make_i5_13b_freeze())
        assert result["cases"]["forward_motion_hard_abstain"]["actual_selected_provider_id"] == ""


def test_ambiguous_single_image_never_selects_across_replays():
    for _ in range(25):
        result = replay_i5_13c_frozen_evidence(i5_13a_admission=make_i5_13a_admission(), i5_13b_freeze=make_i5_13b_freeze())
        for name in (
            "single_image_complete_room_ambiguity_abstain",
            "single_image_low_texture_ambiguity_abstain",
            "single_image_weak_overlap_ambiguity_abstain",
        ):
            assert result["cases"][name]["actual_selected_provider_id"] == ""
