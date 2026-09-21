from __future__ import annotations

import copy

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from tests.i5_13c_helpers import make_binding, make_i5_13a_admission, make_i5_13b_freeze, make_i5_13c_receipt
from core.reconstruction.i5_auto_policy_replay import (
    I513C_PHASE,
    I513C_REPLAY_PROFILE,
    build_i5_13c_replay_qualification,
    expected_i5_13c_replay_manifest,
    replay_i5_13c_frozen_evidence,
    validate_i5_13b_for_i5_13c,
    validate_i5_13c_receipt,
)


def test_constants():
    assert I513C_PHASE == "I5.13C"
    assert I513C_REPLAY_PROFILE == "evidence_bound_conservative_v1"


def test_b_freeze_is_admitted_for_replay():
    frozen = validate_i5_13b_for_i5_13c(make_i5_13b_freeze())
    assert frozen["decision"]["i5_13c_authorized"] is True


def test_b_freeze_rejects_if_production_auto_already_authorized():
    frozen = make_i5_13b_freeze()
    frozen["decision"]["production_auto_authorized"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_13b_for_i5_13c(frozen)


def test_manifest_is_predeclared_and_has_ten_automatic_cases():
    manifest = expected_i5_13c_replay_manifest()
    assert len(manifest) == 10
    assert all(case["request_kind"] == "automatic" for case in manifest.values())
    assert sum(case["expected_decision_class"] == "select" for case in manifest.values()) == 2
    assert sum(case["expected_decision_class"] == "abstain" for case in manifest.values()) == 8


def test_manifest_covers_all_frozen_dense_contexts():
    manifest = expected_i5_13c_replay_manifest()
    dense = {case["evidence_context_id"] for case in manifest.values()}
    assert {
        "i5_6_current_mvs_envelope",
        "i5_current_castle_temporal_envelope",
        "g3_temporal_motion",
        "g3_temporal_static_sweep",
        "g4_single_image_complete_room",
        "g4_single_image_low_texture",
        "g4_single_image_weak_overlap",
    }.issubset(dense)


def test_replay_matches_all_predeclared_expectations():
    result = replay_i5_13c_frozen_evidence(
        i5_13a_admission=make_i5_13a_admission(),
        i5_13b_freeze=make_i5_13b_freeze(),
    )
    assert result["summary"]["case_count"] == 10
    assert result["summary"]["all_expected_decisions_matched"] is True
    assert result["summary"]["all_reason_codes_matched"] is True
    assert result["summary"]["all_same_input_same_decision"] is True
    assert result["summary"]["decision_class_counts"] == {"select": 2, "abstain": 8}


def test_unique_mvs_select_expected():
    result = replay_i5_13c_frozen_evidence(i5_13a_admission=make_i5_13a_admission(), i5_13b_freeze=make_i5_13b_freeze())
    case = result["cases"]["mvs_unique_select"]
    assert case["actual_decision_class"] == "select"
    assert case["actual_selected_provider_id"] == "colmap-mvs-external"


def test_temporal_unique_select_expected():
    result = replay_i5_13c_frozen_evidence(i5_13a_admission=make_i5_13a_admission(), i5_13b_freeze=make_i5_13b_freeze())
    case = result["cases"]["temporal_unique_select"]
    assert case["actual_decision_class"] == "select"
    assert case["actual_selected_provider_id"] == "rolling-depth-temporal"


@pytest.mark.parametrize("case_name", [
    "temporal_motion_zero_clean_abstain",
    "temporal_static_zero_clean_abstain",
    "single_image_complete_room_ambiguity_abstain",
    "single_image_low_texture_ambiguity_abstain",
    "single_image_weak_overlap_ambiguity_abstain",
    "forward_motion_hard_abstain",
    "difficult_handheld_degraded_abstain",
    "visual_stress_degraded_abstain",
])
def test_expected_abstentions(case_name):
    result = replay_i5_13c_frozen_evidence(i5_13a_admission=make_i5_13a_admission(), i5_13b_freeze=make_i5_13b_freeze())
    assert result["cases"][case_name]["actual_decision_class"] == "abstain"
    assert result["cases"][case_name]["actual_selected_provider_id"] == ""


def test_forward_motion_reason_code_is_stable():
    result = replay_i5_13c_frozen_evidence(i5_13a_admission=make_i5_13a_admission(), i5_13b_freeze=make_i5_13b_freeze())
    assert result["cases"]["forward_motion_hard_abstain"]["actual_reason_codes"] == [
        "auto_abstain", "sparse_context_rejected_hard_abstention"
    ]


def test_single_image_ambiguity_reason_code_is_stable():
    result = replay_i5_13c_frozen_evidence(i5_13a_admission=make_i5_13a_admission(), i5_13b_freeze=make_i5_13b_freeze())
    assert result["cases"]["single_image_low_texture_ambiguity_abstain"]["actual_reason_codes"] == [
        "auto_abstain", "multiple_clean_evidence_eligible_providers", "evidence_not_unique_even_after_availability"
    ]


def test_replay_has_no_production_or_execution_effect():
    result = replay_i5_13c_frozen_evidence(i5_13a_admission=make_i5_13a_admission(), i5_13b_freeze=make_i5_13b_freeze())
    assert result["summary"]["provider_execution"] is False
    assert result["summary"]["sfm_execution"] is False
    assert result["summary"]["mvs_execution"] is False
    assert result["summary"]["production_effect"] is False


def test_receipt_validation():
    assert validate_i5_13c_receipt(make_i5_13c_receipt())["pass"] is True


def test_receipt_rejects_provider_execution():
    receipt = make_i5_13c_receipt()
    receipt["scope"]["provider_execution"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_13c_receipt(receipt)


def test_formal_qualification_closes_c_and_only_authorizes_d_review():
    out = build_i5_13c_replay_qualification(
        i5_13a_admission=make_i5_13a_admission(),
        i5_13a_binding=make_binding("a.json", "1"),
        i5_13b_freeze=make_i5_13b_freeze(),
        i5_13b_binding=make_binding("b.json", "2"),
        post_i5_13b_checkpoint_binding=make_binding("post_b.tar.gz", "3"),
        replay_module_binding=make_binding("replay.py", "4"),
        engine_module_binding=make_binding("engine.py", "3"),
        replay_receipt=make_i5_13c_receipt(),
        replay_receipt_binding=make_binding("receipt.json", "5"),
    )
    assert out["pass"] is True
    assert out["quality_status"] == "offline_replay_pass"
    assert out["decision"]["i5_13c_complete"] is True
    assert out["decision"]["i5_13d_authorized"] is True
    assert out["decision"]["production_auto_authorized"] is False


def test_formal_qualification_rejects_engine_hash_drift():
    b = make_i5_13b_freeze()
    with pytest.raises(ReconstructionValidationError):
        build_i5_13c_replay_qualification(
            i5_13a_admission=make_i5_13a_admission(),
            i5_13a_binding=make_binding("a.json", "1"),
            i5_13b_freeze=b,
            i5_13b_binding=make_binding("b.json", "2"),
            post_i5_13b_checkpoint_binding=make_binding("post_b.tar.gz", "3"),
            replay_module_binding=make_binding("replay.py", "4"),
            engine_module_binding=make_binding("engine.py", "9"),
            replay_receipt=make_i5_13c_receipt(),
            replay_receipt_binding=make_binding("receipt.json", "5"),
        )
