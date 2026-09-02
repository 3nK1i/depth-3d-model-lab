from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_difficult_handheld_robustness import (
    I51C_PHASE,
    I51EvidenceBinding,
    build_i5_1c_difficult_handheld_closure,
    validate_i5_1b_for_i5_1c,
    validate_i5_1b_oom_diagnostic,
    validate_i5_1c_receipt,
)
from tests.i5_1b_helpers import passing_i5_1a
from tests.i5_1c_helpers import (
    I51A_SHA,
    I51B_SHA,
    OOM_SHA,
    RECEIPT_SHA,
    clone,
    degraded_i5_1b,
    oom_diagnostic_text,
    passing_i5_1c_receipt,
)


def _binding(path: str, sha: str) -> I51EvidenceBinding:
    return I51EvidenceBinding(path, sha)


def build() -> dict:
    return build_i5_1c_difficult_handheld_closure(
        i5_1a_result=passing_i5_1a(),
        i5_1a_binding=_binding("/evidence/i5_1a.json", I51A_SHA),
        i5_1b_result=degraded_i5_1b(),
        i5_1b_binding=_binding("/evidence/i5_1b.json", I51B_SHA),
        oom_diagnostic=oom_diagnostic_text(),
        oom_diagnostic_binding=_binding("/evidence/i5_1b_failed_attempt_diagnostic.txt", OOM_SHA),
        qualification_receipt=passing_i5_1c_receipt(),
        qualification_receipt_binding=_binding("/evidence/i5_1c_receipt.json", RECEIPT_SHA),
    )


def test_i51c_builds_formal_degraded_closure() -> None:
    result = build()
    assert result["phase"] == I51C_PHASE == "I5.1C"
    assert result["status"] == "difficult_handheld_robustness_qualified"
    assert result["pass"] is True
    q = result["qualification"]
    assert q["quality_status"] == "degraded"
    assert q["fresh_global_selected_qualified_views"] == 2
    assert q["fresh_best_selected_qualified_views"] == 7
    assert q["selected_view_gain_over_historical"] == 5
    assert q["required_connected_qualified_views"] == 10
    assert q["production_dense_gate_reached"] is False
    assert q["operating_envelope_classification"] == "outside_current_production_dense_robustness_envelope"


def test_i51c_closes_i51_and_authorizes_i52_without_starting_it() -> None:
    decision = build()["decision"]
    assert decision["i5_1_complete"] is True
    assert decision["i5_2_authorized"] is True
    assert decision["i5_2_started"] is False
    assert build()["next"] == "I5.2 low overlap / weak parallax."


def test_i51c_preserves_no_threshold_relaxation_or_authority_promotion() -> None:
    result = build()
    assert result["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert result["authority"]["permanent_global_view_cap"] is None
    for key in (
        "provider_selected", "provider_promoted", "provider_ranking_created",
        "universal_provider_winner_declared", "global_scalar_score_created",
        "auto_policy_implemented", "hybrid_authorized",
        "dense_consistency_threshold_relaxation_authorized", "metric_scale_claimed",
        "measurement_authoritative", "certified_metrology", "surface_mesh_generated",
    ):
        assert result["decision"][key] is False


def test_i51c_oom_event_is_bound_but_not_reclassified_as_quality_result() -> None:
    event = build()["robustness_event"]
    assert event["pre_correction_host_oom_observed"] is True
    assert event["failed_attempt_preserved"] is True
    assert event["failed_attempt_reclassified_as_quality_result"] is False
    assert event["failure_stage"] == "i4_1_6b_dense_alignment_projection"
    assert event["last_observed_candidate"] == "147/176"
    assert event["killed_python_anon_rss_kib"] == 10942216
    assert event["corrected_replay_completed"] is True
    assert event["correction_changed_quality_thresholds"] is False
    assert event["i5_11_oom_recovery_closed_early"] is False


def test_i51c_requires_current_degraded_horse_result_not_a_rewritten_pass() -> None:
    replay = clone(degraded_i5_1b())
    replay["quality"]["status"] = "pass"
    with pytest.raises(ReconstructionValidationError):
        validate_i5_1b_for_i5_1c(replay)


@pytest.mark.parametrize("field,value", [
    ("fresh_global_selected_qualified_views", 3),
    ("fresh_best_selected_qualified_views", 8),
    ("fresh_dense_gate_reached", True),
])
def test_i51c_rejects_replay_metric_drift(field: str, value) -> None:
    replay = clone(degraded_i5_1b())
    replay["quality"][field] = value
    with pytest.raises(ReconstructionValidationError):
        validate_i5_1b_for_i5_1c(replay)


def test_i51c_rejects_i51a_binding_mismatch() -> None:
    replay = clone(degraded_i5_1b())
    replay["provenance"]["i5_1a_sha256"] = "f" * 64
    with pytest.raises(ReconstructionValidationError):
        build_i5_1c_difficult_handheld_closure(
            i5_1a_result=passing_i5_1a(),
            i5_1a_binding=_binding("/evidence/i5_1a.json", I51A_SHA),
            i5_1b_result=replay,
            i5_1b_binding=_binding("/evidence/i5_1b.json", I51B_SHA),
            oom_diagnostic=oom_diagnostic_text(),
            oom_diagnostic_binding=_binding("/evidence/oom.txt", OOM_SHA),
            qualification_receipt=passing_i5_1c_receipt(),
            qualification_receipt_binding=_binding("/evidence/receipt.json", RECEIPT_SHA),
        )


def test_i51c_oom_diagnostic_requires_confirmed_python_oom_after_localization() -> None:
    event = validate_i5_1b_oom_diagnostic(oom_diagnostic_text())
    assert event["i4_1_6a_completed_before_oom"] is True
    assert event["last_observed_candidate"] == "147/176"
    with pytest.raises(ReconstructionValidationError):
        validate_i5_1b_oom_diagnostic("ordinary process exit")


def test_i51c_receipt_requires_all_regressions_green_and_no_runtime_rerun() -> None:
    receipt = validate_i5_1c_receipt(passing_i5_1c_receipt())
    assert receipt["counts"]["combined_exercised_tests"] >= 100
    bad = clone(passing_i5_1c_receipt())
    bad["scope"]["quality_replay_rerun"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_1c_receipt(bad)
