from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_difficult_handheld_robustness import (
    I51B_PHASE,
    build_i5_1b_difficult_handheld_replay,
    validate_i5_1a_for_replay,
    validate_i5_1b_dense_ab,
    validate_i5_1b_localization,
    validate_i5_1b_source_identity,
)
from scripts.run_i5_1b_difficult_handheld_replay import (
    build_dense_ab_command,
    build_i1_command,
    build_localization_command,
)
from tests.i5_1b_helpers import (
    BASELINE_SHA,
    SOURCE_SHA,
    clone,
    dense_ab_summary,
    i1_summary,
    localization_summary,
    passing_i5_1a,
    source_identity,
)


def build(**kwargs):
    args = {
        "i5_1a_result": passing_i5_1a(),
        "i5_1a_sha256": BASELINE_SHA,
        "source_identity": source_identity(),
        "i1_summary": i1_summary(),
        "localization_summary": localization_summary(),
        "dense_ab_summary": dense_ab_summary(),
        "stage_return_codes": {"i1": 2, "localization": 0, "dense_ab": 0},
        "replay_parameters": {
            "profile": "efficient",
            "matching": "exhaustive",
            "max_keyframes": 40,
            "max_dense_views": 40,
            "dense_candidate_cap": 120,
            "video_target_fps": 2.0,
            "local_target_fps": 2.0,
            "process_res": 392,
        },
    }
    args.update(kwargs)
    return build_i5_1b_difficult_handheld_replay(**args)


def test_i51b_replay_evidence_passes_independently_of_quality_classification() -> None:
    result = build()
    assert result["phase"] == I51B_PHASE == "I5.1B"
    assert result["pass"] is True
    assert result["status"] == "replay_evidence_complete"
    assert result["quality"]["status"] == "pass"
    assert result["decision"]["i5_1b_complete"] is True
    assert result["decision"]["i5_1c_authorized"] is True
    assert result["decision"]["i5_1_complete"] is False


def test_i51b_fails_closed_without_i51a_authorization() -> None:
    baseline = clone(passing_i5_1a())
    baseline["decision"]["i5_1_current_replay_authorized"] = False
    with pytest.raises(ReconstructionValidationError):
        validate_i5_1a_for_replay(baseline)


def test_i51b_rejects_wrong_predecessor_next_boundary() -> None:
    baseline = clone(passing_i5_1a())
    baseline["next"] = "I5.2 low overlap"
    with pytest.raises(ReconstructionValidationError):
        validate_i5_1a_for_replay(baseline)


def test_i51b_source_identity_is_fresh_sha_bound() -> None:
    source = validate_i5_1b_source_identity(source_identity())
    assert source["sha256"] == SOURCE_SHA
    assert source["size_bytes"] > 0
    assert source["total_frames"] > 0
    assert source["duration_seconds"] > 0


@pytest.mark.parametrize("field", ["sha256", "size_bytes", "total_frames", "fps", "duration_seconds", "width", "height"])
def test_i51b_source_identity_rejects_malformed_fields(field: str) -> None:
    source = source_identity()
    source[field] = "bad" if field == "sha256" else 0
    with pytest.raises(ReconstructionValidationError):
        validate_i5_1b_source_identity(source)


def test_i51b_localization_must_bind_fresh_source_hash() -> None:
    summary = localization_summary()
    summary["source"]["source_sha256"] = "c" * 64
    with pytest.raises(ReconstructionValidationError):
        validate_i5_1b_localization(summary, source_sha256=SOURCE_SHA)


def test_i51b_localization_preserves_single_sfm_authority() -> None:
    loc = validate_i5_1b_localization(localization_summary(), source_sha256=SOURCE_SHA)
    assert loc["decision"]["single_authoritative_sfm_frame"] is True
    assert loc["decision"]["extra_dense_candidates_are_observations_not_second_map"] is True
    assert loc["decision"]["dense_consistency_threshold_relaxation_authorized"] is False


@pytest.mark.parametrize("field", [
    "production_sampling_integration_authorized",
    "production_default_changed",
    "dense_consistency_threshold_relaxation_authorized",
    "metric_scale_claimed",
    "measurement_authoritative",
    "surface_mesh_generated",
])
def test_i51b_localization_rejects_forbidden_promotion(field: str) -> None:
    summary = localization_summary()
    summary["decision"][field] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_1b_localization(summary, source_sha256=SOURCE_SHA)


def test_i51b_dense_ab_preserves_thresholds_and_authority() -> None:
    dense = validate_i5_1b_dense_ab(dense_ab_summary())
    assert dense["thresholds"]["threshold_relaxation"] is False
    assert dense["decision"]["single_authoritative_sfm_frame"] is True
    assert dense["decision"]["extra_dense_candidates_are_observations_not_second_map"] is True


@pytest.mark.parametrize("location,field", [
    ("thresholds", "threshold_relaxation"),
    ("decision", "production_sampling_integration_authorized"),
    ("decision", "production_default_changed"),
    ("decision", "metric_scale_claimed"),
    ("decision", "measurement_authoritative"),
    ("decision", "surface_mesh_generated"),
])
def test_i51b_dense_ab_rejects_forbidden_drift(location: str, field: str) -> None:
    dense = dense_ab_summary()
    dense[location][field] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_1b_dense_ab(dense)


def test_i51b_pass_quality_requires_real_dense_gate() -> None:
    result = build(dense_ab_summary=dense_ab_summary(dense_gate=True, selected=12))
    assert result["quality"]["status"] == "pass"
    assert result["quality"]["fresh_dense_gate_reached"] is True
    assert result["historical_comparison"]["historical_dense_gate_reached"] is False


def test_i51b_degraded_quality_is_valid_evidence_when_dense_gate_still_fails() -> None:
    result = build(dense_ab_summary=dense_ab_summary(dense_gate=False, selected=7))
    assert result["pass"] is True
    assert result["quality"]["status"] == "degraded"
    assert result["quality"]["fresh_dense_gate_reached"] is False
    assert result["decision"]["i5_1c_authorized"] is True


def test_i51b_localization_only_degraded_is_valid_evidence() -> None:
    result = build(
        localization_summary=localization_summary(gate=False),
        dense_ab_summary=None,
        stage_return_codes={"i1": 2, "localization": 3, "dense_ab": None},
    )
    assert result["quality"]["status"] == "degraded"
    assert result["execution"]["dense_ab_executed"] is False


def test_i51b_rejected_sfm_stage_is_valid_robustness_evidence() -> None:
    failed_i1 = {"phase": "I1", "status": "failed", "failed_stage": "sfm"}
    result = build(
        i1_summary=failed_i1,
        localization_summary=None,
        dense_ab_summary=None,
        stage_return_codes={"i1": 2, "localization": None, "dense_ab": None},
    )
    assert result["pass"] is True
    assert result["quality"]["status"] == "rejected"
    assert result["quality"]["failed_stage"] == "sfm"
    assert result["decision"]["i5_1c_authorized"] is True


def test_i51b_historical_comparison_preserves_missing_historical_content_hash() -> None:
    comparison = build()["historical_comparison"]
    assert comparison["historical_raw_video_sha256_available"] is False
    assert comparison["fresh_vs_historical_content_hash_equality_claimed"] is False
    assert comparison["historical_selected_qualified_views"] == 2
    assert comparison["fresh_best_selected_qualified_views"] == 12


def test_i51b_controlled_replay_caps_do_not_become_production_caps() -> None:
    result = build()
    replay = result["replay_parameters"]
    assert replay["max_keyframes"] == 40
    assert replay["max_dense_views"] == 40
    assert replay["dense_candidate_cap"] == 120
    assert replay["controlled_experiment_bounds_only"] is True
    assert result["authority"]["permanent_global_view_cap"] is None


def test_i51b_preserves_no_auto_hybrid_ranking_or_metric_authority() -> None:
    decision = build()["decision"]
    for key in (
        "provider_selected", "provider_promoted", "provider_ranking_created",
        "universal_provider_winner_declared", "global_scalar_score_created",
        "auto_policy_implemented", "hybrid_authorized", "metric_scale_claimed",
        "measurement_authoritative", "certified_metrology", "surface_mesh_generated",
    ):
        assert decision[key] is False


def test_i51b_execution_records_fresh_sfm_and_da3_dense_work() -> None:
    execution = build()["execution"]
    assert execution["current_difficult_handheld_replay_started"] is True
    assert execution["global_i1_executed"] is True
    assert execution["sfm_rerun"] is True
    assert execution["localization_executed"] is True
    assert execution["dense_ab_executed"] is True
    assert execution["da3_extra_candidates_executed"] is True
    assert execution["dense_pair_rerun"] is True


def test_i51b_does_not_close_i51_or_authorize_i52() -> None:
    decision = build()["decision"]
    assert decision["i5_1_complete"] is False
    assert decision["i5_2_authorized"] is False
    assert build()["next"] == "I5.1C difficult-handheld robustness qualification and closure."


def _runner_args() -> argparse.Namespace:
    return argparse.Namespace(
        sfm_python=Path("/env/sfm/python"), da3_python=Path("/env/da3/python"),
        da3_repo=Path("/repo/da3"), profile="efficient", matching="exhaustive",
        max_keyframes=40, max_dense_views=40, dense_candidate_cap=120,
        video_target_fps=2.0, local_target_fps=2.0, process_res=392,
        sfm_timeout_minutes=60.0, da3_timeout_seconds=600.0,
        da3_total_timeout_seconds=7200.0,
    )


def test_i51b_i1_command_is_fresh_exhaustive_global_replay() -> None:
    command = build_i1_command(_runner_args(), video=Path("/data/horse.mp4"), output=Path("/evidence/replay"))
    assert "run_i1_offline_dense_campaign.py" in " ".join(command)
    assert command[command.index("--matching") + 1] == "exhaustive"
    assert command[command.index("--max-keyframes") + 1] == "40"
    assert command[command.index("--max-dense-views") + 1] == "40"
    assert "--reuse-sfm-dir" not in command
    assert "--reuse-dense-dir" not in command


def test_i51b_localization_command_reuses_existing_single_sfm_frame() -> None:
    command = build_localization_command(_runner_args(), video=Path("/data/horse.mp4"), output=Path("/evidence/replay"))
    assert "run_i4_1_6_decoupled_sampling_integration.py" in " ".join(command)
    assert command[command.index("--i4-dir") + 1] == "/evidence/replay"
    assert command[command.index("--dense-candidate-cap") + 1] == "120"


def test_i51b_dense_ab_command_preserves_real_da3_execution_path() -> None:
    command = build_dense_ab_command(_runner_args(), video=Path("/data/horse.mp4"), output=Path("/evidence/replay"))
    assert "run_i4_1_6b_decoupled_dense_ab.py" in " ".join(command)
    assert command[command.index("--process-res") + 1] == "392"
    assert command[command.index("--dense-candidate-cap") + 1] == "120"


@pytest.mark.parametrize("codes", [
    {"i1": 1, "localization": 0, "dense_ab": 0},
    {"i1": 2, "localization": 2, "dense_ab": 0},
    {"i1": 2, "localization": 0, "dense_ab": 3},
])
def test_i51b_rejects_unqualified_stage_return_codes(codes: dict) -> None:
    with pytest.raises(ReconstructionValidationError):
        build(stage_return_codes=codes)


def test_i51b_dense_summary_must_reference_fresh_replay_path() -> None:
    dense = dense_ab_summary()
    dense["source"]["video"] = "/other/video.mp4"
    with pytest.raises(ReconstructionValidationError):
        build(dense_ab_summary=dense)
