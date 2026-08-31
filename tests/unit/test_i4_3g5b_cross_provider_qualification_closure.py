from __future__ import annotations

from copy import deepcopy

import pytest

from core.reconstruction.cross_provider_qualification_closure import (
    G5BClosureBinding,
    build_i4_3g5b_final_closure,
    validate_i4_3g5a_synthesis,
)
from core.reconstruction.errors import ReconstructionValidationError
from tests.i4_3g5b_helpers import passing_g5a, passing_g5b_bindings


def build(**overrides):
    return build_i4_3g5b_final_closure(i4_3g5a_synthesis=overrides.get("g5a", passing_g5a()), evidence_bindings=passing_g5b_bindings(), chain_rebuild_verified=True)


def test_g5b_formally_closes_g_and_authorizes_h() -> None:
    result = build()
    assert result["phase"] == "I4.3G5B"
    assert result["status"] == "formally_closed"
    assert result["pass"] is True
    assert result["decision"]["i4_3g_closed"] is True
    assert result["decision"]["i4_3h_authorized"] is True
    assert result["scope"]["formal_i4_3g_closure_performed"] is True
    assert result["scope"]["i4_3h_work_started"] is False


def test_g5b_preserves_3_11_10_coverage() -> None:
    result = build()
    coverage = result["coverage"]
    assert coverage["qualification_family_count"] == 3
    assert coverage["scenario_count"] == 11
    assert coverage["provider_implementation_count"] == 10
    assert coverage["g2_scenarios"] == 5
    assert coverage["g3_scenarios"] == 3
    assert coverage["g4_scenarios"] == 3


def test_g5b_preserves_exact_terminal_quality_events_as_non_scores() -> None:
    result = build()
    summary = result["operating_envelope_summary"]
    assert summary["terminal_quality_event_count"] == 5
    assert summary["quality_event_counts_are_not_provider_scores"] is True
    assert summary["direct_cross_family_provider_ranking_valid"] is False
    assert {
        (e["family_id"], e["scenario_id"], e["provider_implementation"], e["production_status"])
        for e in summary["terminal_quality_events"]
    } == {
        ("temporal_depth", "g3_temporal_static_sweep", "rolling_depth", "degraded"),
        ("temporal_depth", "g3_temporal_motion", "rolling_depth", "rejected"),
        ("temporal_depth", "g3_temporal_motion", "video_depth_anything", "degraded"),
        ("single_image_depth", "g4_single_image_low_texture", "depth_anything_v2", "rejected"),
        ("single_image_depth", "g4_single_image_low_texture", "marigold", "rejected"),
    }


def test_g5b_preserves_family_roles_without_cross_family_ranking() -> None:
    roles = build()["operating_envelope_summary"]["family_roles"]
    assert set(roles) == {"mvs_da3", "temporal_depth", "single_image_depth"}
    assert all(item["formal_closure_valid"] is True for item in roles.values())
    assert all(item["direct_rank_comparison_with_other_families_valid"] is False for item in roles.values())


def test_g5b_closure_keeps_all_forbidden_decisions_false() -> None:
    result = build()
    for key in (
        "provider_selected", "provider_promoted", "provider_ranking_created",
        "universal_provider_winner_declared", "global_scalar_score_created",
        "auto_policy_implemented", "hybrid_authorized",
    ):
        assert result["decision"][key] is False
    assert result["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert result["authority"]["permanent_global_view_cap"] is None


def test_g5b_rejects_g5a_phase_or_status_drift() -> None:
    g5a = passing_g5a(); g5a["status"] = "formally_closed"
    with pytest.raises(ReconstructionValidationError, match="requires the passing frozen"):
        validate_i4_3g5a_synthesis(g5a)


def test_g5b_rejects_execution_or_rerun_drift() -> None:
    for key in ("provider_execution_started", "sfm_rerun", "matched_view_rerun", "manual_review_rerun"):
        g5a = passing_g5a(); g5a["scope"][key] = True
        with pytest.raises(ReconstructionValidationError, match="contract drift"):
            build(g5a=g5a)


def test_g5b_rejects_premature_g_closure_in_g5a() -> None:
    g5a = passing_g5a(); g5a["decision"]["i4_3g_closed"] = True
    with pytest.raises(ReconstructionValidationError, match="i4_3g_closed"):
        build(g5a=g5a)


def test_g5b_rejects_coverage_drift() -> None:
    g5a = passing_g5a(); g5a["coverage"]["scenario_count"] = 10
    with pytest.raises(ReconstructionValidationError, match="coverage contract drift"):
        build(g5a=g5a)


def test_g5b_rejects_terminal_quality_event_drift() -> None:
    g5a = passing_g5a(); g5a["cross_family_synthesis"]["terminal_quality_events"].pop()
    with pytest.raises(ReconstructionValidationError, match="terminal quality-event set drifted"):
        build(g5a=g5a)


def test_g5b_rejects_provider_family_role_drift() -> None:
    g5a = passing_g5a(); g5a["family_synthesis"]["temporal_depth"]["family_role"] = "static"
    with pytest.raises(ReconstructionValidationError, match="family temporal_depth contract drift"):
        build(g5a=g5a)


def test_g5b_rejects_authority_drift() -> None:
    g5a = passing_g5a(); g5a["authority"]["metric_scale_claimed"] = True
    with pytest.raises(ReconstructionValidationError, match="metric_scale_claimed"):
        build(g5a=g5a)


def test_g5b_rejects_malformed_or_incomplete_g5a_bindings() -> None:
    g5a = passing_g5a(); del g5a["evidence_bindings"]["i4_3g2_closure"]
    with pytest.raises(ReconstructionValidationError, match="binding set drifted"):
        build(g5a=g5a)


def test_g5b_records_all_five_final_evidence_bindings() -> None:
    bindings = {
        key: G5BClosureBinding(path=f"/evidence/{key}.json", sha256=f"{idx:064x}")
        for idx, key in enumerate((
            "i4_3g1_snapshot", "i4_3g2_closure", "i4_3g3c_closure",
            "i4_3g4c_closure", "i4_3g5a_synthesis",
        ), start=1)
    }
    result = build_i4_3g5b_final_closure(i4_3g5a_synthesis=passing_g5a(), evidence_bindings=bindings, chain_rebuild_verified=True)
    assert set(result["evidence_bindings"]) == set(bindings)
    assert result["evidence_bindings"]["i4_3g5a_synthesis"]["sha256"] == f"{5:064x}"


def test_g5b_refuses_unverified_chain_rebuild() -> None:
    with pytest.raises(ReconstructionValidationError, match="verified deterministic"):
        build_i4_3g5b_final_closure(
            i4_3g5a_synthesis=passing_g5a(),
            evidence_bindings=passing_g5b_bindings(),
            chain_rebuild_verified=False,
        )


def test_g5b_does_not_mutate_g5a() -> None:
    g5a = passing_g5a(); before = deepcopy(g5a)
    build(g5a=g5a)
    assert g5a == before
