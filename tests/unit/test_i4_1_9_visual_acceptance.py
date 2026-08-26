from __future__ import annotations

import copy

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_i4_visual_acceptance import (
    assess_visual_plan,
    build_visual_acceptance_decision,
    build_visual_binding,
    choose_visual_plan,
    finalize_visual_acceptance,
    validate_i4_1_7_visual_source,
)


def _plan(name: str, planned: int, aligned: int, *, pair: float, selected: int, full_span: bool = True) -> dict:
    rows = []
    for i in range(planned):
        accepted = i < aligned
        rows.append(
            {
                "plan_order": i,
                "source_sequence_index": i * 10,
                "alignment_accepted": accepted,
                "cloud_npz": f"/tmp/{name}-{i}.npz" if accepted else None,
            }
        )
    return {
        "plan_name": name,
        "planned_candidates": planned,
        "alignment_accepted": aligned,
        "full_span": full_span,
        "pair_pass_fraction": pair,
        "selected_qualified_views": selected,
        "candidate_rows": rows,
    }


def _bonsai_dense() -> dict:
    return {
        "phase": "I4.1.6B",
        "plans": [
            _plan("uniform_localized_dense_pool", 84, 70, pair=0.06818181818181818, selected=4),
            _plan("global_anchors_plus_local_neighbors", 118, 91, pair=0.15695067264573992, selected=4),
        ],
        "decision": {
            "single_authoritative_sfm_frame": True,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "surface_mesh_generated": False,
            "production_default_changed": False,
            "recommended_plan": "global_anchors_plus_local_neighbors",
            "real_dense_ab_validation_pass": False,
            "final_fusion_cap_preserved": 40,
        },
        "thresholds": {"threshold_relaxation": False},
    }


def _degraded_i417() -> dict:
    return {
        "phase": "I4.1.7",
        "status": "degraded",
        "authority": {
            "single_authoritative_sfm_frame": True,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "decision": {
            "clean_static_orbit_qualified": False,
            "i4_1_8_production_binding_authorized": False,
            "dense_consistency_threshold_relaxation_authorized": False,
        },
    }


def test_bonsai_visual_selector_prefers_alignment_fraction_not_pair_fraction() -> None:
    selected = choose_visual_plan(_bonsai_dense(), visual_fusion_cap=40)
    assert selected["visual_plan_available"] is True
    assert selected["selected_plan"] == "uniform_localized_dense_pool"
    assessments = {item["plan_name"]: item for item in selected["plan_assessments"]}
    assert assessments["uniform_localized_dense_pool"]["alignment_fraction"] == pytest.approx(70 / 84)
    assert assessments["uniform_localized_dense_pool"]["visual_gate_pass"] is True
    assert assessments["global_anchors_plus_local_neighbors"]["alignment_fraction"] == pytest.approx(91 / 118)
    assert assessments["global_anchors_plus_local_neighbors"]["visual_gate_pass"] is False
    assert selected["strict_dense_ab_recommended_plan"] == "global_anchors_plus_local_neighbors"
    assert selected["strict_dense_ab_validation_pass"] is False
    assert selected["strict_pair_consistency_required"] is False
    assert selected["dense_consistency_thresholds_changed"] is False


def test_visual_binding_caps_at_40_and_preserves_aligned_pool_endpoints() -> None:
    selection = choose_visual_plan(_bonsai_dense(), visual_fusion_cap=40)
    binding = build_visual_binding(selection, visual_fusion_cap=40)
    assert binding.plan_name == "uniform_localized_dense_pool"
    assert binding.selected_view_count == 40
    assert binding.selected_positions[0] == 0
    assert binding.selected_positions[-1] == 69
    assert binding.selected_source_sequence_indices[0] == 0
    assert binding.selected_source_sequence_indices[-1] == 690
    assert binding.source_span_preserved_within_aligned_pool is True


def test_visual_plan_fails_closed_when_alignment_fraction_is_insufficient() -> None:
    plan = _plan("weak", 100, 79, pair=1.0, selected=79)
    assessment = assess_visual_plan(plan)
    assert assessment["visual_gate_pass"] is False
    assert assessment["gates"]["minimum_alignment_fraction"] is False
    # Excellent strict pair evidence does not override the visual alignment gate.
    assert assessment["pair_pass_fraction"] == 1.0


def test_visual_plan_fails_closed_when_aligned_observations_lose_source_span() -> None:
    plan = _plan("cropped", 100, 90, pair=0.0, selected=1)
    # Keep 90 accepted observations, but only in the middle 90% of source indices.
    for row in plan["candidate_rows"]:
        row["alignment_accepted"] = 5 <= row["plan_order"] < 95
    plan["alignment_accepted"] = 90
    assessment = assess_visual_plan(plan, minimum_aligned_source_span_fraction=0.95)
    assert assessment["visual_gate_pass"] is False
    assert assessment["gates"]["minimum_aligned_source_span_fraction"] is False


def test_dense_threshold_relaxation_or_authority_promotion_is_rejected() -> None:
    payload = _bonsai_dense()
    payload["thresholds"]["threshold_relaxation"] = True
    with pytest.raises(ReconstructionValidationError):
        choose_visual_plan(payload)

    payload = _bonsai_dense()
    payload["decision"]["metric_scale_claimed"] = True
    with pytest.raises(ReconstructionValidationError):
        choose_visual_plan(payload)

    payload = _bonsai_dense()
    payload["decision"]["production_default_changed"] = True
    with pytest.raises(ReconstructionValidationError):
        choose_visual_plan(payload)


def test_degraded_strict_i417_is_valid_source_but_geometry_remains_deferred() -> None:
    source = validate_i4_1_7_visual_source(_degraded_i417())
    assert source["strict_geometry_qualification_passed"] is False
    assert source["single_authoritative_sfm_frame"] is True
    assert source["metric_scale_claimed"] is False


def test_automated_visual_pass_does_not_promote_geometry_or_i42() -> None:
    selection = choose_visual_plan(_bonsai_dense(), visual_fusion_cap=40)
    binding = build_visual_binding(selection, visual_fusion_cap=40)
    decision = build_visual_acceptance_decision(
        source_summary=_degraded_i417(),
        visual_plan_selection=selection,
        binding=binding,
        fusion_pass=True,
        filtering_pass=True,
        persistence_pass=True,
        i2_automated_pass=True,
    )
    assert decision["status"] == "ready_for_manual_visual_review"
    assert decision["automated_visual_reconstruction_pass"] is True
    assert decision["manual_visual_review_required"] is True
    assert decision["geometry_qualification"]["geometry_accuracy_deferred"] is True
    assert decision["authority"]["metric_scale_claimed"] is False
    assert decision["authority"]["measurement_authoritative"] is False
    assert decision["authority"]["certified_metrology"] is False
    assert decision["decision"]["i4_2_detail_preservation_authorized"] is False


def test_automated_visual_pass_requires_unchanged_fusion_filter_persistence_i2_chain() -> None:
    selection = choose_visual_plan(_bonsai_dense(), visual_fusion_cap=40)
    binding = build_visual_binding(selection, visual_fusion_cap=40)
    for failed in ("fusion", "filtering", "persistence", "i2"):
        flags = {"fusion": True, "filtering": True, "persistence": True, "i2": True}
        flags[failed] = False
        decision = build_visual_acceptance_decision(
            source_summary=_degraded_i417(),
            visual_plan_selection=selection,
            binding=binding,
            fusion_pass=flags["fusion"],
            filtering_pass=flags["filtering"],
            persistence_pass=flags["persistence"],
            i2_automated_pass=flags["i2"],
        )
        assert decision["automated_visual_reconstruction_pass"] is False
        assert decision["status"] == "degraded"


def test_manual_closure_is_explicit_and_only_then_authorizes_i42() -> None:
    selection = choose_visual_plan(_bonsai_dense(), visual_fusion_cap=40)
    binding = build_visual_binding(selection, visual_fusion_cap=40)
    automated = build_visual_acceptance_decision(
        source_summary=_degraded_i417(),
        visual_plan_selection=selection,
        binding=binding,
        fusion_pass=True,
        filtering_pass=True,
        persistence_pass=True,
        i2_automated_pass=True,
    )
    with pytest.raises(ReconstructionValidationError):
        finalize_visual_acceptance(automated, manual_visual_review_pass=False)
    closure = finalize_visual_acceptance(automated, manual_visual_review_pass=True)
    assert closure["status"] == "visual_reconstruction_qualified_geometry_deferred"
    assert closure["visual_reconstruction_qualified"] is True
    assert closure["strict_geometry_qualification_passed"] is False
    assert closure["decision"]["i4_1_visual_reconstruction_closed"] is True
    assert closure["decision"]["i4_2_detail_preservation_authorized"] is True
    assert closure["authority"]["metric_scale_claimed"] is False
    assert closure["authority"]["surface_mesh_generated"] is False


def test_i417_source_refuses_relaxed_thresholds_and_metric_claims() -> None:
    source = _degraded_i417()
    source["decision"]["dense_consistency_threshold_relaxation_authorized"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i4_1_7_visual_source(source)

    source = _degraded_i417()
    source["authority"]["measurement_authoritative"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i4_1_7_visual_source(source)
