from __future__ import annotations

from copy import deepcopy

import pytest

from core.reconstruction.cross_provider_evidence_synthesis import (
    G5AEvidenceBinding,
    build_i4_3g5a_cross_provider_synthesis,
    validate_g1_framework,
)
from core.reconstruction.errors import ReconstructionValidationError
from tests.i4_3g5a_helpers import passing_g1, passing_g2, passing_g3, passing_g4


def build(**overrides):
    return build_i4_3g5a_cross_provider_synthesis(
        i4_3g1_snapshot=overrides.get("g1", passing_g1()),
        i4_3g2_closure=overrides.get("g2", passing_g2()),
        i4_3g3c_closure=overrides.get("g3", passing_g3()),
        i4_3g4c_closure=overrides.get("g4", passing_g4()),
    )


def test_g5a_builds_three_family_synthesis_without_closing_g() -> None:
    result = build()
    assert result["phase"] == "I4.3G5A"
    assert result["status"] == "synthesis_complete"
    assert result["pass"] is True
    assert result["coverage"]["qualification_family_count"] == 3
    assert result["coverage"]["scenario_count"] == 11
    assert result["decision"]["g5a_cross_provider_synthesis_complete"] is True
    assert result["decision"]["i4_3g5b_authorized"] is True
    assert result["decision"]["i4_3g_closed"] is False


def test_g5a_preserves_terminal_quality_outcomes_without_scoring() -> None:
    result = build()
    events = result["cross_family_synthesis"]["terminal_quality_events"]
    assert {tuple(sorted(item.items())) for item in events} == {
        tuple(sorted({"family_id": "temporal_depth", "scenario_id": "g3_temporal_static_sweep", "provider_implementation": "rolling_depth", "production_status": "degraded"}.items())),
        tuple(sorted({"family_id": "temporal_depth", "scenario_id": "g3_temporal_motion", "provider_implementation": "rolling_depth", "production_status": "rejected"}.items())),
        tuple(sorted({"family_id": "temporal_depth", "scenario_id": "g3_temporal_motion", "provider_implementation": "video_depth_anything", "production_status": "degraded"}.items())),
        tuple(sorted({"family_id": "single_image_depth", "scenario_id": "g4_single_image_low_texture", "provider_implementation": "depth_anything_v2", "production_status": "rejected"}.items())),
        tuple(sorted({"family_id": "single_image_depth", "scenario_id": "g4_single_image_low_texture", "provider_implementation": "marigold", "production_status": "rejected"}.items())),
    }
    assert result["cross_family_synthesis"]["manual_preferences_aggregated_into_score"] is False
    assert result["decision"]["global_scalar_score_created"] is False
    assert result["decision"]["provider_ranking_created"] is False
    assert result["decision"]["universal_provider_winner_declared"] is False


def test_g5a_keeps_family_roles_non_rankable() -> None:
    result = build()
    families = result["family_synthesis"]
    assert all(item["direct_rank_comparison_with_other_families_valid"] is False for item in families.values())
    assert result["cross_family_synthesis"]["direct_cross_family_provider_ranking_valid"] is False


def test_g5a_accepts_g2_formal_closure_without_inventing_scenario_schema() -> None:
    result = build(g2=passing_g2(with_scenario_detail=False))
    g2 = result["family_synthesis"]["mvs_da3"]
    assert g2["scenario_detail_source"] == "formal_closure_synthesis_only"
    assert g2["scenario_detail"] is None
    assert g2["formal_closure_valid"] is True


def test_g5a_carries_g2_scenario_detail_when_present() -> None:
    result = build(g2=passing_g2(with_scenario_detail=True))
    g2 = result["family_synthesis"]["mvs_da3"]
    assert g2["scenario_detail_source"] == "scenario_synthesis"
    assert set(g2["scenario_detail"]) == set(g2["scenario_ids"])


def test_g5a_rejects_g1_scenario_provider_drift() -> None:
    g1 = passing_g1()
    g1["scenario_matrix"][0]["provider_implementations"] = ["da3", "colmap_mvs"]
    with pytest.raises(ReconstructionValidationError, match="provider matrix drift"):
        validate_g1_framework(g1)


def test_g5a_rejects_g2_authority_drift() -> None:
    g2 = passing_g2()
    g2["synthesis"]["provider_selected"] = True
    with pytest.raises(ReconstructionValidationError, match="G2 synthesis authority drift"):
        build(g2=g2)


def test_g5a_rejects_g3_invalid_status() -> None:
    g3 = passing_g3()
    g3["scenario_synthesis"]["g3_temporal_motion"]["production_statuses"]["rolling_depth"] = "failed"
    with pytest.raises(ReconstructionValidationError, match="invalid terminal statuses"):
        build(g3=g3)


def test_g5a_rejects_g4_rejection_override() -> None:
    g4 = passing_g4()
    g4["scenario_synthesis"]["g4_single_image_low_texture"]["production_rejection_overridden"] = True
    with pytest.raises(ReconstructionValidationError, match="overrides a production rejection"):
        build(g4=g4)


def test_g5a_rejects_g4_authority_drift() -> None:
    g4 = passing_g4()
    g4["authority"]["metric_scale_claimed"] = True
    with pytest.raises(ReconstructionValidationError, match="forbidden authority drift"):
        build(g4=g4)


def test_g5a_records_explicit_evidence_bindings() -> None:
    bindings = {
        key: G5AEvidenceBinding(path=f"/evidence/{key}.json", sha256=f"{idx:064x}")
        for idx, key in enumerate(("i4_3g1_snapshot", "i4_3g2_closure", "i4_3g3c_closure", "i4_3g4c_closure"), start=1)
    }
    result = build_i4_3g5a_cross_provider_synthesis(
        i4_3g1_snapshot=passing_g1(),
        i4_3g2_closure=passing_g2(),
        i4_3g3c_closure=passing_g3(),
        i4_3g4c_closure=passing_g4(),
        evidence_bindings=bindings,
    )
    assert set(result["evidence_bindings"]) == set(bindings)
    assert result["evidence_bindings"]["i4_3g4c_closure"]["sha256"] == f"{4:064x}"


def test_g5a_does_not_mutate_input_evidence() -> None:
    g3 = passing_g3(); g4 = passing_g4(); before3 = deepcopy(g3); before4 = deepcopy(g4)
    build(g3=g3, g4=g4)
    assert g3 == before3
    assert g4 == before4
