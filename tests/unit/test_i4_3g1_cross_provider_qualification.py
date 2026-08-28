from __future__ import annotations

import copy

import pytest

from core.reconstruction.cross_provider_qualification import (
    ComparisonMode,
    MetricDirection,
    QualificationMetricSpec,
    build_i4_3g1_framework_snapshot,
    qualification_decision_rules,
    qualification_metric_registry,
    qualification_run_evidence_schema,
    qualification_scenario_matrix,
    validate_i4_3g1_predecessors,
    validate_metric_value,
)
from core.reconstruction.dense_provider_architecture import DenseProviderImplementation
from core.reconstruction.errors import ReconstructionValidationError
from tests.i4_3g1_helpers import passing_i4_3a_snapshot, passing_i4_3f3_snapshot


def test_metric_registry_has_unique_ids_and_no_global_score() -> None:
    metrics = qualification_metric_registry()
    ids = [item.metric_id for item in metrics]
    assert len(ids) == len(set(ids))
    assert "wall_time_s" in ids
    assert "peak_vram_mib" in ids
    assert "authoritative_sfm_preserved" in ids
    rules = qualification_decision_rules()
    assert rules["global_scalar_score_forbidden"] is True
    assert rules["scenario_specific_verdicts_required"] is True


def test_scenario_matrix_freezes_expected_g2_g3_g4_shape() -> None:
    scenarios = qualification_scenario_matrix()
    assert len(scenarios) == 11
    assert sum(item.substep == "I4.3G2" for item in scenarios) == 5
    assert sum(item.substep == "I4.3G3" for item in scenarios) == 3
    assert sum(item.substep == "I4.3G4" for item in scenarios) == 3
    assert all(item.required for item in scenarios)


def test_scenario_matrix_covers_all_ten_provider_implementations() -> None:
    covered = {impl for scenario in qualification_scenario_matrix() for impl in scenario.provider_implementations}
    assert covered == set(DenseProviderImplementation)


def test_complete_room_is_primary_new_real_g2_comparison() -> None:
    scenario = next(item for item in qualification_scenario_matrix() if item.scenario_id == "g2_complete_room_mvs_da3")
    assert scenario.new_real_execution_required is True
    assert scenario.manual_review_required is True
    assert ComparisonMode.PRODUCTION_NATIVE in scenario.comparison_modes
    assert ComparisonMode.MATCHED_VIEW_DIAGNOSTIC in scenario.comparison_modes
    assert scenario.provider_implementations == (
        DenseProviderImplementation.COLMAP_MVS,
        DenseProviderImplementation.DA3,
    )


def test_house_context_is_historical_only_not_falsely_new() -> None:
    scenario = next(item for item in qualification_scenario_matrix() if item.scenario_id == "g2_house_mvs_da3_historical")
    assert scenario.comparison_modes == (ComparisonMode.HISTORICAL_READ_ONLY,)
    assert scenario.new_real_execution_required is False


def test_temporal_scenarios_require_order_and_only_temporal_providers() -> None:
    temporal = [item for item in qualification_scenario_matrix() if item.substep == "I4.3G3"]
    assert temporal
    for scenario in temporal:
        assert scenario.temporal_order_required is True
        assert set(scenario.provider_implementations) == {
            DenseProviderImplementation.ROLLING_DEPTH,
            DenseProviderImplementation.VIDEO_DEPTH_ANYTHING,
        }


def test_single_image_matrix_contains_exact_six_providers() -> None:
    expected = {
        DenseProviderImplementation.METRIC3D_V2,
        DenseProviderImplementation.UNIDEPTH_V2,
        DenseProviderImplementation.MOGE_2,
        DenseProviderImplementation.DEPTH_PRO,
        DenseProviderImplementation.DEPTH_ANYTHING_V2,
        DenseProviderImplementation.MARIGOLD,
    }
    for scenario in [item for item in qualification_scenario_matrix() if item.substep == "I4.3G4"]:
        assert set(scenario.provider_implementations) == expected


def test_run_evidence_schema_requires_identity_provenance_resources_and_authority() -> None:
    schema = qualification_run_evidence_schema()
    required = set(schema["required_top_level"])
    assert {"source_identity", "authoritative_sfm_identity", "adaptive_plan_identity"} <= required
    assert {"provider_provenance", "runtime_resources", "metrics", "artifacts", "authority"} <= required
    assert "metric_scale_claimed" in schema["authority_flags_must_remain_false"]


def test_metric_value_validation_boolean_gate() -> None:
    spec = next(item for item in qualification_metric_registry() if item.direction is MetricDirection.BOOLEAN_GATE_TRUE)
    validate_metric_value(spec, True)
    with pytest.raises(ReconstructionValidationError):
        validate_metric_value(spec, False)


def test_metric_value_validation_numeric_and_optional_missing() -> None:
    required = next(item for item in qualification_metric_registry() if item.metric_id == "wall_time_s")
    optional = next(item for item in qualification_metric_registry() if item.metric_id == "peak_host_memory_mib")
    validate_metric_value(required, 1.25)
    validate_metric_value(optional, None)
    with pytest.raises(ReconstructionValidationError):
        validate_metric_value(required, None)
    with pytest.raises(ReconstructionValidationError):
        validate_metric_value(required, float("nan"))


def test_g1_snapshot_closes_only_framework_and_authorizes_g2() -> None:
    snapshot = build_i4_3g1_framework_snapshot(
        i4_3a_snapshot=passing_i4_3a_snapshot(),
        i4_3f3_snapshot=passing_i4_3f3_snapshot(),
    )
    assert snapshot["status"] == "pass"
    assert snapshot["scope"]["real_provider_execution_started"] is False
    assert snapshot["scope"]["quality_results_consumed"] is False
    assert snapshot["decision"]["i4_3g1_framework_closed"] is True
    assert snapshot["decision"]["i4_3g2_mvs_da3_qualification_authorized"] is True
    assert snapshot["decision"]["global_scalar_score_forbidden"] is True
    assert snapshot["authority"]["provider_selected"] is False
    assert snapshot["authority"]["auto_policy_implemented"] is False
    assert snapshot["authority"]["hybrid_authorized"] is False
    assert snapshot["authority"]["permanent_global_view_cap"] is None


@pytest.mark.parametrize(
    "field",
    [
        "provider_selected",
        "provider_promoted",
        "auto_policy_implemented",
        "hybrid_authorized",
        "metric_scale_claimed",
        "measurement_authoritative",
        "certified_metrology",
        "surface_mesh_generated",
    ],
)
def test_g1_refuses_f3_authority_drift(field: str) -> None:
    f3 = passing_i4_3f3_snapshot()
    f3["authority"][field] = True
    with pytest.raises(ReconstructionValidationError, match="F3 authority drift"):
        validate_i4_3g1_predecessors(i4_3a_snapshot=passing_i4_3a_snapshot(), i4_3f3_snapshot=f3)


def test_g1_refuses_f3_permanent_cap() -> None:
    f3 = passing_i4_3f3_snapshot()
    f3["decision"]["permanent_global_view_cap"] = 96
    with pytest.raises(ReconstructionValidationError, match="permanent global view cap"):
        validate_i4_3g1_predecessors(i4_3a_snapshot=passing_i4_3a_snapshot(), i4_3f3_snapshot=f3)


def test_g1_refuses_missing_complete_room_requirement() -> None:
    a = passing_i4_3a_snapshot()
    a["qualification_plan"]["complete_room_mvs_vs_da3_required"] = False
    with pytest.raises(ReconstructionValidationError, match="complete_room_mvs_vs_da3_required"):
        validate_i4_3g1_predecessors(i4_3a_snapshot=a, i4_3f3_snapshot=passing_i4_3f3_snapshot())


def test_g1_refuses_provider_set_or_order_drift() -> None:
    f3 = passing_i4_3f3_snapshot()
    f3["validated_provider_implementations"] = list(reversed(f3["validated_provider_implementations"]))
    with pytest.raises(ReconstructionValidationError, match="exact 10 providers"):
        validate_i4_3g1_predecessors(i4_3a_snapshot=passing_i4_3a_snapshot(), i4_3f3_snapshot=f3)


def test_decision_rules_freeze_rerun_and_manual_review_fairness() -> None:
    rules = qualification_decision_rules()
    assert rules["infrastructure_failure_may_be_rerun_with_attempt_provenance"] is True
    assert rules["quality_failure_may_not_be_erased_by_untracked_retuning"] is True
    assert rules["same_manual_review_camera_and_framing_required"] is True
    assert rules["independent_visual_auto_fit_forbidden"] is True
