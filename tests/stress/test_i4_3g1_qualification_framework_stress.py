from __future__ import annotations

import copy

import pytest

from core.reconstruction.cross_provider_qualification import (
    build_i4_3g1_framework_snapshot,
    qualification_scenario_matrix,
    validate_i4_3g1_predecessors,
)
from core.reconstruction.errors import ReconstructionValidationError
from tests.i4_3g1_helpers import passing_i4_3a_snapshot, passing_i4_3f3_snapshot


def test_framework_build_is_deterministic_across_many_repetitions() -> None:
    expected = build_i4_3g1_framework_snapshot(
        i4_3a_snapshot=passing_i4_3a_snapshot(), i4_3f3_snapshot=passing_i4_3f3_snapshot()
    )
    for _ in range(250):
        actual = build_i4_3g1_framework_snapshot(
            i4_3a_snapshot=passing_i4_3a_snapshot(), i4_3f3_snapshot=passing_i4_3f3_snapshot()
        )
        assert actual == expected


def test_every_manual_scenario_has_nonempty_frozen_dimensions() -> None:
    for scenario in qualification_scenario_matrix():
        if scenario.manual_review_required:
            assert len(scenario.manual_review_dimensions) >= 6
            assert len(set(scenario.manual_review_dimensions)) == len(scenario.manual_review_dimensions)


@pytest.mark.parametrize("field", [
    "backend_started", "provider_selected", "provider_promoted", "auto_policy_implemented",
    "hybrid_implementation_authorized", "authoritative_sparse_modified", "metric_scale_claimed",
    "measurement_authoritative", "certified_metrology", "surface_mesh_generated",
])
def test_framework_fails_closed_on_i4_3a_authority_drift(field: str) -> None:
    a = passing_i4_3a_snapshot()
    a["authority"][field] = True
    with pytest.raises(ReconstructionValidationError, match="I4.3A authority drift"):
        validate_i4_3g1_predecessors(i4_3a_snapshot=a, i4_3f3_snapshot=passing_i4_3f3_snapshot())


def test_framework_fails_closed_if_g_authorization_is_removed() -> None:
    f3 = passing_i4_3f3_snapshot()
    f3["decision"]["i4_3g_cross_provider_qualification_authorized"] = False
    with pytest.raises(ReconstructionValidationError, match="F3 closure"):
        validate_i4_3g1_predecessors(i4_3a_snapshot=passing_i4_3a_snapshot(), i4_3f3_snapshot=f3)


def test_framework_fails_closed_if_hybrid_was_pre_authorized() -> None:
    a = passing_i4_3a_snapshot()
    a["qualification_plan"]["hybrid_implementation_authorized"] = True
    with pytest.raises(ReconstructionValidationError, match="hybrid authorization"):
        validate_i4_3g1_predecessors(i4_3a_snapshot=a, i4_3f3_snapshot=passing_i4_3f3_snapshot())
