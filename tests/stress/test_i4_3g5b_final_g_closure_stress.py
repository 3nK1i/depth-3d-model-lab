from __future__ import annotations

from copy import deepcopy

import pytest

from core.reconstruction.cross_provider_qualification_closure import build_i4_3g5b_final_closure
from core.reconstruction.errors import ReconstructionValidationError
from tests.i4_3g5b_helpers import passing_g5a, passing_g5b_bindings


def test_g5b_repeated_closure_is_deterministic() -> None:
    first = build_i4_3g5b_final_closure(i4_3g5a_synthesis=passing_g5a(), evidence_bindings=passing_g5b_bindings(), chain_rebuild_verified=True)
    for _ in range(50):
        assert build_i4_3g5b_final_closure(i4_3g5a_synthesis=passing_g5a(), evidence_bindings=passing_g5b_bindings(), chain_rebuild_verified=True) == first


@pytest.mark.parametrize("field", [
    "provider_selected", "provider_promoted", "auto_policy_implemented", "hybrid_authorized",
    "metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated",
])
def test_g5b_repeatedly_fails_closed_on_authority_drift(field: str) -> None:
    for _ in range(5):
        g5a = passing_g5a(); g5a["authority"][field] = True
        with pytest.raises(ReconstructionValidationError):
            build_i4_3g5b_final_closure(i4_3g5a_synthesis=g5a, evidence_bindings=passing_g5b_bindings(), chain_rebuild_verified=True)


def test_g5b_preserves_input_under_repeated_calls() -> None:
    g5a = passing_g5a(); before = deepcopy(g5a)
    for _ in range(30):
        build_i4_3g5b_final_closure(i4_3g5a_synthesis=g5a, evidence_bindings=passing_g5b_bindings(), chain_rebuild_verified=True)
    assert g5a == before
