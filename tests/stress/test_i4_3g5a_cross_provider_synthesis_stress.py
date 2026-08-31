from __future__ import annotations

from copy import deepcopy

import pytest

from core.reconstruction.cross_provider_evidence_synthesis import build_i4_3g5a_cross_provider_synthesis
from core.reconstruction.errors import ReconstructionValidationError
from tests.i4_3g5a_helpers import passing_g1, passing_g2, passing_g3, passing_g4


def test_g5a_repeated_synthesis_is_deterministic() -> None:
    first = build_i4_3g5a_cross_provider_synthesis(
        i4_3g1_snapshot=passing_g1(), i4_3g2_closure=passing_g2(),
        i4_3g3c_closure=passing_g3(), i4_3g4c_closure=passing_g4(),
    )
    for _ in range(40):
        assert build_i4_3g5a_cross_provider_synthesis(
            i4_3g1_snapshot=passing_g1(), i4_3g2_closure=passing_g2(),
            i4_3g3c_closure=passing_g3(), i4_3g4c_closure=passing_g4(),
        ) == first


@pytest.mark.parametrize("field", [
    "provider_selected", "provider_promoted", "auto_policy_implemented", "hybrid_authorized",
    "metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated",
])
def test_g5a_repeatedly_fails_closed_on_g4_authority_drift(field: str) -> None:
    for _ in range(5):
        g4 = passing_g4()
        g4["authority"][field] = True
        with pytest.raises(ReconstructionValidationError):
            build_i4_3g5a_cross_provider_synthesis(
                i4_3g1_snapshot=passing_g1(), i4_3g2_closure=passing_g2(),
                i4_3g3c_closure=passing_g3(), i4_3g4c_closure=g4,
            )


def test_g5a_preserves_inputs_under_repeated_calls() -> None:
    g1, g2, g3, g4 = passing_g1(), passing_g2(with_scenario_detail=True), passing_g3(), passing_g4()
    originals = deepcopy((g1, g2, g3, g4))
    for _ in range(25):
        build_i4_3g5a_cross_provider_synthesis(
            i4_3g1_snapshot=g1, i4_3g2_closure=g2, i4_3g3c_closure=g3, i4_3g4c_closure=g4,
        )
    assert (g1, g2, g3, g4) == originals
