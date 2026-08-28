from copy import deepcopy

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.temporal_provider_qualification import build_i4_3g3a_preflight_snapshot
from tests.i4_3g3a_helpers import all_bindings, passing_g1_snapshot, passing_g2_closure, runtime_probe


def test_g3a_repeated_build_is_deterministic_and_does_not_mutate_inputs() -> None:
    g1 = passing_g1_snapshot(); g2 = passing_g2_closure(); bindings = all_bindings()
    g1_before = deepcopy(g1); g2_before = deepcopy(g2)
    results = [
        build_i4_3g3a_preflight_snapshot(
            i4_3g1_snapshot=g1, i4_3g2_closure=g2, scenario_bindings=bindings,
            video_depth_anything_runtime=runtime_probe("video_depth_anything"),
            rolling_depth_runtime=runtime_probe("rolling_depth"),
        )
        for _ in range(25)
    ]
    assert all(item == results[0] for item in results[1:])
    assert g1 == g1_before and g2 == g2_before


@pytest.mark.parametrize("key", [
    "metric_scale_claimed", "measurement_authoritative", "certified_metrology", "provider_selected",
])
def test_g3a_fails_closed_on_g1_authority_drift(key: str) -> None:
    g1 = passing_g1_snapshot(); g1["authority"][key] = True
    with pytest.raises(ReconstructionValidationError, match="authority drift"):
        build_i4_3g3a_preflight_snapshot(
            i4_3g1_snapshot=g1, i4_3g2_closure=passing_g2_closure(), scenario_bindings=all_bindings(),
            video_depth_anything_runtime=runtime_probe("video_depth_anything"),
            rolling_depth_runtime=runtime_probe("rolling_depth"),
        )
