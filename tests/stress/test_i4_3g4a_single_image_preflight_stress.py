from __future__ import annotations

from copy import deepcopy

from core.reconstruction.single_image_provider_qualification import build_i4_3g4a_preflight_snapshot
from tests.i4_3g4a_helpers import all_bindings, all_runtime_probes, passing_g1_snapshot, passing_g3c_closure


def test_g4a_preflight_build_is_deterministic_and_does_not_mutate_inputs_under_repetition() -> None:
    g1 = passing_g1_snapshot()
    g3c = passing_g3c_closure()
    bindings = all_bindings()
    runtimes = all_runtime_probes()
    frozen = (deepcopy(g1), deepcopy(g3c), deepcopy(runtimes))
    first = None
    for _ in range(100):
        snapshot = build_i4_3g4a_preflight_snapshot(
            i4_3g1_snapshot=g1,
            i4_3g3c_closure=g3c,
            scenario_bindings=bindings,
            runtime_probes=runtimes,
            free_disk_bytes=64 * 1024**3,
            minimum_free_disk_bytes=32 * 1024**3,
        )
        if first is None:
            first = snapshot
        else:
            assert snapshot == first
    assert g1 == frozen[0]
    assert g3c == frozen[1]
    assert runtimes == frozen[2]
