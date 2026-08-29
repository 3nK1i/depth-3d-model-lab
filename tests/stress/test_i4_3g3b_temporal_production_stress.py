from __future__ import annotations

from copy import deepcopy

from core.reconstruction.temporal_provider_production_qualification import (
    build_g3b_runtime_preflight_snapshot,
    validate_temporal_production_plan,
    validate_temporal_provider_run_evidence,
)
from tests.i4_3g3b_helpers import (
    current_rolling_runtime,
    current_vda_runtime,
    passing_g3a_snapshot,
    production_plan,
    provider_evidence,
    sfm_runtime,
)


def test_g3b_contract_validation_is_deterministic_and_non_mutating() -> None:
    snapshot = passing_g3a_snapshot()
    original = deepcopy(snapshot)
    for _ in range(50):
        for scenario_id in (
            "g3_temporal_static_sweep",
            "g3_temporal_motion",
            "g3_temporal_low_texture",
        ):
            preflight = build_g3b_runtime_preflight_snapshot(
                i4_3g3a_snapshot=snapshot,
                scenario_id=scenario_id,
                sfm_runtime=sfm_runtime(),
                video_depth_anything_runtime=current_vda_runtime(snapshot),
                rolling_depth_runtime=current_rolling_runtime(snapshot),
                verify_files=False,
            )
            assert preflight["pass"] is True
        for implementation in ("rolling_depth", "video_depth_anything"):
            validate_temporal_production_plan(production_plan(implementation), implementation=implementation)
            validate_temporal_provider_run_evidence(provider_evidence(implementation))
    assert snapshot == original
