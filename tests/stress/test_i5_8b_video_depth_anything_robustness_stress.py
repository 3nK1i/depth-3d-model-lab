from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_video_depth_anything_robustness_replay import analyze_i5_8b_provider_summary, build_i5_8b_video_depth_anything_robustness_replay
from tests.i5_8b_helpers import build_args, clone, provider_summary, runtime_preflight, scenario_record


def _resources() -> dict:
    return {"peak_vram_mib": 2048.0, "peak_system_ram_used_mib": 8192.0}


def test_i58b_repeated_analysis_is_deterministic() -> None:
    summary = provider_summary("degraded", rejected_views=3)
    expected = None
    for _ in range(40):
        current = analyze_i5_8b_provider_summary(summary, wall_time_s=2.0, resource_observation=_resources(), expected_runtime_config=runtime_preflight()["runtime_config"])
        if expected is None:
            expected = current
        assert current == expected


def test_i58b_repeated_builder_does_not_mutate_inputs() -> None:
    args = build_args()
    before = clone(args["scenario_records"])
    for _ in range(25):
        out = build_i5_8b_video_depth_anything_robustness_replay(**args)
        assert out["pass"] is True
    assert args["scenario_records"] == before


@pytest.mark.parametrize("index", [1, 2, 3, 4, 5])
def test_i58b_rejects_nonmonotonic_worker_source_order(index: int) -> None:
    summary = provider_summary(views=7)
    summary["worker"]["frame_records"][index]["source_sequence_index"] = 0
    with pytest.raises(ReconstructionValidationError):
        analyze_i5_8b_provider_summary(summary, wall_time_s=2.0, resource_observation=_resources(), expected_runtime_config=runtime_preflight()["runtime_config"])


@pytest.mark.parametrize("scenario", ["castle_low_overlap", "castle_weak_parallax", "castle_overexposure", "castle_symmetry", "castle_dynamic_proxy"])
def test_i58b_degraded_stress_scenario_never_triggers_retry_policy(scenario: str) -> None:
    args = build_args()
    args["scenario_records"][scenario] = scenario_record(scenario, "degraded", rejected_views=1)
    out = build_i5_8b_video_depth_anything_robustness_replay(**args)
    assert out["quality_status"] == "degraded"
    assert out["scope"]["retry_until_pass"] is False
    assert out["scope"]["thresholds_modified"] is False
