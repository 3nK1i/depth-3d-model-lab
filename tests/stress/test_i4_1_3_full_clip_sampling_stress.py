from __future__ import annotations

import numpy as np

from core.reconstruction import (
    FrameSamplingMode,
    FrameSamplingPlanMode,
    FrameSamplingPolicy,
    build_frame_sampling_plan,
)


def _plan(total: int, fps: float, target: float, maximum: int):
    return build_frame_sampling_plan(
        FrameSamplingPolicy(
            mode=FrameSamplingMode.TARGET_FPS,
            target_fps_hz=target,
            maximum_sampled_frames=maximum,
        ),
        estimated_total_frames=total,
        expected_frame_rate_hz=fps,
    )


def test_random_long_source_plans_are_monotonic_unique_and_full_span() -> None:
    rng = np.random.default_rng(413)
    for _ in range(250):
        maximum = int(rng.integers(3, 100))
        total = int(rng.integers(maximum + 20, 20000))
        fps = float(rng.uniform(15.0, 120.0))
        target = float(rng.uniform(1.0, min(12.0, fps)))
        plan = _plan(total, fps, target, maximum)
        if plan.mode is FrameSamplingPlanMode.UNIFORM_FULL_SPAN:
            indices = plan.target_sequence_indices
            assert len(indices) == maximum
            assert indices[0] == 0 and indices[-1] == total - 1
            assert all(a < b for a, b in zip(indices, indices[1:]))


def test_large_sources_never_allocate_more_than_cap_targets() -> None:
    for total in (10_000, 100_000, 1_000_000):
        plan = _plan(total, 30.0, 2.0, 80)
        assert plan.mode is FrameSamplingPlanMode.UNIFORM_FULL_SPAN
        assert len(plan.target_sequence_indices) == 80


def test_plans_are_bitwise_deterministic_across_repeated_builds() -> None:
    reference = _plan(1516, 29.97, 2.0, 40).as_dict()
    for _ in range(500):
        assert _plan(1516, 29.97, 2.0, 40).as_dict() == reference


def test_short_source_native_mode_never_exceeds_cap_nominal_count() -> None:
    for total in range(3, 200):
        plan = _plan(total, 30.0, 2.0, 40)
        if plan.mode is FrameSamplingPlanMode.POLICY_NATIVE:
            assert plan.planned_sample_count is None or plan.planned_sample_count <= 40


def test_full_span_spacing_has_at_most_one_frame_quantization_jitter() -> None:
    for total, maximum in ((1516, 40), (1001, 37), (9999, 80)):
        plan = _plan(total, 30.0, 2.0, maximum)
        indices = np.asarray(plan.target_sequence_indices, dtype=int)
        steps = np.diff(indices)
        assert steps.max() - steps.min() <= 1
