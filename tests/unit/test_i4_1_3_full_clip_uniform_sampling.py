from __future__ import annotations

import pytest

from core.contracts import ImageSize
from core.reconstruction import (
    FrameSampler,
    FrameSamplingMode,
    FrameSamplingPlanMode,
    FrameSamplingPolicy,
    ReconstructionFrame,
    ReconstructionValidationError,
    build_frame_sampling_plan,
)

SESSION = "00000000-0000-4000-8000-000000008413"


def _frame(index: int, *, valid: bool = True) -> ReconstructionFrame:
    kwargs = {}
    if not valid:
        from core.reconstruction import FrameInvalidReason, FrameValidity
        kwargs = {
            "validity": FrameValidity.INVALID,
            "invalid_reasons": (FrameInvalidReason.INSUFFICIENT_QUALITY,),
        }
    return ReconstructionFrame(
        frame_id=f"frame-{index}",
        session_id=SESSION,
        source_id="video-source",
        sequence_index=index,
        timestamp_ns=index * 100_000_000,
        image_size=ImageSize(96, 72),
        **kwargs,
    )


def _policy(*, target: float = 2.0, maximum: int = 40) -> FrameSamplingPolicy:
    return FrameSamplingPolicy(
        mode=FrameSamplingMode.TARGET_FPS,
        target_fps_hz=target,
        maximum_sampled_frames=maximum,
    )


def test_long_video_switches_to_uniform_full_span_plan() -> None:
    plan = build_frame_sampling_plan(
        _policy(), estimated_total_frames=1516, expected_frame_rate_hz=29.97
    )
    assert plan.mode is FrameSamplingPlanMode.UNIFORM_FULL_SPAN
    assert plan.full_span_cap_active is True
    assert plan.planned_sample_count == 40
    assert len(plan.target_sequence_indices) == 40
    assert plan.target_sequence_indices[0] == 0
    assert plan.target_sequence_indices[-1] == 1515


def test_horse_like_effective_rate_spans_full_source() -> None:
    plan = build_frame_sampling_plan(
        _policy(), estimated_total_frames=1516, expected_frame_rate_hz=29.97
    )
    assert plan.estimated_duration_seconds == pytest.approx(1515 / 29.97)
    assert plan.effective_target_fps_hz == pytest.approx(39 / (1515 / 29.97))
    assert 0.76 < float(plan.effective_target_fps_hz) < 0.78


def test_short_video_preserves_historical_target_fps_behavior() -> None:
    plan = build_frame_sampling_plan(
        _policy(), estimated_total_frames=301, expected_frame_rate_hz=30.0
    )
    assert plan.mode is FrameSamplingPlanMode.POLICY_NATIVE
    assert plan.requested_target_fps_hz == 2.0
    assert plan.effective_target_fps_hz == 2.0
    assert plan.nominal_sample_count == 21
    assert plan.target_sequence_indices == ()


def test_exact_cap_boundary_does_not_change_sampling_mode() -> None:
    # 19.5 s at 2 Hz -> exactly 40 nominal samples including the first frame.
    plan = build_frame_sampling_plan(
        _policy(), estimated_total_frames=586, expected_frame_rate_hz=30.0
    )
    assert plan.nominal_sample_count == 40
    assert plan.mode is FrameSamplingPlanMode.POLICY_NATIVE


def test_uniform_targets_are_strictly_monotonic_unique_and_cover_middle() -> None:
    plan = build_frame_sampling_plan(
        _policy(maximum=8), estimated_total_frames=101, expected_frame_rate_hz=10.0
    )
    indices = plan.target_sequence_indices
    assert indices == tuple(sorted(set(indices)))
    assert indices[0] == 0 and indices[-1] == 100
    assert any(40 <= value <= 60 for value in indices)


def test_uniform_sampler_selects_planned_indices() -> None:
    policy = _policy(target=5.0, maximum=5)
    plan = build_frame_sampling_plan(
        policy, estimated_total_frames=21, expected_frame_rate_hz=10.0
    )
    sampler = FrameSampler(policy, plan=plan)
    selected = [i for i in range(21) if sampler.should_select(_frame(i))]
    assert selected == list(plan.target_sequence_indices)
    assert selected[0] == 0 and selected[-1] == 20


def test_invalid_target_uses_first_following_valid_frame() -> None:
    policy = _policy(target=5.0, maximum=5)
    plan = build_frame_sampling_plan(
        policy, estimated_total_frames=21, expected_frame_rate_hz=10.0
    )
    assert plan.target_sequence_indices[1] == 5
    sampler = FrameSampler(policy, plan=plan)
    selected = []
    for i in range(21):
        if sampler.should_select(_frame(i, valid=i != 5)):
            selected.append(i)
    assert selected[:2] == [0, 6]
    assert sampler.selected_count == 5


def test_every_frame_policy_remains_native_even_with_cap_metadata() -> None:
    policy = FrameSamplingPolicy(maximum_sampled_frames=40)
    plan = build_frame_sampling_plan(
        policy, estimated_total_frames=2000, expected_frame_rate_hz=30.0
    )
    assert plan.mode is FrameSamplingPlanMode.POLICY_NATIVE
    assert plan.full_span_cap_active is False


def test_missing_source_metadata_fails_safe_to_native_policy() -> None:
    plan = build_frame_sampling_plan(_policy())
    assert plan.mode is FrameSamplingPlanMode.POLICY_NATIVE
    assert plan.estimated_total_frames is None
    assert plan.target_sequence_indices == ()


def test_plan_serialization_records_full_span_provenance() -> None:
    plan = build_frame_sampling_plan(
        _policy(), estimated_total_frames=1516, expected_frame_rate_hz=29.97
    )
    payload = plan.as_dict()
    assert payload["mode"] == "uniform_full_span"
    assert payload["full_span_cap_active"] is True
    assert payload["requested_target_fps_hz"] == 2.0
    assert payload["planned_sample_count"] == 40
    assert payload["target_sequence_indices"][0] == 0
    assert payload["target_sequence_indices"][-1] == 1515


def test_invalid_source_count_fails_closed() -> None:
    with pytest.raises(ReconstructionValidationError, match="estimated_total_frames"):
        build_frame_sampling_plan(
            _policy(), estimated_total_frames=0, expected_frame_rate_hz=30.0
        )


def test_invalid_source_rate_fails_closed() -> None:
    with pytest.raises(ReconstructionValidationError, match="expected_frame_rate_hz"):
        build_frame_sampling_plan(
            _policy(), estimated_total_frames=100, expected_frame_rate_hz=0.0
        )
