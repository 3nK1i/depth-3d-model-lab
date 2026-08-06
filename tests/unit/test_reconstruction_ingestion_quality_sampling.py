from __future__ import annotations

import numpy as np
import pytest

from core.contracts import ImageSize
from core.reconstruction import (
    FrameQualityPolicy,
    FrameSampler,
    FrameSamplingMode,
    FrameSamplingPolicy,
    KeyframeSelectionPolicy,
    KeyframeSelector,
    ReconstructionFrame,
    ReconstructionValidationError,
    analyze_frame_quality,
)
from tests.reconstruction_ingestion_helpers import (
    INGESTION_SESSION_ID,
    flat_frame,
    textured_frame,
)


def _frame(index: int, timestamp_ns: int) -> ReconstructionFrame:
    return ReconstructionFrame(
        frame_id=f"frame-{index}",
        session_id=INGESTION_SESSION_ID,
        source_id="source",
        sequence_index=index,
        timestamp_ns=timestamp_ns,
        image_size=ImageSize(96, 72),
    )


def test_quality_metrics_accept_textured_and_reject_flat_frames() -> None:
    textured = analyze_frame_quality(textured_frame(1))
    flat = analyze_frame_quality(flat_frame())
    assert textured.accepted
    assert textured.quality_score > flat.quality_score
    assert not flat.accepted
    assert "blur" in flat.rejection_details


def test_quality_policy_rejects_invalid_thresholds() -> None:
    with pytest.raises(ReconstructionValidationError, match="Luma thresholds"):
        FrameQualityPolicy(minimum_mean_luma=200.0, maximum_mean_luma=100.0)


def test_every_frame_and_fixed_interval_sampling_are_deterministic() -> None:
    every = FrameSampler(FrameSamplingPolicy())
    assert [every.should_select(_frame(i, i * 100)) for i in range(4)] == [True] * 4
    fixed = FrameSampler(
        FrameSamplingPolicy(mode=FrameSamplingMode.FIXED_INTERVAL, frame_interval=2)
    )
    assert [fixed.should_select(_frame(i, i * 100)) for i in range(5)] == [
        True,
        False,
        True,
        False,
        True,
    ]


def test_target_fps_sampling_uses_timestamps_and_limit() -> None:
    sampler = FrameSampler(
        FrameSamplingPolicy(
            mode=FrameSamplingMode.TARGET_FPS,
            target_fps_hz=2.0,
            maximum_sampled_frames=2,
        )
    )
    timestamps = [0, 200_000_000, 500_000_000, 1_000_000_000, 1_500_000_000]
    assert [sampler.should_select(_frame(i, ts)) for i, ts in enumerate(timestamps)] == [
        True,
        False,
        True,
        False,
        False,
    ]


def test_keyframe_selection_uses_change_then_forced_spacing() -> None:
    policy = KeyframeSelectionPolicy(
        minimum_spacing_ns=100,
        forced_spacing_ns=400,
        minimum_visual_change=0.03,
        maximum_keyframes=4,
    )
    selector = KeyframeSelector(policy)
    q0 = analyze_frame_quality(textured_frame(0))
    first = selector.consider(_frame(0, 0), textured_frame(0), q0)
    assert first is not None and first.reason.value == "first_frame"
    too_close = selector.consider(_frame(1, 50), textured_frame(1), analyze_frame_quality(textured_frame(1)))
    assert too_close is None
    changed_image = 255 - textured_frame(0)
    changed = selector.consider(_frame(2, 150), changed_image, analyze_frame_quality(changed_image))
    assert changed is not None and changed.reason.value == "visual_change"
    same_image = changed_image.copy()
    forced = selector.consider(_frame(3, 600), same_image, analyze_frame_quality(same_image))
    assert forced is not None and forced.reason.value == "temporal_spacing"
    assert [item.selection_sequence for item in selector.keyframes] == [0, 1, 2]
