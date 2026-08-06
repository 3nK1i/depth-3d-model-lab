from __future__ import annotations

from core.reconstruction import (
    FrameQualityPolicy,
    FrameSamplingPolicy,
    IngestionCachePolicy,
    KeyframeSelectionPolicy,
    ReconstructionIngestionPipeline,
)
from tests.reconstruction_ingestion_helpers import (
    INGESTION_SESSION_ID,
    SyntheticFrameSource,
    candidate,
    make_ingestion_metadata,
    textured_frame,
)


def _run_large_sequence():
    candidates = tuple(
        candidate(index, textured_frame(index), timestamp_ns=index * 50_000_000)
        for index in range(1000)
    )
    pipeline = ReconstructionIngestionPipeline(
        session_id=INGESTION_SESSION_ID,
        source=SyntheticFrameSource(make_ingestion_metadata(fps=20.0), candidates),
        sampling_policy=FrameSamplingPolicy(maximum_sampled_frames=300),
        keyframe_policy=KeyframeSelectionPolicy(
            minimum_spacing_ns=100_000_000,
            forced_spacing_ns=500_000_000,
            minimum_visual_change=0.04,
            maximum_keyframes=20,
        ),
        quality_policy=FrameQualityPolicy(),
        cache_policy=IngestionCachePolicy(
            maximum_cpu_bytes=5 * 96 * 72 * 3,
            maximum_frames=5,
            gpu_staging_bytes=0,
        ),
    )
    return pipeline.run()


def test_large_sequence_is_deterministic_and_cache_remains_bounded() -> None:
    first = _run_large_sequence()
    second = _run_large_sequence()
    assert first.decoded_frames == 1000
    assert first.valid_frames == 1000
    assert first.sampled_frames == 300
    assert len(first.keyframes) <= 20
    assert first.cache_stats.current_frames <= 5
    assert first.cache_stats.current_bytes <= first.cache_policy.maximum_cpu_bytes
    assert [item.frame_id for item in first.keyframes] == [item.frame_id for item in second.keyframes]
