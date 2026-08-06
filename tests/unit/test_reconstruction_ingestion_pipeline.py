from __future__ import annotations

import pytest

from core.reconstruction import (
    FrameInvalidReason,
    FrameQualityPolicy,
    FrameSamplingPolicy,
    IngestionCachePolicy,
    KeyframeSelectionPolicy,
    ReconstructionCancelledError,
    ReconstructionIngestionPipeline,
    ReconstructionOperationControl,
    ReconstructionQualityProfile,
    ReconstructionSession,
)
from tests.reconstruction_ingestion_helpers import (
    INGESTION_SESSION_ID,
    SyntheticFrameSource,
    candidate,
    decode_issue,
    flat_frame,
    make_ingestion_metadata,
    textured_frame,
)


def _pipeline(source: SyntheticFrameSource) -> ReconstructionIngestionPipeline:
    return ReconstructionIngestionPipeline(
        session_id=INGESTION_SESSION_ID,
        source=source,
        sampling_policy=FrameSamplingPolicy(),
        keyframe_policy=KeyframeSelectionPolicy(
            minimum_spacing_ns=100_000_000,
            forced_spacing_ns=500_000_000,
            minimum_visual_change=0.03,
            maximum_keyframes=8,
        ),
        quality_policy=FrameQualityPolicy(),
        cache_policy=IngestionCachePolicy(
            maximum_cpu_bytes=16 * 1024**2,
            maximum_frames=8,
            gpu_staging_bytes=0,
        ),
    )


def test_pipeline_marks_duplicate_timestamp_dimension_and_quality_failures() -> None:
    image0 = textured_frame(0)
    source = SyntheticFrameSource(
        make_ingestion_metadata(),
        (
            candidate(0, image0, timestamp_ns=0),
            candidate(1, image0.copy(), timestamp_ns=100_000_000),
            candidate(2, textured_frame(2), timestamp_ns=100_000_000),
            candidate(3, textured_frame(3, width=80), timestamp_ns=300_000_000),
            candidate(4, flat_frame(), timestamp_ns=400_000_000),
            candidate(5, textured_frame(5), timestamp_ns=600_000_000),
        ),
        (decode_issue(6),),
    )
    result = _pipeline(source).run()
    assert result.decoded_frames == 6
    assert result.valid_frames == 2
    assert result.rejected_frames == 5
    assert result.sampled_frames == 2
    assert len(result.keyframes) == 2
    assert len(result.keyframe_images) == 2

    reasons = [set(record.frame.invalid_reasons) for record in result.records]
    assert FrameInvalidReason.DUPLICATE in reasons[1]
    assert FrameInvalidReason.TIMESTAMP_INVALID in reasons[2]
    assert FrameInvalidReason.DIMENSION_MISMATCH in reasons[3]
    assert FrameInvalidReason.INSUFFICIENT_QUALITY in reasons[4]
    assert [record.frame.timestamp_ns for record in result.records] == sorted(
        record.frame.timestamp_ns for record in result.records
    )


def test_result_applies_ordered_contracts_to_matching_session() -> None:
    metadata = make_ingestion_metadata()
    source = SyntheticFrameSource(
        metadata,
        tuple(candidate(i, textured_frame(i), timestamp_ns=i * 300_000_000) for i in range(4)),
    )
    result = _pipeline(source).run()
    session = ReconstructionSession(
        metadata,
        ReconstructionQualityProfile.EFFICIENT,
        session_id=INGESTION_SESSION_ID,
    )
    session.prepare()
    result.apply_to_session(session)
    snapshot = session.snapshot()
    assert len(snapshot.frames) == 4
    assert len(snapshot.keyframes) == len(result.keyframes)


def test_pipeline_honors_pre_requested_cancellation() -> None:
    source = SyntheticFrameSource(
        make_ingestion_metadata(),
        (candidate(0, textured_frame(0)),),
    )
    control = ReconstructionOperationControl()
    control.request_cancel()
    with pytest.raises(ReconstructionCancelledError, match="cancellation"):
        _pipeline(source).run(control=control)


def test_pipeline_rejects_unsupported_pixel_layout() -> None:
    import numpy as np

    source = SyntheticFrameSource(
        make_ingestion_metadata(),
        (candidate(0, np.zeros((72, 96, 4), dtype=np.uint8)),),
    )
    result = _pipeline(source).run()
    assert result.valid_frames == 0
    assert result.records[0].quality is None
    assert FrameInvalidReason.UNSUPPORTED_FORMAT in result.records[0].frame.invalid_reasons


def test_pipeline_rejects_non_increasing_source_indices() -> None:
    source = SyntheticFrameSource(
        make_ingestion_metadata(),
        (
            candidate(1, textured_frame(1), timestamp_ns=100_000_000),
            candidate(0, textured_frame(0), timestamp_ns=200_000_000),
        ),
    )
    with pytest.raises(Exception, match="sequence indices"):
        _pipeline(source).run()
