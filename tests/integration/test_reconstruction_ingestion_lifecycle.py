from __future__ import annotations

import json
from pathlib import Path

import cv2

from core.reconstruction import (
    FrameQualityPolicy,
    FrameSamplingMode,
    FrameSamplingPolicy,
    ImageSequenceReader,
    IngestionCachePolicy,
    KeyframeSelectionPolicy,
    ReconstructionEventRecorder,
    ReconstructionIngestionPipeline,
    ReconstructionQualityProfile,
    ReconstructionSession,
)
from tests.reconstruction_ingestion_helpers import INGESTION_SESSION_ID, textured_frame


def test_image_sequence_ingestion_to_session_and_report(tmp_path: Path) -> None:
    sequence_dir = tmp_path / "sequence"
    sequence_dir.mkdir()
    for index in (10, 2, 1, 3, 4, 5, 6, 7):
        assert cv2.imwrite(str(sequence_dir / f"frame{index}.png"), textured_frame(index))
    (sequence_dir / "frame8.png").write_bytes(b"corrupt image")

    reader = ImageSequenceReader.from_directory(sequence_dir, frame_rate_hz=8.0)
    pipeline = ReconstructionIngestionPipeline(
        session_id=INGESTION_SESSION_ID,
        source=reader,
        sampling_policy=FrameSamplingPolicy(
            mode=FrameSamplingMode.FIXED_INTERVAL,
            frame_interval=2,
        ),
        keyframe_policy=KeyframeSelectionPolicy(
            minimum_spacing_ns=100_000_000,
            forced_spacing_ns=500_000_000,
            minimum_visual_change=0.02,
            maximum_keyframes=8,
        ),
        quality_policy=FrameQualityPolicy(),
        cache_policy=IngestionCachePolicy(
            maximum_cpu_bytes=32 * 1024**2,
            maximum_frames=8,
            gpu_staging_bytes=0,
        ),
    )
    events = ReconstructionEventRecorder(INGESTION_SESSION_ID)
    result = pipeline.run(events=events)

    assert result.decoded_frames == 8
    assert len(result.read_issues) == 1
    assert result.valid_frames == 8
    assert result.sampled_frames == 5
    assert 1 <= len(result.keyframes) <= 4
    assert events.events[0].stage.value == "ingestion"
    assert events.events[-1].stage.value == "keyframe_selection"
    ingestion_progress = [
        event.progress_fraction
        for event in events.events
        if event.stage.value == "ingestion" and event.progress_fraction is not None
    ]
    assert ingestion_progress[-1] == 1.0

    session = ReconstructionSession(
        reader.metadata,
        ReconstructionQualityProfile.EFFICIENT,
        session_id=INGESTION_SESSION_ID,
    )
    session.prepare()
    result.apply_to_session(session)
    snapshot = session.snapshot()
    assert len(snapshot.frames) == 8
    assert len(snapshot.keyframes) == len(result.keyframes)

    report_path = tmp_path / "ingestion-report.json"
    report_path.write_text(json.dumps(result.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    restored = json.loads(report_path.read_text(encoding="utf-8"))
    assert restored["schema_version"] == 1
    assert restored["summary"]["decoded_frames"] == 8
    assert restored["summary"]["keyframes"] == len(result.keyframes)
