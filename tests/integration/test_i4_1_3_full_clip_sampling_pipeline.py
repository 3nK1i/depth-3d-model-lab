from __future__ import annotations

import json

from core.contracts import ImageSize
from core.reconstruction import (
    FrameQualityPolicy,
    FrameSamplingMode,
    FrameSamplingPolicy,
    IngestionCachePolicy,
    InputSourceKind,
    InputSourceMetadata,
    KeyframeSelectionPolicy,
    ReconstructionEventRecorder,
    ReconstructionIngestionPipeline,
    TimestampOrigin,
)
from tests.reconstruction_ingestion_helpers import (
    INGESTION_SESSION_ID,
    SyntheticFrameSource,
    candidate,
    textured_frame,
)


def _video_source(count: int, *, fps: float = 10.0) -> SyntheticFrameSource:
    metadata = InputSourceMetadata(
        source_id="i4-1-3-video",
        kind=InputSourceKind.RECORDED_VIDEO,
        display_name="synthetic.mp4",
        source_uri="file:///synthetic.mp4",
        expected_image_size=ImageSize(96, 72),
        expected_frame_rate_hz=fps,
        timestamp_origin=TimestampOrigin.PRESENTATION_TIMESTAMP,
        fingerprint="i4-1-3-fingerprint",
        metadata={"reported_frame_count": count},
    )
    interval_ns = int(round(1_000_000_000 / fps))
    frames = tuple(
        candidate(i, textured_frame(i), timestamp_ns=i * interval_ns)
        for i in range(count)
    )
    return SyntheticFrameSource(metadata, frames)


def _run(count: int, *, fps: float = 10.0, target: float = 2.0, maximum: int = 10, events=None):
    return ReconstructionIngestionPipeline(
        session_id=INGESTION_SESSION_ID,
        source=_video_source(count, fps=fps),
        sampling_policy=FrameSamplingPolicy(
            mode=FrameSamplingMode.TARGET_FPS,
            target_fps_hz=target,
            maximum_sampled_frames=maximum,
        ),
        keyframe_policy=KeyframeSelectionPolicy(
            minimum_spacing_ns=1,
            forced_spacing_ns=1,
            minimum_visual_change=0.0,
            minimum_quality_score=0.0,
            maximum_keyframes=maximum,
        ),
        quality_policy=FrameQualityPolicy(),
        cache_policy=IngestionCachePolicy(
            maximum_cpu_bytes=64 * 1024**2,
            maximum_frames=maximum,
            gpu_staging_bytes=0,
        ),
    ).run(events=events)


def test_pipeline_distributes_capped_samples_across_entire_video() -> None:
    result = _run(101, maximum=10)
    sampled = [record.frame.sequence_index for record in result.records if record.sampled]
    assert result.sampling_plan.mode.value == "uniform_full_span"
    assert len(sampled) == 10
    assert sampled[0] == 0
    assert sampled[-1] == 100
    assert any(40 <= value <= 60 for value in sampled)


def test_short_video_keeps_native_two_hz_sampling() -> None:
    result = _run(31, maximum=10)
    sampled = [record.frame.sequence_index for record in result.records if record.sampled]
    assert result.sampling_plan.mode.value == "policy_native"
    assert sampled == [0, 5, 10, 15, 20, 25, 30]


def test_ingestion_json_records_resolved_sampling_plan() -> None:
    result = _run(101, maximum=10)
    payload = json.loads(json.dumps(result.as_dict()))
    plan = payload["sampling_plan"]
    assert plan["mode"] == "uniform_full_span"
    assert plan["planned_sample_count"] == 10
    assert plan["target_sequence_indices"][0] == 0
    assert plan["target_sequence_indices"][-1] == 100


def test_keyframes_follow_full_span_samples_when_quality_is_valid() -> None:
    result = _run(101, maximum=10)
    sampled_ids = {record.frame.frame_id for record in result.records if record.sampled}
    assert len(result.keyframes) == 10
    assert {keyframe.frame_id for keyframe in result.keyframes} == sampled_ids


def test_event_contract_reports_full_span_sampling_mode() -> None:
    events = ReconstructionEventRecorder(INGESTION_SESSION_ID)
    _run(101, maximum=10, events=events)
    final = [event for event in events.events if event.stage.value == "keyframe_selection"][-1]
    metadata = final.as_dict()["metadata"]
    assert metadata["sampling_plan_mode"] == "uniform_full_span"
    assert metadata["full_span_cap_active"] is True


def test_real_recorded_video_preflight_reports_full_span_plan(tmp_path) -> None:
    import cv2
    import numpy as np
    import subprocess
    import sys
    from pathlib import Path

    video = tmp_path / "long.mp4"
    writer = cv2.VideoWriter(
        str(video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    for i in range(101):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :, 0] = i % 255
        writer.write(frame)
    writer.release()
    output = tmp_path / "preflight.json"
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts/run_i4_1_3_sampling_preflight.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--video",
            str(video),
            "--target-fps",
            "2",
            "--max-keyframes",
            "10",
            "--output",
            str(output),
        ],
        cwd=root,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(output.read_text())
    assert payload["sampling_plan"]["mode"] == "uniform_full_span"
    assert payload["sampling_plan"]["planned_sample_count"] == 10
    assert payload["planned_source_span_fraction"] == 1.0
