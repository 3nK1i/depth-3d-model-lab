from __future__ import annotations

from pathlib import Path

import cv2

from core.reconstruction import (
    FrameQualityPolicy,
    FrameSamplingMode,
    FrameSamplingPolicy,
    ImageSequenceReader,
    IngestionCachePolicy,
    KeyframeSelectionPolicy,
    ReconstructionIngestionPipeline,
)
from tests.reconstruction_ingestion_helpers import textured_frame

SFM_SESSION_ID = "00000000-0000-4000-8000-000000008003"


def write_sfm_sequence(directory: Path, *, count: int = 6) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        image = textured_frame(index, width=160, height=120)
        path = directory / f"frame{index + 1}.png"
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not write test image: {path}")
    return directory


def make_sfm_ingestion_result(directory: Path, *, count: int = 6):
    write_sfm_sequence(directory, count=count)
    source = ImageSequenceReader.from_directory(directory, frame_rate_hz=10.0)
    pipeline = ReconstructionIngestionPipeline(
        session_id=SFM_SESSION_ID,
        source=source,
        sampling_policy=FrameSamplingPolicy(mode=FrameSamplingMode.EVERY_FRAME),
        keyframe_policy=KeyframeSelectionPolicy(
            minimum_spacing_ns=1,
            forced_spacing_ns=1,
            minimum_visual_change=0.0,
            minimum_quality_score=0.0,
            maximum_keyframes=count,
        ),
        quality_policy=FrameQualityPolicy(
            minimum_blur_variance=0.0,
            minimum_contrast_stddev=0.0,
            minimum_mean_luma=0.0,
            maximum_mean_luma=255.0,
            maximum_clipped_fraction=1.0,
            minimum_quality_score=0.0,
        ),
        cache_policy=IngestionCachePolicy(
            maximum_cpu_bytes=32 * 1024 * 1024,
            maximum_frames=2,
            gpu_staging_bytes=0,
        ),
    )
    return pipeline.run()
