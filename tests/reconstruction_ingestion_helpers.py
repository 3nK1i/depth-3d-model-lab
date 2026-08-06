from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np

from core.contracts import ImageSize
from core.reconstruction import (
    DecodedFrameCandidate,
    FrameInvalidReason,
    InputSourceKind,
    InputSourceMetadata,
    SourceReadIssue,
    TimestampOrigin,
)

INGESTION_SESSION_ID = "00000000-0000-4000-8000-000000008002"


def textured_frame(index: int, *, width: int = 96, height: int = 72) -> np.ndarray:
    yy, xx = np.indices((height, width))
    checker = (((xx // 6 + yy // 6 + index) % 2) * 180 + 35).astype(np.uint8)
    image = cv2.cvtColor(checker, cv2.COLOR_GRAY2BGR)
    cv2.circle(
        image,
        (int((12 + index * 9) % max(width, 1)), int((15 + index * 5) % max(height, 1))),
        max(3, min(width, height) // 12),
        (20 + index % 200, 220, 80),
        -1,
    )
    image[0, 0] = (index % 251, (index * 7) % 251, (index * 13) % 251)
    return image


def flat_frame(value: int = 127, *, width: int = 96, height: int = 72) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def make_ingestion_metadata(
    *,
    source_id: str = "phase82-source",
    width: int | None = 96,
    height: int | None = 72,
    fps: float = 10.0,
) -> InputSourceMetadata:
    return InputSourceMetadata(
        source_id=source_id,
        kind=InputSourceKind.IMAGE_SEQUENCE,
        display_name="Phase 8.2 synthetic sequence",
        source_uri="sequence://phase82",
        expected_image_size=(
            ImageSize(width, height) if width is not None and height is not None else None
        ),
        expected_frame_rate_hz=fps,
        timestamp_origin=TimestampOrigin.SYNTHETIC,
        fingerprint="phase82-fingerprint",
    )


@dataclass
class SyntheticFrameSource:
    metadata: InputSourceMetadata
    candidates: tuple[DecodedFrameCandidate, ...]
    source_issues: tuple[SourceReadIssue, ...] = ()

    @property
    def estimated_total_frames(self) -> int:
        return len(self.candidates) + len(self.source_issues)

    @property
    def issues(self) -> tuple[SourceReadIssue, ...]:
        return self.source_issues

    def __iter__(self) -> Iterator[DecodedFrameCandidate]:
        yield from self.candidates


def candidate(
    index: int,
    image: np.ndarray,
    *,
    timestamp_ns: int | None = None,
    origin: TimestampOrigin = TimestampOrigin.SYNTHETIC,
) -> DecodedFrameCandidate:
    return DecodedFrameCandidate(
        sequence_index=index,
        timestamp_ns=index * 100_000_000 if timestamp_ns is None else timestamp_ns,
        timestamp_origin=origin,
        image=image,
        source_locator=f"synthetic://frame/{index}",
    )


def decode_issue(index: int) -> SourceReadIssue:
    return SourceReadIssue(
        sequence_index=index,
        source_locator=f"synthetic://frame/{index}",
        reasons=(FrameInvalidReason.DECODE_FAILED,),
        message="Synthetic decode failure.",
    )
