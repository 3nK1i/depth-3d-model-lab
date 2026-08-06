"""Dependency-light offline input readers for reconstruction ingestion."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

import cv2
import numpy as np

from core.contracts import ImageSize
from core.contracts.validation import require_nonempty
from core.reconstruction.contracts import (
    FrameInvalidReason,
    InputSourceKind,
    InputSourceMetadata,
    TimestampOrigin,
)
from core.reconstruction.errors import ReconstructionValidationError

SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
_NATURAL_PARTS = re.compile(r"(\d+)")
_FINGERPRINT_CHUNK_BYTES = 1024 * 1024


def natural_path_key(path: str | Path) -> tuple[tuple[int, object], ...]:
    """Sort paths naturally, so frame2 precedes frame10."""

    text = Path(path).as_posix().casefold()
    parts: list[tuple[int, object]] = []
    for part in _NATURAL_PARTS.split(text):
        if not part:
            continue
        parts.append((0, int(part)) if part.isdigit() else (1, part))
    return tuple(parts)


def _hash_file_chunks(path: Path, *, full_file: bool) -> str:
    digest = hashlib.sha256()
    size = path.stat().st_size
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as handle:
        if full_file or size <= 2 * _FINGERPRINT_CHUNK_BYTES:
            for block in iter(lambda: handle.read(_FINGERPRINT_CHUNK_BYTES), b""):
                digest.update(block)
        else:
            digest.update(handle.read(_FINGERPRINT_CHUNK_BYTES))
            handle.seek(max(0, size - _FINGERPRINT_CHUNK_BYTES))
            digest.update(handle.read(_FINGERPRINT_CHUNK_BYTES))
    return digest.hexdigest()


def image_sequence_fingerprint(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8", errors="surrogatepass"))
        digest.update(_hash_file_chunks(path, full_file=True).encode("ascii"))
    return digest.hexdigest()


def video_file_fingerprint(path: Path) -> str:
    return _hash_file_chunks(path, full_file=False)


@dataclass(frozen=True, slots=True)
class SourceReadIssue:
    sequence_index: int
    source_locator: str
    reasons: tuple[FrameInvalidReason, ...]
    message: str

    def __post_init__(self) -> None:
        if isinstance(self.sequence_index, bool) or int(self.sequence_index) < 0:
            raise ReconstructionValidationError("sequence_index must be non-negative.")
        object.__setattr__(self, "sequence_index", int(self.sequence_index))
        object.__setattr__(
            self, "source_locator", require_nonempty(self.source_locator, field_name="source_locator")
        )
        reasons = tuple(self.reasons)
        if not reasons or any(not isinstance(reason, FrameInvalidReason) for reason in reasons):
            raise ReconstructionValidationError(
                "Source read issues require FrameInvalidReason values."
            )
        object.__setattr__(self, "reasons", tuple(dict.fromkeys(reasons)))
        object.__setattr__(self, "message", require_nonempty(self.message, field_name="message"))

    def as_dict(self) -> dict[str, object]:
        return {
            "sequence_index": self.sequence_index,
            "source_locator": self.source_locator,
            "reasons": [reason.value for reason in self.reasons],
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class DecodedFrameCandidate:
    sequence_index: int
    timestamp_ns: int
    timestamp_origin: TimestampOrigin
    image: np.ndarray
    source_locator: str

    def __post_init__(self) -> None:
        if isinstance(self.sequence_index, bool) or int(self.sequence_index) < 0:
            raise ReconstructionValidationError("sequence_index must be non-negative.")
        if isinstance(self.timestamp_ns, bool) or int(self.timestamp_ns) != self.timestamp_ns:
            raise ReconstructionValidationError("timestamp_ns must be an integer.")
        if not isinstance(self.timestamp_origin, TimestampOrigin):
            raise TypeError("timestamp_origin must be a TimestampOrigin.")
        if not isinstance(self.image, np.ndarray):
            raise TypeError("image must be a NumPy array.")
        object.__setattr__(self, "sequence_index", int(self.sequence_index))
        object.__setattr__(self, "timestamp_ns", int(self.timestamp_ns))
        object.__setattr__(
            self, "source_locator", require_nonempty(self.source_locator, field_name="source_locator")
        )


@runtime_checkable
class ReconstructionFrameSource(Protocol):
    metadata: InputSourceMetadata

    @property
    def estimated_total_frames(self) -> int | None: ...

    @property
    def issues(self) -> tuple[SourceReadIssue, ...]: ...

    def __iter__(self) -> Iterator[DecodedFrameCandidate]: ...


class ImageSequenceReader:
    """Lazy image-sequence decoder with deterministic natural ordering."""

    def __init__(
        self,
        paths: tuple[Path, ...],
        *,
        frame_rate_hz: float = 30.0,
        preserve_explicit_order: bool = False,
    ) -> None:
        if not paths:
            raise ReconstructionValidationError("An image sequence requires at least one path.")
        rate = float(frame_rate_hz)
        if not isfinite(rate) or rate <= 0.0:
            raise ReconstructionValidationError("frame_rate_hz must be finite and positive.")
        normalized = tuple(Path(path).expanduser().resolve() for path in paths)
        if not preserve_explicit_order:
            normalized = tuple(sorted(normalized, key=natural_path_key))
        if len(set(normalized)) != len(normalized):
            raise ReconstructionValidationError("Image sequence paths must be unique.")
        missing = [str(path) for path in normalized if not path.is_file()]
        if missing:
            raise ReconstructionValidationError(f"Image sequence file is missing: {missing[0]}")
        unsupported = [str(path) for path in normalized if path.suffix.casefold() not in SUPPORTED_IMAGE_EXTENSIONS]
        if unsupported:
            raise ReconstructionValidationError(
                f"Unsupported image-sequence file extension: {unsupported[0]}"
            )

        fingerprint = image_sequence_fingerprint(normalized)
        common_parent = Path(Path(normalized[0]).parent)
        source_id = f"image-sequence-{fingerprint[:16]}"
        self.metadata = InputSourceMetadata(
            source_id=source_id,
            kind=InputSourceKind.IMAGE_SEQUENCE,
            display_name=common_parent.name or "Image sequence",
            source_uri=str(common_parent),
            expected_frame_rate_hz=rate,
            timestamp_origin=TimestampOrigin.SYNTHETIC,
            fingerprint=fingerprint,
            metadata={"frame_count": len(normalized)},
        )
        self.paths = normalized
        self.frame_rate_hz = rate
        self._issues: list[SourceReadIssue] = []

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        *,
        frame_rate_hz: float = 30.0,
        recursive: bool = False,
    ) -> "ImageSequenceReader":
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            raise ReconstructionValidationError(f"Image-sequence directory does not exist: {root}")
        iterator = root.rglob("*") if recursive else root.iterdir()
        paths = tuple(
            path for path in iterator if path.is_file() and path.suffix.casefold() in SUPPORTED_IMAGE_EXTENSIONS
        )
        if not paths:
            raise ReconstructionValidationError(f"No supported images were found in: {root}")
        return cls(paths, frame_rate_hz=frame_rate_hz)

    @classmethod
    def from_paths(
        cls,
        paths: tuple[str | Path, ...] | list[str | Path],
        *,
        frame_rate_hz: float = 30.0,
        preserve_order: bool = True,
    ) -> "ImageSequenceReader":
        return cls(
            tuple(Path(path) for path in paths),
            frame_rate_hz=frame_rate_hz,
            preserve_explicit_order=preserve_order,
        )

    @property
    def estimated_total_frames(self) -> int:
        return len(self.paths)

    @property
    def issues(self) -> tuple[SourceReadIssue, ...]:
        return tuple(self._issues)

    def __iter__(self) -> Iterator[DecodedFrameCandidate]:
        self._issues = []
        interval_ns = int(round(1_000_000_000 / self.frame_rate_hz))
        for index, path in enumerate(self.paths):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                self._issues.append(
                    SourceReadIssue(
                        sequence_index=index,
                        source_locator=str(path),
                        reasons=(FrameInvalidReason.DECODE_FAILED,),
                        message=f"Could not decode image: {path.name}",
                    )
                )
                continue
            yield DecodedFrameCandidate(
                sequence_index=index,
                timestamp_ns=index * interval_ns,
                timestamp_origin=TimestampOrigin.SYNTHETIC,
                image=image,
                source_locator=str(path),
            )


class RecordedVideoReader:
    """Lazy OpenCV video reader with PTS-to-synthetic timestamp fallback."""

    def __init__(
        self,
        path: str | Path,
        *,
        fallback_frame_rate_hz: float = 30.0,
    ) -> None:
        video_path = Path(path).expanduser().resolve()
        if not video_path.is_file():
            raise ReconstructionValidationError(f"Recorded video does not exist: {video_path}")
        fallback = float(fallback_frame_rate_hz)
        if not isfinite(fallback) or fallback <= 0.0:
            raise ReconstructionValidationError(
                "fallback_frame_rate_hz must be finite and positive."
            )
        probe = cv2.VideoCapture(str(video_path))
        if not probe.isOpened():
            probe.release()
            raise ReconstructionValidationError(f"Could not open recorded video: {video_path}")
        native_fps = float(probe.get(cv2.CAP_PROP_FPS))
        frame_count = int(max(0.0, probe.get(cv2.CAP_PROP_FRAME_COUNT)))
        probe.release()
        if not isfinite(native_fps) or native_fps <= 0.0:
            native_fps = fallback
        fingerprint = video_file_fingerprint(video_path)
        source_id = f"recorded-video-{fingerprint[:16]}"
        self.metadata = InputSourceMetadata(
            source_id=source_id,
            kind=InputSourceKind.RECORDED_VIDEO,
            display_name=video_path.name,
            source_uri=str(video_path),
            expected_frame_rate_hz=native_fps,
            timestamp_origin=TimestampOrigin.PRESENTATION_TIMESTAMP,
            fingerprint=fingerprint,
            metadata={"reported_frame_count": frame_count},
        )
        self.path = video_path
        self.frame_rate_hz = native_fps
        self._estimated_total_frames = frame_count or None
        self._issues: list[SourceReadIssue] = []

    @property
    def estimated_total_frames(self) -> int | None:
        return self._estimated_total_frames

    @property
    def issues(self) -> tuple[SourceReadIssue, ...]:
        return tuple(self._issues)

    def __iter__(self) -> Iterator[DecodedFrameCandidate]:
        self._issues = []
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            self._issues.append(
                SourceReadIssue(
                    sequence_index=0,
                    source_locator=str(self.path),
                    reasons=(FrameInvalidReason.DECODE_FAILED,),
                    message="Could not reopen recorded video for decoding.",
                )
            )
            return
        interval_ns = int(round(1_000_000_000 / self.frame_rate_hz))
        last_timestamp_ns = -1
        index = 0
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                pts_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                pts_ns = int(round(pts_ms * 1_000_000)) if isfinite(pts_ms) and pts_ms >= 0.0 else -1
                if pts_ns > last_timestamp_ns:
                    timestamp_ns = pts_ns
                    origin = TimestampOrigin.PRESENTATION_TIMESTAMP
                else:
                    timestamp_ns = 0 if index == 0 else last_timestamp_ns + interval_ns
                    origin = TimestampOrigin.SYNTHETIC
                last_timestamp_ns = timestamp_ns
                yield DecodedFrameCandidate(
                    sequence_index=index,
                    timestamp_ns=timestamp_ns,
                    timestamp_origin=origin,
                    image=image,
                    source_locator=f"{self.path}#frame={index}",
                )
                index += 1
        finally:
            capture.release()

        expected = self._estimated_total_frames
        if index == 0:
            self._issues.append(
                SourceReadIssue(
                    sequence_index=0,
                    source_locator=str(self.path),
                    reasons=(FrameInvalidReason.DECODE_FAILED,),
                    message="Recorded video contained no decodable frames.",
                )
            )
        elif expected is not None and index + 1 < expected:
            self._issues.append(
                SourceReadIssue(
                    sequence_index=index,
                    source_locator=str(self.path),
                    reasons=(FrameInvalidReason.DECODE_FAILED,),
                    message=(
                        f"Video decoding stopped early after {index} frames; "
                        f"container reported approximately {expected}."
                    ),
                )
            )


def make_usb_source_metadata(
    device_index: int,
    *,
    width: int | None = None,
    height: int | None = None,
    frame_rate_hz: float | None = None,
) -> InputSourceMetadata:
    """Declare a future live USB source without opening it in Phase 8.2."""

    if isinstance(device_index, bool) or int(device_index) < 0:
        raise ReconstructionValidationError("device_index must be non-negative.")
    metadata: dict[str, object] = {"device_index": int(device_index)}
    if (width is None) != (height is None):
        raise ReconstructionValidationError(
            "width and height must either both be provided or both be omitted."
        )
    expected_size = None
    if width is not None and height is not None:
        if isinstance(width, bool) or int(width) <= 0:
            raise ReconstructionValidationError("width must be positive when provided.")
        if isinstance(height, bool) or int(height) <= 0:
            raise ReconstructionValidationError("height must be positive when provided.")
        expected_size = ImageSize(int(width), int(height))
        metadata["requested_width"] = int(width)
        metadata["requested_height"] = int(height)
    identity = f"usb:{int(device_index)}:{width or 0}x{height or 0}:{frame_rate_hz or 0}"
    fingerprint = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return InputSourceMetadata(
        source_id=f"usb-camera-{fingerprint[:16]}",
        kind=InputSourceKind.USB_CAMERA,
        display_name=f"USB camera {int(device_index)}",
        source_uri=f"usb://{int(device_index)}",
        expected_image_size=expected_size,
        expected_frame_rate_hz=frame_rate_hz,
        timestamp_origin=TimestampOrigin.CAPTURE_CLOCK,
        fingerprint=fingerprint,
        metadata=metadata,
    )


def make_rtsp_source_metadata(url: str) -> InputSourceMetadata:
    """Declare future RTSP ingestion while redacting credentials from metadata."""

    text = require_nonempty(url, field_name="url")
    sanitized = re.sub(r"(rtsp://[^:/@]+:)[^@]+(@)", r"\1***\2", text, count=1)
    fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return InputSourceMetadata(
        source_id=f"rtsp-stream-{fingerprint[:16]}",
        kind=InputSourceKind.RTSP_STREAM,
        display_name="RTSP stream",
        source_uri=sanitized,
        timestamp_origin=TimestampOrigin.CAPTURE_CLOCK,
        fingerprint=fingerprint,
    )
