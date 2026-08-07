"""Safe live-source metadata factories used by the application capture bridge."""

from __future__ import annotations

import hashlib
from pathlib import Path

from core.contracts import ImageSize
from core.contracts.validation import require_nonempty
from core.reconstruction.contracts import InputSourceKind, InputSourceMetadata, TimestampOrigin
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.sources import video_file_fingerprint


def make_rtsp_source_metadata_from_identity(
    *,
    sanitized_url: str,
    source_fingerprint: str,
    expected_image_size: ImageSize | None = None,
    frame_rate_hz: float | None = None,
) -> InputSourceMetadata:
    """Build RTSP metadata without re-exposing plaintext credentials."""
    uri = require_nonempty(sanitized_url, field_name="sanitized_url")
    fingerprint = require_nonempty(source_fingerprint, field_name="source_fingerprint").lower()
    if "***" not in uri and "@" in uri and "://" in uri:
        # Username-only URLs are valid, but any user:password@ form must already be redacted.
        authority = uri.split("://", 1)[1].split("/", 1)[0]
        if ":" in authority.split("@", 1)[0] and "@" in authority:
            raise ReconstructionValidationError("RTSP live metadata must redact credentials.")
    return InputSourceMetadata(
        source_id=f"rtsp-stream-{fingerprint[:16]}",
        kind=InputSourceKind.RTSP_STREAM,
        display_name="RTSP stream",
        source_uri=uri,
        expected_image_size=expected_image_size,
        expected_frame_rate_hz=frame_rate_hz,
        timestamp_origin=TimestampOrigin.CAPTURE_CLOCK,
        fingerprint=fingerprint,
    )


def make_recorded_video_live_metadata(
    path: str | Path,
    *,
    expected_image_size: ImageSize,
    frame_rate_hz: float | None = None,
) -> InputSourceMetadata:
    """Describe VideoSource playback as capture-clock simulated-live input."""
    video_path = Path(path).expanduser().resolve()
    if not video_path.is_file():
        raise ReconstructionValidationError(f"Recorded live video does not exist: {video_path}")
    if not isinstance(expected_image_size, ImageSize):
        raise TypeError("expected_image_size must be an ImageSize.")
    fingerprint = video_file_fingerprint(video_path)
    return InputSourceMetadata(
        source_id=f"recorded-live-{fingerprint[:16]}",
        kind=InputSourceKind.RECORDED_VIDEO,
        display_name=video_path.name,
        source_uri=str(video_path),
        expected_image_size=expected_image_size,
        expected_frame_rate_hz=frame_rate_hz,
        timestamp_origin=TimestampOrigin.CAPTURE_CLOCK,
        fingerprint=fingerprint,
        metadata={"simulated_live": True},
    )


def rtsp_identity_fingerprint(raw_url: str) -> str:
    """Hash an in-memory RTSP locator; callers must never persist the raw URL."""
    text = require_nonempty(raw_url, field_name="raw_url")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
