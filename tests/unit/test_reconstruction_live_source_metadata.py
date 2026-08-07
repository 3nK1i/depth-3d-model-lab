from __future__ import annotations

from pathlib import Path

from core.contracts import ImageSize
from core.reconstruction import (
    InputSourceKind,
    TimestampOrigin,
    make_recorded_video_live_metadata,
    make_rtsp_source_metadata_from_identity,
    rtsp_identity_fingerprint,
)


def test_rtsp_identity_factory_never_persists_plaintext_password() -> None:
    raw = "rtsp://operator:secret-password@10.0.0.5/live"
    fingerprint = rtsp_identity_fingerprint(raw)
    source = make_rtsp_source_metadata_from_identity(
        sanitized_url="rtsp://operator:***@10.0.0.5/live",
        source_fingerprint=fingerprint,
        expected_image_size=ImageSize(1280, 720),
    )
    assert source.kind is InputSourceKind.RTSP_STREAM
    assert source.timestamp_origin is TimestampOrigin.CAPTURE_CLOCK
    assert "secret-password" not in source.source_uri
    assert source.fingerprint == fingerprint


def test_recorded_video_live_factory_uses_capture_clock(tmp_path: Path) -> None:
    path = tmp_path / "capture.avi"
    path.write_bytes(b"deterministic-video-identity")
    source = make_recorded_video_live_metadata(
        path,
        expected_image_size=ImageSize(752, 480),
        frame_rate_hz=20.0,
    )
    assert source.kind is InputSourceKind.RECORDED_VIDEO
    assert source.timestamp_origin is TimestampOrigin.CAPTURE_CLOCK
    assert dict(source.metadata)["simulated_live"] is True
