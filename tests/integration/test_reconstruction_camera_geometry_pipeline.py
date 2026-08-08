from __future__ import annotations

from pathlib import Path

from core.contracts import ImageSize
from core.reconstruction import (
    CameraGeometryPurpose,
    CameraIntrinsicsPolicy,
    InputSourceKind,
    InputSourceMetadata,
    LiveSlamRequest,
    ReconstructionQualityProfile,
    TimestampOrigin,
    camera_geometry_profile_from_json,
    camera_geometry_profile_to_json,
    estimate_camera_geometry_from_horizontal_fov,
    stella_monocular_config_text,
)
from tests.reconstruction_helpers import SESSION_ID


def test_estimated_camera_profile_flows_to_explicit_provisional_live_request(tmp_path: Path) -> None:
    profile = estimate_camera_geometry_from_horizontal_fov(
        profile_id="rtsp-phone-estimated",
        camera_id="phone-rtsp",
        image_size=ImageSize(640, 480),
        horizontal_fov_degrees=72.0,
        focal_length_relative_uncertainty=0.20,
        notes="Phase 8.6A no-calibration simulation",
    )
    stored = tmp_path / "camera_geometry.json"
    stored.write_text(camera_geometry_profile_to_json(profile) + "\n", encoding="utf-8")
    restored = camera_geometry_profile_from_json(stored.read_text(encoding="utf-8"))
    source = InputSourceMetadata(
        source_id="phone-rtsp",
        kind=InputSourceKind.RTSP_STREAM,
        display_name="Phone RTSP simulation",
        source_uri="rtsp://user:***@example.invalid/stream",
        expected_image_size=restored.camera.image_size,
        expected_frame_rate_hz=30.0,
        timestamp_origin=TimestampOrigin.CAPTURE_CLOCK,
    )
    request = LiveSlamRequest(
        session_id=SESSION_ID,
        source=source,
        camera=restored.camera,
        profile=ReconstructionQualityProfile.EFFICIENT,
        backend_id="development-live-slam-mock",
        camera_intrinsics_policy=CameraIntrinsicsPolicy.ALLOW_ESTIMATED,
    )
    config = stella_monocular_config_text(request)
    assert restored.assess(CameraGeometryPurpose.RELATIVE_RECONSTRUCTION).allowed is True
    assert "intrinsics_source=estimated" in config
    assert request.camera.intrinsics.calibration_record_id == ""
    assert "unit_to_metre" not in config
