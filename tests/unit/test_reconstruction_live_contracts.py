from __future__ import annotations

import numpy as np
import pytest

from core.contracts import CalibrationStatus, CameraIntrinsics
from core.models.metadata import IntrinsicsSource
from core.reconstruction import (
    LIVE_SLAM_RECONSTRUCTION_FRAME,
    LiveFrameBufferReference,
    LiveFramePixelFormat,
    LiveMapSnapshot,
    LiveSlamObservation,
    LiveSlamPose,
    LiveSlamRequest,
    LiveTrackingQuality,
    LiveTrackingState,
    ReconstructionGauge,
    ReconstructionQualityProfile,
    ReconstructionValidationError,
    SfmScaleStatus,
    make_usb_source_metadata,
)
from tests.reconstruction_helpers import SESSION_ID
from tests.reconstruction_live_helpers import LIVE_IMAGE_SIZE, make_calibrated_live_camera


def _buffer() -> LiveFrameBufferReference:
    return LiveFrameBufferReference(
        shared_memory_name="test-live-ring",
        slot_index=0,
        generation=1,
        byte_length=LIVE_IMAGE_SIZE.width * LIVE_IMAGE_SIZE.height * 3,
        image_size=LIVE_IMAGE_SIZE,
        pixel_format=LiveFramePixelFormat.BGR8,
    )


def test_live_pose_is_explicitly_arbitrary_scale() -> None:
    pose = LiveSlamPose(
        pose_id="pose-1",
        camera_id="live-camera-1",
        frame_id="live-frame-1",
        timestamp_ns=1_000,
        camera_frame_id="live-camera-frame",
        T_reconstruction_from_camera=np.eye(4),
    )
    assert pose.reconstruction_frame_id == LIVE_SLAM_RECONSTRUCTION_FRAME
    assert pose.gauge.scale_status is SfmScaleStatus.ARBITRARY
    assert pose.gauge.unit_to_metre is None
    payload = pose.as_dict()
    assert payload["pose_convention"] == "T_reconstruction_from_camera"
    assert "translation_world_m" not in payload
    assert payload["gauge"]["unit_to_metre"] is None


def test_live_pose_rejects_metric_gauge() -> None:
    with pytest.raises(ReconstructionValidationError, match="arbitrary-scale"):
        LiveSlamPose(
            pose_id="pose-1",
            camera_id="live-camera-1",
            frame_id="live-frame-1",
            timestamp_ns=1_000,
            camera_frame_id="live-camera-frame",
            T_reconstruction_from_camera=np.eye(4),
            gauge=ReconstructionGauge(
                scale_status=SfmScaleStatus.ESTIMATED_METRIC,
                unit_to_metre=1.0,
                scale_source="test",
            ),
        )


def test_tracking_observation_requires_pose_when_tracking() -> None:
    quality = LiveTrackingQuality(100, 80, 2_000_000, tracking_confidence=0.9)
    with pytest.raises(ReconstructionValidationError, match="TRACKING"):
        LiveSlamObservation(
            session_id=SESSION_ID,
            frame_id="live-frame-1",
            sequence_index=1,
            tracking_state=LiveTrackingState.TRACKING,
            quality=quality,
        )


def test_tracking_quality_derives_inlier_ratio() -> None:
    quality = LiveTrackingQuality(200, 150, 2_000_000, tracking_confidence=0.9)
    assert quality.inlier_ratio == pytest.approx(0.75)


def test_tracking_quality_rejects_ratio_without_tracked_features() -> None:
    with pytest.raises(ReconstructionValidationError, match="tracked_features is zero"):
        LiveTrackingQuality(0, 0, 2_000_000, inlier_ratio=0.0)


def test_live_request_requires_explicit_calibration() -> None:
    camera = make_calibrated_live_camera()
    estimated = CameraIntrinsics(
        camera.intrinsics.matrix,
        camera.intrinsics.pixel_space_id,
        IntrinsicsSource.ESTIMATED,
        CalibrationStatus.NOT_CALIBRATED,
    )
    uncalibrated_camera = type(camera)(
        camera_id=camera.camera_id,
        image_size=camera.image_size,
        intrinsics=estimated,
        camera_frame=camera.camera_frame,
        distortion_model=camera.distortion_model,
        distortion_coefficients=camera.distortion_coefficients,
    )
    with pytest.raises(ReconstructionValidationError, match="calibrated"):
        LiveSlamRequest(
            session_id=SESSION_ID,
            source=make_usb_source_metadata(0, width=320, height=240, frame_rate_hz=30),
            camera=uncalibrated_camera,
            profile=ReconstructionQualityProfile.EFFICIENT,
            backend_id="mock",
        )


def test_phase8_4_request_forbids_loop_and_global_optimization() -> None:
    from tests.reconstruction_live_helpers import make_live_request

    base = make_live_request()
    with pytest.raises(ReconstructionValidationError, match="Phase 8.5"):
        LiveSlamRequest(
            session_id=base.session_id,
            source=base.source,
            camera=base.camera,
            profile=base.profile,
            backend_id=base.backend_id,
            enable_loop_closure=True,
        )


def test_map_snapshot_is_nonnegative() -> None:
    assert LiveMapSnapshot(2, 3, 400).as_dict()["landmark_count"] == 400
    with pytest.raises(ReconstructionValidationError):
        LiveMapSnapshot(-1, 0, 0)


def test_recorded_video_requires_explicit_simulated_live_mode() -> None:
    from core.reconstruction import (
        InputSourceKind,
        InputSourceMetadata,
        ReconstructionQualityProfile,
        TimestampOrigin,
    )

    source = InputSourceMetadata(
        source_id="recorded-live",
        kind=InputSourceKind.RECORDED_VIDEO,
        display_name="Recorded live simulation",
        source_uri="video:///tmp/test.avi",
        expected_image_size=LIVE_IMAGE_SIZE,
        expected_frame_rate_hz=20.0,
        timestamp_origin=TimestampOrigin.PRESENTATION_TIMESTAMP,
    )
    camera = make_calibrated_live_camera()
    with pytest.raises(ReconstructionValidationError, match="simulated-live"):
        LiveSlamRequest(
            session_id=SESSION_ID,
            source=source,
            camera=camera,
            profile=ReconstructionQualityProfile.EFFICIENT,
            backend_id="mock",
        )
    request = LiveSlamRequest(
        session_id=SESSION_ID,
        source=source,
        camera=camera,
        profile=ReconstructionQualityProfile.EFFICIENT,
        backend_id="mock",
        simulated_live=True,
    )
    assert request.simulated_live is True


def test_live_request_rejects_plaintext_rtsp_credentials() -> None:
    from core.reconstruction import (
        InputSourceKind,
        InputSourceMetadata,
        ReconstructionQualityProfile,
        TimestampOrigin,
    )

    source = InputSourceMetadata(
        source_id="rtsp-plain",
        kind=InputSourceKind.RTSP_STREAM,
        display_name="RTSP",
        source_uri="rtsp://user:secret@example.invalid/stream",
        expected_image_size=LIVE_IMAGE_SIZE,
        expected_frame_rate_hz=20.0,
        timestamp_origin=TimestampOrigin.CAPTURE_CLOCK,
    )
    with pytest.raises(ReconstructionValidationError, match="plaintext credentials"):
        LiveSlamRequest(
            session_id=SESSION_ID,
            source=source,
            camera=make_calibrated_live_camera(),
            profile=ReconstructionQualityProfile.EFFICIENT,
            backend_id="mock",
        )
