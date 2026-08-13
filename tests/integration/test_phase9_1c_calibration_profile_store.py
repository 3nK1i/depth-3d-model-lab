from __future__ import annotations

import json
from pathlib import Path

from core.contracts import ImageSize
from core.reconstruction import (
    CalibrationApplicationMode,
    CameraCalibrationMethod,
    CameraCalibrationProfileStore,
    CameraCalibrationValidationState,
    CameraDistortionModel,
    apply_camera_calibration_profile,
    create_manual_camera_calibration_profile,
    live_camera_calibration_dict,
)
from tests.reconstruction_live_helpers import make_calibrated_live_camera


CREATED_AT = "2026-08-13T16:35:00+00:00"


def test_manual_profile_persists_and_remains_resolution_safe(tmp_path: Path) -> None:
    store = CameraCalibrationProfileStore(tmp_path / "profiles")
    profile = create_manual_camera_calibration_profile(
        width=1920,
        height=1080,
        fx=1400.0,
        fy=1404.0,
        cx=960.0,
        cy=540.0,
        distortion_model=CameraDistortionModel.BROWN_CONRADY,
        distortion_coefficients=(-0.02, 0.004, 0.0, 0.0, 0.0),
        profile_id="manual-1080p",
        calibration_record_id="manual-1080p-record",
        camera_id="factory-camera-a",
        provenance="phase9-1c-integration",
        created_at_utc=CREATED_AT,
    )
    store.save(profile)
    restored = store.load("manual-1080p")

    applied = apply_camera_calibration_profile(
        restored,
        ImageSize(1280, 720),
        allow_same_aspect_full_frame_resize=True,
    )

    assert restored.method is CameraCalibrationMethod.MANUAL
    assert restored.validation_state is CameraCalibrationValidationState.UNVALIDATED
    assert applied.mode is CalibrationApplicationMode.SAME_ASPECT_FULL_FRAME_RESIZE
    assert applied.camera.intrinsics.calibration_record_id == "manual-1080p-record"
    assert float(applied.camera.intrinsics.matrix[0, 0]) == 1400.0 * (1280.0 / 1920.0)


def test_legacy_phase8_file_can_be_imported_stored_and_reloaded(tmp_path: Path) -> None:
    legacy_path = tmp_path / "phase8-camera.json"
    legacy_path.write_text(
        json.dumps(live_camera_calibration_dict(make_calibrated_live_camera())),
        encoding="utf-8",
    )
    store = CameraCalibrationProfileStore(tmp_path / "profiles")

    imported = store.import_file(
        legacy_path,
        legacy_profile_id="legacy-camera-profile",
        created_at_utc=CREATED_AT,
    )
    restored = store.load("legacy-camera-profile")

    assert imported.profile.method is CameraCalibrationMethod.LEGACY_LIVE_JSON
    assert restored.validation_state is CameraCalibrationValidationState.LEGACY_COMPATIBLE
    assert restored.has_phase9_validation_evidence is False
    original = make_calibrated_live_camera()
    assert restored.camera.camera_id == original.camera_id
    assert restored.camera.image_size == original.image_size
    assert restored.camera.intrinsics.matrix.tolist() == original.intrinsics.matrix.tolist()
    assert restored.camera.intrinsics.calibration_record_id == original.intrinsics.calibration_record_id
    assert restored.camera.distortion_model == original.distortion_model
    assert restored.camera.distortion_coefficients == original.distortion_coefficients
    assert restored.camera.camera_frame == original.camera_frame
