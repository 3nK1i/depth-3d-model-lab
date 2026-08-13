#!/usr/bin/env python3
"""Non-UI Phase 9.1C manual/import/profile-store smoke."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.contracts import ImageSize
from core.reconstruction import (
    CameraCalibrationProfileStore,
    CameraDistortionModel,
    apply_camera_calibration_profile,
    create_manual_camera_calibration_profile,
)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="phase9_1c_calibration_") as directory:
        root = Path(directory)
        store = CameraCalibrationProfileStore(root / "profiles")
        profile = create_manual_camera_calibration_profile(
            width=1920,
            height=1080,
            fx=1400.0,
            fy=1404.0,
            cx=960.0,
            cy=540.0,
            distortion_model=CameraDistortionModel.BROWN_CONRADY,
            distortion_coefficients=(-0.02, 0.004, 0.0, 0.0, 0.0),
            profile_id="phase9-1c-smoke-profile",
            calibration_record_id="phase9-1c-smoke-record",
            camera_id="phase9-1c-smoke-camera",
            provenance="phase9_1c_smoke",
            created_at_utc="2026-08-13T16:40:00+00:00",
        )
        store.save(profile)
        restored = store.load(profile.profile_id)
        applied = apply_camera_calibration_profile(
            restored,
            ImageSize(1280, 720),
            allow_same_aspect_full_frame_resize=True,
        )
        payload = {
            "status": "PASS",
            "profile_id": restored.profile_id,
            "calibration_record_id": restored.calibration_record_id,
            "method": restored.method.value,
            "validation_state": restored.validation_state.value,
            "stored_profile_count": len(store.list_profiles()),
            "application_mode": applied.mode.value,
            "target_resolution": applied.camera.image_size.as_dict(),
            "metric_qualification_changed": False,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
