#!/usr/bin/env python3
"""Dependency-light Phase 8.6A camera-geometry smoke."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.contracts import ImageSize
from core.reconstruction import (
    CameraGeometryPurpose,
    CameraIntrinsicsPolicy,
    LiveSlamRequest,
    ReconstructionQualityProfile,
    ReconstructionValidationError,
    camera_geometry_from_calibrated_camera,
    camera_geometry_profile_from_json,
    camera_geometry_profile_to_json,
    estimate_camera_geometry_from_horizontal_fov,
    make_usb_source_metadata,
    stella_monocular_config_text,
)
from tests.reconstruction_helpers import SESSION_ID
from tests.reconstruction_live_helpers import make_calibrated_live_camera


def main() -> int:
    estimated = estimate_camera_geometry_from_horizontal_fov(
        profile_id="phase8-6a-estimated-720p",
        camera_id="usb-camera-0",
        image_size=ImageSize(1280, 720),
        horizontal_fov_degrees=70.0,
        focal_length_relative_uncertainty=0.15,
        notes="No-calibration V1 simulation profile",
    )
    restored = camera_geometry_profile_from_json(camera_geometry_profile_to_json(estimated))
    source = make_usb_source_metadata(0, width=1280, height=720, frame_rate_hz=30.0)

    strict_rejected = False
    try:
        LiveSlamRequest(
            session_id=SESSION_ID,
            source=source,
            camera=restored.camera,
            profile=ReconstructionQualityProfile.EFFICIENT,
            backend_id="development-live-slam-mock",
        )
    except ReconstructionValidationError:
        strict_rejected = True

    provisional = LiveSlamRequest(
        session_id=SESSION_ID,
        source=source,
        camera=restored.camera,
        profile=ReconstructionQualityProfile.EFFICIENT,
        backend_id="development-live-slam-mock",
        camera_intrinsics_policy=CameraIntrinsicsPolicy.ALLOW_ESTIMATED,
    )
    config = stella_monocular_config_text(provisional)
    calibrated = camera_geometry_from_calibrated_camera(
        profile_id="phase8-6a-calibrated",
        camera=make_calibrated_live_camera(),
    )

    relative = restored.assess(CameraGeometryPurpose.RELATIVE_RECONSTRUCTION)
    metric_rays = restored.assess(CameraGeometryPurpose.CALIBRATED_RAY_GEOMETRY)
    passed = all(
        (
            strict_rejected,
            provisional.camera_intrinsics_policy is CameraIntrinsicsPolicy.ALLOW_ESTIMATED,
            relative.allowed and relative.provisional,
            not metric_rays.allowed,
            restored.is_provisional,
            calibrated.is_calibrated,
            "intrinsics_source=estimated" in config,
            "calibration_status=not_calibrated" in config,
            "unit_to_metre" not in config,
        )
    )

    matrix = restored.camera.intrinsics.matrix
    print(f"Phase 8.6A camera geometry smoke: {'PASS' if passed else 'FAIL'}")
    print(
        f"estimated_profile={restored.profile_id} mode={restored.mode.value} "
        f"method={restored.method.value}"
    )
    print(
        f"image_size={restored.camera.image_size.width}x{restored.camera.image_size.height} "
        f"fx={float(matrix[0,0]):.3f} fy={float(matrix[1,1]):.3f}"
    )
    print(
        f"source={restored.camera.intrinsics.source.value} "
        f"calibration_status={restored.camera.intrinsics.calibration_status.value}"
    )
    print(f"strict_default_rejected={strict_rejected} provisional_opt_in_accepted=True")
    print(f"relative_reconstruction_allowed={relative.allowed} provisional={relative.provisional}")
    print(f"calibrated_ray_geometry_allowed={metric_rays.allowed}")
    print("scale_status=arbitrary unit_to_metre=None")
    print("reconstruction_ui_enabled=False")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
