#!/usr/bin/env python3
"""Dependency-light Phase 9.1G calibrated Measure & Inspect integration smoke."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from core.alignment import (
    declare_full_frame_resize,
    depth_input_from_inference_result,
    make_depth_space,
    transform_intrinsics,
)
from core.alignment.calibration import rectify_bgr_with_camera_calibration
from core.contracts import FrameReference, ImageSize, PixelCoordinateSpace, PixelSpaceRole
from core.reconstruction import (
    CameraDistortionModel,
    arbitrary_scale_reference,
    assess_reconstruction_geometry,
    camera_geometry_from_calibrated_camera,
)
from core.reconstruction.calibration_management import create_manual_camera_calibration_profile
from core.runtime.messages import InferenceResult


def main() -> int:
    profile = create_manual_camera_calibration_profile(
        width=640,
        height=480,
        fx=520.0,
        fy=522.0,
        cx=320.0,
        cy=240.0,
        distortion_model=CameraDistortionModel.BROWN_CONRADY,
        distortion_coefficients=(-0.08, 0.01, 0.0, 0.0, 0.0),
        profile_id="phase9-1g-smoke-profile",
        calibration_record_id="phase9-1g-smoke-record",
        camera_id="phase9-1g-smoke-camera",
    )
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[96:384, 128:512] = (40, 120, 220)
    prepared = rectify_bgr_with_camera_calibration(image, profile)

    frame = FrameReference(91, 1, 0.0, ImageSize(640, 480))
    rgb_space = PixelCoordinateSpace(
        prepared.application.camera.intrinsics.pixel_space_id,
        frame,
        ImageSize(640, 480),
        PixelSpaceRole.RGB,
    )
    depth = np.full((240, 320), 2.0, dtype=np.float32)
    depth_space = make_depth_space(frame, depth)
    transform = declare_full_frame_resize(rgb_space, depth_space)
    depth_intrinsics = transform_intrinsics(prepared.application.camera.intrinsics, transform)
    inference = InferenceResult(
        session_id=91,
        frame_id=1,
        model_id="relative-depth-smoke",
        depth=depth,
        latency_ms=1.0,
        intrinsics=None,
        depth_scale="relative",
        intrinsics_source="none",
    )
    depth_input = depth_input_from_inference_result(
        inference,
        frame,
        intrinsics_override=depth_intrinsics,
    )

    geometry = camera_geometry_from_calibrated_camera(
        profile_id="phase9-1g-smoke-camera-geometry",
        camera=prepared.application.camera,
    )
    authority = assess_reconstruction_geometry(
        geometry,
        arbitrary_scale_reference(profile_id="phase9-1g-smoke-arbitrary-scale"),
    )

    result = {
        "status": "PASS",
        "profile_id": profile.profile_id,
        "calibration_record_id": profile.calibration_record_id,
        "calibration_validation_state": profile.validation_state.value,
        "application_mode": prepared.application.mode.value,
        "distortion_rectified": prepared.rectified,
        "native_depth_intrinsics_source": depth_input.intrinsics.source.value,
        "native_depth_calibration_status": depth_input.intrinsics.calibration_status.value,
        "depth_scale_preserved": depth_input.depth_scale.value,
        "metric_coordinates_from_calibration_alone": authority.metric_coordinates_allowed,
        "certified_metrology": authority.certified_metrology,
    }
    if result["native_depth_intrinsics_source"] != "calibrated":
        raise RuntimeError("Calibrated intrinsics did not reach the native depth grid.")
    if result["depth_scale_preserved"] != "relative":
        raise RuntimeError("Camera calibration changed depth-scale semantics.")
    if result["metric_coordinates_from_calibration_alone"]:
        raise RuntimeError("Camera calibration incorrectly promoted metric authority.")
    if result["certified_metrology"]:
        raise RuntimeError("Phase 9.1G must not claim certified metrology.")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
