from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from core.alignment import (
    AlignmentRequest,
    align_frame,
    declare_full_frame_resize,
    depth_input_from_inference_result,
    make_depth_space,
    resample_boolean_mask,
    transform_intrinsics,
)
from core.alignment.calibration import rectify_bgr_with_camera_calibration
from core.contracts import (
    AxisConvention,
    CalibrationStatus,
    CameraIntrinsics,
    CoordinateFrame,
    CoordinateFrameKind,
    FrameReference,
    ImageSize,
    ModelProvenance,
    PixelCoordinateSpace,
    PixelSpaceRole,
)
from core.models.metadata import IntrinsicsSource
from core.object_geometry.projection import project_selected_pixels
from core.perception.result import PerceptionFrameResult
from core.reconstruction import (
    CameraCalibrationMethod,
    CameraCalibrationProfile,
    CameraCalibrationValidationState,
    CameraDistortionModel,
    ReconstructionCamera,
)
from core.runtime.messages import InferenceResult


def _profile() -> CameraCalibrationProfile:
    size = ImageSize(640, 480)
    return CameraCalibrationProfile(
        profile_id="phase9-1g-integration-profile",
        camera=ReconstructionCamera(
            camera_id="phase9-1g-camera",
            image_size=size,
            intrinsics=CameraIntrinsics(
                np.array([[520.0, 0.0, 320.0], [0.0, 520.0, 240.0], [0.0, 0.0, 1.0]]),
                "calibrated-rgb",
                IntrinsicsSource.CALIBRATED,
                CalibrationStatus.CALIBRATED,
                calibration_record_id="phase9-1g-integration-record",
            ),
            camera_frame=CoordinateFrame(
                "phase9-1g-camera-frame",
                CoordinateFrameKind.CAMERA_GENERIC,
                AxisConvention.X_RIGHT_Y_DOWN_Z_FORWARD,
                parent_frame_id="phase9-1g-reconstruction",
            ),
            distortion_model=CameraDistortionModel.BROWN_CONRADY,
            distortion_coefficients=(-0.12, 0.02, 0.0, 0.0, 0.0),
        ),
        method=CameraCalibrationMethod.MANUAL,
        validation_state=CameraCalibrationValidationState.UNVALIDATED,
        provenance="phase9-1g-integration",
        created_at_utc="2026-08-13T18:00:00+00:00",
    )


def test_rectified_calibration_reaches_native_depth_projection_without_changing_depth_scale() -> None:
    source_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    source_bgr[80:400, 100:540] = (40, 120, 220)
    prepared = rectify_bgr_with_camera_calibration(source_bgr, _profile())
    rgb = prepared.image_bgr[..., ::-1].copy()

    frame = FrameReference(11, 12, 0.0, ImageSize(640, 480))
    rgb_space = PixelCoordinateSpace(
        prepared.application.camera.intrinsics.pixel_space_id,
        frame,
        ImageSize(640, 480),
        PixelSpaceRole.RGB,
    )
    depth = np.full((240, 320), 2.0, dtype=np.float32)
    depth_space = make_depth_space(frame, depth)
    transform = declare_full_frame_resize(rgb_space, depth_space)
    calibrated_depth_intrinsics = transform_intrinsics(
        prepared.application.camera.intrinsics,
        transform,
    )
    inference = InferenceResult(
        session_id=11,
        frame_id=12,
        model_id="qualified-metric-depth",
        depth=depth,
        latency_ms=1.0,
        intrinsics=None,
        depth_scale="metric",
        intrinsics_source="none",
        model_name="Qualified Metric Depth",
    )
    depth_input = depth_input_from_inference_result(
        inference,
        frame,
        intrinsics_override=calibrated_depth_intrinsics,
    )
    perception = PerceptionFrameResult(
        frame_reference=frame,
        observations=(),
        model_provenance=(
            ModelProvenance("seg", "Segmentation", "instance_segmentation"),
        ),
    )
    aligned = align_frame(
        AlignmentRequest(frame, rgb, rgb_space, depth_input, perception, (transform,))
    )
    calibrated_support = resample_boolean_mask(prepared.support_mask, transform)
    aligned = replace(
        aligned,
        rgb_support_mask=np.logical_and(aligned.rgb_support_mask, calibrated_support),
    )

    assert aligned.intrinsics.source is IntrinsicsSource.CALIBRATED
    assert aligned.intrinsics.calibration_status is CalibrationStatus.CALIBRATED
    assert aligned.intrinsics.calibration_record_id == "phase9-1g-integration-record"
    assert aligned.depth_scale.value == "metric"
    assert np.count_nonzero(aligned.rgb_support_mask) > 0
    assert np.count_nonzero(aligned.rgb_support_mask) <= aligned.rgb_support_mask.size

    # The ideal rectified pinhole centre projects exactly onto the optical axis.
    point = project_selected_pixels(
        aligned.depth,
        np.array([[160, 120]], dtype=np.int64),
        aligned.intrinsics,
    )[0]
    assert point == pytest.approx(np.array([0.0, 0.0, 2.0]), abs=1e-9)


def test_isolated_measurement_job_persists_selected_calibration_provenance(tmp_path) -> None:
    import json
    from pathlib import Path
    import subprocess
    import sys

    from core.inspection import load_inspection_bundle
    from core.reconstruction.calibration_management import create_manual_camera_calibration_profile
    from core.reconstruction.calibration_profile import save_camera_calibration_profile

    height, width = 72, 96
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = np.linspace(20, 180, width, dtype=np.uint8)
    image[..., 1] = 90
    input_path = tmp_path / "frame.npy"
    np.save(input_path, image)
    config = tmp_path / "depth.json"
    config.write_text(
        json.dumps({"device": "cpu", "precision": "fp32", "latency_ms": 0.0}),
        encoding="utf-8",
    )
    profile = create_manual_camera_calibration_profile(
        width=width,
        height=height,
        fx=80.0,
        fy=81.0,
        cx=48.0,
        cy=36.0,
        profile_id="phase9-1g-job-profile",
        calibration_record_id="phase9-1g-job-record",
        camera_id="phase9-1g-job-camera",
    )
    profile_path = save_camera_calibration_profile(profile, tmp_path / "calibration.json")
    output = tmp_path / "calibrated-job"
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "run_measurement_inspection_job.py"),
            "--input",
            str(input_path),
            "--output",
            str(output),
            "--depth-model",
            "development-mock",
            "--depth-config",
            str(config),
            "--perception-model",
            "development-perception-mock",
            "--max-objects",
            "1",
            "--calibration-profile",
            str(profile_path),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    summary, _arrays = load_inspection_bundle(output)
    assert summary.intrinsics_source == "calibrated"
    assert summary.calibration_status == "calibrated"
    assert summary.calibration_profile_id == "phase9-1g-job-profile"
    assert summary.calibration_record_id == "phase9-1g-job-record"
    assert summary.calibration_method == "manual"
    assert summary.calibration_validation_state == "unvalidated"
    assert summary.calibration_application_mode == "exact_resolution"
    assert summary.calibration_rectified is False
