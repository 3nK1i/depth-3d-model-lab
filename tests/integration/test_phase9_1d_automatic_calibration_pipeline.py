from __future__ import annotations

import cv2
import numpy as np
import pytest

from core.contracts import ImageSize
from core.reconstruction import (
    AutomaticCalibrationQualityPolicy,
    CameraCalibrationMethod,
    CameraCalibrationProfileStore,
    CameraCalibrationValidationState,
    CharucoCalibrationTarget,
    CheckerboardCalibrationTarget,
    MetricGeometryStatus,
    apply_camera_calibration_profile,
    arbitrary_scale_reference,
    assess_reconstruction_geometry,
    calibrate_charuco_image_set,
    calibrate_checkerboard_image_set,
    camera_geometry_from_calibrated_camera,
    inspect_opencv_calibration_capabilities,
)


_POSES = (
    (0.15, -0.10, 0.02, -0.12, -0.10, 0.75),
    (0.25, 0.08, -0.05, -0.08, -0.08, 0.80),
    (-0.15, 0.15, 0.08, -0.10, -0.05, 0.85),
    (0.10, 0.22, -0.10, -0.12, -0.12, 0.90),
    (-0.22, -0.12, 0.05, -0.05, -0.08, 0.80),
    (0.18, -0.20, 0.12, -0.10, -0.04, 0.88),
    (0.30, 0.05, 0.15, -0.08, -0.10, 0.92),
    (-0.10, 0.30, -0.12, -0.13, -0.05, 0.95),
    (0.05, -0.28, 0.02, -0.08, -0.12, 0.86),
    (0.22, 0.18, -0.08, -0.11, -0.08, 0.82),
)


def _policy() -> AutomaticCalibrationQualityPolicy:
    return AutomaticCalibrationQualityPolicy(8, 1.0, 1.0, 1.5)


def _projected_checkerboard_images() -> list[tuple[str, np.ndarray]]:
    target = CheckerboardCalibrationTarget(7, 6, 0.03)
    width, height = 640, 480
    matrix = np.array([[700.0, 0.0, 320.0], [0.0, 710.0, 240.0], [0.0, 0.0, 1.0]])
    squares_x = target.inner_corners_x + 1
    squares_y = target.inner_corners_y + 1
    cell, margin = 60, 70
    board_width, board_height = squares_x * cell, squares_y * cell
    texture = np.full((board_height + 2 * margin, board_width + 2 * margin), 255, np.uint8)
    for y in range(squares_y):
        for x in range(squares_x):
            if (x + y) % 2 == 0:
                cv2.rectangle(
                    texture,
                    (margin + x * cell, margin + y * cell),
                    (margin + (x + 1) * cell, margin + (y + 1) * cell),
                    0,
                    -1,
                )
    object_corners = np.array(
        [
            [0, 0, 0],
            [squares_x * target.square_length, 0, 0],
            [squares_x * target.square_length, squares_y * target.square_length, 0],
            [0, squares_y * target.square_length, 0],
        ],
        np.float32,
    )
    source = np.array(
        [
            [margin, margin],
            [margin + board_width, margin],
            [margin + board_width, margin + board_height],
            [margin, margin + board_height],
        ],
        np.float32,
    )
    images = []
    for index, pose in enumerate(_POSES):
        projected, _ = cv2.projectPoints(
            object_corners,
            np.asarray(pose[:3], float),
            np.asarray(pose[3:], float),
            matrix,
            np.zeros(5),
        )
        homography = cv2.getPerspectiveTransform(
            source, np.asarray(projected, np.float32).reshape(4, 2)
        )
        image = cv2.warpPerspective(texture, homography, (width, height), borderValue=200)
        images.append((f"checker-{index}", cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)))
    return images


def _projected_charuco_images(target: CharucoCalibrationTarget) -> list[tuple[str, np.ndarray]]:
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, target.dictionary_name))
    if hasattr(aruco, "CharucoBoard"):
        board = aruco.CharucoBoard(
            (target.squares_x, target.squares_y),
            target.square_length,
            target.marker_length,
            dictionary,
        )
    else:
        board = aruco.CharucoBoard_create(
            target.squares_x,
            target.squares_y,
            target.square_length,
            target.marker_length,
            dictionary,
        )
    base = board.generateImage((500, 700))
    margin = 80
    texture = cv2.copyMakeBorder(base, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255)
    matrix = np.array([[700.0, 0.0, 320.0], [0.0, 710.0, 240.0], [0.0, 0.0, 1.0]])
    object_corners = np.array(
        [
            [0, 0, 0],
            [target.squares_x * target.square_length, 0, 0],
            [target.squares_x * target.square_length, target.squares_y * target.square_length, 0],
            [0, target.squares_y * target.square_length, 0],
        ],
        np.float32,
    )
    source = np.array(
        [
            [margin, margin],
            [margin + base.shape[1], margin],
            [margin + base.shape[1], margin + base.shape[0]],
            [margin, margin + base.shape[0]],
        ],
        np.float32,
    )
    images = []
    for index, pose in enumerate(_POSES):
        projected, _ = cv2.projectPoints(
            object_corners,
            np.asarray(pose[:3], float),
            np.asarray(pose[3:], float),
            matrix,
            np.zeros(5),
        )
        homography = cv2.getPerspectiveTransform(
            source, np.asarray(projected, np.float32).reshape(4, 2)
        )
        image = cv2.warpPerspective(texture, homography, (640, 480), borderValue=200)
        images.append((f"charuco-{index}", cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)))
    return images


def test_checkerboard_profile_round_trip_and_resolution_application(tmp_path) -> None:
    result = calibrate_checkerboard_image_set(
        _projected_checkerboard_images(),
        CheckerboardCalibrationTarget(7, 6, 0.03),
        quality_policy=_policy(),
        profile_id="phase9-1d-checkerboard",
        calibration_record_id="phase9-1d-checkerboard-record",
        camera_id="phase9-1d-camera",
    )
    store = CameraCalibrationProfileStore(tmp_path / "profiles")
    store.save(result.profile)
    restored = store.load(result.profile.profile_id)
    applied = apply_camera_calibration_profile(
        restored,
        ImageSize(320, 240),
        allow_same_aspect_full_frame_resize=True,
    )
    assert restored.validation_state is CameraCalibrationValidationState.VALIDATED
    assert restored.method is CameraCalibrationMethod.CHECKERBOARD
    assert applied.was_rescaled is True
    assert applied.camera.image_size == ImageSize(320, 240)


def test_rejected_detection_view_is_preserved_in_quality_counts() -> None:
    images = _projected_checkerboard_images()
    images.append(("blank", np.full((480, 640, 3), 127, np.uint8)))
    result = calibrate_checkerboard_image_set(
        images,
        CheckerboardCalibrationTarget(7, 6, 0.03),
        quality_policy=_policy(),
        profile_id="phase9-1d-rejection-evidence",
        calibration_record_id="phase9-1d-rejection-record",
        camera_id="phase9-1d-camera",
    )
    assert result.profile.quality is not None
    assert result.profile.quality.accepted_view_count == 10
    assert result.profile.quality.rejected_view_count == 1
    assert result.rejected_view_count == 1


def test_automatic_camera_still_requires_separate_scale_evidence() -> None:
    result = calibrate_checkerboard_image_set(
        _projected_checkerboard_images(),
        CheckerboardCalibrationTarget(7, 6, 0.03),
        quality_policy=_policy(),
        profile_id="phase9-1d-authority",
        calibration_record_id="phase9-1d-authority-record",
        camera_id="phase9-1d-camera",
    )
    camera = camera_geometry_from_calibrated_camera(
        profile_id="phase9-1d-camera-geometry", camera=result.profile.camera
    )
    assessment = assess_reconstruction_geometry(
        camera, arbitrary_scale_reference(profile_id="phase9-1d-no-scale")
    )
    assert assessment.metric_status is MetricGeometryStatus.ARBITRARY
    assert assessment.metric_coordinates_allowed is False
    assert assessment.certified_metrology is False


@pytest.mark.skipif(
    not inspect_opencv_calibration_capabilities().charuco,
    reason="active OpenCV build does not expose ChArUco primitives",
)
def test_charuco_pipeline_uses_same_profile_contract_and_quality_evidence(tmp_path) -> None:
    target = CharucoCalibrationTarget(5, 7, 0.04, 0.02, "DICT_4X4_50")
    result = calibrate_charuco_image_set(
        _projected_charuco_images(target),
        target,
        quality_policy=_policy(),
        profile_id="phase9-1d-charuco",
        calibration_record_id="phase9-1d-charuco-record",
        camera_id="phase9-1d-charuco-camera",
    )
    assert result.profile.method is CameraCalibrationMethod.CHARUCO
    assert result.profile.validation_state is CameraCalibrationValidationState.VALIDATED
    assert result.profile.quality is not None
    assert result.profile.quality.accepted_view_count >= 8
    store = CameraCalibrationProfileStore(tmp_path / "profiles")
    store.save(result.profile)
    restored = store.load(result.profile.profile_id)
    assert restored.method is CameraCalibrationMethod.CHARUCO
    assert restored.quality is not None
    assert restored.quality.as_dict() == result.profile.quality.as_dict()
