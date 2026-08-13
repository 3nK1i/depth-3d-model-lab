#!/usr/bin/env python3
"""Non-UI Phase 9.1D automatic checkerboard/ChArUco calibration smoke."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (  # noqa: E402
    AutomaticCalibrationQualityPolicy,
    CharucoCalibrationTarget,
    CheckerboardCalibrationTarget,
    MetricGeometryStatus,
    arbitrary_scale_reference,
    assess_reconstruction_geometry,
    calibrate_charuco_image_set,
    calibrate_checkerboard_image_set,
    camera_geometry_from_calibrated_camera,
    inspect_opencv_calibration_capabilities,
)


POSES = (
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


def _opencv_distributions() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in (
        "opencv-python",
        "opencv-contrib-python",
        "opencv-python-headless",
        "opencv-contrib-python-headless",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return result


def _matrix() -> np.ndarray:
    return np.array([[700.0, 0.0, 320.0], [0.0, 710.0, 240.0], [0.0, 0.0, 1.0]])


def _checkerboard_images(target: CheckerboardCalibrationTarget):
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
    for index, pose in enumerate(POSES):
        projected, _ = cv2.projectPoints(
            object_corners,
            np.asarray(pose[:3], float),
            np.asarray(pose[3:], float),
            _matrix(),
            np.zeros(5),
        )
        transform = cv2.getPerspectiveTransform(source, np.asarray(projected, np.float32).reshape(4, 2))
        image = cv2.warpPerspective(texture, transform, (640, 480), borderValue=200)
        images.append((f"checker-{index}", cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)))
    return images


def _charuco_images(target: CharucoCalibrationTarget):
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
    if target.legacy_pattern:
        board.setLegacyPattern(True)
    base = board.generateImage((500, 700))
    margin = 80
    texture = cv2.copyMakeBorder(base, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255)
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
    for index, pose in enumerate(POSES):
        projected, _ = cv2.projectPoints(
            object_corners,
            np.asarray(pose[:3], float),
            np.asarray(pose[3:], float),
            _matrix(),
            np.zeros(5),
        )
        transform = cv2.getPerspectiveTransform(source, np.asarray(projected, np.float32).reshape(4, 2))
        image = cv2.warpPerspective(texture, transform, (640, 480), borderValue=200)
        images.append((f"charuco-{index}", cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)))
    return images


def _summary(result) -> dict[str, object]:
    quality = result.profile.quality
    assert quality is not None
    return {
        "method": result.profile.method.value,
        "validation_state": result.profile.validation_state.value,
        "accepted_views": quality.accepted_view_count,
        "rejected_views": quality.rejected_view_count,
        "rms_reprojection_error_px": quality.rms_reprojection_error_px,
        "mean_reprojection_error_px": quality.mean_reprojection_error_px,
        "max_reprojection_error_px": quality.max_reprojection_error_px,
    }


def main() -> int:
    capabilities = inspect_opencv_calibration_capabilities()
    policy = AutomaticCalibrationQualityPolicy(8, 1.0, 1.0, 1.5)
    checker_target = CheckerboardCalibrationTarget(7, 6, 0.03)
    checker = calibrate_checkerboard_image_set(
        _checkerboard_images(checker_target),
        checker_target,
        quality_policy=policy,
        profile_id="phase9-1d-smoke-checkerboard",
        calibration_record_id="phase9-1d-smoke-checkerboard-record",
        camera_id="phase9-1d-smoke-camera",
    )
    camera = camera_geometry_from_calibrated_camera(
        profile_id="phase9-1d-smoke-camera-geometry",
        camera=checker.profile.camera,
    )
    authority = assess_reconstruction_geometry(
        camera,
        arbitrary_scale_reference(profile_id="phase9-1d-smoke-no-scale"),
    )
    payload: dict[str, object] = {
        "status": "PASS" if capabilities.charuco else "BLOCKED_CHARUCO_CAPABILITY",
        "opencv_distributions": _opencv_distributions(),
        "capabilities": capabilities.as_dict(),
        "checkerboard": _summary(checker),
        "metric_status_without_scale_reference": authority.metric_status.value,
        "metric_coordinates_allowed_without_scale_reference": authority.metric_coordinates_allowed,
        "certified_metrology": authority.certified_metrology,
    }
    if capabilities.charuco:
        charuco_target = CharucoCalibrationTarget(5, 7, 0.04, 0.02, "DICT_4X4_50")
        charuco = calibrate_charuco_image_set(
            _charuco_images(charuco_target),
            charuco_target,
            quality_policy=policy,
            profile_id="phase9-1d-smoke-charuco",
            calibration_record_id="phase9-1d-smoke-charuco-record",
            camera_id="phase9-1d-smoke-camera",
        )
        payload["charuco"] = _summary(charuco)
    else:
        payload["charuco"] = {
            "status": "UNAVAILABLE",
            "missing_features": list(capabilities.charuco_missing_features),
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if authority.metric_status is not MetricGeometryStatus.ARBITRARY:
        return 4
    if authority.metric_coordinates_allowed or authority.certified_metrology:
        return 5
    return 0 if capabilities.charuco else 3


if __name__ == "__main__":
    raise SystemExit(main())
