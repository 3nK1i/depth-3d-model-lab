from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.contracts import ImageSize
from core.reconstruction import (
    AutomaticCalibrationQualityPolicy,
    CameraCalibrationMethod,
    CameraCalibrationValidationState,
    CharucoCalibrationTarget,
    CheckerboardCalibrationTarget,
    MetricGeometryStatus,
    arbitrary_scale_reference,
    assess_reconstruction_geometry,
    calibrate_charuco_image_set,
    calibrate_checkerboard_image_set,
    camera_geometry_from_calibrated_camera,
    inspect_opencv_calibration_capabilities,
    load_calibration_image_set,
)
from core.reconstruction.errors import ReconstructionValidationError


_CHECKER_POSES = (
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


def _camera_matrix(width: int = 640, height: int = 480) -> np.ndarray:
    return np.array(
        [[700.0, 0.0, width / 2.0], [0.0, 710.0, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _checkerboard_view(
    pose: tuple[float, float, float, float, float, float],
    *,
    target: CheckerboardCalibrationTarget,
    size: tuple[int, int] = (640, 480),
) -> np.ndarray:
    squares_x = target.inner_corners_x + 1
    squares_y = target.inner_corners_y + 1
    cell = 60
    margin = 70
    board_width = squares_x * cell
    board_height = squares_y * cell
    texture = np.full(
        (board_height + 2 * margin, board_width + 2 * margin), 255, dtype=np.uint8
    )
    for row in range(squares_y):
        for column in range(squares_x):
            if (row + column) % 2 == 0:
                cv2.rectangle(
                    texture,
                    (margin + column * cell, margin + row * cell),
                    (margin + (column + 1) * cell, margin + (row + 1) * cell),
                    0,
                    -1,
                )
    width, height = size
    object_corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [squares_x * target.square_length, 0.0, 0.0],
            [squares_x * target.square_length, squares_y * target.square_length, 0.0],
            [0.0, squares_y * target.square_length, 0.0],
        ],
        dtype=np.float32,
    )
    projected, _ = cv2.projectPoints(
        object_corners,
        np.asarray(pose[:3], dtype=np.float64),
        np.asarray(pose[3:], dtype=np.float64),
        _camera_matrix(width, height),
        np.zeros(5, dtype=np.float64),
    )
    source = np.array(
        [
            [margin, margin],
            [margin + board_width, margin],
            [margin + board_width, margin + board_height],
            [margin, margin + board_height],
        ],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(
        source,
        np.asarray(projected, dtype=np.float32).reshape(4, 2),
    )
    rendered = cv2.warpPerspective(
        texture,
        homography,
        (width, height),
        borderValue=200,
    )
    return cv2.cvtColor(rendered, cv2.COLOR_GRAY2BGR)


def _checkerboard_images() -> list[tuple[str, np.ndarray]]:
    target = CheckerboardCalibrationTarget(7, 6, 0.03)
    return [
        (f"checker-{index:02d}.png", _checkerboard_view(pose, target=target))
        for index, pose in enumerate(_CHECKER_POSES)
    ]


def _policy() -> AutomaticCalibrationQualityPolicy:
    return AutomaticCalibrationQualityPolicy(
        minimum_accepted_views=8,
        maximum_rms_reprojection_error_px=1.0,
        maximum_mean_reprojection_error_px=1.0,
        maximum_view_reprojection_error_px=1.5,
    )


def test_opencv_capability_probe_requires_only_actual_features() -> None:
    capabilities = inspect_opencv_calibration_capabilities()
    assert capabilities.opencv_version
    assert capabilities.checkerboard is True
    assert isinstance(capabilities.charuco, bool)
    assert capabilities.as_dict()["checkerboard"] is True


def test_calibration_targets_validate_geometry() -> None:
    with pytest.raises(ReconstructionValidationError, match="inner_corners_x"):
        CheckerboardCalibrationTarget(1, 6, 0.03)
    with pytest.raises(ReconstructionValidationError, match="square_length"):
        CheckerboardCalibrationTarget(7, 6, 0.0)
    with pytest.raises(ReconstructionValidationError, match="marker_length"):
        CharucoCalibrationTarget(5, 7, 0.04, 0.04)


def test_quality_policy_is_explicit_and_validated() -> None:
    with pytest.raises(ReconstructionValidationError, match="minimum_accepted_views"):
        AutomaticCalibrationQualityPolicy(2, 1.0, 1.0, 1.5)
    with pytest.raises(ReconstructionValidationError, match="maximum_rms"):
        AutomaticCalibrationQualityPolicy(8, 0.0, 1.0, 1.5)


def test_checkerboard_calibration_produces_validated_profile_with_quality() -> None:
    result = calibrate_checkerboard_image_set(
        _checkerboard_images(),
        CheckerboardCalibrationTarget(7, 6, 0.03),
        quality_policy=_policy(),
        profile_id="checkerboard-profile",
        calibration_record_id="checkerboard-record",
        camera_id="checkerboard-camera",
    )
    profile = result.profile
    assert profile.method is CameraCalibrationMethod.CHECKERBOARD
    assert profile.validation_state is CameraCalibrationValidationState.VALIDATED
    assert profile.quality is not None
    assert profile.quality.accepted_view_count == 10
    assert profile.quality.rejected_view_count == 0
    assert profile.quality.rms_reprojection_error_px is not None
    assert profile.quality.rms_reprojection_error_px < 1.0
    assert result.validation_reasons == ()
    assert all(item.reprojection_error_px is not None for item in result.diagnostics)


def test_checkerboard_without_policy_remains_unvalidated() -> None:
    result = calibrate_checkerboard_image_set(
        _checkerboard_images(),
        CheckerboardCalibrationTarget(7, 6, 0.03),
        profile_id="unvalidated-profile",
        calibration_record_id="unvalidated-record",
        camera_id="unvalidated-camera",
    )
    assert result.profile.validation_state is CameraCalibrationValidationState.UNVALIDATED
    assert result.validation_reasons == ("quality_policy_not_applied",)
    assert result.profile.has_phase9_validation_evidence is False


def test_checkerboard_strict_policy_rejects_quality_without_losing_evidence() -> None:
    result = calibrate_checkerboard_image_set(
        _checkerboard_images(),
        CheckerboardCalibrationTarget(7, 6, 0.03),
        quality_policy=AutomaticCalibrationQualityPolicy(8, 0.001, 0.001, 0.001),
        profile_id="rejected-profile",
        calibration_record_id="rejected-record",
        camera_id="rejected-camera",
    )
    assert result.profile.validation_state is CameraCalibrationValidationState.REJECTED
    assert result.profile.quality is not None
    assert "rms_reprojection_error_exceeds_policy" in result.validation_reasons
    assert result.profile.is_accepted is False


def test_mixed_resolution_image_set_is_rejected_before_calibration() -> None:
    images = _checkerboard_images()[:3]
    images.append(("different.png", np.zeros((240, 320, 3), dtype=np.uint8)))
    with pytest.raises(ReconstructionValidationError, match="same resolution"):
        calibrate_checkerboard_image_set(
            images,
            CheckerboardCalibrationTarget(7, 6, 0.03),
        )


def test_calibration_does_not_create_metric_scene_authority() -> None:
    result = calibrate_checkerboard_image_set(
        _checkerboard_images(),
        CheckerboardCalibrationTarget(7, 6, 0.03),
        quality_policy=_policy(),
        profile_id="authority-profile",
        calibration_record_id="authority-record",
        camera_id="authority-camera",
    )
    camera_geometry = camera_geometry_from_calibrated_camera(
        profile_id="automatic-camera-geometry",
        camera=result.profile.camera,
    )
    assessment = assess_reconstruction_geometry(
        camera_geometry,
        arbitrary_scale_reference(profile_id="no-scale-reference"),
    )
    assert assessment.metric_status is MetricGeometryStatus.ARBITRARY
    assert assessment.metric_coordinates_allowed is False
    assert assessment.certified_metrology is False


def test_load_calibration_image_set_preserves_deterministic_names(tmp_path: Path) -> None:
    images = _checkerboard_images()[:2]
    paths = []
    for name, image in images:
        path = tmp_path / name
        assert cv2.imwrite(str(path), image)
        paths.append(path)
    loaded = load_calibration_image_set(paths)
    assert [name for name, _image in loaded] == [path.name for path in paths]
    assert all(image.shape == (480, 640, 3) for _name, image in loaded)


def test_charuco_unavailable_fails_closed_or_capability_is_usable() -> None:
    capabilities = inspect_opencv_calibration_capabilities()
    target = CharucoCalibrationTarget(5, 7, 0.04, 0.02)
    blank = [(f"blank-{index}", np.full((480, 640, 3), 255, dtype=np.uint8)) for index in range(3)]
    if not capabilities.charuco:
        with pytest.raises(ReconstructionValidationError, match="ChArUco calibration is unavailable"):
            calibrate_charuco_image_set(blank, target)
    else:
        with pytest.raises(ReconstructionValidationError, match="at least three detected"):
            calibrate_charuco_image_set(blank, target)
