from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from core.contracts import (
    AxisConvention,
    CalibrationStatus,
    CameraIntrinsics,
    CoordinateFrame,
    CoordinateFrameKind,
    ImageSize,
)
from core.models.metadata import IntrinsicsSource
from core.reconstruction.camera import CameraDistortionModel, ReconstructionCamera
from scripts.run_d2_three_view_overlap_diagnostics import (
    _assess_direction,
    _bilinear_sample,
    _directed_reprojection_diagnostic,
    _pair_status,
    _project_camera_points_to_pixels,
    _robust_scene_scale,
    _spatial_nn_diagnostic,
)


def _camera(width: int = 8, height: int = 6) -> ReconstructionCamera:
    matrix = np.asarray([[10.0, 0.0, 3.5], [0.0, 10.0, 2.5], [0.0, 0.0, 1.0]])
    return ReconstructionCamera(
        camera_id="cam0",
        image_size=ImageSize(width=width, height=height),
        intrinsics=CameraIntrinsics(
            matrix=matrix,
            pixel_space_id="test",
            source=IntrinsicsSource.ESTIMATED,
            calibration_status=CalibrationStatus.NOT_CALIBRATED,
        ),
        camera_frame=CoordinateFrame(
            frame_id="cam0:frame",
            kind=CoordinateFrameKind.CAMERA_RGB,
            axis_convention=AxisConvention.X_RIGHT_Y_DOWN_Z_FORWARD,
            parent_frame_id="world",
        ),
        distortion_model=CameraDistortionModel.NONE,
    )


def _plane_points(camera: ReconstructionCamera, depth: float) -> np.ndarray:
    h, w = camera.image_size.height, camera.image_size.width
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    K = camera.intrinsics.matrix
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    z = np.full_like(x, depth)
    return np.column_stack((x.ravel(), y.ravel(), z.ravel()))


def test_bilinear_sampling_matches_known_values() -> None:
    image = np.asarray([[0.0, 2.0], [4.0, 6.0]])
    value = _bilinear_sample(image, np.asarray([0.5]), np.asarray([0.5]))
    assert np.allclose(value, [3.0])


def test_pinhole_projection_round_trips_plane_pixels() -> None:
    camera = _camera()
    points = _plane_points(camera, 2.0)
    pixels = _project_camera_points_to_pixels(camera, points)
    expected_u, expected_v = np.meshgrid(
        np.arange(camera.image_size.width), np.arange(camera.image_size.height)
    )
    expected = np.column_stack((expected_u.ravel(), expected_v.ravel()))
    assert np.allclose(pixels, expected)


def test_directed_reprojection_accepts_identical_plane() -> None:
    camera = _camera()
    points = _plane_points(camera, 2.0)
    depth = np.full((camera.image_size.height, camera.image_size.width), 2.0)
    metrics, diagnostics = _directed_reprojection_diagnostic(
        points,
        target_camera=camera,
        target_T_reconstruction_from_camera=np.eye(4),
        target_aligned_depth=depth,
        maximum_points=10_000,
        consistency_relative_tolerance=0.12,
        occlusion_relative_margin=0.20,
    )
    assert metrics["comparable_samples"] == points.shape[0]
    assert metrics["consistent_ratio_of_comparable"] == 1.0
    assert metrics["median_absolute_relative_depth_residual"] < 1e-12
    assert np.all(diagnostics["classification"] == 0)


def test_directed_reprojection_distinguishes_occlusion_from_front_conflict() -> None:
    camera = _camera()
    target = np.full((camera.image_size.height, camera.image_size.width), 2.0)
    behind = _plane_points(camera, 3.0)
    behind_metrics, behind_diag = _directed_reprojection_diagnostic(
        behind,
        target_camera=camera,
        target_T_reconstruction_from_camera=np.eye(4),
        target_aligned_depth=target,
        maximum_points=10_000,
        consistency_relative_tolerance=0.12,
        occlusion_relative_margin=0.20,
    )
    assert behind_metrics["occluded_ratio_of_valid_target"] == 1.0
    assert np.all(behind_diag["classification"] == 3)

    front = _plane_points(camera, 1.5)
    front_metrics, front_diag = _directed_reprojection_diagnostic(
        front,
        target_camera=camera,
        target_T_reconstruction_from_camera=np.eye(4),
        target_aligned_depth=target,
        maximum_points=10_000,
        consistency_relative_tolerance=0.12,
        occlusion_relative_margin=0.20,
    )
    assert front_metrics["front_conflict_ratio_of_comparable"] == 1.0
    assert np.all(front_diag["classification"] == 1)


def test_robust_scene_scale_uses_sparse_quantiles_and_ignores_extreme_outlier() -> None:
    rng = np.random.default_rng(0)
    sparse = rng.uniform(-1.0, 1.0, size=(1000, 3))
    sparse[0] = [10_000.0, 10_000.0, 10_000.0]
    diagonal, source, low, high = _robust_scene_scale(sparse, [sparse])
    assert source == "sparse_sfm_landmarks"
    assert 2.5 < diagonal < 4.0
    assert max(high) < 2.0
    assert min(low) > -2.0


def test_spatial_nn_diagnostic_reports_tight_identical_support() -> None:
    rng = np.random.default_rng(1)
    points = rng.normal(size=(500, 3))
    result = _spatial_nn_diagnostic(
        points,
        points.copy(),
        scene_scale=10.0,
        maximum_points=500,
        support_scene_fraction=0.05,
    )
    assert result["a_to_b"]["support_ratio"] == 1.0
    assert result["b_to_a"]["median_nn_scene_fraction_all"] == 0.0


def test_direction_assessment_covers_pass_degraded_and_inconclusive() -> None:
    good = {
        "comparable_samples": 5000,
        "consistent_ratio_of_comparable": 0.8,
        "front_conflict_ratio_of_comparable": 0.05,
        "median_absolute_relative_depth_residual": 0.04,
        "p90_absolute_relative_depth_residual": 0.12,
    }
    status, failures = _assess_direction(
        good,
        minimum_comparable_samples=1000,
        minimum_consistent_ratio=0.5,
        maximum_median_relative_residual=0.10,
        maximum_p90_relative_residual=0.25,
        maximum_front_conflict_ratio=0.25,
    )
    assert status == "pass"
    assert failures == []

    poor = dict(good, consistent_ratio_of_comparable=0.2, p90_absolute_relative_depth_residual=0.4)
    status, failures = _assess_direction(
        poor,
        minimum_comparable_samples=1000,
        minimum_consistent_ratio=0.5,
        maximum_median_relative_residual=0.10,
        maximum_p90_relative_residual=0.25,
        maximum_front_conflict_ratio=0.25,
    )
    assert status == "degraded"
    assert "low_consistent_overlap_ratio" in failures
    assert "p90_depth_disagreement" in failures

    tiny = dict(good, comparable_samples=20)
    status, failures = _assess_direction(
        tiny,
        minimum_comparable_samples=1000,
        minimum_consistent_ratio=0.5,
        maximum_median_relative_residual=0.10,
        maximum_p90_relative_residual=0.25,
        maximum_front_conflict_ratio=0.25,
    )
    assert status == "inconclusive"
    assert failures == ["insufficient_comparable_overlap"]


def test_pair_status_propagates_degraded_then_inconclusive() -> None:
    assert _pair_status("pass", "pass") == "pass"
    assert _pair_status("pass", "inconclusive") == "inconclusive"
    assert _pair_status("inconclusive", "degraded") == "degraded"
