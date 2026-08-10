from __future__ import annotations

import numpy as np
import pytest

from app.offline_reconstruction_spatial import build_offline_sparse_spatial_coverage
from app.offline_reconstruction_visualization import (
    OFFLINE_VISUALIZATION_SCHEMA,
    parse_offline_sparse_visualization,
)


def _pose(x: float, y: float = 0.0, z: float = 0.0) -> list[list[float]]:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = (x, y, z)
    return matrix.tolist()


def _visual(points: np.ndarray, *, landmark_total: int | None = None):
    points = np.asarray(points, dtype=np.float64)
    return parse_offline_sparse_visualization(
        {
            "schema": OFFLINE_VISUALIZATION_SCHEMA,
            "session_id": "d4-session",
            "backend_id": "pycolmap-offline-sfm",
            "backend_version": "test",
            "pose_convention": "T_reconstruction_from_camera",
            "scale_status": "arbitrary",
            "alignment_status": "arbitrary",
            "registered_camera_total": 4,
            "registered_cameras": [
                {"keyframe_id": "kf-0", "T_reconstruction_from_camera": _pose(-2.0, 0.0, 0.0)},
                {"keyframe_id": "kf-1", "T_reconstruction_from_camera": _pose(0.0, -2.0, 0.0)},
                {"keyframe_id": "kf-2", "T_reconstruction_from_camera": _pose(2.0, 0.0, 0.0)},
                {"keyframe_id": "kf-3", "T_reconstruction_from_camera": _pose(0.0, 2.0, 0.0)},
            ],
            "landmark_total": int(points.shape[0] if landmark_total is None else landmark_total),
            "landmarks_reconstruction": points.tolist(),
            "mean_reprojection_error_px": 0.5,
        }
    )


def test_d4_spatial_coverage_uses_robust_sparse_sample_envelope() -> None:
    # One extreme point must not dominate a 2-98% presentation envelope.
    x = np.linspace(-1.0, 1.0, 100)
    points = np.column_stack((x, 0.5 * x, 0.25 * x))
    points = np.vstack((points, np.asarray([[500.0, 500.0, 500.0]])))
    coverage = build_offline_sparse_spatial_coverage(_visual(points, landmark_total=50_000))

    assert coverage.has_envelope is True
    assert coverage.sampled_landmark_count == 101
    assert coverage.landmark_total == 50_000
    assert coverage.extents_reconstruction is not None
    assert max(coverage.extents_reconstruction) < 10.0
    assert coverage.envelope_diagonal_gauge is not None
    assert coverage.envelope_diagonal_gauge < 10.0
    assert "gauge" in coverage.envelope_text
    assert "101/50,000 sparse points" in coverage.basis_text
    assert "presentation envelope only" in coverage.basis_text
    assert "not object dimensions" in coverage.envelope_tooltip
    assert "metres" in coverage.envelope_tooltip


def test_d4_camera_span_and_standoff_remain_arbitrary_gauge() -> None:
    points = np.asarray(
        [
            [-1.0, -0.5, -0.25],
            [-0.5, 0.25, 0.0],
            [0.0, 0.0, 0.0],
            [0.5, -0.25, 0.0],
            [1.0, 0.5, 0.25],
        ],
        dtype=np.float64,
    )
    coverage = build_offline_sparse_spatial_coverage(_visual(points))

    assert coverage.camera_span_gauge == pytest.approx(np.sqrt(4.0**2 + 4.0**2))
    assert coverage.camera_standoff_min_gauge is not None
    assert coverage.camera_standoff_median_gauge is not None
    assert coverage.camera_standoff_max_gauge is not None
    assert coverage.camera_standoff_min_gauge <= coverage.camera_standoff_median_gauge
    assert coverage.camera_standoff_median_gauge <= coverage.camera_standoff_max_gauge
    assert "centers" in coverage.camera_span_text
    assert "gauge span" in coverage.camera_span_text
    assert "gauge min / med / max" in coverage.standoff_text


def test_d4_too_few_sparse_points_stays_unavailable_without_fake_dimensions() -> None:
    coverage = build_offline_sparse_spatial_coverage(
        _visual(np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64))
    )
    assert coverage.has_envelope is False
    assert coverage.envelope_text == "Unavailable"
    assert coverage.standoff_text == "Unavailable"
    assert coverage.camera_span_text == "Unavailable"
    assert "presentation envelope only" in coverage.basis_text


def test_d4_rejects_wrong_visual_type() -> None:
    with pytest.raises(TypeError, match="OfflineSparseVisualization"):
        build_offline_sparse_spatial_coverage(object())  # type: ignore[arg-type]
