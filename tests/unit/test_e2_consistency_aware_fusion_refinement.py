from __future__ import annotations

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from scripts.run_e2_consistency_aware_fusion_refinement import (
    _aggregate_frame_voxels,
    _attached_single_view_fringe,
    _classify_supported_voxels,
    _refinement_confidence,
    _reconstruct_e1_input_points,
    _robust_observation_statistics,
    _topology_metrics,
    _validate_e1_summary,
    _voxel_indices,
)


def _e1_summary(*, passed: bool = True, authorized: bool = True, capacity_clean: bool = True):
    return {
        "phase": "E1",
        "pass": passed,
        "status": "pass" if passed else "degraded",
        "decision": {"authorizes_next_fusion_refinement_milestone": authorized},
        "fusion": {"capacity_clean": capacity_clean},
        "frames": [{"workspace_order": i, "cloud_npz": f"cloud-{i}.npz"} for i in range(10)],
        "authority": {"metric_scale_claimed": False, "measurement_authoritative": False},
    }


def test_validate_e1_summary_requires_passing_authorized_capacity_clean_source():
    _validate_e1_summary(_e1_summary())
    with pytest.raises(ReconstructionValidationError):
        _validate_e1_summary(_e1_summary(passed=False))
    with pytest.raises(ReconstructionValidationError):
        _validate_e1_summary(_e1_summary(authorized=False))
    with pytest.raises(ReconstructionValidationError):
        _validate_e1_summary(_e1_summary(capacity_clean=False))


def test_voxel_indices_and_frame_aggregation_are_unique_per_view():
    points = np.asarray([[0.01, 0.01, 0.01], [0.04, 0.02, 0.03], [0.11, 0.01, 0.01]])
    colors = np.asarray([[10, 20, 30], [30, 40, 50], [100, 110, 120]], dtype=np.uint8)
    assert _voxel_indices(points, 0.1).tolist() == [[0, 0, 0], [0, 0, 0], [1, 0, 0]]
    keys, centroids, mean_colors = _aggregate_frame_voxels(points, colors, 0.1)
    assert keys.tolist() == [[0, 0, 0], [1, 0, 0]]
    assert centroids[0].tolist() == pytest.approx([0.025, 0.015, 0.02])
    assert mean_colors[0].tolist() == [20, 30, 40]


def test_robust_observation_statistics_use_median_center_and_voxel_normalization():
    observations = np.asarray([[0.00, 0.0, 0.0], [0.02, 0.0, 0.0], [0.04, 0.0, 0.0]])
    colors = np.asarray([[10, 20, 30], [20, 30, 40], [30, 40, 50]], dtype=np.uint8)
    center, color, median, p90, maximum = _robust_observation_statistics(observations, colors, 0.1)
    assert center.tolist() == pytest.approx([0.02, 0.0, 0.0])
    assert color.tolist() == [20, 30, 40]
    assert median == pytest.approx(0.2)
    assert p90 == pytest.approx(0.2)
    assert maximum == pytest.approx(0.2)


def test_topology_metrics_find_component_size_and_high_support_neighbors():
    keys = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0], [10, 10, 10]], dtype=np.int64)
    counts = np.asarray([2, 3, 4, 2], dtype=np.int64)
    n26, radius2, high26, component, largest, component_count = _topology_metrics(keys, counts)
    assert n26.tolist() == [1, 2, 1, 0]
    assert high26.tolist() == [1, 1, 1, 0]
    assert radius2[:3].tolist() == [2, 2, 2]
    assert component.tolist() == [3, 3, 3, 1]
    assert largest == 3
    assert component_count == 2


def test_refinement_confidence_rewards_support_and_low_dispersion():
    confidence = _refinement_confidence(
        np.asarray([2, 5]),
        np.asarray([0.8, 0.1]),
        np.asarray([4, 30]),
        np.asarray([0, 6]),
    )
    assert np.all((0.0 <= confidence) & (confidence <= 1.0))
    assert confidence[1] > confidence[0]


def test_classification_keeps_dense_low_dispersion_core_and_rejects_sparse_noise():
    support = np.asarray([4, 2, 2, 3])
    dispersion = np.asarray([0.2, 0.3, 1.2, 0.5])
    radius2 = np.asarray([20, 12, 20, 2])
    high = np.asarray([4, 3, 5, 1])
    components = np.asarray([100, 100, 100, 100])
    core, extended = _classify_supported_voxels(
        support, dispersion, radius2, high, components,
        minimum_component_size=64,
        core_max_dispersion_voxels=0.70,
        support2_core_max_dispersion_voxels=0.45,
        extended_max_dispersion_voxels=0.95,
        support2_core_min_high_neighbors=2,
        support2_core_min_radius2_neighbors=8,
        extended_min_radius2_neighbors=6,
    )
    assert core.tolist() == [True, True, False, False]
    assert extended.tolist() == [True, True, False, False]


def test_attached_single_view_fringe_requires_multiple_extended_neighbors():
    all_keys = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 5]], dtype=np.int64)
    counts = np.asarray([1, 2, 2, 1])
    extended = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.int64)
    fringe = _attached_single_view_fringe(all_keys, counts, extended, minimum_extended_neighbors=2)
    assert fringe.tolist() == [True, False, False, False]


def test_classification_respects_component_gate_even_for_high_support():
    core, extended = _classify_supported_voxels(
        np.asarray([6]), np.asarray([0.1]), np.asarray([20]), np.asarray([8]), np.asarray([5]),
        minimum_component_size=64,
        core_max_dispersion_voxels=0.70,
        support2_core_max_dispersion_voxels=0.45,
        extended_max_dispersion_voxels=0.95,
        support2_core_min_high_neighbors=2,
        support2_core_min_radius2_neighbors=8,
        extended_min_radius2_neighbors=6,
    )
    assert core.tolist() == [False]
    assert extended.tolist() == [False]


def test_reconstruct_e1_input_points_reproduces_float32_aligned_depth_quantization():
    # This is a real boundary-crossing shape: D4's float64 projection is in
    # voxel 215 while E1's saved float32 aligned depth lands in voxel 214.
    # Use an array operation here because D4 multiplies a float32 depth *array*
    # by the Python scale.  NumPy 1.x and 2.x differ in scalar-promotion rules
    # for ``np.float32_scalar * Python-float``; the production path is the
    # array operation and remains float32 on both supported NumPy generations.
    source_depth_grid = np.asarray([[4.196056842803955]], dtype=np.float32)
    scale = 4.008
    precise_depth = float(np.float64(source_depth_grid[0, 0]) * scale)
    saved_aligned_depth = np.asarray(source_depth_grid * scale, dtype=np.float64)
    e1_depth = float(np.asarray(saved_aligned_depth, dtype=np.float32)[0, 0])
    assert np.floor(precise_depth / 0.07822230229029022) == 215
    assert np.floor(e1_depth / 0.07822230229029022) == 214

    canonical = np.asarray([[0.0, 0.0, precise_depth]], dtype=np.float64)
    reconstructed = _reconstruct_e1_input_points(
        canonical, saved_aligned_depth, np.eye(4, dtype=np.float64)
    )
    assert reconstructed[0, 2] == pytest.approx(e1_depth, abs=1e-12)
    assert _voxel_indices(reconstructed, 0.07822230229029022).tolist() == [[0, 0, 214]]


def test_reconstruct_e1_input_points_preserves_camera_ray_and_pose():
    angle = np.deg2rad(20.0)
    rotation = np.asarray([
        [np.cos(angle), 0.0, np.sin(angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)],
    ])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray([1.0, -2.0, 0.5])
    camera_point = np.asarray([[0.8, -0.4, 4.0]], dtype=np.float64)
    canonical = camera_point @ rotation.T + transform[:3, 3]
    target = np.asarray([[np.float32(3.5)]], dtype=np.float64)

    reconstructed = _reconstruct_e1_input_points(canonical, target, transform)
    recovered_camera = (reconstructed - transform[:3, 3]) @ rotation
    assert recovered_camera[0, 2] == pytest.approx(3.5)
    assert recovered_camera[0, 0] / recovered_camera[0, 2] == pytest.approx(0.8 / 4.0)
    assert recovered_camera[0, 1] / recovered_camera[0, 2] == pytest.approx(-0.4 / 4.0)


def test_e2_preview_uses_same_concrete_canvas_and_sampling_imports_as_working_e1_preview():
    """Keep E2 preview imports aligned with the proven E1 preview entry point."""
    from pathlib import Path

    preview = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "run_e2_consistency_aware_fusion_preview.py"
    )
    source = preview.read_text(encoding="utf-8")
    assert "from app.widgets.viewport.vispy_canvas import PointCloudCanvas" in source
    assert "from app.widgets.viewport import PointCloudCanvas" not in source
    assert "from core.geometry.sampling import enforce_point_budget" in source
    assert "from core.visualization_sampling import enforce_point_budget" not in source
