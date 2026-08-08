from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from core.contracts import ImageSize
from core.reconstruction import (
    DenseDepthAlignmentMode,
    DenseDepthUnit,
    DenseFusionFrame,
    DenseFusionPolicy,
    DenseSceneProfile,
    DenseVoxelFusion,
    LiveTrajectoryPose,
    LiveTrajectoryPoseLayer,
    MetricGeometryStatus,
    ReconstructionValidationError,
    arbitrary_scale_reference,
    estimate_camera_geometry_from_horizontal_fov,
    estimate_depth_alignment_from_correspondences,
    explicit_depth_scale_alignment,
    fuse_dense_frames,
    known_dimension_scale_reference,
    metric_depth_alignment,
    reconstruction_unit_depth_alignment,
)


def _camera(width: int = 64, height: int = 48):
    return estimate_camera_geometry_from_horizontal_fov(
        profile_id="dense-estimated",
        camera_id="dense-camera",
        image_size=ImageSize(width, height),
        horizontal_fov_degrees=70.0,
        focal_length_relative_uncertainty=0.15,
        camera_frame_id="dense-camera-frame",
        reconstruction_frame_id="dense-reconstruction",
    )


def _pose(tx: float = 0.0, ty: float = 0.0, tz: float = 0.0) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = (tx, ty, tz)
    return matrix


def _frame(index: int, *, tx: float = 0.0, depth_value: float = 2.0) -> DenseFusionFrame:
    h, w = 48, 64
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = 40 + index
    rgb[..., 1] = 100
    rgb[..., 2] = 180
    depth = np.full((h, w), depth_value, dtype=np.float32)
    return DenseFusionFrame(
        frame_id=f"frame-{index}",
        timestamp_ns=1_000_000_000 + index * 50_000_000,
        rgb=rgb,
        depth=depth,
        T_reconstruction_from_camera=_pose(tx),
        depth_unit=DenseDepthUnit.RECONSTRUCTION_UNITS,
        depth_alignment=reconstruction_unit_depth_alignment(provenance="synthetic-plane"),
        pose_layer="optimized_propagated",
    )


def _policy(*, minimum_views: int = 1, max_voxels: int = 50_000) -> DenseFusionPolicy:
    return DenseFusionPolicy(
        scene_profile=DenseSceneProfile.OBJECT_INSPECTION,
        voxel_size_reconstruction_units=0.08,
        pixel_stride=2,
        minimum_view_observations=minimum_views,
        max_voxels=max_voxels,
    )


def test_reconstruction_unit_alignment_is_identity() -> None:
    alignment = reconstruction_unit_depth_alignment(provenance="already-aligned")
    assert alignment.mode is DenseDepthAlignmentMode.RECONSTRUCTION_UNITS
    assert alignment.scale_to_reconstruction_units == 1.0
    assert alignment.relative_scale_uncertainty is None


def test_metric_depth_alignment_requires_metric_scale_profile() -> None:
    with pytest.raises(ReconstructionValidationError):
        metric_depth_alignment(arbitrary_scale_reference(profile_id="raw"), provenance="depth-pro")


def test_metric_depth_alignment_uses_inverse_unit_to_metre() -> None:
    scale = known_dimension_scale_reference(
        profile_id="metric",
        reference_id="known-20mm",
        known_length_m=0.02,
        measured_length_reconstruction_units=2.0,
        provenance="unit-test",
    )
    alignment = metric_depth_alignment(scale, provenance="metric-depth")
    assert alignment.scale_to_reconstruction_units == pytest.approx(100.0)
    assert alignment.scale_reference_profile_id == "metric"


def test_sparse_correspondence_alignment_is_robust_to_one_outlier() -> None:
    source = np.array([1.0, 2.0, 1.5, 3.0, 2.5, 1.25, 2.2, 1.8, 2.8, 50.0])
    target = source * 4.0
    target[-1] = 1.0
    alignment = estimate_depth_alignment_from_correspondences(
        source,
        target,
        provenance="sparse-landmarks",
        minimum_correspondences=8,
    )
    assert alignment.mode is DenseDepthAlignmentMode.SPARSE_CORRESPONDENCE_SCALE
    assert alignment.scale_to_reconstruction_units == pytest.approx(4.0)
    assert alignment.relative_scale_uncertainty == pytest.approx(0.0)
    assert "inliers=9/10" in alignment.provenance


def test_source_model_units_require_explicit_alignment() -> None:
    rgb = np.zeros((8, 8, 3), np.uint8)
    depth = np.ones((8, 8), np.float32)
    with pytest.raises(ReconstructionValidationError):
        DenseFusionFrame(
            frame_id="x",
            timestamp_ns=1,
            rgb=rgb,
            depth=depth,
            T_reconstruction_from_camera=np.eye(4),
            depth_unit=DenseDepthUnit.SOURCE_MODEL_UNITS,
            depth_alignment=reconstruction_unit_depth_alignment(provenance="invalid"),
        )


def test_dense_multiview_plane_fuses_into_reconstruction_coordinates() -> None:
    result = fuse_dense_frames(
        (_frame(0, tx=-0.06), _frame(1), _frame(2, tx=0.06)),
        session_id="dense-unit",
        camera_profile=_camera(),
        scale_profile=arbitrary_scale_reference(profile_id="raw"),
        policy=_policy(minimum_views=2),
        reconstruction_frame_id="dense-reconstruction",
    )
    assert result.metric_geometry_status is MetricGeometryStatus.ARBITRARY
    assert result.unit_to_metre is None
    assert result.statistics.frames_integrated == 3
    assert result.statistics.multi_view_voxels > 0
    assert len(result.point_cloud.points_reconstruction) > 100
    assert np.median(result.point_cloud.points_reconstruction[:, 2]) == pytest.approx(2.0, abs=1e-9)
    assert np.min(result.point_cloud.view_observation_counts) >= 2
    assert result.source_pose_layers == ("optimized_propagated",) * 3
    assert result.certified_metrology is False


def test_dense_fusion_is_deterministic() -> None:
    kwargs = dict(
        session_id="dense-deterministic",
        camera_profile=_camera(),
        scale_profile=arbitrary_scale_reference(profile_id="raw"),
        policy=_policy(minimum_views=1),
    )
    first = fuse_dense_frames((_frame(0), _frame(1, tx=0.03)), **kwargs)
    second = fuse_dense_frames((_frame(0), _frame(1, tx=0.03)), **kwargs)
    assert np.array_equal(first.point_cloud.points_reconstruction, second.point_cloud.points_reconstruction)
    assert np.array_equal(first.point_cloud.colors_rgb, second.point_cloud.colors_rgb)
    assert np.array_equal(first.point_cloud.view_observation_counts, second.point_cloud.view_observation_counts)


def test_metric_depth_is_converted_back_to_reconstruction_gauge_and_exported_in_metres() -> None:
    camera = _camera(32, 24)
    scale = known_dimension_scale_reference(
        profile_id="known-scale",
        reference_id="known-20mm",
        known_length_m=0.02,
        measured_length_reconstruction_units=2.0,
        provenance="known connector dimension",
    )
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    depth_metres = np.full((24, 32), 2.0, dtype=np.float32)
    frame = DenseFusionFrame(
        frame_id="metric-frame",
        timestamp_ns=1,
        rgb=rgb,
        depth=depth_metres,
        T_reconstruction_from_camera=np.eye(4),
        depth_unit=DenseDepthUnit.METRES,
        depth_alignment=metric_depth_alignment(scale, provenance="depth-pro metric output"),
    )
    result = fuse_dense_frames(
        (frame,),
        session_id="dense-metric",
        camera_profile=camera,
        scale_profile=scale,
        policy=DenseFusionPolicy(
            scene_profile=DenseSceneProfile.OBJECT_INSPECTION,
            voxel_size_reconstruction_units=8.0,
            pixel_stride=2,
        ),
    )
    assert result.metric_geometry_status is MetricGeometryStatus.PROVISIONAL_METRIC
    assert result.unit_to_metre == pytest.approx(0.01)
    metres = result.point_cloud.points_metres(scale)
    assert np.median(metres[:, 2]) == pytest.approx(2.0, abs=1e-8)


def test_arbitrary_dense_cloud_refuses_metric_conversion() -> None:
    result = fuse_dense_frames(
        (_frame(0),),
        session_id="dense-arbitrary",
        camera_profile=_camera(),
        scale_profile=arbitrary_scale_reference(profile_id="raw"),
        policy=_policy(),
    )
    with pytest.raises(ReconstructionValidationError):
        result.point_cloud.points_metres(arbitrary_scale_reference(profile_id="raw-2"))


def test_capacity_is_bounded_and_reported() -> None:
    fusion = DenseVoxelFusion(
        session_id="capacity",
        camera_profile=_camera(),
        scale_profile=arbitrary_scale_reference(profile_id="raw"),
        policy=DenseFusionPolicy(
            scene_profile=DenseSceneProfile.SPATIAL_DIGITAL_TWIN,
            voxel_size_reconstruction_units=0.005,
            pixel_stride=1,
            max_voxels=100,
        ),
    )
    fusion.integrate(_frame(0))
    result = fusion.finalize()
    assert len(result.point_cloud.points_reconstruction) <= 100
    assert result.statistics.dropped_new_voxels_capacity > 0


def test_frame_order_and_ids_are_strict() -> None:
    fusion = DenseVoxelFusion(
        session_id="order",
        camera_profile=_camera(),
        scale_profile=arbitrary_scale_reference(profile_id="raw"),
        policy=_policy(),
    )
    fusion.integrate(_frame(1))
    with pytest.raises(ReconstructionValidationError):
        fusion.integrate(_frame(1))
    earlier = DenseFusionFrame(
        frame_id="earlier",
        timestamp_ns=1,
        rgb=np.zeros((48, 64, 3), np.uint8),
        depth=np.ones((48, 64), np.float32),
        T_reconstruction_from_camera=np.eye(4),
        depth_unit=DenseDepthUnit.RECONSTRUCTION_UNITS,
        depth_alignment=reconstruction_unit_depth_alignment(provenance="test"),
    )
    with pytest.raises(ReconstructionValidationError):
        fusion.integrate(earlier)


def test_live_trajectory_factory_preserves_optimized_pose_layer() -> None:
    pose = LiveTrajectoryPose(
        session_id=str(uuid4()),
        frame_id="live-1",
        sequence_index=1,
        camera_id="camera",
        timestamp_ns=123,
        raw_pose_id="raw-1",
        layer=LiveTrajectoryPoseLayer.OPTIMIZED_PROPAGATED,
        T_reconstruction_from_camera=np.eye(4),
        source_keyframe_id="kf-0",
        optimization_revision=3,
        optimizer_epoch=1,
        native_optimized_source_pose_id="native-0",
    )
    frame = DenseFusionFrame.from_live_trajectory_pose(
        pose=pose,
        rgb=np.zeros((4, 5, 3), np.uint8),
        depth=np.ones((4, 5), np.float32),
        depth_unit=DenseDepthUnit.RECONSTRUCTION_UNITS,
        depth_alignment=reconstruction_unit_depth_alignment(provenance="aligned"),
    )
    assert frame.pose_layer == "optimized_propagated"
    assert np.array_equal(frame.T_reconstruction_from_camera, pose.T_reconstruction_from_camera)


def test_explicit_alignment_records_uncertainty_without_claiming_metric_scale() -> None:
    alignment = explicit_depth_scale_alignment(
        scale_to_reconstruction_units=2.5,
        relative_scale_uncertainty=0.08,
        provenance="manual sparse alignment",
    )
    assert alignment.mode is DenseDepthAlignmentMode.EXPLICIT_SCALE_FACTOR
    assert alignment.scale_to_reconstruction_units == pytest.approx(2.5)
    assert alignment.relative_scale_uncertainty == pytest.approx(0.08)
