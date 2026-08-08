#!/usr/bin/env python3
"""Phase 8.6C deterministic dense multi-view fusion smoke."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.contracts import ImageSize
from core.reconstruction import (
    DenseDepthUnit,
    DenseFusionFrame,
    DenseFusionPolicy,
    DenseSceneProfile,
    MetricGeometryStatus,
    arbitrary_scale_reference,
    estimate_camera_geometry_from_horizontal_fov,
    estimate_depth_alignment_from_correspondences,
    fuse_dense_frames,
    known_dimension_scale_reference,
    metric_depth_alignment,
    reconstruction_unit_depth_alignment,
)


def _pose(tx: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = tx
    return matrix


def _camera(width: int, height: int):
    return estimate_camera_geometry_from_horizontal_fov(
        profile_id="phase8-6c-estimated-camera",
        camera_id="phase8-6c-camera",
        image_size=ImageSize(width, height),
        horizontal_fov_degrees=72.0,
        focal_length_relative_uncertainty=0.15,
        camera_frame_id="phase8-6c-camera-frame",
        reconstruction_frame_id="phase8-6c-reconstruction",
        notes="Validation-only estimated camera profile",
    )


def main() -> int:
    width, height = 96, 72
    camera = _camera(width, height)
    raw_scale = arbitrary_scale_reference(profile_id="phase8-6c-arbitrary")
    aligned = reconstruction_unit_depth_alignment(provenance="synthetic plane in reconstruction gauge")
    frames = []
    for index, tx in enumerate((-0.12, -0.06, 0.0, 0.06, 0.12)):
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[..., 0] = np.arange(width, dtype=np.uint8)[None, :]
        rgb[..., 1] = 80 + index * 20
        rgb[..., 2] = 180
        depth = np.full((height, width), 2.0, dtype=np.float32)
        frames.append(
            DenseFusionFrame(
                frame_id=f"arbitrary-{index}",
                timestamp_ns=1_000_000_000 + index * 50_000_000,
                rgb=rgb,
                depth=depth,
                T_reconstruction_from_camera=_pose(tx),
                depth_unit=DenseDepthUnit.RECONSTRUCTION_UNITS,
                depth_alignment=aligned,
                pose_layer="optimized_keyframe" if index in {0, 4} else "optimized_propagated",
            )
        )
    arbitrary = fuse_dense_frames(
        frames,
        session_id="phase8-6c-arbitrary-smoke",
        camera_profile=camera,
        scale_profile=raw_scale,
        policy=DenseFusionPolicy(
            scene_profile=DenseSceneProfile.SPATIAL_DIGITAL_TWIN,
            voxel_size_reconstruction_units=0.075,
            pixel_stride=2,
            minimum_view_observations=2,
            max_voxels=100_000,
        ),
        reconstruction_frame_id="phase8-6c-reconstruction",
    )

    metric_scale = known_dimension_scale_reference(
        profile_id="phase8-6c-known-dimension",
        reference_id="known-20mm",
        known_length_m=0.02,
        measured_length_reconstruction_units=2.0,
        known_length_relative_uncertainty=0.005,
        measurement_relative_uncertainty=0.02,
        provenance="Phase 8.6C validation known dimension",
    )
    metric_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    metric_rgb[..., 2] = 255
    metric_frame = DenseFusionFrame(
        frame_id="metric-0",
        timestamp_ns=2_000_000_000,
        rgb=metric_rgb,
        depth=np.full((height, width), 1.5, dtype=np.float32),
        T_reconstruction_from_camera=np.eye(4, dtype=np.float64),
        depth_unit=DenseDepthUnit.METRES,
        depth_alignment=metric_depth_alignment(metric_scale, provenance="metric validation depth"),
        pose_layer="optimized_keyframe",
    )
    metric = fuse_dense_frames(
        (metric_frame,),
        session_id="phase8-6c-metric-smoke",
        camera_profile=camera,
        scale_profile=metric_scale,
        policy=DenseFusionPolicy(
            scene_profile=DenseSceneProfile.OBJECT_INSPECTION,
            voxel_size_reconstruction_units=6.0,
            pixel_stride=3,
            minimum_view_observations=1,
            max_voxels=50_000,
        ),
        reconstruction_frame_id="phase8-6c-reconstruction",
    )
    metric_points_m = metric.point_cloud.points_metres(metric_scale)

    source = np.array([0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 20.0])
    target = source * 3.25
    target[-1] = 1.0
    sparse_alignment = estimate_depth_alignment_from_correspondences(
        source,
        target,
        provenance="synthetic sparse landmark correspondences",
        minimum_correspondences=8,
    )

    arbitrary_plane_z = float(np.median(arbitrary.point_cloud.points_reconstruction[:, 2]))
    metric_plane_z_m = float(np.median(metric_points_m[:, 2]))
    passed = (
        arbitrary.metric_geometry_status is MetricGeometryStatus.ARBITRARY
        and arbitrary.unit_to_metre is None
        and arbitrary.statistics.frames_integrated == 5
        and arbitrary.statistics.multi_view_voxels > 200
        and len(arbitrary.point_cloud.points_reconstruction) > 200
        and abs(arbitrary_plane_z - 2.0) < 1e-9
        and set(arbitrary.source_pose_layers) == {"optimized_keyframe", "optimized_propagated"}
        and metric.metric_geometry_status is MetricGeometryStatus.PROVISIONAL_METRIC
        and metric.unit_to_metre == 0.01
        and abs(metric_plane_z_m - 1.5) < 1e-8
        and abs(sparse_alignment.scale_to_reconstruction_units - 3.25) < 1e-12
        and sparse_alignment.relative_scale_uncertainty < 1e-12
        and not arbitrary.certified_metrology
        and not metric.certified_metrology
    )

    output_dir = Path("/tmp/phase8_6c_dense_multiview")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "phase": "8.6C",
        "pass": passed,
        "arbitrary": arbitrary.as_dict(),
        "arbitrary_plane_z_reconstruction_units": arbitrary_plane_z,
        "metric": metric.as_dict(),
        "metric_plane_z_metres": metric_plane_z_m,
        "sparse_alignment": sparse_alignment.as_dict(),
        "camera_geometry_mode": camera.mode.value,
        "camera_calibration_status": camera.camera.intrinsics.calibration_status.value,
        "reconstruction_ui_enabled": False,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Phase 8.6C dense multi-view fusion smoke: {'PASS' if passed else 'FAIL'}")
    print(
        f"arbitrary_frames={arbitrary.statistics.frames_integrated} "
        f"points={len(arbitrary.point_cloud.points_reconstruction)} "
        f"multi_view_voxels={arbitrary.statistics.multi_view_voxels} "
        f"plane_z={arbitrary_plane_z:.6f}"
    )
    print(
        f"metric_status={metric.metric_geometry_status.value} "
        f"unit_to_metre={metric.unit_to_metre} plane_z_m={metric_plane_z_m:.6f}"
    )
    print(
        f"sparse_alignment_scale={sparse_alignment.scale_to_reconstruction_units:.6f} "
        f"uncertainty={sparse_alignment.relative_scale_uncertainty:.6f}"
    )
    print("pose_convention=T_reconstruction_from_camera")
    print("arbitrary_scale_preserved=True certified_metrology=False")
    print(f"summary={summary_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
