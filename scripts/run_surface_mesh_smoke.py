#!/usr/bin/env python3
"""Phase 8.7A deterministic quality-aware surface mesh smoke."""

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
    DenseGeometryEvidence,
    DenseGeometryQualityPolicy,
    DenseSceneProfile,
    MetricGeometryStatus,
    arbitrary_scale_reference,
    assess_dense_geometry_quality,
    estimate_camera_geometry_from_horizontal_fov,
    fuse_dense_frames,
    generate_quality_aware_surface_mesh,
    known_dimension_scale_reference,
    metric_depth_alignment,
    reconstruction_unit_depth_alignment,
    write_surface_mesh_obj,
)


def _pose(tx: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = tx
    return matrix


def _camera(width: int, height: int):
    return estimate_camera_geometry_from_horizontal_fov(
        profile_id="phase8-7a-estimated-camera",
        camera_id="phase8-7a-camera",
        image_size=ImageSize(width, height),
        horizontal_fov_degrees=72.0,
        focal_length_relative_uncertainty=0.15,
        camera_frame_id="phase8-7a-camera-frame",
        reconstruction_frame_id="phase8-7a-reconstruction",
        notes="Validation-only estimated camera profile",
    )


def _good_evidence(count: int, *, angle: float = 28.0) -> DenseGeometryEvidence:
    return DenseGeometryEvidence(
        point_count=count,
        depth_relative_residual=np.full(count, 0.01, dtype=np.float32),
        view_angle_span_degrees=np.full(count, angle, dtype=np.float32),
        visibility_ratio=np.full(count, 0.96, dtype=np.float32),
        provenance="Phase 8.7A synthetic trusted surface evidence",
    )


def _arbitrary_surface(width: int, height: int, camera):
    scale = arbitrary_scale_reference(profile_id="phase8-7a-arbitrary")
    alignment = reconstruction_unit_depth_alignment(
        provenance="Phase 8.7A synthetic plane already in reconstruction units"
    )
    frames = []
    for index, tx in enumerate((-0.12, -0.06, 0.0, 0.06, 0.12)):
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[..., 0] = np.arange(width, dtype=np.uint8)[None, :]
        rgb[..., 1] = 70 + index * 25
        rgb[..., 2] = 190
        frames.append(
            DenseFusionFrame(
                frame_id=f"arbitrary-{index}",
                timestamp_ns=1_000_000_000 + index * 50_000_000,
                rgb=rgb,
                depth=np.full((height, width), 2.0, dtype=np.float32),
                T_reconstruction_from_camera=_pose(tx),
                depth_unit=DenseDepthUnit.RECONSTRUCTION_UNITS,
                depth_alignment=alignment,
                confidence=np.full((height, width), 0.96, dtype=np.float32),
                pose_layer="optimized_keyframe" if index in {0, 4} else "optimized_propagated",
            )
        )
    dense = fuse_dense_frames(
        frames,
        session_id="phase8-7a-arbitrary-smoke",
        camera_profile=camera,
        scale_profile=scale,
        policy=DenseFusionPolicy(
            scene_profile=DenseSceneProfile.SPATIAL_DIGITAL_TWIN,
            voxel_size_reconstruction_units=0.075,
            pixel_stride=2,
            minimum_view_observations=2,
            max_voxels=100_000,
        ),
        reconstruction_frame_id="phase8-7a-reconstruction",
    )
    quality = assess_dense_geometry_quality(
        dense,
        camera_profile=camera,
        scale_profile=scale,
        evidence=_good_evidence(len(dense.point_cloud.points_reconstruction)),
    )
    return dense, quality, scale


def _metric_surface(width: int, height: int, camera):
    scale = known_dimension_scale_reference(
        profile_id="phase8-7a-known-dimension",
        reference_id="known-20mm",
        known_length_m=0.02,
        measured_length_reconstruction_units=2.0,
        known_length_relative_uncertainty=0.005,
        measurement_relative_uncertainty=0.02,
        provenance="Phase 8.7A validation known dimension",
    )
    frames = []
    for index, tx in enumerate((-3.0, 0.0, 3.0)):
        frames.append(
            DenseFusionFrame(
                frame_id=f"metric-{index}",
                timestamp_ns=2_000_000_000 + index * 50_000_000,
                rgb=np.full((height, width, 3), (80, 150, 210), dtype=np.uint8),
                depth=np.full((height, width), 1.5, dtype=np.float32),
                T_reconstruction_from_camera=_pose(tx),
                depth_unit=DenseDepthUnit.METRES,
                depth_alignment=metric_depth_alignment(
                    scale, provenance="Phase 8.7A validation metric depth"
                ),
                confidence=np.full((height, width), 0.96, dtype=np.float32),
                pose_layer="optimized_keyframe",
            )
        )
    dense = fuse_dense_frames(
        frames,
        session_id="phase8-7a-metric-smoke",
        camera_profile=camera,
        scale_profile=scale,
        policy=DenseFusionPolicy(
            scene_profile=DenseSceneProfile.OBJECT_INSPECTION,
            voxel_size_reconstruction_units=8.0,
            pixel_stride=3,
            minimum_view_observations=1,
            max_voxels=50_000,
        ),
        reconstruction_frame_id="phase8-7a-reconstruction",
    )
    quality = assess_dense_geometry_quality(
        dense,
        camera_profile=camera,
        scale_profile=scale,
        policy=DenseGeometryQualityPolicy(
            scene_profile=DenseSceneProfile.OBJECT_INSPECTION,
            target_view_observations=2,
            minimum_view_observations=1,
            minimum_confidence=0.65,
            desired_view_angle_span_degrees=20.0,
            maximum_depth_relative_residual=0.04,
            high_score_threshold=0.80,
            medium_score_threshold=0.60,
            minimum_input_depth_coverage=0.65,
        ),
        evidence=_good_evidence(len(dense.point_cloud.points_reconstruction)),
    )
    return dense, quality, scale


def main() -> int:
    width, height = 96, 72
    camera = _camera(width, height)
    arbitrary_dense, arbitrary_quality, arbitrary_scale = _arbitrary_surface(width, height, camera)
    arbitrary_mesh = generate_quality_aware_surface_mesh(arbitrary_dense, arbitrary_quality)

    metric_dense, metric_quality, metric_scale = _metric_surface(width, height, camera)
    metric_mesh = generate_quality_aware_surface_mesh(metric_dense, metric_quality)
    metric_vertices = metric_mesh.vertices_metres(metric_scale)
    metric_uncertainty = metric_mesh.uncertainty_metres(metric_scale)

    output_dir = Path("/tmp/phase8_7a_surface_mesh")
    output_dir.mkdir(parents=True, exist_ok=True)
    arbitrary_obj = write_surface_mesh_obj(arbitrary_mesh, output_dir / "arbitrary_surface.obj")
    metric_obj = write_surface_mesh_obj(metric_mesh, output_dir / "metric_surface.obj")

    source_points = arbitrary_dense.point_cloud.points_reconstruction[
        arbitrary_mesh.mesh.source_point_indices.astype(np.int64)
    ]
    positions_preserved = np.array_equal(source_points, arbitrary_mesh.mesh.vertices_reconstruction)

    passed = (
        arbitrary_mesh.metric_geometry_status is MetricGeometryStatus.ARBITRARY
        and arbitrary_mesh.unit_to_metre is None
        and arbitrary_mesh.mesh.vertex_count > 100
        and arbitrary_mesh.mesh.face_count > 100
        and arbitrary_mesh.mesh.component_count >= 1
        and float(np.min(arbitrary_mesh.mesh.vertex_quality_scores))
        >= arbitrary_mesh.policy.minimum_vertex_quality_score
        and positions_preserved
        and metric_mesh.metric_geometry_status is MetricGeometryStatus.PROVISIONAL_METRIC
        and metric_mesh.unit_to_metre == 0.01
        and metric_mesh.mesh.vertex_count > 50
        and metric_mesh.mesh.face_count > 50
        and np.isfinite(metric_vertices).all()
        and np.isfinite(metric_uncertainty).all()
        and np.all(metric_uncertainty > 0.0)
        and not arbitrary_mesh.certified_metrology
        and not metric_mesh.certified_metrology
        and arbitrary_obj.exists()
        and metric_obj.exists()
    )

    summary = {
        "phase": "8.7A",
        "pass": passed,
        "arbitrary_mesh": arbitrary_mesh.as_dict(),
        "metric_mesh": metric_mesh.as_dict(),
        "positions_preserved": positions_preserved,
        "metric_vertex_median_z_m": float(np.median(metric_vertices[:, 2])),
        "metric_uncertainty_median_m": float(np.median(metric_uncertainty)),
        "relative_scale_uncertainty": metric_scale.gauge.relative_scale_uncertainty,
        "arbitrary_obj": str(arbitrary_obj),
        "metric_obj": str(metric_obj),
        "reconstruction_ui_enabled": False,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Phase 8.7A quality-aware surface mesh smoke: {'PASS' if passed else 'FAIL'}")
    print(
        f"arbitrary_vertices={arbitrary_mesh.mesh.vertex_count} "
        f"faces={arbitrary_mesh.mesh.face_count} "
        f"components={arbitrary_mesh.mesh.component_count} "
        f"positions_preserved={positions_preserved}"
    )
    print(
        f"metric_status={metric_mesh.metric_geometry_status.value} "
        f"unit_to_metre={metric_mesh.unit_to_metre} "
        f"vertices={metric_mesh.mesh.vertex_count} faces={metric_mesh.mesh.face_count}"
    )
    print(
        f"metric_uncertainty_median_m={float(np.median(metric_uncertainty)):.6f} "
        f"relative_scale_uncertainty={metric_scale.gauge.relative_scale_uncertainty:.6f}"
    )
    print("pose_convention=T_reconstruction_from_camera")
    print("certified_metrology=False reconstruction_ui_enabled=False")
    print(f"mesh_obj={arbitrary_obj}")
    print(f"summary={summary_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
