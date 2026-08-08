#!/usr/bin/env python3
"""Phase 8.6D deterministic geometry-quality and uncertainty smoke."""

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
    GeometryQualityLevel,
    MetricGeometryStatus,
    arbitrary_scale_reference,
    assess_dense_geometry_quality,
    estimate_camera_geometry_from_horizontal_fov,
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
        profile_id="phase8-6d-estimated-camera",
        camera_id="phase8-6d-camera",
        image_size=ImageSize(width, height),
        horizontal_fov_degrees=72.0,
        focal_length_relative_uncertainty=0.15,
        camera_frame_id="phase8-6d-camera-frame",
        reconstruction_frame_id="phase8-6d-reconstruction",
        notes="Validation-only estimated camera profile",
    )


def _arbitrary_dense(width: int, height: int, camera):
    scale = arbitrary_scale_reference(profile_id="phase8-6d-arbitrary")
    alignment = reconstruction_unit_depth_alignment(
        provenance="Phase 8.6D synthetic plane in reconstruction units"
    )
    frames = []
    for index, tx in enumerate((-0.12, -0.06, 0.0, 0.06, 0.12)):
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[..., 0] = np.arange(width, dtype=np.uint8)[None, :]
        rgb[..., 1] = 80 + index * 20
        rgb[..., 2] = 180
        frames.append(
            DenseFusionFrame(
                frame_id=f"arbitrary-{index}",
                timestamp_ns=1_000_000_000 + index * 50_000_000,
                rgb=rgb,
                depth=np.full((height, width), 2.0, dtype=np.float32),
                T_reconstruction_from_camera=_pose(tx),
                depth_unit=DenseDepthUnit.RECONSTRUCTION_UNITS,
                depth_alignment=alignment,
                confidence=np.full((height, width), 0.95, dtype=np.float32),
                pose_layer="optimized_keyframe" if index in {0, 4} else "optimized_propagated",
            )
        )
    dense = fuse_dense_frames(
        frames,
        session_id="phase8-6d-arbitrary-smoke",
        camera_profile=camera,
        scale_profile=scale,
        policy=DenseFusionPolicy(
            scene_profile=DenseSceneProfile.SPATIAL_DIGITAL_TWIN,
            voxel_size_reconstruction_units=0.075,
            pixel_stride=2,
            minimum_view_observations=2,
            max_voxels=100_000,
        ),
        reconstruction_frame_id="phase8-6d-reconstruction",
    )
    return dense, scale


def _metric_dense(width: int, height: int, camera):
    scale = known_dimension_scale_reference(
        profile_id="phase8-6d-known-dimension",
        reference_id="known-20mm",
        known_length_m=0.02,
        measured_length_reconstruction_units=2.0,
        known_length_relative_uncertainty=0.005,
        measurement_relative_uncertainty=0.02,
        provenance="Phase 8.6D validation known dimension",
    )
    frames = []
    for index, tx in enumerate((-3.0, 0.0, 3.0)):
        frames.append(
            DenseFusionFrame(
                frame_id=f"metric-{index}",
                timestamp_ns=2_000_000_000 + index * 50_000_000,
                rgb=np.zeros((height, width, 3), dtype=np.uint8),
                depth=np.full((height, width), 1.5, dtype=np.float32),
                T_reconstruction_from_camera=_pose(tx),
                depth_unit=DenseDepthUnit.METRES,
                depth_alignment=metric_depth_alignment(
                    scale, provenance="Phase 8.6D metric validation depth"
                ),
                confidence=np.full((height, width), 0.95, dtype=np.float32),
                pose_layer="optimized_keyframe",
            )
        )
    dense = fuse_dense_frames(
        frames,
        session_id="phase8-6d-metric-smoke",
        camera_profile=camera,
        scale_profile=scale,
        policy=DenseFusionPolicy(
            scene_profile=DenseSceneProfile.OBJECT_INSPECTION,
            voxel_size_reconstruction_units=8.0,
            pixel_stride=3,
            minimum_view_observations=1,
            max_voxels=50_000,
        ),
        reconstruction_frame_id="phase8-6d-reconstruction",
    )
    return dense, scale


def _good_evidence(count: int) -> DenseGeometryEvidence:
    return DenseGeometryEvidence(
        point_count=count,
        depth_relative_residual=np.full(count, 0.01, dtype=np.float32),
        view_angle_span_degrees=np.full(count, 24.0, dtype=np.float32),
        visibility_ratio=np.full(count, 0.95, dtype=np.float32),
        provenance="synthetic cross-view residual, angle span, and visibility evidence",
    )


def main() -> int:
    width, height = 96, 72
    camera = _camera(width, height)

    arbitrary, raw_scale = _arbitrary_dense(width, height, camera)
    arbitrary_count = len(arbitrary.point_cloud.points_reconstruction)
    good_report = assess_dense_geometry_quality(
        arbitrary,
        camera_profile=camera,
        scale_profile=raw_scale,
        evidence=_good_evidence(arbitrary_count),
    )

    degraded_evidence = DenseGeometryEvidence(
        point_count=arbitrary_count,
        depth_relative_residual=np.full(arbitrary_count, 0.25, dtype=np.float32),
        view_angle_span_degrees=np.full(arbitrary_count, 1.0, dtype=np.float32),
        visibility_ratio=np.full(arbitrary_count, 0.15, dtype=np.float32),
        provenance="intentionally degraded validation evidence",
    )
    degraded_report = assess_dense_geometry_quality(
        arbitrary,
        camera_profile=camera,
        scale_profile=raw_scale,
        evidence=degraded_evidence,
    )

    incomplete_report = assess_dense_geometry_quality(
        arbitrary,
        camera_profile=camera,
        scale_profile=raw_scale,
    )

    metric, metric_scale = _metric_dense(width, height, camera)
    metric_count = len(metric.point_cloud.points_reconstruction)
    metric_report = assess_dense_geometry_quality(
        metric,
        camera_profile=camera,
        scale_profile=metric_scale,
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
        evidence=_good_evidence(metric_count),
    )
    metric_uncertainty = metric_report.point_quality.local_uncertainty_metres(metric_scale)

    passed = (
        good_report.summary.global_level is GeometryQualityLevel.HIGH
        and good_report.summary.accepted_ratio > 0.90
        and good_report.summary.high_quality_ratio > 0.50
        and good_report.metric_geometry_status is MetricGeometryStatus.ARBITRARY
        and good_report.summary.median_local_uncertainty_m is None
        and "arbitrary_scale_no_metric_claim" in good_report.quality_limiters
        and degraded_report.summary.global_level is GeometryQualityLevel.LOW
        and degraded_report.summary.low_quality_ratio > 0.50
        and incomplete_report.summary.evidence_completeness_ratio == 0.0
        and incomplete_report.summary.global_level is not GeometryQualityLevel.HIGH
        and metric_report.metric_geometry_status is MetricGeometryStatus.PROVISIONAL_METRIC
        and metric_report.summary.median_local_uncertainty_m is not None
        and np.isclose(
            float(np.median(metric_uncertainty)),
            metric_report.summary.median_local_uncertainty_m,
            rtol=1e-6,
            atol=1e-12,
        )
        and np.isclose(
            metric_report.summary.relative_scale_uncertainty,
            metric_scale.gauge.relative_scale_uncertainty,
            rtol=0.0,
            atol=1e-15,
        )
        and not good_report.certified_metrology
        and not metric_report.certified_metrology
    )

    output_dir = Path("/tmp/phase8_6d_geometry_quality")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "phase": "8.6D",
        "pass": passed,
        "good_relative_geometry": good_report.as_dict(),
        "degraded_relative_geometry": degraded_report.as_dict(),
        "incomplete_evidence_geometry": incomplete_report.as_dict(),
        "provisional_metric_geometry": metric_report.as_dict(),
        "metric_local_uncertainty_median_m": float(np.median(metric_uncertainty)),
        "scale_relative_uncertainty": metric_scale.gauge.relative_scale_uncertainty,
        "camera_geometry_mode": camera.mode.value,
        "camera_calibration_status": camera.camera.intrinsics.calibration_status.value,
        "reconstruction_ui_enabled": False,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Phase 8.6D geometry quality/uncertainty smoke: {'PASS' if passed else 'FAIL'}")
    print(
        f"good_quality={good_report.summary.global_level.value} "
        f"score={good_report.summary.global_score:.3f} "
        f"high_ratio={good_report.summary.high_quality_ratio:.3f} "
        f"accepted_ratio={good_report.summary.accepted_ratio:.3f}"
    )
    print(
        f"degraded_quality={degraded_report.summary.global_level.value} "
        f"score={degraded_report.summary.global_score:.3f} "
        f"low_ratio={degraded_report.summary.low_quality_ratio:.3f}"
    )
    print(
        f"incomplete_evidence_quality={incomplete_report.summary.global_level.value} "
        f"evidence_completeness={incomplete_report.summary.evidence_completeness_ratio:.3f}"
    )
    print(
        f"metric_status={metric_report.metric_geometry_status.value} "
        f"median_local_uncertainty_m={metric_report.summary.median_local_uncertainty_m:.6f} "
        f"relative_scale_uncertainty={metric_report.summary.relative_scale_uncertainty:.6f}"
    )
    print("unobserved_semantics=sampled_input_depth_only")
    print("pose_convention=T_reconstruction_from_camera")
    print("certified_metrology=False reconstruction_ui_enabled=False")
    print(f"summary={summary_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
