#!/usr/bin/env python3
"""Dependency-light Phase 9.1E metric qualification smoke."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.contracts import (  # noqa: E402
    AxisConvention,
    CalibrationStatus,
    CameraIntrinsics,
    CoordinateFrame,
    CoordinateFrameKind,
    ImageSize,
)
from core.models.metadata import IntrinsicsSource  # noqa: E402
from core.reconstruction import (  # noqa: E402
    CameraDistortionModel,
    MetricEvidenceKind,
    MetricEvidenceValidationState,
    MetricGeometryStatus,
    MetricQualificationState,
    ReconstructionCamera,
    assess_metric_qualification,
    assess_reconstruction_geometry,
    camera_geometry_from_calibrated_camera,
    future_metric_source_evidence,
    known_size_benchmark_supporting_evidence,
    known_size_reference_evidence,
    metric_depth_scale_evidence,
    reference_target_metric_evidence,
)


def _calibrated_camera() -> ReconstructionCamera:
    size = ImageSize(1280, 720)
    return ReconstructionCamera(
        camera_id="phase9-1e-camera",
        image_size=size,
        intrinsics=CameraIntrinsics(
            matrix=np.array(
                [[900.0, 0.0, 640.0], [0.0, 905.0, 360.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            pixel_space_id="phase9-1e-1280x720",
            source=IntrinsicsSource.CALIBRATED,
            calibration_status=CalibrationStatus.CALIBRATED,
            calibration_record_id="phase9-1e-calibration",
        ),
        camera_frame=CoordinateFrame(
            frame_id="phase9-1e-camera-frame",
            kind=CoordinateFrameKind.CAMERA_RGB,
            axis_convention=AxisConvention.X_RIGHT_Y_DOWN_Z_FORWARD,
        ),
        distortion_model=CameraDistortionModel.BROWN_CONRADY,
        distortion_coefficients=(0.01, -0.02, 0.0005, -0.0004, 0.003),
    )


def main() -> int:
    known = known_size_reference_evidence(
        evidence_id="phase9-1e-known-size",
        reconstruction_frame_id="reconstruction",
        reference_id="fixture-width-250mm",
        known_length_m=0.25,
        measured_length_reconstruction_units=5.0,
        known_length_relative_uncertainty=0.004,
        measurement_relative_uncertainty=0.02,
        provenance="synthetic known-size smoke evidence",
    )
    known_q = assess_metric_qualification(known, scale_profile_id="phase9-1e-known-scale")

    target = reference_target_metric_evidence(
        evidence_id="phase9-1e-target",
        reconstruction_frame_id="reconstruction",
        reference_id="qualified-target-100mm",
        target_length_m=0.1,
        measured_length_reconstruction_units=2.0,
        target_relative_uncertainty=0.001,
        measurement_relative_uncertainty=0.008,
        provenance="synthetic validated target smoke evidence",
        validation_state=MetricEvidenceValidationState.VALIDATED,
        validation_record_id="phase9-1e-target-validation",
    )
    target_q = assess_metric_qualification(target, scale_profile_id="phase9-1e-target-scale")

    metric_depth = metric_depth_scale_evidence(
        evidence_id="phase9-1e-metric-depth",
        reconstruction_frame_id="reconstruction",
        reference_id="qualified-metric-depth-source",
        unit_to_metre=1.0,
        relative_scale_uncertainty=0.03,
        provenance="synthetic qualified metric-depth smoke evidence",
        validation_state=MetricEvidenceValidationState.VALIDATED,
        validation_record_id="phase9-1e-metric-depth-validation",
    )
    metric_depth_q = assess_metric_qualification(
        metric_depth,
        scale_profile_id="phase9-1e-metric-depth-scale",
    )

    benchmark = known_size_benchmark_supporting_evidence(
        {
            "schema_version": 1,
            "passed": True,
            "summary": {"case_count": 3, "available_dimension_count": 9},
            "certified_metrology": False,
        },
        evidence_id="phase9-1e-benchmark",
        provenance="synthetic known-size benchmark smoke report",
        validation_record_id="phase9-1e-known-size-report",
    )
    benchmark_q = assess_metric_qualification(
        benchmark,
        scale_profile_id="phase9-1e-benchmark-must-not-scale",
    )

    future = future_metric_source_evidence(
        evidence_id="phase9-1e-future-rgbd",
        kind=MetricEvidenceKind.FUTURE_RGBD,
        provenance="reserved future RGB-D provider",
    )
    future_q = assess_metric_qualification(future, scale_profile_id="future-no-scale")

    camera_profile = camera_geometry_from_calibrated_camera(
        profile_id="phase9-1e-calibrated-camera",
        camera=_calibrated_camera(),
    )
    assert target_q.scale_profile is not None
    geometry = assess_reconstruction_geometry(camera_profile, target_q.scale_profile)

    ok = all(
        (
            known_q.state is MetricQualificationState.PROVISIONAL,
            known_q.authorizes_metric_coordinates,
            not known_q.authorizes_calibrated_metric_scale,
            target_q.state is MetricQualificationState.QUALIFIED,
            target_q.authorizes_calibrated_metric_scale,
            metric_depth_q.state is MetricQualificationState.QUALIFIED,
            benchmark_q.state is MetricQualificationState.SUPPORTING_ONLY,
            not benchmark_q.authorizes_metric_coordinates,
            future_q.state is MetricQualificationState.UNSUPPORTED,
            not future_q.authorizes_metric_coordinates,
            geometry.metric_status is MetricGeometryStatus.CALIBRATED_METRIC,
            geometry.metric_coordinates_allowed,
            geometry.calibrated_metric_geometry_allowed,
            not geometry.certified_metrology,
        )
    )

    payload = {
        "status": "PASS" if ok else "FAIL",
        "known_size": {
            "qualification_state": known_q.state.value,
            "metric_coordinates_allowed": known_q.authorizes_metric_coordinates,
            "calibrated_metric_scale": known_q.authorizes_calibrated_metric_scale,
            "relative_scale_uncertainty": known_q.relative_scale_uncertainty,
        },
        "reference_target": {
            "qualification_state": target_q.state.value,
            "metric_coordinates_allowed": target_q.authorizes_metric_coordinates,
            "calibrated_metric_scale": target_q.authorizes_calibrated_metric_scale,
        },
        "qualified_metric_depth": {
            "qualification_state": metric_depth_q.state.value,
            "relative_scale_uncertainty": metric_depth_q.relative_scale_uncertainty,
        },
        "known_size_benchmark": {
            "qualification_state": benchmark_q.state.value,
            "metric_coordinates_allowed": benchmark_q.authorizes_metric_coordinates,
        },
        "future_rgbd": {
            "qualification_state": future_q.state.value,
            "metric_coordinates_allowed": future_q.authorizes_metric_coordinates,
        },
        "calibrated_camera_plus_validated_target": geometry.metric_status.value,
        "certified_metrology": geometry.certified_metrology,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
