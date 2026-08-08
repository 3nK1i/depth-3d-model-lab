#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from core.contracts import ImageSize
from core.reconstruction import (
    MetricGeometryStatus,
    assess_reconstruction_geometry,
    arbitrary_scale_reference,
    estimate_camera_geometry_from_horizontal_fov,
    known_dimension_scale_reference,
    reconstruction_distance_to_metres,
    reconstruction_points_to_metres,
    reference_target_scale_reference,
)
from core.reconstruction.errors import ReconstructionValidationError


def main() -> int:
    camera = estimate_camera_geometry_from_horizontal_fov(
        profile_id="phase8-6b-estimated-camera",
        camera_id="phase8-6b-camera",
        image_size=ImageSize(1280, 720),
        horizontal_fov_degrees=79.0,
        focal_length_relative_uncertainty=0.20,
    )
    arbitrary = arbitrary_scale_reference(profile_id="phase8-6b-arbitrary")
    arbitrary_rejected = False
    try:
        reconstruction_distance_to_metres(1.0, arbitrary)
    except ReconstructionValidationError:
        arbitrary_rejected = True

    known = known_dimension_scale_reference(
        profile_id="phase8-6b-known-dimension",
        reference_id="connector-width-20mm",
        known_length_m=0.020,
        measured_length_reconstruction_units=2.5,
        known_length_relative_uncertainty=0.005,
        measurement_relative_uncertainty=0.020,
        provenance="operator-entered known connector dimension",
    )
    known_assessment = assess_reconstruction_geometry(camera, known)
    metric_distance = reconstruction_distance_to_metres(2.5, known)
    metric_points = reconstruction_points_to_metres(
        np.array([[0.0, 0.0, 0.0], [2.5, 1.25, -0.5]], dtype=np.float64),
        known,
    )

    target = reference_target_scale_reference(
        profile_id="phase8-6b-target",
        reference_id="validated-target-40mm",
        target_length_m=0.040,
        measured_length_reconstruction_units=5.0,
        target_relative_uncertainty=0.001,
        measurement_relative_uncertainty=0.010,
        validated_target=True,
        provenance="validated reference-target record",
    )
    target_assessment = assess_reconstruction_geometry(camera, target)

    ok = all(
        (
            arbitrary_rejected,
            arbitrary.gauge.unit_to_metre is None,
            abs(float(known.gauge.unit_to_metre or 0.0) - 0.008) < 1e-12,
            abs(metric_distance - 0.020) < 1e-12,
            np.allclose(metric_points[1], [0.020, 0.010, -0.004]),
            known_assessment.metric_status is MetricGeometryStatus.PROVISIONAL_METRIC,
            known_assessment.metric_coordinates_allowed,
            not known_assessment.calibrated_metric_geometry_allowed,
            not known_assessment.certified_metrology,
            target.is_calibrated_metric,
            target_assessment.metric_status is MetricGeometryStatus.PROVISIONAL_METRIC,
            not target_assessment.calibrated_metric_geometry_allowed,
            not target_assessment.certified_metrology,
        )
    )
    print(f"Phase 8.6B scale/reference smoke: {'PASS' if ok else 'FAIL'}")
    print(
        "arbitrary_scale_rejected_for_metric_conversion="
        f"{arbitrary_rejected} unit_to_metre={arbitrary.gauge.unit_to_metre}"
    )
    print(
        "known_dimension_status="
        f"{known.gauge.scale_status.value} unit_to_metre={known.gauge.unit_to_metre:.6f} "
        f"relative_uncertainty={known.gauge.relative_scale_uncertainty:.6f}"
    )
    print(
        "estimated_camera_plus_known_dimension="
        f"{known_assessment.metric_status.value} metric_coordinates_allowed="
        f"{known_assessment.metric_coordinates_allowed}"
    )
    print(
        "estimated_camera_plus_validated_target="
        f"{target_assessment.metric_status.value} calibrated_metric_geometry_allowed="
        f"{target_assessment.calibrated_metric_geometry_allowed}"
    )
    print(f"certified_metrology={known_assessment.certified_metrology}")
    print("world_alignment=arbitrary")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
