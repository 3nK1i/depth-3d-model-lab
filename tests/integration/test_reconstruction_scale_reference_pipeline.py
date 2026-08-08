from __future__ import annotations

import numpy as np

from core.contracts import ImageSize
from core.reconstruction import (
    MetricGeometryStatus,
    assess_reconstruction_geometry,
    estimate_camera_geometry_from_horizontal_fov,
    known_dimension_scale_reference,
    reconstruction_points_to_metres,
)


def test_estimated_camera_known_dimension_pipeline_remains_provisional_but_metric_units_are_traceable() -> None:
    camera = estimate_camera_geometry_from_horizontal_fov(
        profile_id="phone-estimated",
        camera_id="phone-rtsp",
        image_size=ImageSize(1280, 720),
        horizontal_fov_degrees=75.0,
        focal_length_relative_uncertainty=0.20,
    )
    scale = known_dimension_scale_reference(
        profile_id="connector-reference",
        reference_id="connector-body-width",
        known_length_m=0.024,
        measured_length_reconstruction_units=3.0,
        known_length_relative_uncertainty=0.005,
        measurement_relative_uncertainty=0.025,
        provenance="operator-entered drawing dimension",
    )
    assessment = assess_reconstruction_geometry(camera, scale)
    assert assessment.metric_status is MetricGeometryStatus.PROVISIONAL_METRIC
    assert assessment.metric_coordinates_allowed is True
    assert assessment.calibrated_metric_geometry_allowed is False
    assert assessment.certified_metrology is False

    raw_points = np.array([[0.0, 0.0, 0.0], [3.0, 1.5, -0.75]], dtype=np.float64)
    metric_points = reconstruction_points_to_metres(raw_points, scale)
    np.testing.assert_allclose(metric_points[1], [0.024, 0.012, -0.006])
