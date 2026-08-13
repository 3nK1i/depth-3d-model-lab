from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from core.contracts import ImageSize
from core.reconstruction import (
    MetricEvidenceValidationState,
    MetricGeometryStatus,
    MetricQualificationState,
    assess_metric_qualification,
    assess_reconstruction_geometry,
    camera_geometry_from_calibrated_camera,
    estimate_camera_geometry_from_horizontal_fov,
    known_size_benchmark_supporting_evidence,
    known_size_reference_evidence,
    metric_depth_scale_evidence,
    reconstruction_points_to_metres,
    reference_target_metric_evidence,
)
from core.validation import run_known_size_validation
from tests.reconstruction_live_helpers import make_calibrated_live_camera


FIXED_TIME = "2026-08-13T17:00:00+00:00"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_existing_known_size_validator_flows_to_supporting_only_evidence(tmp_path: Path) -> None:
    export = tmp_path / "inspection_export"
    export.mkdir()
    measurements = {
        "schema_version": 1,
        "certified_metrology": False,
        "objects": [
            {
                "observation_id": "box-0",
                "object_index": 0,
                "class_name": "box",
                "preferred_box_id": "box-0:camera_aligned",
                "mask_source": "native_instance",
                "mask_revision": 0,
                "measurements": [
                    {"kind": "width", "status": "available", "value_si": 0.402, "expanded_uncertainty_si": 0.008, "reliability": "high", "rejection_reasons": []},
                    {"kind": "height", "status": "available", "value_si": 0.299, "expanded_uncertainty_si": 0.006, "reliability": "high", "rejection_reasons": []},
                    {"kind": "depth", "status": "available", "value_si": 0.251, "expanded_uncertainty_si": 0.007, "reliability": "high", "rejection_reasons": []},
                ],
            }
        ],
    }
    measurements_path = export / "measurements.json"
    measurements_path.write_text(json.dumps(measurements), encoding="utf-8")
    marker = export / "report.txt"
    marker.write_text("integration evidence\n", encoding="utf-8")
    (export / "manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {"path": "measurements.json", "sha256": _sha256(measurements_path)},
                    {"path": "report.txt", "sha256": _sha256(marker)},
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "benchmark_name": "Phase 9.1E known-size bridge",
        "operator": "integration",
        "measurement_instrument": "synthetic",
        "thresholds_confirmed": True,
        "thresholds": {
            "minimum_cases": 1,
            "minimum_available_dimensions": 3,
            "require_all_case_dimensions_available": True,
            "max_mean_absolute_error_mm": 5.0,
            "max_mean_relative_error_percent": 2.0,
            "max_single_dimension_error_mm": 5.0,
            "max_single_dimension_relative_error_percent": 2.0,
            "minimum_uncertainty_coverage_percent": 100.0,
        },
        "cases": [
            {
                "case_id": "box",
                "export_path": str(export),
                "observation_id": "box-0",
                "ground_truth_mm": {"width": 400.0, "height": 300.0, "depth": 250.0},
                "measurement_mapping": {"width": "width", "height": "height", "depth": "depth"},
            }
        ],
    }
    manifest_path = tmp_path / "benchmark.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = run_known_size_validation(manifest_path)
    assert report.passed is True
    evidence = known_size_benchmark_supporting_evidence(
        report.payload,
        evidence_id="known-size-benchmark",
        provenance="existing core.validation.known_size output",
        validation_record_id="known-size-report",
        created_at_utc=FIXED_TIME,
    )
    qualification = assess_metric_qualification(evidence, scale_profile_id="must-not-be-used")
    assert qualification.state is MetricQualificationState.SUPPORTING_ONLY
    assert qualification.scale_profile is None
    assert qualification.authorizes_metric_coordinates is False
    assert qualification.certified_metrology is False


def test_known_size_reference_emits_existing_scale_profile_for_geometry_conversion() -> None:
    evidence = known_size_reference_evidence(
        evidence_id="known-size",
        reconstruction_frame_id="reconstruction",
        reference_id="fixture-width",
        known_length_m=0.25,
        measured_length_reconstruction_units=5.0,
        known_length_relative_uncertainty=0.004,
        measurement_relative_uncertainty=0.02,
        provenance="independently measured fixture",
        created_at_utc=FIXED_TIME,
    )
    qualification = assess_metric_qualification(evidence, scale_profile_id="scale-known-size")
    assert qualification.scale_profile is not None
    points = reconstruction_points_to_metres(
        np.array([[0.0, 0.0, 0.0], [5.0, 2.0, -1.0]], dtype=np.float64),
        qualification.scale_profile,
    )
    np.testing.assert_allclose(points[1], [0.25, 0.10, -0.05])

    estimated_camera = estimate_camera_geometry_from_horizontal_fov(
        profile_id="estimated-camera",
        camera_id="camera",
        image_size=ImageSize(1280, 720),
        horizontal_fov_degrees=75.0,
        focal_length_relative_uncertainty=0.20,
    )
    geometry = assess_reconstruction_geometry(estimated_camera, qualification.scale_profile)
    assert geometry.metric_status is MetricGeometryStatus.PROVISIONAL_METRIC
    assert geometry.certified_metrology is False


def test_validated_reference_target_plus_calibrated_camera_yields_calibrated_metric_geometry() -> None:
    evidence = reference_target_metric_evidence(
        evidence_id="target",
        reconstruction_frame_id="reconstruction",
        reference_id="target-100mm",
        target_length_m=0.1,
        measured_length_reconstruction_units=10.0,
        target_relative_uncertainty=0.001,
        measurement_relative_uncertainty=0.008,
        provenance="qualified target certificate",
        validation_state=MetricEvidenceValidationState.VALIDATED,
        validation_record_id="target-validation",
        created_at_utc=FIXED_TIME,
    )
    qualification = assess_metric_qualification(evidence, scale_profile_id="target-scale")
    camera = camera_geometry_from_calibrated_camera(
        profile_id="camera-calibrated",
        camera=make_calibrated_live_camera(),
    )
    assert qualification.scale_profile is not None
    geometry = assess_reconstruction_geometry(camera, qualification.scale_profile)
    assert qualification.state is MetricQualificationState.QUALIFIED
    assert geometry.metric_status is MetricGeometryStatus.CALIBRATED_METRIC
    assert geometry.calibrated_metric_geometry_allowed is True
    assert geometry.certified_metrology is False


def test_qualified_metric_depth_maps_to_external_metric_prior_without_inventing_scale() -> None:
    evidence = metric_depth_scale_evidence(
        evidence_id="metric-depth",
        reconstruction_frame_id="reconstruction",
        reference_id="qualified-metric-depth-run",
        unit_to_metre=0.001,
        relative_scale_uncertainty=0.025,
        provenance="qualified metric-depth source with explicit reconstruction-unit mapping",
        validation_state=MetricEvidenceValidationState.VALIDATED,
        validation_record_id="metric-depth-validation",
        created_at_utc=FIXED_TIME,
    )
    qualification = assess_metric_qualification(evidence, scale_profile_id="metric-depth-scale")
    assert qualification.state is MetricQualificationState.QUALIFIED
    assert qualification.scale_profile is not None
    assert qualification.scale_profile.gauge.unit_to_metre == 0.001
    assert qualification.scale_profile.gauge.relative_scale_uncertainty == 0.025
    assert qualification.certified_metrology is False
