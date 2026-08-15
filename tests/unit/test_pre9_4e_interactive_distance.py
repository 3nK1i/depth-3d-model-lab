from __future__ import annotations

import numpy as np
import pytest

from core.contracts.spatial import CalibrationStatus
from core.measurement.interactive_distance import (
    InteractiveDistanceEstimate,
    InteractiveDistanceState,
    estimate_depth_pixel_distance,
    estimate_reconstruction_distance,
    format_interactive_distance,
)
from core.models.metadata import DepthScale, IntrinsicsSource


def _depth_inputs():
    depth = np.full((80, 100), 2.0, dtype=np.float32)
    valid = np.ones_like(depth, dtype=bool)
    K = np.asarray([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return depth, valid, K


def test_metric_depth_returns_estimated_real_world_distance_with_uncertainty_before_calibration() -> None:
    depth, valid, K = _depth_inputs()
    result = estimate_depth_pixel_distance(
        depth=depth,
        depth_valid_mask=valid,
        intrinsics_matrix=K,
        pixel_a_xy=(30, 40),
        pixel_b_xy=(70, 40),
        depth_scale=DepthScale.METRIC,
        intrinsics_source=IntrinsicsSource.MODEL_PREDICTED,
        calibration_status=CalibrationStatus.NOT_CALIBRATED,
    )
    assert result.state is InteractiveDistanceState.ESTIMATED
    assert result.distance_m == pytest.approx(0.8, rel=1e-5)
    assert result.expanded_uncertainty_m is not None and result.expanded_uncertainty_m > 0.0
    assert result.certified_metrology is False
    primary, secondary = format_interactive_distance(result)
    assert primary.startswith("800.0 ±")
    assert "ESTIMATED" in secondary
    assert "certified metrology: no" in secondary


def test_calibrated_metric_depth_can_be_qualified_but_never_certified() -> None:
    depth, valid, K = _depth_inputs()
    result = estimate_depth_pixel_distance(
        depth=depth,
        depth_valid_mask=valid,
        intrinsics_matrix=K,
        pixel_a_xy=(40, 40),
        pixel_b_xy=(60, 40),
        depth_scale="metric",
        intrinsics_source="calibrated",
        calibration_status="calibrated",
    )
    assert result.state is InteractiveDistanceState.QUALIFIED
    assert result.distance_m == pytest.approx(0.4, rel=1e-5)
    assert result.certified_metrology is False


def test_relative_depth_keeps_native_distance_but_never_fabricates_metric_value() -> None:
    depth, valid, K = _depth_inputs()
    result = estimate_depth_pixel_distance(
        depth=depth,
        depth_valid_mask=valid,
        intrinsics_matrix=K,
        pixel_a_xy=(30, 40),
        pixel_b_xy=(70, 40),
        depth_scale=DepthScale.RELATIVE,
        intrinsics_source=IntrinsicsSource.ESTIMATED,
        calibration_status=CalibrationStatus.NOT_CALIBRATED,
    )
    assert result.state is InteractiveDistanceState.UNAVAILABLE
    assert result.distance_source_units == pytest.approx(0.8, rel=1e-5)
    assert result.distance_m is None
    assert result.expanded_uncertainty_m is None


def test_invalid_depth_click_rejects_without_metric_value() -> None:
    depth, valid, K = _depth_inputs()
    valid[:, :] = False
    result = estimate_depth_pixel_distance(
        depth=depth,
        depth_valid_mask=valid,
        intrinsics_matrix=K,
        pixel_a_xy=(30, 40),
        pixel_b_xy=(70, 40),
        depth_scale="metric",
        intrinsics_source="estimated",
        calibration_status="not_calibrated",
    )
    assert result.state is InteractiveDistanceState.REJECTED
    assert result.distance_m is None
    assert result.reasons


def test_provisional_canonical_scale_returns_estimated_metric_distance_plus_uncertainty() -> None:
    result = estimate_reconstruction_distance(
        point_a_reconstruction=(0.0, 0.0, 0.0),
        point_b_reconstruction=(2.0, 0.0, 0.0),
        unit_to_metre=0.5,
        relative_scale_uncertainty=0.04,
        scale_state="provisional",
        calibration_available=False,
    )
    assert result.state is InteractiveDistanceState.ESTIMATED
    assert result.distance_source_units == pytest.approx(2.0)
    assert result.distance_m == pytest.approx(1.0)
    assert result.expanded_uncertainty_m is not None and result.expanded_uncertainty_m >= 0.04
    assert result.certified_metrology is False


def test_arbitrary_canonical_scene_exposes_only_reconstruction_units() -> None:
    result = estimate_reconstruction_distance(
        point_a_reconstruction=(0.0, 0.0, 0.0),
        point_b_reconstruction=(3.0, 4.0, 0.0),
        unit_to_metre=None,
        relative_scale_uncertainty=None,
        scale_state="not_qualified",
        calibration_available=False,
    )
    assert result.state is InteractiveDistanceState.UNAVAILABLE
    assert result.distance_source_units == pytest.approx(5.0)
    assert result.distance_m is None
    primary, secondary = format_interactive_distance(result)
    assert primary == "Metric estimate unavailable"
    assert "5.0000 reconstruction units" in secondary


def test_interactive_contract_rejects_certified_metrology_claim() -> None:
    with pytest.raises(ValueError, match="certified metrology"):
        InteractiveDistanceEstimate(
            state=InteractiveDistanceState.ESTIMATED,
            source_kind="image_depth",
            point_a_source=(0.0, 0.0, 1.0),
            point_b_source=(1.0, 0.0, 1.0),
            distance_source_units=1.0,
            distance_m=1.0,
            expanded_uncertainty_m=0.1,
            uncertainty_basis="test",
            calibration_status="not_calibrated",
            scale_status="metric",
            certified_metrology=True,
        )
