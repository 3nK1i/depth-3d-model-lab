from core.measurement import compute_measurements, make_measurement_request
from core.measurement.contracts import BoundingBoxKind
from core.measurement.policy import MeasurementComputationPolicy, MeasurementSelectionPolicy
from tests.measurement_computation_helpers import make_chain


def _compute(kind):
    bundle, extraction, fitting = make_chain(outliers=2)
    policy = MeasurementComputationPolicy(
        selection=MeasurementSelectionPolicy(forced_box_kind=kind)
    )
    return compute_measurements(make_measurement_request(bundle, extraction, fitting, policy=policy)).object_results[0]


def test_operator_can_force_camera_aligned_measurement_box():
    result = _compute(BoundingBoxKind.CAMERA_ALIGNED)
    assert "operator-selected camera_aligned" in result.diagnostics.selected_box_method
    assert result.diagnostics.selected_box_id.endswith("camera_aligned")
    assert all(estimate.method == "camera_percentile" for estimate in result.measurement_result.estimates if estimate.status.value == "available")


def test_operator_can_force_oriented_measurement_box():
    result = _compute(BoundingBoxKind.ORIENTED)
    assert "operator-selected oriented" in result.diagnostics.selected_box_method
    assert result.diagnostics.selected_box_id.endswith("oriented")
