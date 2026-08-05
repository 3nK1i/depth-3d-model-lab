from core.measurement import MeasurementKind, MeasurementStatus, compute_measurements, make_measurement_request
from tests.measurement_computation_helpers import make_chain

def test_measurement_perturbations_remain_finite_across_object_rotations():
    for angle in (0.0,0.2,0.5,0.9,1.2):
        b,e,f=make_chain(angle=angle)
        r=compute_measurements(make_measurement_request(b,e,f))
        for x in r.object_results[0].measurement_result.estimates:
            if x.status is MeasurementStatus.AVAILABLE:
                assert x.uncertainty.expanded_uncertainty_si>=0
        assert any(x.kind is MeasurementKind.CAMERA_DISTANCE and x.status is MeasurementStatus.AVAILABLE for x in r.object_results[0].measurement_result.estimates)
