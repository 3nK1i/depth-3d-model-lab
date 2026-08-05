from core.measurement import MeasurementKind, MeasurementStatus, compute_measurements, make_measurement_request
from tests.measurement_computation_helpers import make_chain

def test_step4_step5_step6_contract_chain():
    bundle, extraction, fitting = make_chain()
    result = compute_measurements(make_measurement_request(bundle, extraction, fitting))
    estimates=result.object_results[0].measurement_result.estimates
    assert any(x.kind is MeasurementKind.WIDTH and x.status is MeasurementStatus.AVAILABLE for x in estimates)
    assert all(x.model_provenance for x in estimates if x.status is MeasurementStatus.AVAILABLE)
