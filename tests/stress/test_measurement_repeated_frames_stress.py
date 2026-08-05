from core.measurement import compute_measurements, make_measurement_request
from tests.measurement_computation_helpers import make_chain

def test_measurement_repeated_computation_is_deterministic():
    b,e,f=make_chain()
    values=[]
    for _ in range(20):
        r=compute_measurements(make_measurement_request(b,e,f))
        values.append(tuple((x.kind,x.status,x.value_si,x.uncertainty.expanded_uncertainty_si if x.uncertainty else None) for x in r.object_results[0].measurement_result.estimates))
    assert all(v==values[0] for v in values)
