from core.measurement import compute_measurements, make_measurement_request
from tests.measurement_computation_helpers import make_chain

def test_measurement_selected_object_path_repeatedly():
    b,e,f=make_chain()
    for _ in range(25):
        r=compute_measurements(make_measurement_request(b,e,f,observation_ids=('object-0',)))
        assert len(r.object_results)==1
