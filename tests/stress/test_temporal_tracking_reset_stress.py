from core.measurement.contracts import MeasurementKind
from core.tracking import TemporalTrackingEngine,make_tracking_request
from tests.tracking_helpers import frame,measurement_batch,observation,perception

def test_repeated_context_resets_do_not_accumulate_tracks():
    engine=TemporalTrackingEngine()
    for fid in range(1,101):
        f=frame(fid,session=fid//10,size=(100+(fid%2)*10,80)); oid=f'o-{fid}'
        engine.update(make_tracking_request(perception(f,[observation(f,oid)]),measurement_batch(f,{oid:{MeasurementKind.WIDTH:(1.0,.01)}}),depth_model_id=f'depth-{fid%3}',perception_model_id='seg',worker_generation=fid%4))
        assert engine.active_track_count<=1
