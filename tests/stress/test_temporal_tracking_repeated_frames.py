from core.measurement.contracts import MeasurementKind
from core.tracking import TemporalTrackingEngine,make_tracking_request
from tests.tracking_helpers import frame,measurement_batch,observation,perception

def test_repeated_frames_keep_bounded_history():
    engine=TemporalTrackingEngine()
    for fid in range(1,301):
        f=frame(fid); oid=f'o-{fid}'
        result=engine.update(make_tracking_request(perception(f,[observation(f,oid,box=(20+(fid%3),20,50+(fid%3),50))]),measurement_batch(f,{oid:{MeasurementKind.WIDTH:(1+(fid%5)*.001,.01)}}),depth_model_id='depth',perception_model_id='seg'))
    assert engine.active_track_count==1
    assert result.tracked_objects[0].track_state.measurement_history_count<=15
