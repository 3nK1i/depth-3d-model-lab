from core.measurement.contracts import MeasurementKind
from core.tracking import TemporalEstimateStatus,TemporalTrackingEngine,TrackingState,make_tracking_request
from tests.tracking_helpers import frame,measurement_batch,observation,perception

def test_sequence_with_noise_outlier_loss_and_reacquisition():
    engine=TemporalTrackingEngine(); widths=[1.0,1.02,4.0,.99,1.01]
    results=[]
    for fid,width in enumerate(widths,1):
        f=frame(fid); oid=f'obs-{fid}'; results.append(engine.update(make_tracking_request(perception(f,[observation(f,oid,box=(20+fid,20,50+fid,50))]),measurement_batch(f,{oid:{MeasurementKind.WIDTH:(width,.02)}}),depth_model_id='depth',perception_model_id='seg')))
    stable=next(v for v in results[-1].tracked_objects[0].stabilized_measurements if v.kind is MeasurementKind.WIDTH)
    assert stable.status is TemporalEstimateStatus.STABLE and abs(stable.value_si-1.01)<.04
    f6=frame(6); lost=engine.update(make_tracking_request(perception(f6,[]),measurement_batch(f6,{}),depth_model_id='depth',perception_model_id='seg'))
    assert lost.tracked_objects[0].track_state.state is TrackingState.LOST
    f7=frame(7); reacquired=engine.update(make_tracking_request(perception(f7,[observation(f7,'obs-7',box=(27,20,57,50))]),measurement_batch(f7,{'obs-7':{MeasurementKind.WIDTH:(1.0,.02)}}),depth_model_id='depth',perception_model_id='seg'))
    assert reacquired.tracked_objects[0].track_state.state is TrackingState.REACQUIRED
