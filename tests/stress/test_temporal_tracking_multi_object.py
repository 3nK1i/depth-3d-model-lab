from core.measurement.contracts import MeasurementKind
from core.tracking import TemporalTrackingEngine,make_tracking_request
from tests.tracking_helpers import frame,measurement_batch,observation,perception

def test_many_objects_remain_distinct():
    engine=TemporalTrackingEngine()
    for fid in range(1,31):
        f=frame(fid,size=(400,200)); observations=[]
        for index in range(12): observations.append(observation(f,f'{index}-{fid}',class_id=index%3,box=(5+index*30,20+(index%2)*60,25+index*30,50+(index%2)*60)))
        values={o.observation_id:{MeasurementKind.WIDTH:(.5+o.class_id*.1,.01)} for o in observations}
        result=engine.update(make_tracking_request(perception(f,observations),measurement_batch(f,values),depth_model_id='depth',perception_model_id='seg'))
    assert len(result.tracked_objects)==12
