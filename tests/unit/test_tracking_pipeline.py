from core.measurement.contracts import MeasurementKind
from core.tracking import TemporalEstimateStatus,TemporalTrackingEngine,TemporalTrackingPolicy,TrackingResetReason,TrackingState,make_tracking_request
from tests.tracking_helpers import frame,measurement_batch,observation,perception

def _update(engine,frame_id,x=20,width=1.0,*,session=1,size=(100,80),depth='depth',worker=0,obs_id=None):
    f=frame(frame_id,session=session,size=size); oid=obs_id or f'obs-{frame_id}'
    p=perception(f,[observation(f,oid,box=(x,20,x+30,50))])
    m=measurement_batch(f,{oid:{MeasurementKind.WIDTH:(width,.02),MeasurementKind.CAMERA_DISTANCE:(2.0,.03)}})
    return engine.update(make_tracking_request(p,m,depth_model_id=depth,perception_model_id='seg',worker_generation=worker))

def test_track_id_persists_and_becomes_confirmed_then_stable():
    engine=TemporalTrackingEngine()
    r1=_update(engine,1); r2=_update(engine,2,x=22,width=1.02); r3=_update(engine,3,x=24,width=.99)
    track_ids=[r.tracked_objects[0].track_state.track_id for r in (r1,r2,r3)]
    assert len(set(track_ids))==1
    assert r1.tracked_objects[0].track_state.state is TrackingState.TENTATIVE
    assert r2.tracked_objects[0].track_state.state is TrackingState.CONFIRMED
    width=next(v for v in r3.tracked_objects[0].stabilized_measurements if v.kind is MeasurementKind.WIDTH)
    assert width.status is TemporalEstimateStatus.STABLE

def test_track_is_lost_reacquired_and_then_confirmed():
    engine=TemporalTrackingEngine(TemporalTrackingPolicy(maximum_missed_frames=2))
    _update(engine,1); _update(engine,2)
    f3=frame(3); empty=engine.update(make_tracking_request(perception(f3,[]),measurement_batch(f3,{}),depth_model_id='depth',perception_model_id='seg'))
    assert empty.tracked_objects[0].track_state.state is TrackingState.LOST
    reacquired=_update(engine,4,x=21)
    assert reacquired.tracked_objects[0].track_state.state is TrackingState.REACQUIRED
    assert reacquired.tracked_objects[0].track_state.reacquisition_count==1
    confirmed=_update(engine,5,x=22)
    assert confirmed.tracked_objects[0].track_state.state is TrackingState.CONFIRMED

def test_track_terminates_after_missed_limit():
    engine=TemporalTrackingEngine(TemporalTrackingPolicy(maximum_missed_frames=1))
    _update(engine,1); _update(engine,2)
    for fid in (3,4):
        f=frame(fid); result=engine.update(make_tracking_request(perception(f,[]),measurement_batch(f,{}),depth_model_id='depth',perception_model_id='seg'))
    assert result.tracked_objects[0].track_state.state is TrackingState.TERMINATED
    assert engine.active_track_count==0

def test_model_worker_and_resolution_changes_reset_history():
    engine=TemporalTrackingEngine(); _update(engine,1); _update(engine,2)
    changed=_update(engine,3,depth='other-depth',worker=1,size=(120,80))
    reasons=set(changed.reset_events[0].reasons)
    assert {TrackingResetReason.DEPTH_MODEL_CHANGED,TrackingResetReason.WORKER_GENERATION_CHANGED,TrackingResetReason.RESOLUTION_CHANGED} <= reasons
    assert changed.tracked_objects[0].track_state.state is TrackingState.TENTATIVE
    assert changed.tracked_objects[0].track_state.measurement_history_count==1

def test_multiple_same_class_objects_keep_distinct_ids():
    engine=TemporalTrackingEngine()
    for fid in (1,2,3):
        f=frame(fid); observations=(observation(f,f'a-{fid}',box=(10+fid,10,30+fid,30)),observation(f,f'b-{fid}',box=(60-fid,40,80-fid,60)))
        values={o.observation_id:{MeasurementKind.WIDTH:(.5,.01)} for o in observations}
        result=engine.update(make_tracking_request(perception(f,observations),measurement_batch(f,values),depth_model_id='depth',perception_model_id='seg'))
    assert len(result.tracked_objects)==2
    assert len({r.track_state.track_id for r in result.tracked_objects})==2

def test_source_session_perception_and_alignment_changes_reset():
    engine=TemporalTrackingEngine(); _update(engine,1); _update(engine,2)
    f=frame(1,session=2); oid='new-session'
    result=engine.update(make_tracking_request(
        perception(f,[observation(f,oid)]),
        measurement_batch(f,{oid:{MeasurementKind.WIDTH:(1.0,.02)}}),
        depth_model_id='depth',perception_model_id='other-seg',alignment_config_digest='new-align',
    ))
    reasons=set(result.reset_events[0].reasons)
    assert {TrackingResetReason.SOURCE_SESSION_CHANGED,TrackingResetReason.PERCEPTION_MODEL_CHANGED,TrackingResetReason.ALIGNMENT_CONFIGURATION_CHANGED} <= reasons


def test_stable_history_survives_one_unavailable_measurement_frame():
    engine=TemporalTrackingEngine(); _update(engine,1); _update(engine,2); _update(engine,3)
    f=frame(4); oid='obs-4'
    result=engine.update(make_tracking_request(
        perception(f,[observation(f,oid,box=(24,20,54,50))]),
        measurement_batch(f,{}),depth_model_id='depth',perception_model_id='seg',
    ))
    width=next(v for v in result.tracked_objects[0].stabilized_measurements if v.kind is MeasurementKind.WIDTH)
    assert width.status is TemporalEstimateStatus.STABLE
    assert width.source_status.value=='unavailable'
