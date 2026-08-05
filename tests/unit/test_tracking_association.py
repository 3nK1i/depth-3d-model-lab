from core.tracking.association import association_score,observation_geometry
from core.tracking.policy import AssociationPolicy
from tests.tracking_helpers import frame,observation

def test_same_object_shift_associates_using_mask_box_and_centre():
    f1,f2=frame(1),frame(2)
    previous=observation_geometry(observation(f1,'a',box=(20,20,50,50)))
    current=observation_geometry(observation(f2,'b',box=(23,20,53,50)))
    score=association_score(previous,current,AssociationPolicy())
    assert score is not None and score.score>.6 and score.mask_iou is not None

def test_different_classes_never_associate():
    previous=observation_geometry(observation(frame(1),'a',class_id=1))
    current=observation_geometry(observation(frame(2),'b',class_id=2))
    assert association_score(previous,current,AssociationPolicy()) is None

def test_far_box_below_threshold_is_rejected():
    previous=observation_geometry(observation(frame(1),'a',box=(0,0,10,10)))
    current=observation_geometry(observation(frame(2),'b',box=(85,65,95,75)))
    assert association_score(previous,current,AssociationPolicy(minimum_score=.5)) is None
