from pathlib import Path
import numpy as np
from core.reconstruction.da3 import da3_opencv_w2c_to_reconstruction_from_camera, validate_da3_depth_confidence, validate_da3_intrinsics

def test_da3_pose_normalization_is_deterministic_under_repetition():
    mats=np.tile(np.eye(4)[None],(8,1,1)); mats[:,0,3]=np.linspace(0,1,8)
    first=da3_opencv_w2c_to_reconstruction_from_camera(mats[:,:3,:])
    for _ in range(200): np.testing.assert_allclose(da3_opencv_w2c_to_reconstruction_from_camera(mats[:,:3,:]),first)

def test_da3_array_validation_repeated_without_mutation():
    depth=np.ones((4,32,48),np.float32); conf=np.ones_like(depth); K=np.tile(np.array([[100,0,24],[0,100,16],[0,0,1]],float)[None],(4,1,1))
    for _ in range(100):
        d,c=validate_da3_depth_confidence(depth,conf); k=validate_da3_intrinsics(K,frame_count=4); assert not d.flags.writeable and not c.flags.writeable and not k.flags.writeable
