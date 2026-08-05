import pickle, numpy as np, pytest
from core.contracts.common import FrameReference, ImageSize
from core.contracts.spatial import PixelCoordinateSpace, PixelSpaceRole
from core.perception.messages import FramePerceptionRequest, PerceptionPipelineRequest, PerceptionWorkerConfiguration
from core.contracts.validation import ContractValidationError
from core.perception.result import DenseMaskPayload
from core.perception.metadata import PerceptionTask

def test_worker_configuration_and_request_are_pickle_safe():
    c = PerceptionWorkerConfiguration('development-perception-mock', (('emit_native_masks', True),))
    f = FrameReference(1, 1, 0, ImageSize(4, 3))
    s = PixelCoordinateSpace('rgb', f, f.source_size, PixelSpaceRole.RGB)
    r = PerceptionPipelineRequest(1, FramePerceptionRequest(f, np.zeros((3, 4, 3), np.uint8), s, (PerceptionTask.DETECTION,)))
    assert pickle.loads(pickle.dumps(c)) == c
    assert pickle.loads(pickle.dumps(r)).request_id == 1

def test_pickled_arrays_restore_read_only_contract():
    f = FrameReference(1, 2, 0, ImageSize(4, 3))
    s = PixelCoordinateSpace('rgb', f, f.source_size, PixelSpaceRole.RGB)
    req = FramePerceptionRequest(f, np.zeros((3, 4, 3), np.uint8), s, (PerceptionTask.DETECTION,))
    restored = pickle.loads(pickle.dumps(req))
    assert not restored.rgb.flags.writeable
    payload = pickle.loads(pickle.dumps(DenseMaskPayload(np.ones((3, 4), bool))))
    assert not payload.data.flags.writeable

def test_refiner_configuration_requires_explicit_safe_configuration():
    with pytest.raises(ContractValidationError):
        PerceptionWorkerConfiguration('development-perception-mock', refiner_config=(('checkpoint', 'x'),))
    with pytest.raises(ContractValidationError):
        PerceptionWorkerConfiguration('development-perception-mock', refiner_model_id='ultralytics-sam2-prompted', enable_prompted_refinement=False)
    with pytest.raises(ContractValidationError):
        PerceptionWorkerConfiguration('development-perception-mock', allow_refiner_co_residency=True)
    c = PerceptionWorkerConfiguration('development-perception-mock', refiner_model_id='ultralytics-sam2-prompted', allow_refiner_co_residency=True)
    assert c.allow_refiner_co_residency
